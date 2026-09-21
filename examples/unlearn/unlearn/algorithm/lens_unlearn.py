"""Unlearning via entropy maximization at frozen tuned lens layers.

Supports LoRA (DDP) and SFT (FSDP2) with AdamW or Muon optimizer.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module
from peft import LoraConfig, get_peft_model
from simple_parsing import ArgumentParser
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import seed_worker
from tuned_lens import TunedLens

from unlearn.utils.hook import ActivationCapture, resolve_layer_names
from unlearn.utils.keyword_masks import apply_keyword_masks
from unlearn.utils.muon import MuonAdamW
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer


def load_tuned_lenses(lens_path, model, device):
    lens = TunedLens.from_model(model, bias=True)
    lens_state_dict = torch.load(f"{lens_path}/params.pt", map_location=device)

    mapped_state_dict = {}
    num_layers = len(lens)
    for i in range(num_layers):
        src_weight_key = f"{i}.weight"
        src_bias_key = f"{i}.bias"
        dst_weight_key = f"layer_translators.{i}.weight"
        dst_bias_key = f"layer_translators.{i}.bias"
        if src_weight_key in lens_state_dict:
            mapped_state_dict[dst_weight_key] = lens_state_dict[src_weight_key]
        if src_bias_key in lens_state_dict:
            mapped_state_dict[dst_bias_key] = lens_state_dict[src_bias_key]

    current_state = lens.state_dict()
    for key in current_state:
        if key.startswith("unembed"):
            mapped_state_dict[key] = current_state[key]

    lens.load_state_dict(mapped_state_dict)

    for submodule in lens.modules():
        remove_hook_from_module(submodule)

    lens = lens.to(device=device, dtype=torch.bfloat16)
    lens.eval()
    for param in lens.parameters():
        param.requires_grad = False

    return lens


def _compute_forget_loss(
    lens,
    act_capturer,
    target_layers,
    layer_id_to_name,
    forget_attention_mask,
    target_device,
    dummy_loss=None,
):
    """Shared forget loss: entropy maximization via tuned lens."""
    layer_losses = []
    lens_device = next(lens.parameters()).device
    vocab_size = None
    for layer_idx in target_layers:
        name = layer_id_to_name[layer_idx]
        if name not in act_capturer.activations:
            continue

        hidden = act_capturer.activations[name]
        hidden_bf16 = hidden.to(device=lens_device, dtype=torch.bfloat16)

        lens_logits = lens(hidden_bf16, idx=layer_idx)
        batch_size, seq_len, vocab_size = lens_logits.shape

        random_targets = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=lens_logits.device
        )

        mask = forget_attention_mask.bool()
        logits_flat = lens_logits[mask]
        targets_flat = random_targets[mask]

        if logits_flat.numel() > 0:
            ce_loss = F.cross_entropy(
                logits_flat.float(), targets_flat, reduction="mean"
            )
            layer_losses.append(ce_loss)

    if vocab_size is None:
        vocab_size = 50257
    log_vocab = torch.log(torch.tensor(float(vocab_size), device=target_device))
    if layer_losses:
        mean_ce = torch.stack(layer_losses).mean()
        forget_loss = mean_ce / log_vocab
        mean_entropy = -mean_ce
    else:
        forget_loss = torch.tensor(0.0, device=target_device)
        mean_entropy = torch.tensor(0.0)

    if dummy_loss is not None:
        forget_loss = forget_loss + dummy_loss

    return forget_loss, mean_entropy


class LensUnlearningTrainer(Trainer):
    def __init__(
        self,
        run_args,
        model,
        args,
        train_dataset,
        tokenizer,
        target_layers,
        lens=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        self.run_args = run_args
        self.current_training_step = 0
        self.tokenizer = tokenizer
        self.target_layers = target_layers
        self.model = model
        self.retain_coef = run_args.retain_coef
        self.remove_coef = run_args.remove_coef
        self.lens = lens

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))  # type: ignore

    def _prepare_for_training(
        self, max_steps, train_dataloader, resume_from_checkpoint
    ):
        """Skip accelerate's model wrapping when FSDP2 is applied externally."""
        if not self.run_args.lora:
            self.create_optimizer()
            self.optimizer = self.accelerator.prepare(self.optimizer)
            self.create_scheduler(num_training_steps=max_steps)
            self.model_wrapped = self.model
            return self.model, train_dataloader
        return super()._prepare_for_training(
            max_steps, train_dataloader, resume_from_checkpoint
        )

    def _get_scheduled_coeffs(self):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        scheduled_coeff = min(
            1.0,
            self.current_training_step
            / (self.run_args.num_train_examples / (self.run_args.pdbs * world_size)),
        )
        retain_coeff = self.retain_coef * scheduled_coeff
        forget_coeff = self.remove_coef * (1 - 0.25 * scheduled_coeff)
        return retain_coeff, forget_coeff


class LensSFTTrainer(LensUnlearningTrainer):
    """SFT trainer with frozen reference model for retain loss."""

    def __init__(
        self,
        *args,
        reference_model=None,
        target_modules: list[str],
        model,
        use_muon=False,
        **kwargs,
    ):
        if use_muon:
            self.muon_param_names = {
                name
                for name, p in model.named_parameters()
                if p.ndim >= 2 and p.size(0) < 50000
            }

        super().__init__(*args, **kwargs)
        self.reference_model = reference_model
        self.target_modules = target_modules
        self.use_muon = use_muon

        self.layer_id_to_name = {int(m.split(".")[-1]): m for m in target_modules}

        self.act_capturer = ActivationCapture(
            model, target_modules, accelerator=self.accelerator
        )
        self.act_capturer.register()

        if reference_model is not None:
            self.ref_capturer = ActivationCapture(reference_model, target_modules)
            self.ref_capturer.register()

    def create_optimizer(self, model=None):
        if self.use_muon:
            self.optimizer = MuonAdamW(
                self.model.named_parameters(),
                lr=self.args.learning_rate,
                muon_momentum=self.run_args.muon_momentum,
                weight_decay=self.args.weight_decay,
                muon_param_names=self.muon_param_names,
            )
            return self.optimizer
        return super().create_optimizer()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        target_device = inputs["input_ids"].device

        retain_input_ids = inputs.get("input_ids").to(target_device)  # type: ignore
        forget_input_ids = inputs.get("bio_remove_input_ids").to(target_device)  # type: ignore
        forget_attention_mask = inputs.get("bio_remove_attention_mask").to(  # type: ignore
            target_device
        )

        retain_coeff, forget_coeff = self._get_scheduled_coeffs()

        model.train()

        # Retain loss
        if retain_coeff > 0 and self.reference_model is not None:
            retain_loss_type = getattr(self.run_args, "retain_loss_type", "kl")
            if retain_loss_type == "kl":
                with torch.no_grad():
                    ref_outputs = self.reference_model(
                        retain_input_ids, attention_mask=None
                    )
                    ref_logits = ref_outputs.logits.to(target_device)

                current_logits = model(
                    input_ids=retain_input_ids, attention_mask=None
                ).logits

                retain_loss = F.kl_div(
                    input=F.log_softmax(current_logits, dim=-1),
                    target=F.softmax(ref_logits, dim=-1),
                    reduction="batchmean",
                )
                self.act_capturer.clear()
            else:
                retain_attention_mask = inputs.get("attention_mask").to(target_device)  # type: ignore
                retain_mask = retain_attention_mask.unsqueeze(-1)

                with torch.no_grad():
                    self.reference_model(
                        input_ids=retain_input_ids,
                        attention_mask=retain_attention_mask,
                    )
                orig_retain_acts = {}
                for mod in self.target_modules:
                    if mod in self.ref_capturer.activations:
                        layer_idx = int(mod.split(".")[-1])
                        orig_retain_acts[layer_idx] = (
                            self.ref_capturer.activations[mod].detach() * retain_mask
                        )
                self.ref_capturer.clear()

                model(
                    input_ids=retain_input_ids,
                    attention_mask=retain_attention_mask,
                )

                n_layers = len(self.target_modules)
                retain_loss = torch.tensor(0.0, device=target_device)
                for mod in self.target_modules:
                    if mod in self.act_capturer.activations:
                        layer_idx = int(mod.split(".")[-1])
                        lora_h = self.act_capturer.activations[mod] * retain_mask
                        retain_loss = (
                            retain_loss
                            + torch.norm(
                                lora_h - orig_retain_acts[layer_idx],
                                dim=-1,
                                p=2,
                                dtype=torch.float,
                            ).nanmean()
                        )
                retain_loss = retain_loss / n_layers
                self.act_capturer.clear()
        else:
            retain_loss = torch.tensor(0.0, device=target_device)

        # Forget loss
        if forget_coeff > 0:
            outputs = model(
                input_ids=forget_input_ids,
                attention_mask=forget_attention_mask,
            )

            # Dummy loss connects to output for FSDP backward bookkeeping
            if hasattr(outputs, "logits"):
                dummy_loss = outputs.logits.mean() * 0.0
            elif isinstance(outputs, tuple):
                dummy_loss = outputs[0].mean() * 0.0
            else:
                dummy_loss = outputs.mean() * 0.0

            forget_loss, mean_entropy = _compute_forget_loss(
                self.lens,
                self.act_capturer,
                self.target_layers,
                self.layer_id_to_name,
                forget_attention_mask,
                target_device,
                dummy_loss=dummy_loss,
            )
            self.act_capturer.clear()
        else:
            forget_loss = torch.tensor(0.0, device=target_device)
            mean_entropy = torch.tensor(0.0)

        loss = retain_coeff * retain_loss + forget_coeff * forget_loss

        if (
            self.current_training_step % 32 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            entropy_val = (
                mean_entropy.item()
                if isinstance(mean_entropy, torch.Tensor)
                else mean_entropy
            )
            print(
                f"retain_coeff: {retain_coeff:.4f} || "
                f"forget_coeff: {forget_coeff:.4f} || "
                f"retain_loss: {retain_loss:.4f} || "
                f"forget_loss: {forget_loss:.4f} || "
                f"mean_entropy: {entropy_val:.4f}"
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


class LensLoRATrainer(LensUnlearningTrainer):
    """LoRA trainer using disable_adapter for retain loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_id_to_name = resolve_layer_names(self.model, self.target_layers)
        self.target_module_names = list(self.layer_id_to_name.values())

        if getattr(self.run_args, "update_coef", 0.0) > 0:
            self.initial_params: dict[str, torch.Tensor] = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.initial_params[name] = param.data.clone().cpu()
        else:
            self.initial_params = {}

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        unwrapped_model = unwrap_model(model)
        target_device = inputs["input_ids"].device

        retain_input_ids = inputs.get("input_ids").to(target_device)  # type: ignore
        retain_attention_mask = inputs.get("attention_mask").to(target_device)  # type: ignore
        cb_input_ids = inputs.get("bio_remove_input_ids").to(target_device)  # type: ignore
        cb_attention_mask = inputs.get("bio_remove_attention_mask").to(  # type: ignore
            target_device
        )

        retain_inputs_dict = dict(
            input_ids=retain_input_ids,
            attention_mask=retain_attention_mask,
            output_hidden_states=False,
        )
        cb_inputs_dict = dict(
            input_ids=cb_input_ids,
            attention_mask=cb_attention_mask,
            output_hidden_states=False,
        )

        retain_coeff, forget_coeff = self._get_scheduled_coeffs()

        retain_mask = retain_attention_mask.unsqueeze(-1)
        capturer = ActivationCapture(unwrapped_model, self.target_module_names)

        # Reference pass (adapter disabled)
        with unwrapped_model.disable_adapter():  # type: ignore
            unwrapped_model.eval()  # type: ignore
            capturer.register()
            with torch.no_grad():
                if retain_coeff > 0:
                    unwrapped_model(**retain_inputs_dict)
                    orig_retain_acts = {}
                    for l in self.target_layers:
                        name = self.layer_id_to_name[l]
                        orig_retain_acts[l] = (
                            capturer.activations[name].detach() * retain_mask
                        )
            capturer.remove()

        unwrapped_model.train()  # type: ignore

        # Training pass (adapter enabled)
        capturer.register()

        # Retain L2 loss
        if retain_coeff > 0:
            unwrapped_model(**retain_inputs_dict)
            n_layers = len(self.target_layers)
            retain_loss = torch.tensor(0.0, device=target_device)
            for l in self.target_layers:
                name = self.layer_id_to_name[l]
                lora_h = capturer.activations[name] * retain_mask
                retain_loss = (
                    retain_loss
                    + torch.norm(
                        lora_h - orig_retain_acts[l], dim=-1, p=2, dtype=torch.float
                    ).nanmean()
                )
            retain_loss = retain_loss / n_layers
            capturer.clear()
        else:
            retain_loss = torch.tensor(0.0, device=target_device)

        # Forget loss
        if forget_coeff > 0:
            unwrapped_model(**cb_inputs_dict)
            forget_loss, mean_entropy = _compute_forget_loss(
                self.lens,
                capturer,
                self.target_layers,
                self.layer_id_to_name,
                cb_attention_mask,
                target_device,
            )
        else:
            forget_loss = torch.tensor(0.0, device=target_device)
            mean_entropy = torch.tensor(0.0)

        capturer.remove()

        loss = retain_coeff * retain_loss + forget_coeff * forget_loss

        update_coef = getattr(self.run_args, "update_coef", 0.0)
        update_norm = torch.tensor(0.0, device=target_device)
        if update_coef > 0 and self.initial_params:
            for name, param in model.named_parameters():
                key = name.removeprefix("module.")
                if param.requires_grad and key in self.initial_params:
                    init_p = self.initial_params[key].to(param.device)
                    diff = param - init_p.detach()
                    update_norm = update_norm + diff.pow(2).sum()
            loss = loss - update_coef * update_norm

        if (
            self.current_training_step % 32 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            entropy_val = (
                mean_entropy.item()
                if isinstance(mean_entropy, torch.Tensor)
                else mean_entropy
            )
            log_parts = [
                f"retain_coeff: {retain_coeff:.4f}",
                f"forget_coeff: {forget_coeff:.4f}",
                f"retain_loss: {retain_loss:.4f}",
                f"forget_loss: {forget_loss:.4f}",
                f"mean_entropy: {entropy_val:.4f}",
            ]
            if update_coef > 0:
                log_parts.append(f"update_norm: {update_norm.item():.4f}")
            print(" || ".join(log_parts))

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


@dataclass
class LensUnlearnConfig:
    num_train_examples: int = 0
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    wmdp_eval_limit: int | None = None
    mmlu_agieval_limit: int | None = None
    lr: float = 1e-3
    pdbs: int = 4
    retain_coef: float = 5.0
    remove_coef: float = 5.0
    retain_loss_type: Literal["l2", "kl"] = "kl"
    lora_r: int = 16
    lora: bool = False
    layers: list[int] = field(default_factory=lambda: list(range(32)))
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    revision: str = "main"
    lens_path: str = ""
    epochs: int = 1
    update_coef: float = 0.0
    warmup_ratio: float = 0.0
    optimizer: Literal["adamw", "muon"] = "adamw"
    muon_momentum: float = 0.95
    dtype: Literal["bf16", "fp16"] = "bf16"
    use_ultrachat: bool = False


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    NUM_PROC = (os.cpu_count() or 32) // 2

    parser = ArgumentParser()
    parser.add_arguments(LensUnlearnConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    if "smollm2" in run_cfg.model_name:
        run_cfg.layers = [l for l in run_cfg.layers if l < 24]

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"  {arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision, dtype=run_cfg.dtype
    )
    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)
    train_dataset.tokenized_bio_remove_dataset = apply_keyword_masks(
        train_dataset.tokenized_bio_remove_dataset, run_cfg, tokenizer
    )

    # Load frozen tuned lens
    print(f"Loading tuned lens from: {run_cfg.lens_path}")
    device = next(model.parameters()).device
    lens = load_tuned_lenses(run_cfg.lens_path, model, device)
    print(f"Loaded lens with {len(lens)} layer translators (frozen)")

    if run_cfg.lora:
        # LoRA mode: DDP
        lora_layers_to_transform = list(range(max(run_cfg.layers) + 1))
        lora_config = LoraConfig(
            r=run_cfg.lora_r,
            lora_alpha=16,
            target_modules=(
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
                if "OLMo" in run_cfg.model_name
                else None
            ),
            lora_dropout=0.05,
            bias="none",
            layers_to_transform=lora_layers_to_transform,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # SFT mode: enable gradients on all params
        for param in model.parameters():
            param.requires_grad = True

    model = cast(PreTrainedModel, model)
    model.enable_input_require_grads()

    # Load frozen reference model for SFT retain loss
    reference_model = None
    if not run_cfg.lora and run_cfg.retain_coef > 0:
        print("Loading frozen reference model for retain loss (bf16)...")
        reference_model = AutoModelForCausalLM.from_pretrained(
            run_cfg.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
        )
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        print(f"Reference model loaded on GPU {local_rank}")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = 32
    grad_acc_steps = max(1, global_batch_size // (run_cfg.pdbs * world_size))

    print(
        f"Running with {world_size} GPUs. Per device batch: {run_cfg.pdbs}. "
        f"Grad Acc steps: {grad_acc_steps}."
    )

    # Apply FSDP2 for SFT mode
    if not run_cfg.lora:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        for layer in model.gpt_neox.layers:
            fully_shard(layer)
        fully_shard(model)
        print(f"FSDP2 applied: {sum(1 for _ in model.parameters())} params as DTensors")

    use_muon = run_cfg.optimizer == "muon"

    if run_cfg.lora:
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=run_cfg.lr,
            warmup_ratio=run_cfg.warmup_ratio,
            gradient_accumulation_steps=grad_acc_steps,
            per_device_train_batch_size=run_cfg.pdbs,
            per_device_eval_batch_size=run_cfg.pdbs,
            num_train_epochs=run_cfg.epochs,
            weight_decay=0.01,
            gradient_checkpointing=True,
            fp16=run_cfg.dtype == "fp16",
            bf16=run_cfg.dtype == "bf16",
            max_grad_norm=1.0,
            save_strategy="no",
            ddp_find_unused_parameters=False,
        )
    else:
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=run_cfg.lr,
            warmup_ratio=run_cfg.warmup_ratio,
            gradient_accumulation_steps=grad_acc_steps,
            per_device_train_batch_size=run_cfg.pdbs,
            per_device_eval_batch_size=run_cfg.pdbs,
            num_train_epochs=run_cfg.epochs,
            weight_decay=0.01,
            gradient_checkpointing=True,
            fp16=run_cfg.dtype == "fp16",
            bf16=run_cfg.dtype == "bf16",
            max_grad_norm=1.0,
            save_strategy="no",
            optim="adamw_torch",
            ddp_find_unused_parameters=False,
        )

    if run_cfg.lora:
        trainer = LensLoRATrainer(
            run_cfg,
            model,
            training_args,
            train_dataset,
            tokenizer,
            run_cfg.layers,
            lens=lens,
        )
    else:
        if use_muon:
            print(f"Using Muon optimizer (lr={run_cfg.lr})")

        target_modules = [f"gpt_neox.layers.{i}" for i in run_cfg.layers]
        trainer = LensSFTTrainer(
            run_cfg,
            model,
            training_args,
            train_dataset,
            tokenizer,
            run_cfg.layers,
            lens=lens,
            use_muon=use_muon,
            reference_model=reference_model,
            target_modules=target_modules,
            model=model,
        )

    model.train()
    trainer.train()

    if run_cfg.save_path:
        if run_cfg.lora:
            trainer.accelerator.wait_for_everyone()
            if trainer.accelerator.is_main_process:
                unwrapped = trainer.accelerator.unwrap_model(trainer.model)
                adapter_path = os.path.join(run_cfg.save_path, "adapter")
                unwrapped.save_pretrained(adapter_path, safe_serialization=True)
                tokenizer.save_pretrained(adapter_path)
                print(f"Saved adapter to {adapter_path}")

                merged_path = os.path.join(run_cfg.save_path, "merged")
                merged_model = unwrapped.merge_and_unload()
                merged_model.save_pretrained(merged_path, safe_serialization=True)
                tokenizer.save_pretrained(merged_path)
                print(f"Saved merged model to {merged_path}")
            trainer.accelerator.wait_for_everyone()
        else:
            state_dict = get_model_state_dict(
                model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            if local_rank == 0:
                os.makedirs(run_cfg.save_path, exist_ok=True)
                model.save_pretrained(
                    run_cfg.save_path,
                    state_dict=state_dict,
                    safe_serialization=True,
                )
                tokenizer.save_pretrained(run_cfg.save_path)
                config_path = os.path.join(run_cfg.save_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        config_dict = json.load(f)
                    if config_dict.get("dtype") is None:
                        config_dict["dtype"] = config_dict.get("torch_dtype", "float32")
                        with open(config_path, "w") as f:
                            json.dump(config_dict, f, indent=2)
            if dist.is_initialized():
                dist.barrier()

    print("Done")
