# A base script for prototyping unlearning methods.
# Uses Cas's circuit breakers implementation with DDP enabled.

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import torch.distributed as dist
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

from unlearn.utils.hook import ActivationCapture, resolve_layer_names
from unlearn.utils.keyword_masks import apply_keyword_masks
from unlearn.utils.muon import MuonAdamW
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer, save_checkpoint


class UnlearningTrainer(Trainer):
    def __init__(
        self,
        run_args,
        model: PreTrainedModel,
        args,
        train_dataset,
        tokenizer,
        target_layers,
        use_lora: bool = True,
        use_muon: bool = False,
        reference_model=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        self.run_args = run_args
        self.num_training_steps = self.args.max_steps
        self.current_training_step = 0
        self.tokenizer = tokenizer
        self.target_layers = target_layers
        self.model = model
        self.retain_coef = self.run_args.retain_coef
        self.remove_coef = self.run_args.remove_coef
        self.orth_coef = self.run_args.orth_coef
        self.trainer_tokenizer = tokenizer
        self.use_lora = use_lora
        self.use_muon = use_muon
        self.reference_model = reference_model
        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False

        self.layer_id_to_name = resolve_layer_names(model, target_layers)
        self.target_module_names = list(self.layer_id_to_name.values())
        if self.reference_model is not None:
            self.reference_layer_id_to_name = resolve_layer_names(
                self.reference_model, target_layers
            )
            self.reference_target_module_names = list(
                self.reference_layer_id_to_name.values()
            )

    def create_optimizer(self, model=None):
        if self.use_muon:
            self.optimizer = MuonAdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer
        return super().create_optimizer()

    def _prepare_for_training(
        self, max_steps, train_dataloader, resume_from_checkpoint
    ):
        """Skip accelerate model wrapping when FSDP2 has already been applied."""
        if self.run_args.use_fsdp2 and not self.use_lora:
            self.create_optimizer()
            self.optimizer = self.accelerator.prepare(self.optimizer)
            self.create_scheduler(num_training_steps=max_steps)
            self.model_wrapped = self.model
            return self.model, train_dataloader
        return super()._prepare_for_training(
            max_steps, train_dataloader, resume_from_checkpoint
        )

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


class RRTrainer(UnlearningTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.use_lora and self.reference_model is None:
            self._orig_param_data = {
                name: param.data.clone()
                for name, param in self.model.named_parameters()
            }

    @contextmanager
    def _original_params(self, model):
        """Swap model parameters to their pre-training values for the reference pass."""
        saved = {}
        for name, param in model.named_parameters():
            saved[name] = param.data
            param.data = self._orig_param_data[name]
        yield
        for name, param in model.named_parameters():
            param.data = saved[name]

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        unwrapped_model = unwrap_model(model)

        target_device = (
            inputs["input_ids"].device
            if hasattr(inputs["input_ids"], "device")
            else unwrapped_model.device
        )

        # === retain ===
        retain_input_ids = inputs.get("input_ids").to(target_device)  # type: ignore
        retain_attention_mask = inputs.get("attention_mask").to(target_device)  # type: ignore
        # ==== cb ====
        circuit_breaker_input_ids = inputs.get("bio_remove_input_ids").to(target_device)  # type: ignore
        circuit_breaker_attention_mask = inputs.get("bio_remove_attention_mask").to(  # type: ignore
            target_device  # type: ignore
        )
        if "bio_remove_keyword_mask" in inputs:
            keyword_mask = inputs["bio_remove_keyword_mask"].to(target_device)
            circuit_breaker_attention_mask = (
                circuit_breaker_attention_mask & keyword_mask.bool()
            )

        # ==== Inputs ====
        retain_inputs_dict = dict(
            input_ids=retain_input_ids,
            attention_mask=retain_attention_mask,
            output_hidden_states=False,
        )
        cb_inputs_dict = dict(
            input_ids=circuit_breaker_input_ids,
            attention_mask=circuit_breaker_attention_mask,
            output_hidden_states=False,
        )

        # ===== Step Coeff ====
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        scheduled_coeff = min(
            [
                1.0,
                self.current_training_step
                / (
                    self.run_args.num_train_examples / (self.run_args.pdbs * world_size)
                ),
            ]
        )
        if self.run_args.retain_warmup:
            retain_coeff = self.retain_coef * (0.1 + 0.9 * scheduled_coeff)
        else:
            retain_coeff = self.retain_coef * scheduled_coeff
        circuit_breaker_coeff = self.remove_coef * (1 - 0.25 * scheduled_coeff)
        orth_coeff = self.orth_coef * scheduled_coeff

        retain_mask = retain_attention_mask.unsqueeze(-1)

        # Initialize Hook Manager
        # We use unwrapped_model here to ensure names match what we resolved in __init__
        capturer = ActivationCapture(unwrapped_model, self.target_module_names)

        # --- Forward Pass 1: Reference (No Adapter) ---
        if self.reference_model is not None:
            ref_capturer = ActivationCapture(
                self.reference_model, self.reference_target_module_names
            )
            ref_capturer.register()
            with torch.no_grad():
                if retain_coeff > 0:
                    self.reference_model(**retain_inputs_dict)
                    orig_retain_acts = {}
                    for l in self.target_layers:
                        ref_name = self.reference_layer_id_to_name[l]
                        orig_retain_acts[l] = (
                            ref_capturer.activations[ref_name].detach() * retain_mask
                        )
                ref_capturer.clear()

                if circuit_breaker_coeff > 0:
                    self.reference_model(**cb_inputs_dict)
                    orig_cb_acts = {}
                    for l in self.target_layers:
                        ref_name = self.reference_layer_id_to_name[l]
                        orig_cb_acts[l] = ref_capturer.activations[ref_name].detach()
            ref_capturer.remove()
        else:
            ref_ctx = (
                unwrapped_model.disable_adapter()
                if self.use_lora
                else self._original_params(unwrapped_model)
            )  # type: ignore
            with ref_ctx:
                unwrapped_model.eval()
                capturer.register()  # Attach hooks

                with torch.no_grad():
                    ### Retain control
                    if retain_coeff > 0:
                        model(**retain_inputs_dict)
                        orig_retain_acts = {}
                        for l in self.target_layers:
                            name = self.layer_id_to_name[l]
                            orig_retain_acts[l] = (
                                capturer.activations[name].detach() * retain_mask
                            )

                    capturer.clear()

                    ### Circuit Breaker control
                    if circuit_breaker_coeff > 0:
                        model(**cb_inputs_dict)
                        orig_cb_acts = {}
                        for l in self.target_layers:
                            name = self.layer_id_to_name[l]
                            orig_cb_acts[l] = capturer.activations[name].detach()

                capturer.remove()  # Remove hooks

        unwrapped_model.train()

        # --- Forward Pass 2: Training (With Adapter) ---

        # Re-register hooks for the training pass
        capturer.register()

        ### Retain control
        if retain_coeff > 0:
            model(**retain_inputs_dict)

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

        ### Circuit Breaker control
        if circuit_breaker_coeff > 0:
            model(**cb_inputs_dict)

            denom = circuit_breaker_attention_mask.sum() * len(self.target_layers)
            cb_loss_total = torch.tensor(0.0, device=target_device)

            cb_mask = circuit_breaker_attention_mask.unsqueeze(-1)  # [B, S, 1]
            if self.orth_coef > 0:
                cb_mask_sum = circuit_breaker_attention_mask.sum(
                    dim=1, keepdim=True
                ).clamp(min=1.0)
                batch_size = circuit_breaker_input_ids.shape[0]
                diag_mask = 1.0 - torch.eye(
                    batch_size, device=target_device, dtype=torch.float
                )
                orth_loss_total = torch.tensor(0.0, device=target_device)

            for l in self.target_layers:
                name = self.layer_id_to_name[l]
                lora_h = capturer.activations[name]
                ref_h = orig_cb_acts[l]

                norm_lora = lora_h / torch.norm(
                    lora_h, dim=-1, keepdim=True, dtype=torch.float
                )
                norm_ref = ref_h / torch.norm(
                    ref_h, dim=-1, keepdim=True, dtype=torch.float
                )
                cos_sim = (norm_lora * norm_ref).sum(dim=-1)
                cos_sim = cos_sim * circuit_breaker_attention_mask
                cb_loss_total = cb_loss_total + torch.relu(cos_sim).sum()

                if self.orth_coef > 0:
                    masked = lora_h * cb_mask
                    pooled = masked.sum(dim=1) / cb_mask_sum
                    pooled_n = pooled / (
                        torch.norm(pooled, dim=-1, keepdim=True, dtype=torch.float)
                        + 1e-8
                    )
                    sim = pooled_n @ pooled_n.T
                    off_diag = sim * diag_mask
                    orth_loss_total = orth_loss_total + torch.relu(off_diag).sum()

            circuit_breaker_loss = cb_loss_total / (denom + 1e-6)

            if self.orth_coef > 0:
                num_pairs = batch_size * (batch_size - 1) * len(self.target_layers)
                mean_seq_len = cb_mask_sum.mean()
                orth_loss = orth_loss_total / (num_pairs + 1e-6) * mean_seq_len
            else:
                orth_loss = torch.tensor(0.0, device=target_device)
        else:
            circuit_breaker_loss = torch.tensor(0.0, device=target_device)
            orth_loss = torch.tensor(0.0, device=target_device)

        capturer.remove()  # Clean up

        loss = (
            retain_coeff * retain_loss
            + circuit_breaker_coeff * circuit_breaker_loss
            + orth_coeff * orth_loss
        )

        log_now = self.current_training_step == 0 or (
            (self.current_training_step + 1) % self.args.gradient_accumulation_steps
            == 0
        )
        if log_now:
            self.log(
                {
                    "retain_loss": retain_loss.detach().item(),
                    "cb_loss": circuit_breaker_loss.detach().item(),
                    "orth_loss": orth_loss.detach().item(),
                    "retain_coeff": retain_coeff,
                    "cb_coeff": circuit_breaker_coeff,
                    "orth_coeff": orth_coeff,
                }
            )

        if (
            self.current_training_step % 32 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            print(
                f"retain_coeff: {retain_coeff:.4f} || cb_coeff: "
                f"{circuit_breaker_coeff:.4f} || orth_coeff: {orth_coeff:.4f} || "
                f"retain_loss: {retain_loss:.4f} || "
                f"cb_loss: {circuit_breaker_loss:.4f} "
                f"|| orth_loss: {orth_loss:.4f}"
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


@dataclass
class OrthCircuitBreakerConfig:
    num_train_examples: int = 0
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    lr: float = 1e-3
    pdbs: int = 4
    alg: Literal["rr", "lat", "rr-lat"] = "rr"
    retain_coef: float = 2.0
    remove_coef: float = 23.0
    orth_coef: float = 10.0
    lora_r: int = 16
    adv_lr: float = 2e-3
    attack_iters: int = 8
    lora: bool = True
    layers: list[int] = field(default_factory=lambda: list(range(32)))
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    blocklist_path: str = ""
    keyword_mask_method: Literal["regex", "activation", "sae", "probe"] = "regex"
    activation_mask_threshold: float = 0.2
    activation_mask_layer: int = 16
    sae_latents_path: str = ""
    sae_mask_frac: float = 0.115
    probe_mask_path: str = ""
    probe_mask_layer: int = 11
    probe_mask_frac: float = 0.105
    revision: str = "main"
    hidden_dim: int = 4096
    lora_target: Literal["attn", "mlp", "all"] = "all"
    lora_all_layers: bool = False
    exclude_lora_layers: list[int] = field(default_factory=list)
    optimizer: Literal["adamw", "muon"] = "adamw"
    max_steps: int = -1
    num_train_epochs: int = 1
    dtype: Literal["bf16", "fp16"] = "bf16"
    retain_warmup: bool = False
    use_ultrachat: bool = False
    use_fsdp2: bool = True


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    parser = ArgumentParser()
    parser.add_arguments(OrthCircuitBreakerConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"{arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision, dtype=run_cfg.dtype
    )

    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)
    train_dataset.tokenized_bio_remove_dataset = apply_keyword_masks(
        train_dataset.tokenized_bio_remove_dataset, run_cfg, tokenizer
    )

    if run_cfg.lora_all_layers:
        num_layers = model.config.num_hidden_layers
        lora_layers_to_transform = list(range(num_layers))
    else:
        lora_layers_to_transform = [i for i in range(max(run_cfg.layers) + 1)]

    if run_cfg.exclude_lora_layers:
        lora_layers_to_transform = [
            l for l in lora_layers_to_transform if l not in run_cfg.exclude_lora_layers
        ]
        print(f"Excluding LoRA layers: {run_cfg.exclude_lora_layers}")

    # GPT-NeoX style
    attn_modules = ["query_key_value", "dense"]
    mlp_modules = ["dense_h_to_4h", "dense_4h_to_h"]

    if run_cfg.lora_target == "attn":
        target_modules = attn_modules
    elif run_cfg.lora_target == "mlp":
        target_modules = mlp_modules
    else:  # "all"
        target_modules = attn_modules + mlp_modules

    if run_cfg.lora:
        # Use rank-stabilized LoRA scaling (alpha=r)
        # so effective LR doesn't depend on rank
        lora_config = LoraConfig(
            r=run_cfg.lora_r,
            lora_alpha=run_cfg.lora_r,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            layers_to_transform=lora_layers_to_transform,
            task_type="CAUSAL_LM",
        )
        print(
            f"LoRA config: r={run_cfg.lora_r}, alpha={run_cfg.lora_r}, "
            f"target={run_cfg.lora_target}, "
            f"modules={target_modules}, layers={len(lora_layers_to_transform)}"
        )

        model = get_peft_model(model, lora_config)
    model = cast(PreTrainedModel, model)
    model.enable_input_require_grads()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = 32
    grad_acc_steps = max(1, global_batch_size // (run_cfg.pdbs * world_size))

    use_fsdp2 = run_cfg.use_fsdp2 and not run_cfg.lora and world_size > 1
    if use_fsdp2:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        for layer in model.gpt_neox.layers:
            fully_shard(layer)
        fully_shard(model)
        print(f"FSDP2 applied: {sum(1 for _ in model.parameters())} params as DTensors")

    reference_model = None
    if use_fsdp2:
        reference_torch_dtype = (
            torch.float16 if run_cfg.dtype == "fp16" else torch.bfloat16
        )
        reference_model = AutoModelForCausalLM.from_pretrained(
            run_cfg.model_name,
            revision=run_cfg.revision,
            torch_dtype=reference_torch_dtype,
            device_map={"": local_rank},
            use_cache=False,
        )
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        print("Loaded frozen reference model.")

    print(
        f"Running with {world_size} GPUs. Per device batch: {run_cfg.pdbs}. "
        f"Grad Acc steps: {grad_acc_steps}."
    )

    output_dir = run_cfg.save_path or "./results"
    use_muon = run_cfg.optimizer == "muon"
    if world_size > 1 and not run_cfg.lora and not use_fsdp2:
        raise ValueError("Distributed full-rank cb runs must use FSDP2.")

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=run_cfg.lr,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=run_cfg.pdbs,
        per_device_eval_batch_size=run_cfg.pdbs,
        max_steps=run_cfg.max_steps if run_cfg.max_steps > 0 else -1,
        num_train_epochs=run_cfg.num_train_epochs,
        weight_decay=0.01,
        gradient_checkpointing=run_cfg.lora and not use_fsdp2,
        fp16=run_cfg.dtype == "fp16",
        bf16=run_cfg.dtype == "bf16",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        report_to=["wandb"],
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
    )

    trainer = RRTrainer(
        run_cfg,
        model,
        training_args,
        train_dataset,
        tokenizer,
        run_cfg.layers,
        use_lora=run_cfg.lora,
        use_muon=use_muon,
        reference_model=reference_model,
    )

    model.train()
    trainer.train()

    if run_cfg.lora:
        print("\nMerging LoRA weights...")
        model = model.merge_and_unload()  # type: ignore
        trainer.model = model

    if run_cfg.save_path:
        if use_fsdp2:
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
        else:
            save_checkpoint(trainer, run_cfg.save_path, tokenizer)

    print("Done :)")
