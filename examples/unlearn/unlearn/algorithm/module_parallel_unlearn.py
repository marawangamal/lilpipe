"""Module-parallel SFT unlearning. FSDP2. Only supports transformers that use nn.Linear.

Launch with ``torchrun --nproc_per_node=N``. Do not pass an outer FSDP config.
"""

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from simple_parsing import ArgumentParser
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import seed_worker

import wandb
from unlearn.utils.keyword_masks import apply_keyword_masks
from unlearn.utils.math import max_entropy_kl_loss
from unlearn.utils.muon import MuonAdamW
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.utils import assert_type
from unlearn.utils.worker_utils import get_model_and_tokenizer


def get_layer_list(model: PreTrainedModel) -> tuple[str, nn.ModuleList]:
    """Get the list of layers to train SAEs on."""
    N = assert_type(int, model.config.num_hidden_layers)
    candidates = [
        (name, mod)
        for (name, mod) in model.base_model.named_modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]


def get_target_modules(model: PreTrainedModel):
    N = model.config.num_hidden_layers
    layers = list(range(0, N))

    # Now convert layers to hookpoints
    layers_name, _ = get_layer_list(model)
    return [f"{layers_name}.{i}" for i in layers]


@contextmanager
def patch_weights(
    target_model: PreTrainedModel, target_module_name: str, source_module: nn.Linear
):
    target_module = dict(target_model.base_model.named_modules())[target_module_name]
    assert isinstance(target_module, nn.Linear)

    original_weight = target_module.weight
    original_bias = target_module.bias

    source_weight = assert_type(DTensor, source_module.weight)
    weight_local = source_weight.full_tensor().detach().requires_grad_(True)
    # We patch via the dict to prevent the model from realizing that we are
    # sneaking another computational graph into it
    target_module.__dict__["weight"] = weight_local

    if source_module.bias is not None:
        source_bias = assert_type(DTensor, source_module.bias)
        bias_local = source_bias.full_tensor().detach().requires_grad_(True)
        target_module.__dict__["bias"] = bias_local
    else:
        bias_local = None

    try:
        yield weight_local, bias_local
    finally:
        # Restore via __dict__ as well, then fix the Parameter registration
        target_module.__dict__.pop("weight", None)
        target_module.__dict__.pop("bias", None)
        target_module.weight = original_weight
        target_module.bias = original_bias


class ModuleParallelTrainer(Trainer):

    model: PreTrainedModel

    def __init__(
        self,
        run_args,
        model,
        args,
        train_dataset,
        tokenizer,
        frozen_model,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        self.run_args = run_args
        self.tokenizer = tokenizer
        self.frozen_model = frozen_model
        self.current_training_step = 0

        self.target_modules = get_target_modules(model)
        print("target_modules", self.target_modules)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(
            DataLoader(self.train_dataset, **dataloader_params)  # type: ignore
        )

    def _prepare_for_training(
        self, max_steps, train_dataloader, resume_from_checkpoint
    ):
        """Skip accelerate's model wrapping (FSDP2 applied externally)."""
        self.create_optimizer()
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.create_scheduler(num_training_steps=max_steps)
        self.model_wrapped = self.model
        return self.model, train_dataloader

    def create_optimizer(self, model=None):
        if self.run_args.optimizer == "muon":
            muon_param_names = {
                name
                for name, p in self.model.named_parameters()
                if p.ndim >= 2 and p.size(0) < 50000
            }

            self.optimizer = MuonAdamW(
                self.model.named_parameters(),
                lr=self.run_args.lr,
                muon_momentum=self.run_args.muon_momentum,
                weight_decay=self.args.weight_decay,
                muon_param_names=muon_param_names,
            )
            return self.optimizer

        result = super().create_optimizer()
        for group in self.optimizer.param_groups:
            group["fused"] = False
        self.optimizer.defaults["fused"] = False
        return result

    def _compute_retain_loss(self, model, inputs, target_device):
        """The KL divergence of the updated model's predictions on retain data
        from the original model's predictions."""
        retain_input_ids = inputs["input_ids"].to(target_device)
        retain_attention_mask = inputs["attention_mask"].to(target_device)
        with torch.no_grad():
            ref_outputs = self.frozen_model(retain_input_ids, attention_mask=None)
            ref_logits = ref_outputs.logits.to(target_device)

        current_logits = model(
            input_ids=retain_input_ids,
            attention_mask=retain_attention_mask,
        ).logits

        # Use log_target=True for numerical safety
        # Divide by seq_len (current_logits.size(1))
        # because 'batchmean' only averages over batch_size
        loss = F.kl_div(
            input=F.log_softmax(current_logits, dim=-1),
            target=F.log_softmax(ref_logits, dim=-1),
            reduction="batchmean",
            log_target=True,
        )

        return loss / current_logits.size(1)

    def _compute_nll_retain_loss(self, model, inputs, target_device):
        """Standard cross-entropy on retain tokens with teacher forcing."""
        retain_input_ids = inputs["input_ids"].to(target_device)

        logits = model(
            input_ids=retain_input_ids,
            attention_mask=None,
        ).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = retain_input_ids[:, 1:].contiguous()

        pad_token_id = self.tokenizer.pad_token_id  # type: ignore
        if pad_token_id is not None:
            mask = shift_labels != pad_token_id
            shift_logits = shift_logits[mask]
            shift_labels = shift_labels[mask]

        return F.cross_entropy(shift_logits.float(), shift_labels, reduction="mean")

    def _compute_l2_retain_loss(self, model, inputs, target_layers, target_device):
        """L2 distance between current and original hidden states,
        averaged over layers."""
        if isinstance(target_layers, int):
            target_layers = [target_layers]
        retain_input_ids = inputs["input_ids"].to(target_device)

        with torch.no_grad():
            ref_outputs = self.frozen_model(
                retain_input_ids, attention_mask=None, output_hidden_states=True
            )

        current_outputs = model(
            input_ids=retain_input_ids,
            attention_mask=None,
            output_hidden_states=True,
        )

        # hidden_states[L+1] is the output of layer L
        total_loss = torch.tensor(0.0, device=target_device)
        for layer in target_layers:
            ref_h = ref_outputs.hidden_states[layer + 1].to(target_device)
            cur_h = current_outputs.hidden_states[layer + 1]
            total_loss = (
                total_loss
                + torch.norm(cur_h - ref_h, dim=-1, p=2, dtype=torch.float).mean()
            )
        # Gradient anchor: the L2 loss only uses intermediate hidden states,
        # so lm_head/final_norm FSDP units don't receive gradients. With
        # gradient accumulation (no_sync), their _post_backward_called stays
        # True from the forget backward, causing an assertion on sync steps.
        # Adding a zero-valued logits term ensures all FSDP units participate.
        return total_loss / len(target_layers) + current_outputs.logits.sum() * 0

    def _compute_forget_loss(self, model, inputs, target_device):
        """Forget loss using the model's actual output logits.

        Gradient flows through the full model, but _freeze_and_log zeroes
        non-target-layer gradients so only the target layer is updated.
        """
        forget_input_ids = inputs["bio_remove_input_ids"].to(target_device)
        forget_attention_mask = inputs["bio_remove_attention_mask"].to(target_device)

        logits = model(
            input_ids=forget_input_ids,
            attention_mask=forget_attention_mask,
        ).logits

        mask = forget_attention_mask.bool()
        if "bio_remove_keyword_mask" in inputs:
            keyword_mask = inputs["bio_remove_keyword_mask"].bool().to(target_device)
            mask = mask & keyword_mask
        logits_masked = logits[mask]

        if logits_masked.numel() == 0:
            return logits.mean() * 0.0

        if self.run_args.use_dpo_forget:
            orig_device = next(self.frozen_model.parameters()).device
            with torch.no_grad():
                ref_logits = self.frozen_model(
                    forget_input_ids.to(orig_device),
                    attention_mask=forget_attention_mask.to(orig_device),
                ).logits.to(target_device)

            shift_labels = forget_input_ids[:, 1:]
            shift_mask = forget_attention_mask[:, 1:].bool()
            total_tokens = shift_mask.sum().clamp(min=1)
            beta = self.run_args.dpo_beta

            shift_current = logits[:, :-1, :].float()
            shift_ref = ref_logits[:, :-1, :].float()

            current_lp = F.log_softmax(shift_current, dim=-1)
            ref_lp = F.log_softmax(shift_ref, dim=-1)

            current_token_lp = current_lp.gather(
                -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            ref_token_lp = ref_lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

            if self.run_args.dpo_per_token:
                per_token_npo = -F.logsigmoid(-beta * (current_token_lp - ref_token_lp))
                loss = (per_token_npo * shift_mask).sum() / total_tokens
            else:
                seq_lengths = shift_mask.sum(dim=-1).clamp(min=1)
                token_log_ratios = (current_token_lp - ref_token_lp) * shift_mask
                avg_log_ratios = token_log_ratios.sum(dim=-1) / seq_lengths
                loss = -F.logsigmoid(-beta * avg_log_ratios).mean()

            return loss

        elif self.run_args.use_top_k_entropy:
            k = self.run_args.top_k
            log_k = torch.log(torch.tensor(float(k), device=target_device))
            topk_logits, _ = torch.topk(logits_masked, k, dim=-1)
            probs = F.softmax(topk_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            self._topk_entropy = entropy.item()
            self._topk_entropy_ratio = entropy.item() / log_k.item()
            return max_entropy_kl_loss(topk_logits) / log_k

        elif self.run_args.use_max_entropy_kl:
            vocab_size = logits.shape[-1]
            log_vocab = torch.log(torch.tensor(float(vocab_size), device=target_device))
            return max_entropy_kl_loss(logits_masked) / log_vocab
        else:
            vocab_size = logits.shape[-1]
            batch_size, seq_len = logits.shape[:2]
            random_targets = torch.randint(
                0, vocab_size, (batch_size, seq_len), device=logits.device
            )
            targets_flat = random_targets[mask]
            ce_loss = F.cross_entropy(
                logits_masked.float(), targets_flat, reduction="mean"
            )
            log_vocab = torch.log(torch.tensor(float(vocab_size), device=target_device))
            return ce_loss / log_vocab

    def compute_grad_norm(self, model, target_modules):
        """Compute L2 gradient norm for target_modules."""
        total_norm_sq = torch.tensor(0.0, device="cuda")
        for param in model.parameters():
            if param.grad is not None:
                g = param.grad.detach()
                if hasattr(g, "to_local"):
                    g = g.to_local()
                total_norm_sq += g.float().pow(2).sum()

        if dist.is_initialized():
            dist.all_reduce(total_norm_sq)

        return total_norm_sq.sqrt().item()

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        accumulated_grads = {
            name: param.grad.clone()
            for name, param in model.base_model.named_parameters()  # type: ignore
            if param.grad is not None
        }
        model.zero_grad(set_to_none=False)  # Isolate this micro-batch

        inputs = self._prepare_inputs(inputs)
        target_device = inputs["input_ids"].device

        amp_ctx = torch.amp.autocast(
            "cuda",
            dtype=torch.float16 if self.args.fp16 else torch.bfloat16,
            enabled=self.args.fp16 or self.args.bf16,
        )

        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Get scheduled loss coefficient
        scheduled_coeff = min(
            1.0,
            self.current_training_step
            / (self.run_args.num_train_examples / (self.run_args.pdbs * world_size)),
        )
        retain_coeff = self.run_args.retain_coef * (0.25 + 0.75 * scheduled_coeff)
        forget_coeff = self.run_args.remove_coef * (1 - 0.25 * scheduled_coeff)

        # Standard retain loss
        with amp_ctx:
            assert self.run_args.retain_loss_type in ["nll", "kl"]
            match self.run_args.retain_loss_type:
                case "nll":
                    retain_loss = self._compute_nll_retain_loss(
                        model, inputs, target_device
                    )
                case "kl":
                    retain_loss = self._compute_retain_loss(
                        model, inputs, target_device
                    )
                case _:
                    assert False

        self.accelerator.backward(retain_coeff * retain_loss)
        retain_grad_norm = self.compute_grad_norm(model, self.target_modules)
        retain_grads = {
            name: param.grad.clone()
            for name, param in model.base_model.named_parameters()  # type: ignore
        }  # type: ignore

        model.zero_grad(set_to_none=False)

        # Forget backward
        forget_grad_norms = []
        forget_losses = []
        forget_grads = {}
        for name, module in tqdm(model.base_model.named_modules()):  # type: ignore
            if not isinstance(module, nn.Linear):
                continue

            self.frozen_model.zero_grad()

            with patch_weights(self.frozen_model, name, module) as (weight, bias):
                if weight.grad is not None:
                    assert (
                        weight.grad.count_nonzero().item() == 0
                    ), "Nonzero weight grad"

                with amp_ctx:
                    forget_loss = self._compute_forget_loss(
                        self.frozen_model, inputs, target_device
                    )

                forget_loss_term = forget_coeff * forget_loss

                # Backpropagate through the frozen model to populate
                # the current module's gradients

                # Scale the loss in FP16
                if self.accelerator.scaler is not None:
                    # Scale the loss to match the retain gradient scale
                    self.accelerator.scaler.scale(forget_loss_term).backward()
                    scale_factor = self.accelerator.scaler.get_scale()
                else:
                    # BF16 or FP32 bypasses scaling
                    forget_loss_term.backward()
                    scale_factor = 1.0

                # Extract the newly computed (and properly scaled) gradients
                forget_grads[f"{name}.weight"] = weight.grad.clone()
                if bias is not None and bias.grad is not None:
                    forget_grads[f"{name}.bias"] = bias.grad.clone()

                # Divide by scale_factor so your wandb logs
                # show the true (unscaled) norm
                forget_grad_norms.append(
                    (
                        forget_grads[f"{name}.weight"].detach().norm() / scale_factor
                    ).item()
                )
                if bias is not None and bias.grad is not None:
                    forget_grad_norms.append(
                        (
                            forget_grads[f"{name}.bias"].detach().norm() / scale_factor
                        ).item()
                    )

                forget_losses.append(forget_loss.detach())

        # The forget loop accumulated gradients into the active model.
        model.zero_grad(set_to_none=False)

        # Combine saved gradients
        for name, param in model.base_model.named_parameters():  # type: ignore
            if name in self.target_modules and name in retain_grads:
                param.grad.copy_(retain_grads[name])

                # Distribute gradient for add
                local_forget_grad = forget_grads[name]
                # Reshard to match the DTensor's placement
                sharded_grad = distribute_tensor(
                    local_forget_grad, param.device_mesh, param.placements
                )
                param.grad.add_(sharded_grad.to(param.grad.dtype))

                if name in accumulated_grads:
                    param.grad.add_(accumulated_grads[name])

        # Logging
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            # Keyword mask stats
            keyword_mask_frac = None
            if "bio_remove_keyword_mask" in inputs:
                km = inputs["bio_remove_keyword_mask"]
                attn = inputs["bio_remove_attention_mask"]
                attn_tokens = attn.sum().item()
                keyword_tokens = (km.bool() & attn.bool()).sum().item()
                keyword_mask_frac = keyword_tokens / max(attn_tokens, 1)

            if self.current_training_step % 4 == 0:
                msg = (
                    f"step {self.current_training_step} | "
                    f"retain_loss: {retain_loss.item():.4f} | "
                    f"forget_loss: {torch.stack(forget_losses).cpu().mean().item():.4f}"
                    f" | retain_grad_norm: {retain_grad_norm:.4f} | "
                    f"forget_grad_norm: {np.mean(forget_grad_norms):.4f}"
                )
                if keyword_mask_frac is not None:
                    msg += f" | kw_mask: {keyword_mask_frac:.3f}"
                if hasattr(self, "_topk_entropy"):
                    msg += f" | topk_H: {self._topk_entropy:.4f} "
                    msg += f"({self._topk_entropy_ratio:.3f})"
                print(msg)

            if wandb.run is not None:
                log_dict = {
                    "retain_loss": retain_loss.item(),
                    "forget_loss": torch.stack(forget_losses).cpu().mean().item(),
                    "retain_grad_norm": retain_grad_norm,
                    "forget_grad_norm": np.mean(forget_grad_norms),
                    "retain_coeff": retain_coeff,
                    "forget_coeff": forget_coeff,
                }
                if keyword_mask_frac is not None:
                    log_dict["keyword_mask_frac"] = keyword_mask_frac
                if hasattr(self, "_topk_entropy"):
                    log_dict["topk_entropy"] = self._topk_entropy
                    log_dict["topk_entropy_ratio"] = self._topk_entropy_ratio
                wandb.log(log_dict, step=self.current_training_step)

        self.current_training_step += 1
        return (
            retain_coeff * retain_loss
            + forget_coeff * torch.stack(forget_losses).cpu().mean().item()
        ).detach()


@dataclass
class ModuleParallelUnlearnConfig:
    num_train_examples: int = 0
    lr: float = 1e-4
    pdbs: int = 1
    retain_coef: float = 20.0
    remove_coef: float = 5.0
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    warmup_ratio: float = 0.0
    use_ultrachat: bool = False
    use_max_entropy_kl: bool = False
    use_top_k_entropy: bool = False
    top_k: int = 100
    asymmetric_filter: bool = False
    max_grad_norm: float = 1.0
    retain_loss_type: Literal["kl", "nll"] = "kl"
    use_dpo_forget: bool = False
    dpo_per_token: bool = False
    dpo_beta: float = 0.1
    deep_layer_step_scale: float = 1.0
    optimizer: Literal["muon", "adamw"] = "adamw"
    muon_momentum: float = 0.95
    wandb_project: str = ""
    wandb_run_name: str = ""
    blocklist_path: str = ""
    keyword_mask_method: Literal["regex", "activation", "sae", "probe"] = "regex"
    activation_mask_threshold: float = 0.2
    activation_mask_layer: int = 16
    sae_latents_path: str = ""
    sae_mask_frac: float = 0.115
    probe_mask_path: str = ""
    probe_mask_layer: int = 11
    probe_mask_frac: float = 0.105
    gradient_checkpointing: bool = False
    maintain_unlearned: bool = False
    dtype: Literal["bf16", "fp16"] = "bf16"

    # Unlearn dataset HPs
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    assert world_size > 1, "This script is designed for use with multiple GPUs"

    parser = ArgumentParser()
    parser.add_arguments(ModuleParallelUnlearnConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"  {arg}: {value}")

    if run_cfg.wandb_project and local_rank == 0:
        from dotenv import load_dotenv

        load_dotenv()
        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(
            project=run_cfg.wandb_project,
            name=run_cfg.wandb_run_name or None,
            config=vars(run_cfg),
        )

    model, tokenizer = get_model_and_tokenizer(run_cfg.model_name)
    model = cast(PreTrainedModel, model)

    for param in model.parameters():
        param.requires_grad = True
    model.enable_input_require_grads()

    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)
    train_dataset.tokenized_bio_remove_dataset = apply_keyword_masks(
        train_dataset.tokenized_bio_remove_dataset, run_cfg, tokenizer
    )

    # Load one reference model per rank
    frozen_model = AutoModelForCausalLM.from_pretrained(
        run_cfg.model_name,
        torch_dtype=torch.float32,
        device_map={"": local_rank},
    )
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad = False

    print("Loaded frozen reference model.")

    # Apply FSDP2 before Trainer creation
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

        module_list_name, module_list = get_layer_list(model)

        for module in module_list:
            fully_shard(module)
        fully_shard(model)

        print(f"FSDP2 applied: {sum(1 for _ in model.parameters())} params as DTensors")

    # Constant global batch size
    global_batch_size = 32
    grad_acc_steps = max(1, global_batch_size // (run_cfg.pdbs * world_size))

    print(
        f"Running with {world_size} GPUs. Per device batch: {run_cfg.pdbs}. "
        f"Grad Acc steps: {grad_acc_steps}. "
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=run_cfg.lr,
        warmup_ratio=run_cfg.warmup_ratio,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=run_cfg.pdbs,
        per_device_eval_batch_size=run_cfg.pdbs,
        num_train_epochs=1,
        weight_decay=0.01,
        gradient_checkpointing=run_cfg.gradient_checkpointing,
        fp16=run_cfg.dtype == "fp16",
        bf16=run_cfg.dtype == "bf16",
        max_grad_norm=run_cfg.max_grad_norm,
        save_strategy="no",
        optim="adamw_torch",
        report_to="none",
    )

    trainer = ModuleParallelTrainer(
        run_args=run_cfg,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        frozen_model=frozen_model,
    )

    model.train()
    trainer.train()

    # Save final checkpoint
    if run_cfg.save_path:
        if world_size > 1:
            state_dict = get_model_state_dict(
                model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        else:
            state_dict = model.state_dict()
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

    print("Training complete")
