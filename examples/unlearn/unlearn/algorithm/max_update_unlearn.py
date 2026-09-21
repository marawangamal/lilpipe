"""Max Update Unlearning.

Maximizes parameter deviation from the initial model while maintaining
retain quality via SFT cross-entropy. No forget loss term -- the theory
is that maximizing parameter change while constraining retain quality
pushes the model away from dangerous capabilities.

Loss = retain_coef * sft_loss - update_coef * ||theta - theta_0||^2
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from simple_parsing import ArgumentParser
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import seed_worker

from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.muon import MuonAdamW
from unlearn.utils.worker_utils import (
    get_model_and_tokenizer,
)


def compute_lm_loss(logits, labels, attention_mask):
    """Standard SFT cross-entropy with attention masking."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()

    _, _, vocab_size = shift_logits.shape
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_mask = shift_mask.view(-1)

    loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
    return (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)


def _compute_update_term(param, w0, element_norm: bool):
    """Differentiable update loss for one parameter.

    Returns (update_term, norm_sq_detached).
    For element_norm: update_term = sqrt(N) * ||diff||, gradient = diff / rms.
    For L2 norm:      update_term = ||diff||,           gradient = diff / ||diff||.
    """
    diff = param - w0
    norm_sq = diff.pow(2).sum()
    eps = 1e-8
    if element_norm:
        return diff.pow(2).mean().add(eps).sqrt() * param.numel(), norm_sq.detach()
    else:
        return (norm_sq + eps).sqrt(), norm_sq.detach()


class MaxUpdateTrainer(Trainer):
    def __init__(
        self,
        run_cfg,
        model,
        args,
        train_dataset,
        tokenizer,
    ):
        if run_cfg.optimizer == "muon":
            self.muon_param_names = {
                name
                for name, p in model.named_parameters()
                if p.ndim >= 2 and p.size(0) < 50000
            }
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        self.run_cfg = run_cfg
        self.current_training_step = 0
        self.tokenizer = tokenizer

        # Clone initial trainable params to CPU (full tensors, not shards)
        self.initial_params: dict[str, torch.Tensor] = {}
        self.total_trainable_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                p = param.data
                if isinstance(p, DTensor):
                    p = p.full_tensor()
                self.initial_params[name] = p.clone().cpu()
                self.total_trainable_params += param.numel()

        print(
            f"Stored {len(self.initial_params)} initial param tensors "
            f"({self.total_trainable_params} params total)"
        )

        # Pre-group initial params by transformer layer module path.
        # Remaining params (embeddings, head, norms) go in _non_layer_init.
        no_split = set(getattr(model.config, "_no_split_modules", None) or [])
        self._layer_hooks_info: list[tuple[str, dict[str, torch.Tensor]]] = []
        layer_param_fullnames: set[str] = set()

        for module_name, module in model.named_modules():
            if module.__class__.__name__ not in no_split:
                continue
            prefix = module_name + "."
            layer_init: dict[str, torch.Tensor] = {}
            for pname in dict(module.named_parameters()):
                full = prefix + pname
                if full in self.initial_params:
                    layer_init[pname] = self.initial_params[full]
                    layer_param_fullnames.add(full)
            if layer_init:
                self._layer_hooks_info.append((module_name, layer_init))

        self._non_layer_init: dict[str, torch.Tensor] = {
            k: v
            for k, v in self.initial_params.items()
            if k not in layer_param_fullnames
        }

    def create_optimizer(self, model=None):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.run_cfg.optimizer == "muon":
            self.optimizer = MuonAdamW(
                self.model.named_parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                muon_param_names=self.muon_param_names,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                fused=False,
            )
        return self.optimizer

    def _prepare_for_training(self, max_steps, train_dataloader, resume_from_checkpoint):
        """Skip accelerate's model wrapping (FSDP2 applied externally)."""
        if not self.run_cfg.lora:
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
            DataLoader(self.train_dataset, **dataloader_params)
        )

    def _register_update_hooks(self, model, update_terms, norm_sq_terms):
        """Register per-layer forward hooks to compute differentiable update
        loss terms while FSDP has gathered that layer's full parameters."""
        fsdp_root = model.module if hasattr(model, "module") else model
        handles = []

        def _make_layer_hook(layer_init_params):
            def hook(module, _input, output):
                for pname, param in module.named_parameters():
                    if pname not in layer_init_params or not param.requires_grad:
                        continue
                    p = param.full_tensor() if isinstance(param, DTensor) else param
                    w0 = layer_init_params[pname].to(p.device, dtype=p.dtype)
                    term, nsq = _compute_update_term(
                        p, w0, self.run_cfg.element_norm
                    )
                    update_terms.append(term)
                    norm_sq_terms.append(nsq)
                return output

            return hook

        for module_name, layer_init in self._layer_hooks_info:
            parts = module_name.split(".")
            module = fsdp_root
            for p in parts:
                if p.isdigit():
                    module = module[int(p)]
                else:
                    module = getattr(module, p)
            handles.append(module.register_forward_hook(_make_layer_hook(layer_init)))

        # Non-layer params (embeddings, head, norms) are in the root FSDP unit.
        # The root's forward hook fires while root-unit params are gathered.
        if self._non_layer_init:
            non_layer = self._non_layer_init

            def _root_hook(module, _input, output):
                for pname, param in module.named_parameters():
                    if pname not in non_layer or not param.requires_grad:
                        continue
                    p = param.full_tensor() if isinstance(param, DTensor) else param
                    w0 = non_layer[pname].to(p.device, dtype=p.dtype)
                    term, nsq = _compute_update_term(
                        p, w0, self.run_cfg.element_norm
                    )
                    update_terms.append(term)
                    norm_sq_terms.append(nsq)
                return output

            handles.append(fsdp_root.register_forward_hook(_root_hook))

        return handles

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        target_device = (
            inputs["input_ids"].device
            if hasattr(inputs["input_ids"], "device")
            else next(model.parameters()).device
        )

        retain_ids = inputs["input_ids"].to(target_device)
        retain_mask = inputs["attention_mask"].to(target_device)
        grad_acc = self.args.gradient_accumulation_steps

        # Per-layer forward hooks capture differentiable update terms while
        # FSDP has each layer's full params gathered.
        update_terms: list[torch.Tensor] = []
        norm_sq_terms: list[torch.Tensor] = []
        handles = self._register_update_hooks(model, update_terms, norm_sq_terms)

        retain_logits = model(
            input_ids=retain_ids,
            attention_mask=retain_mask,
        ).logits

        for h in handles:
            h.remove()

        sft_loss = compute_lm_loss(retain_logits, retain_ids, retain_mask)

        update_loss = (
            torch.stack(update_terms).sum()
            if update_terms
            else torch.tensor(0.0, device=target_device)
        )
        update_norm_sq = (
            torch.stack(norm_sq_terms).sum()
            if norm_sq_terms
            else torch.tensor(0.0, device=target_device)
        )

        # DTensor ops during forward hooks produce DTensors. Extract the local
        # tensor for manual collective ops. During forward, params are gathered
        # (Replicate), so each rank has the full value — no all_reduce needed.
        if isinstance(update_loss, DTensor):
            update_loss = update_loss.to_local()
        if isinstance(update_norm_sq, DTensor):
            update_norm_sq = update_norm_sq.to_local()

        if self.run_cfg.same_sign_grads:
            # Two-pass backward: update first, then SFT, filter by sign.
            self.accelerator.backward(
                -(self.run_cfg.update_coef / grad_acc) * update_loss,
                retain_graph=True,
            )
            update_grads: dict[str, torch.Tensor] = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    g = param.grad
                    if hasattr(g, "to_local"):
                        g = g.to_local()
                    update_grads[name] = g.clone()
            model.zero_grad()

            self.accelerator.backward(self.run_cfg.retain_coef * sft_loss)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in update_grads and param.grad is not None:
                        ug = update_grads[name]
                        g = param.grad
                        if hasattr(g, "to_local"):
                            g = g.to_local()
                        same_sign = ug.sign() == g.sign()
                        param.grad.add_(ug * same_sign)
        elif self.run_cfg.sgd_update:
            # SGD update applied via differentiable loss, then manual LR scaling.
            # backward on SFT only; apply update step directly to params.
            self.accelerator.backward(self.run_cfg.retain_coef * sft_loss)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    key = name.removeprefix("module.")
                    if not (param.requires_grad and key in self.initial_params):
                        continue
                    pd = param.data.full_tensor() if isinstance(param.data, DTensor) else param.data
                    init_p = self.initial_params[key].to(
                        pd.device, dtype=pd.dtype
                    )
                    diff = pd - init_p
                    if self.run_cfg.element_norm:
                        rms = diff.pow(2).mean().sqrt()
                        unit_dir = diff / rms if rms > 1e-8 else diff
                    else:
                        pn = diff.norm()
                        unit_dir = diff / pn if pn > 1e-8 else diff
                    step = (
                        self.args.learning_rate
                        * self.run_cfg.update_coef
                        * unit_dir
                        / grad_acc
                    )
                    param.data.add_(step)
        else:
            # Default: single backward on combined differentiable loss.
            total_loss = (
                self.run_cfg.retain_coef * sft_loss
                - (self.run_cfg.update_coef / grad_acc) * update_loss
            )
            self.accelerator.backward(total_loss)

        loss = (
            self.run_cfg.retain_coef * sft_loss
            - self.run_cfg.update_coef * update_norm_sq.sqrt()
        )

        if (
            self.current_training_step % 8 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            from tqdm import tqdm

            tag = ""
            if self.run_cfg.same_sign_grads:
                tag += " || [same_sign_grads]"
            if self.run_cfg.sgd_update:
                tag += " || [sgd_update]"
            if self.run_cfg.element_norm:
                tag += " || [element_norm]"
            tqdm.write(
                f"step: {self.current_training_step} || "
                f"sft_loss: {sft_loss.item():.4f} || "
                f"update_norm: {update_norm_sq.item():.6f} || "
                f"combined_loss: {loss.item():.4f}"
                f"{tag}"
            )
            sys.stdout.flush()

        self.current_training_step += 1
        return loss.detach()


@dataclass
class MaxUpdateConfig:
    retain_coef: float = 1.0
    update_coef: float = 10.0
    same_sign_grads: bool = False
    sgd_update: bool = False
    element_norm: bool = True
    num_train_examples: int = 0
    epochs: int = 1
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: str = "rewritten"
    lr: float = 2e-4
    pdbs: int = 1
    lora: bool = False
    lora_r: int = 16
    layers: list[int] = field(default_factory=lambda: list(range(32)))
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    optimizer: Literal["adamw", "muon"] = "adamw"
    dtype: Literal["bf16", "fp16"] = "bf16"
    save_path: str = ""
    revision: str = "main"
    hidden_dim: int = 4096
    use_ultrachat: bool = False


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2

    parser = ArgumentParser()
    parser.add_arguments(MaxUpdateConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    assert run_cfg.retain_coef > 0, (
        "retain_coef must be > 0 for max update unlearning. "
        "At initialization diff = param - w0 = 0, so the update term gradient "
        "is zero. Without SFT loss to bootstrap parameter movement, the model "
        "never leaves the fixed point."
    )

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"  {arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision, dtype=run_cfg.dtype
    )

    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)

    if not run_cfg.lora:
        for param in model.parameters():
            param.requires_grad = True

    if run_cfg.lora:
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
    model = cast(PreTrainedModel, model)
    model.enable_input_require_grads()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not run_cfg.lora and world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        for layer in model.gpt_neox.layers:
            fully_shard(layer)
        fully_shard(model)
        print(f"FSDP2 applied: {sum(1 for _ in model.parameters())} params as DTensors")

    global_batch_size = 32
    grad_acc_steps = max(
        1,
        global_batch_size // (run_cfg.pdbs * world_size),
    )

    print(
        f"Running with {world_size} GPUs. "
        f"Per device batch: {run_cfg.pdbs}. "
        f"Grad acc steps: {grad_acc_steps}."
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=run_cfg.lr,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=run_cfg.pdbs,
        per_device_eval_batch_size=run_cfg.pdbs,
        num_train_epochs=run_cfg.epochs,
        weight_decay=0.01,
        gradient_checkpointing=False,
        fp16=run_cfg.dtype == "fp16",
        bf16=run_cfg.dtype == "bf16",
        save_strategy="no",
        optim="adamw_torch",
    )

    trainer = MaxUpdateTrainer(
        run_cfg,
        model,
        training_args,
        train_dataset,
        tokenizer,
    )

    model.train()
    trainer.train()

    if run_cfg.lora:
        model = model.merge_and_unload()

    if run_cfg.save_path:
        if not run_cfg.lora and world_size > 1:
            state_dict = get_model_state_dict(
                model, options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        else:
            state_dict = model.state_dict()
        if local_rank == 0:
            os.makedirs(run_cfg.save_path, exist_ok=True)
            model.save_pretrained(
                run_cfg.save_path, state_dict=state_dict, safe_serialization=True,
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

    print("Done.")
