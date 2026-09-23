"""WMDP/WikiText orthogonal circuit breaker for Axolotl."""

import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.prompt_tokenizers import DatasetWrappingStrategy

WMDP_PATH = "cais/wmdp-bio-forget-corpus"
WIKITEXT_PATH = "EleutherAI/wikitext_document_level"
TARGET_LAYERS = (5, 10, 15, 20, 25, 30)


class TaggedDocumentStrategy(DatasetWrappingStrategy):
    """Produce one tokenized document with a source tag per selected row."""

    def __init__(self, tokenizer, sequence_len: int, path: str):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.path = path

    def wrap_dataset(self, dataset, **kwargs):
        del kwargs
        if self.path == WIKITEXT_PATH:
            dataset = dataset.shuffle(seed=42).select(range(min(1024, len(dataset))))
        return dataset.map(self.tokenize_row, remove_columns=dataset.column_names)

    def tokenize_row(self, row):
        if self.path == WMDP_PATH:
            fields = ("title", "abstract", "text")
            if not all(isinstance(row.get(field), str) for field in fields):
                raise ValueError(
                    "WMDP row must contain title, abstract, and text fields"
                )
            text = "\n\n".join(row[field] for field in fields)
            source = 1
        else:
            text = row.get("page")
            source = 0
        if not isinstance(text, str) or not text.strip():
            raise ValueError("dataset document must be a non-empty string")
        tokens = self.tokenizer(
            text,
            max_length=self.sequence_len,
            truncation=True,
            add_special_tokens=True,
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["input_ids"].copy(),
            "cb_source": source,
        }


def load(tokenizer, cfg, ds_cfg=None):
    """Select the native WMDP or WikiText dataset source."""

    path = getattr(ds_cfg, "path", None)
    if path not in (WMDP_PATH, WIKITEXT_PATH):
        raise ValueError(f"unsupported circuit breaker dataset: {path}")
    return TaggedDocumentStrategy(tokenizer, cfg.sequence_len, path)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(values.dtype).unsqueeze(0).expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1)


def coefficients(microstep: int, total_microsteps: int) -> tuple[float, float, float]:
    """Retain, removal, and orthogonalization weights over a run."""

    progress = min(max(microstep, 0) / max(total_microsteps - 1, 1), 1.0)
    return 1.0 + 9.0 * progress, 23.0 - 5.75 * progress, 5.0 * progress


def orthogonalization_loss(activations: torch.Tensor, mask: torch.Tensor):
    """Penalize positive cosine between different forget sequences."""

    lengths = mask.sum(dim=1).clamp_min(1).to(activations.dtype)
    pooled = (activations * mask[None, :, :, None]).sum(dim=2)
    pooled = F.normalize((pooled / lengths[None, :, None]).float(), dim=-1)
    similarities = pooled @ pooled.transpose(-1, -2)
    off_diagonal = ~torch.eye(
        activations.shape[1], dtype=torch.bool, device=activations.device
    )
    positive_pairs = F.relu(similarities[:, off_diagonal])
    if positive_pairs.numel() == 0:
        return activations.float().sum() * 0
    return positive_pairs.mean() * lengths.float().mean()


class OrthCircuitBreakerTrainer(AxolotlTrainer):
    """Retain WikiText and reroute WMDP activations without token loss."""

    def __init__(self, *args, target_layers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_layers = tuple(target_layers or TARGET_LAYERS)
        num_layers = self.model.config.num_hidden_layers
        if len(set(self.target_layers)) != len(self.target_layers) or any(
            layer < 0 or layer >= num_layers for layer in self.target_layers
        ):
            raise ValueError(
                f"target_layers must be unique indices in 0-{num_layers - 1}"
            )
        self.microstep = 0
        self.total_microsteps = (
            self.args.max_steps * self.args.gradient_accumulation_steps
        )

    def selected_activations(
        self,
        model,
        input_ids,
        attention_mask,
        *,
        disable_adapter=False,
        disable_grad=False,
    ):
        adapter_context = model.disable_adapter() if disable_adapter else nullcontext()
        gradient_context = torch.no_grad() if disable_grad else nullcontext()
        was_training = model.training
        model.eval() if disable_grad else model.train()
        try:
            with adapter_context, gradient_context:
                hidden_states = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                ).hidden_states
                result = torch.stack(
                    tuple(hidden_states[layer + 1] for layer in self.target_layers)
                )
        finally:
            model.train(was_training)
        return result.detach() if disable_grad else result

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_mask_retain = inputs["cb_source"] == 0
        attn_mask_retain = inputs["attention_mask"][sample_mask_retain]
        x_retain = inputs["input_ids"][sample_mask_retain]

        sample_mask_forget = inputs["cb_source"] == 1
        attn_mask_forget = inputs["attention_mask"][sample_mask_forget]
        x_forget = inputs["input_ids"][sample_mask_forget]

        n_retain = int(sample_mask_retain.sum())
        n_forget = int(sample_mask_forget.sum())
        lam_ret, lam_fgt, lam_ortho = coefficients(
            self.microstep, self.total_microsteps
        )

        zero = (
            next(
                parameter for parameter in model.parameters() if parameter.requires_grad
            ).sum()
            * 0
        )
        loss = zero.clone()
        loss_retain = loss_reroute = loss_orth = zero
        z_retain = z_forget = None
        if n_retain:
            z_retain_ref = self.selected_activations(
                model,
                x_retain,
                attn_mask_retain,
                disable_adapter=True,
                disable_grad=True,
            )
            z_retain = self.selected_activations(model, x_retain, attn_mask_retain)
            loss_retain = masked_mean(
                (z_retain.float() - z_retain_ref.float()).norm(dim=-1),
                attn_mask_retain,
            )
            loss += lam_ret * loss_retain

        harmful_cosine = zero.detach()
        if n_forget:
            z_forget_ref = self.selected_activations(
                model,
                x_forget,
                attn_mask_forget,
                disable_adapter=True,
                disable_grad=True,
            )
            z_forget = self.selected_activations(model, x_forget, attn_mask_forget)
            forget_cosine = F.cosine_similarity(
                z_forget.float(), z_forget_ref.float(), dim=-1
            )
            loss_reroute = masked_mean(F.relu(forget_cosine), attn_mask_forget)
            harmful_cosine = masked_mean(forget_cosine.detach(), attn_mask_forget)
            loss += lam_fgt * loss_reroute

        if n_forget >= 2:
            loss_orth = orthogonalization_loss(z_forget, attn_mask_forget)
            loss += lam_ortho * loss_orth
        gradients = [
            parameter.grad.detach().float().norm()
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        gradient_norm = torch.stack(gradients).norm().item() if gradients else 0.0
        self.log(
            {
                "loss_retain": loss_retain.detach().item(),
                "loss_reroute": loss_reroute.detach().item(),
                "loss_orthogonalization": loss_orth.detach().item(),
                "retain_coefficient": lam_ret,
                "reroute_coefficient": lam_fgt,
                "orthogonalization_coefficient": lam_ortho,
                "harmful_cosine": harmful_cosine.item(),
                "gradient_norm": gradient_norm,
                "microstep": self.microstep,
                "retain_count": n_retain,
                "forget_count": n_forget,
                "retain_skipped": not n_retain,
                "reroute_skipped": not n_forget,
                "orthogonalization_skipped": n_forget < 2,
            }
        )
        self.microstep += 1
        if return_outputs:
            return loss, {"retain": z_retain, "forget": z_forget}
        return loss


class BalancedSourceSampler(torch.utils.data.Sampler[int]):
    """Shuffle each source once, then draw half of each per microbatch."""

    def __init__(self, dataset, batch_size: int, seed: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError(
                "balanced source sampling requires a positive even batch size"
            )
        sources = dataset["cb_source"]
        if set(sources) != {0, 1}:
            raise ValueError("balanced source sampling requires both source tags")
        self.retain = [index for index, source in enumerate(sources) if source == 0]
        self.forget = [index for index, source in enumerate(sources) if source == 1]
        if len(self.retain) != len(self.forget) or len(self.retain) % (batch_size // 2):
            raise ValueError(
                "source counts must be equal and divisible by half a batch"
            )
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        retain, forget = self.retain.copy(), self.forget.copy()
        rng.shuffle(retain)
        rng.shuffle(forget)
        half = self.batch_size // 2
        for start in range(0, len(retain), half):
            batch = retain[start : start + half] + forget[start : start + half]
            rng.shuffle(batch)
            yield from batch

    def __len__(self):
        return len(self.retain) + len(self.forget)


class BalancedOrthCircuitBreakerTrainer(OrthCircuitBreakerTrainer):
    """Run the same loss with equal retain and forget rows in every batch."""

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return BalancedSourceSampler(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )
