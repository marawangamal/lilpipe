"""FSDP-safe gradient-difference unlearning with tunable SAM radii."""

import random

import torch

from configs.training.trainers.gd_sam import GDSAMTrainer


class AuthorsMixedSourceSampler(torch.utils.data.Sampler[int]):
    """Shuffle forget rows and draw retain rows with replacement like the paper code."""

    def __init__(self, dataset, batch_size, seed):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("mixed source sampling requires an even batch size")
        sources = dataset["is_forget"]
        self.forget = [index for index, value in enumerate(sources) if value]
        self.retain = [index for index, value in enumerate(sources) if not value]
        if not self.forget or not self.retain:
            raise ValueError("mixed source sampling requires forget and retain rows")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        forget = self.forget.copy()
        rng.shuffle(forget)
        half_batch = self.batch_size // 2
        usable = len(forget) // half_batch * half_batch
        for start in range(0, usable, half_batch):
            batch = forget[start : start + half_batch]
            batch += [rng.choice(self.retain) for _ in range(half_batch)]
            rng.shuffle(batch)
            yield from batch

    def __len__(self):
        half_batch = self.batch_size // 2
        return 2 * (len(self.forget) // half_batch * half_batch)


class AuthorsSamplingMixin:
    """Use the paper code's sampling policy without changing dataset contents."""

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return AuthorsMixedSourceSampler(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.seed,
        )


class GDSAMV2Rho1eMinus3Trainer(GDSAMTrainer):
    """FSDP-backward GD+SAM with rho fixed to 1e-3."""

    rho = 1e-3


class GDSAMV2Rho2p5eMinus3Trainer(GDSAMTrainer):
    """FSDP-backward GD+SAM with rho fixed to 2.5e-3."""

    rho = 2.5e-3


class GDSAMV2Rho5eMinus3Trainer(GDSAMTrainer):
    """FSDP-backward GD+SAM with rho fixed to 5e-3."""

    rho = 5e-3


class GDSAMV2Rho1eMinus2Trainer(GDSAMTrainer):
    """FSDP-backward GD+SAM with rho fixed to 1e-2."""

    rho = 1e-2


class GDSAMV2AuthorsRho5eMinus3Trainer(AuthorsSamplingMixin, GDSAMTrainer):
    """FSDP-backward GD+SAM at rho 5e-3 with authors-compatible sampling."""

    rho = 5e-3


class GDSAMV2AuthorsRho1eMinus2Trainer(AuthorsSamplingMixin, GDSAMTrainer):
    """GD+SAM at the paper's rho 1e-2 with authors-compatible sampling."""

    rho = 1e-2
