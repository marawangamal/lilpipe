"""Balanced and replacement sampling for forget/retain training."""

import random

import torch


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


class PairedSourceSampler(torch.utils.data.Sampler[int]):
    """Pair each shuffled forget row with a sampled retain row."""

    def __init__(self, dataset, batch_size: int, seed: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("paired sampling requires an even micro batch size")
        sources = dataset["cb_source"]
        self.forget = [i for i, source in enumerate(sources) if source == 1]
        self.retain = [i for i, source in enumerate(sources) if source == 0]
        if not self.forget or not self.retain:
            raise ValueError("paired sampling requires forget and retain rows")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        forget = self.forget.copy()
        rng.shuffle(forget)
        for start in range(0, len(forget), self.batch_size // 2):
            batch = []
            for index in forget[start : start + self.batch_size // 2]:
                batch.extend((index, rng.choice(self.retain)))
            rng.shuffle(batch)
            yield from batch

    def __len__(self):
        return len(self.forget) * 2


class PairedSourceSamplerMixin:
    """Use paired sampling for unbalanced forget and retain corpora."""

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        seed = (
            self.args.data_seed if self.args.data_seed is not None else self.args.seed
        )
        return PairedSourceSampler(dataset, self.args.per_device_train_batch_size, seed)
