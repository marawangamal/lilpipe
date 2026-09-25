"""Sampling for batches containing equal numbers of forget and retain rows."""

import random

import torch


class MixedSourceSampler(torch.utils.data.Sampler[int]):
    """Shuffle, truncate, and mix both sources without replacement."""

    def __init__(self, dataset, batch_size: int, seed: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("mixed source sampling requires an even batch size")

        sources = dataset["cb_source"]
        self.forget = [index for index, source in enumerate(sources) if source == 1]
        self.retain = [index for index, source in enumerate(sources) if source == 0]
        if not self.forget or not self.retain:
            raise ValueError("mixed source sampling requires forget and retain rows")

        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    @property
    def source_size(self):
        half_batch = self.batch_size // 2
        return min(len(self.forget), len(self.retain)) // half_batch * half_batch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        forget = self.forget.copy()
        retain = self.retain.copy()
        rng.shuffle(forget)
        rng.shuffle(retain)

        half_batch = self.batch_size // 2
        for start in range(0, self.source_size, half_batch):
            batch = (
                forget[start : start + half_batch] + retain[start : start + half_batch]
            )
            rng.shuffle(batch)
            yield from batch

    def __len__(self):
        return 2 * self.source_size
