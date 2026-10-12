"""Sampling for batches containing equal numbers of forget and retain rows."""

import random

import torch


class MixedSourceSampler(torch.utils.data.Sampler[int]):
    """Shuffle, truncate, and mix both sources without replacement."""

    def __init__(self, dataset, batch_size: int, seed: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("mixed source sampling requires an even batch size")

        sources = dataset["is_forget"]
        self.forget = [index for index, is_forget in enumerate(sources) if is_forget]
        self.retain = [
            index for index, is_forget in enumerate(sources) if not is_forget
        ]
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


class AuthorsMixedSourceSampler(torch.utils.data.Sampler[int]):
    """Shuffle forget rows and sample matching retain rows with replacement."""

    def __init__(self, dataset, batch_size: int, seed: int):
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


class OptmlMixedSourceSampler(torch.utils.data.Sampler[int]):
    """Match OPTML's shuffled forget rows and dynamic retain draws."""

    def __init__(self, dataset, batch_size: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("mixed source sampling requires an even batch size")
        sources = dataset["is_forget"]
        self.forget = [index for index, value in enumerate(sources) if value]
        self.retain = [index for index, value in enumerate(sources) if not value]
        if not self.forget or not self.retain:
            raise ValueError("mixed source sampling requires forget and retain rows")
        self.batch_size = batch_size

    def __iter__(self):
        # OPTML wraps the forget dataset in PyTorch's default RandomSampler.
        # Its paired dataset then draws one retain row from Python's global RNG
        # each time a forget row is fetched.
        order = torch.utils.data.RandomSampler(self.forget)
        half_batch = self.batch_size // 2
        batch = []
        for position in order:
            batch.append(self.forget[position])
            if len(batch) == half_batch:
                yield from batch
                yield from (random.choice(self.retain) for _ in range(half_batch))
                batch.clear()

    def __len__(self):
        half_batch = self.batch_size // 2
        return 2 * (len(self.forget) // half_batch * half_batch)
