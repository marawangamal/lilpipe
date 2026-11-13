from torch.utils.data import Sampler


def dict_dot(first, second):
    """Dot matching tensor mappings, reducing scalars onto the first device."""
    result = None
    for name in first:
        value = (first[name] * second[name]).sum()
        result = value if result is None else result + value.to(result.device)
    if result is None:
        raise ValueError("dict_dot requires nonempty mappings")
    return result


class BalancedSampler(Sampler):
    """Visit one index from each source at a time.

    ``BalancedSampler([0, 3], [1, 2, 4])`` yields ``0, 1, 3, 2``,
    stopping when the shortest source is exhausted.
    """

    def __init__(self, sample_ids_forget, sample_ids_retain):
        self.sample_ids_forget = sample_ids_forget
        self.sample_ids_retain = sample_ids_retain

    def __iter__(self):
        for group in zip(self.sample_ids_forget, self.sample_ids_retain):
            yield from group

    def __len__(self):
        return 2 * min(len(self.sample_ids_forget), len(self.sample_ids_retain))
