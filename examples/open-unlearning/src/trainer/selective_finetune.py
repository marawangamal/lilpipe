import re

from trainer.base import FinetuneTrainer


class SelectiveFinetune(FinetuneTrainer):
    """Fine-tune only parameters selected by full-match regular expressions."""

    def __init__(self, trainable_params_regex=(".*",), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable_params_regex = tuple(trainable_params_regex)

    def create_optimizer(self):
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(
                any(
                    re.fullmatch(pattern, name)
                    for pattern in self.trainable_params_regex
                )
            )
        return super().create_optimizer()
