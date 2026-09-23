"""Plain language-modeling documents for the weight-steering arms."""

from configs.unlearn.utils import TaggedDocumentStrategy, WIKITEXT_PATH, WMDP_PATH


class WeightSteeringDocumentStrategy(TaggedDocumentStrategy):
    """Use the CB document selection and formatting without its source tag."""

    def tokenize_row(self, row):
        tokens = super().tokenize_row(row)
        del tokens["cb_source"]
        return tokens


def load(tokenizer, cfg, ds_cfg=None):
    path = getattr(ds_cfg, "path", None)
    if path not in (WMDP_PATH, WIKITEXT_PATH):
        raise ValueError(f"unsupported weight-steering dataset: {path}")
    return WeightSteeringDocumentStrategy(tokenizer, cfg.sequence_len, path)
