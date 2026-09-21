import re

import torch
from torch.optim import AdamW, Muon

_FSDP_INNER_RE = re.compile(r"\._fsdp_wrapped_module\.|\._checkpoint_wrapped_module\.")
_FSDP_ROOT_RE = re.compile(r"^_fsdp_wrapped_module\.")


def _strip_fsdp_prefix(name: str) -> str:
    """Strip FSDP1 wrapper segments to recover the original parameter name."""
    name = _FSDP_INNER_RE.sub(".", name)
    return _FSDP_ROOT_RE.sub("", name)


class MuonAdamW(torch.optim.Optimizer):
    """
    Hybrid optimizer that applies torch.optim.Muon to 2D hidden matrices
    and torch.optim.AdamW to everything else (embeddings, biases, norms).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        muon_momentum=0.95,
        adam_betas=(0.9, 0.95),
        adam_eps=1e-8,
        weight_decay=0.01,
        muon_param_names: set[str] | None = None,
    ):
        muon_params = []
        adam_params = []

        if muon_param_names is not None:
            for name, p in params:
                if not p.requires_grad:
                    continue
                clean_name = _strip_fsdp_prefix(name)
                if clean_name in muon_param_names:
                    muon_params.append(p)
                else:
                    adam_params.append(p)
        else:
            for p in params:
                if not p.requires_grad:
                    continue
                if p.ndim >= 2 and p.size(0) < 50000:
                    muon_params.append(p)
                else:
                    adam_params.append(p)

        assert muon_params, (
            f"MuonAdamW found 0 Muon-eligible parameters out of "
            f"{len(muon_params) + len(adam_params)} total. "
            f"This usually means FSDP flattened the parameters before the "
            f"optimizer was created. Capture muon_param_names before FSDP "
            f"wrapping and pass them explicitly."
        )

        self.optimizers = []

        if muon_params:
            self.muon = Muon(
                muon_params,
                lr=lr,
                momentum=muon_momentum,
                weight_decay=weight_decay,
                adjust_lr_fn="match_rms_adamw",
            )
            self.optimizers.append(self.muon)

        if adam_params:
            self.adam = AdamW(
                adam_params,
                lr=lr,
                betas=adam_betas,
                eps=adam_eps,
                weight_decay=weight_decay,
            )
            self.optimizers.append(self.adam)

        # 3. Combine param_groups so HF Scheduler can see/update all LRs
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

        # Initialize base class (dummy) to satisfy type checks
        super().__init__(self.param_groups, {})

        print(
            f"MuonAdamW: {len(muon_params)} Muon params, "
            f"{len(adam_params)} AdamW params"
        )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for opt in self.optimizers:
            opt.step()

        return loss

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict() if hasattr(self, "muon") else None,
            "adam": self.adam.state_dict() if hasattr(self, "adam") else None,
        }

    def load_state_dict(self, state_dict):
        if hasattr(self, "muon") and state_dict["muon"]:
            self.muon.load_state_dict(state_dict["muon"])
        if hasattr(self, "adam") and state_dict["adam"]:
            self.adam.load_state_dict(state_dict["adam"])
