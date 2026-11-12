"""Sharpness-aware minimization (SAM) applied to unlearning objectives."""

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.ul_gd import GDTrainer
from trainers.utils import BalancedSampler

INP_FIELDS = ("input_ids", "attention_mask", "labels")


class SAMBaseTrainer(AxolotlTrainer):
    """Apply a small SAM perturbation to gradient-difference unlearning."""

    def __init__(
        self,
        *args,
        source_id_forget,
        source_id_retain,
        rho=0.01,
        coeff_retain=1.0,
        coeff_forget=1.0,
        **kwargs,
    ):
        if rho <= 0:
            raise ValueError("rho must be positive")
        if type(source_id_forget) is not int or type(source_id_retain) is not int:
            raise ValueError("source IDs must be integers")
        if source_id_forget == source_id_retain:
            raise ValueError("forget and retain source IDs must differ")
        super().__init__(*args, **kwargs)
        self.source_id_forget = source_id_forget
        self.source_id_retain = source_id_retain
        self.coeff_retain = coeff_retain
        self.coeff_forget = coeff_forget
        self.rho = rho

    def _get_train_sampler(self, train_dataset=None):
        """Draw equally from the configured forget and retain sources."""
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self._train_batch_size < 2 or self._train_batch_size % 2:
            raise ValueError("SAM sampling requires an even batch size")

        source_ids = dataset["source_id"]
        sample_ids_forget = [
            i
            for i, source_id in enumerate(source_ids)
            if source_id == self.source_id_forget
        ]
        sample_ids_retain = [
            i
            for i, source_id in enumerate(source_ids)
            if source_id == self.source_id_retain
        ]
        if not sample_ids_forget or not sample_ids_retain:
            raise ValueError("SAM sampling requires both forget and retain examples")
        return BalancedSampler(sample_ids_forget, sample_ids_retain)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        forget_mask = inputs["source_ids"] == self.source_id_forget
        retain_mask = inputs["source_ids"] == self.source_id_retain

        x_forget = {field: inputs[field][forget_mask] for field in INP_FIELDS}
        x_retain = {field: inputs[field][retain_mask] for field in INP_FIELDS}

        theta = {name: param for name, param in model.named_parameters()}
        theta_0 = copy.deepcopy(theta)

        # get pertubed theta
        loss_forget = self.compute_loss_forget(model, x_forget)
        grad_forget = compute_grad(loss_forget)
        theta = dict_binary_op(  # perturb
            theta, dict_unary_op(grad_forget, lambda x: (self.rho * x).detach()), sum
        )

        # compute forget loss at perturbed theta
        loss_forget = self.compute_loss_forget(model, x_forget)
        grad = compute_grad(loss_forget)

        # compute retain loss at original theta
        ## restore theta
        set_model_state_dict(model, theta)
        loss_retain = self.compute_loss_forget(model, x_retain)
        grad += compute_grad(loss_retain)

        # populate the grad
        set_model_grad_dict(model, grad)

        # perform the backward step
        return (
            self.coeff_retain * loss_retain + self.coeff_forget * loss_forget
        ).detach(), None


class SAMGDTrainer(SAMBaseTrainer):
    pass
