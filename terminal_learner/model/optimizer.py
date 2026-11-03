"""Accumulate low-precision slow-weight updates in FP32, outside the inner graph."""
from __future__ import annotations

import torch
from .numerics import validate_finite_tree


class SlowWeightAdamW(torch.optim.AdamW):
    """AdamW on FP32 masters for BF16 parameters; native precision otherwise.

    The forward model remains BF16. Autograd computes theta gradients through the
    inner updates first, then this optimizer updates masters and copies rounded
    values back. Fast weights are never members of this optimizer.
    """
    def __init__(self, parameters, **kwargs):
        self.model_parameters = tuple(parameters)
        self.master_parameters = tuple(
            torch.nn.Parameter(p.detach().float().clone()) if p.dtype == torch.bfloat16 else p
            for p in self.model_parameters)
        super().__init__(self.master_parameters, **kwargs)

    def set_gradients(self, gradients):
        gradients = tuple(gradients)
        if len(gradients) != len(self.master_parameters):
            raise ValueError('slow-weight gradient count mismatch')
        for parameter, gradient in zip(self.master_parameters, gradients):
            parameter.grad = gradient.detach().to(device=parameter.device, dtype=parameter.dtype)

    @torch.no_grad()
    def _copy_to_model(self):
        for model, master in zip(self.model_parameters, self.master_parameters):
            if model is not master:
                model.copy_(master)

    def step(self, closure=None):
        if closure is not None:
            raise ValueError('meta-training supplies explicit gradients, not closures')
        result = super().step()
        validate_finite_tree(self.master_parameters, 'optimizer.master_weights')
        validate_finite_tree(list(self.state.values()), 'optimizer.state')
        self._copy_to_model()
        return result

    def state_dict(self):
        state = super().state_dict()
        # Optimizer moments alone cannot recover the sub-BF16 accumulated update.
        state['fp32_master_weights'] = {
            str(i): master.detach().cpu().clone()
            for i, (model, master) in enumerate(zip(self.model_parameters, self.master_parameters))
            if model is not master}
        return state

    def load_state_dict(self, state_dict):
        validate_finite_tree(state_dict, 'optimizer.checkpoint')
        state = dict(state_dict)
        saved = state.pop('fp32_master_weights', None)
        expected = {str(i) for i, (model, master) in enumerate(
            zip(self.model_parameters, self.master_parameters)) if model is not master}
        if saved is not None:
            if set(saved) != expected:
                raise ValueError('checkpoint FP32 master-weight names mismatch')
            for i, value in saved.items():
                master = self.master_parameters[int(i)]
                if value.shape != master.shape or value.dtype != torch.float32:
                    raise ValueError('checkpoint FP32 master-weight shape/dtype mismatch')
        # AdamW converts legacy BF16 moments to the FP32 master's dtype on load.
        super().load_state_dict(state)
        if saved is not None:
            with torch.no_grad():
                for i, value in saved.items():
                    self.master_parameters[int(i)].copy_(value)
            self._copy_to_model()
        # Legacy checkpoints lack masters: start from their saved model weights.
