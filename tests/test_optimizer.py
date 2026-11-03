import copy

import torch

from terminal_learner.model.optimizer import SlowWeightAdamW


def test_bf16_updates_accumulate_and_resume():
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    optimizer = SlowWeightAdamW([parameter], lr=1e-5, weight_decay=0)
    for _ in range(150):
        optimizer.set_gradients([torch.ones_like(parameter)])
        optimizer.step()
    master = optimizer.master_parameters[0]
    assert master.dtype == torch.float32
    assert master.item() < 0.999
    assert optimizer.state[master]['exp_avg'].dtype == torch.float32
    saved = copy.deepcopy(optimizer.state_dict())
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = SlowWeightAdamW([restored_parameter], lr=1e-5, weight_decay=0)
    restored.load_state_dict(saved)
    for _ in range(150):
        for opt in (optimizer, restored):
            opt.set_gradients([torch.ones_like(parameter)])
            opt.step()
    assert parameter.item() < 1  # BF16 model eventually receives accumulated changes.
    assert torch.equal(parameter, restored_parameter)
    assert torch.equal(master, restored.master_parameters[0])


def test_float32_matches_native_adamw_and_legacy_state():
    parameter = torch.nn.Parameter(torch.tensor([0.1, 1.0]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = SlowWeightAdamW([parameter], lr=1e-3)
    native = torch.optim.AdamW([reference], lr=1e-3)
    for _ in range(3):
        grad = torch.tensor([0.3, -0.2])
        optimizer.set_gradients([grad])
        reference.grad = grad
        optimizer.step()
        native.step()
    assert torch.equal(parameter, reference)
    optimizer.load_state_dict(native.state_dict())
    assert optimizer.state_dict()['fp32_master_weights'] == {}


def test_legacy_bf16_moments_upgrade_to_float32():
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    native = torch.optim.AdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    native.step()
    optimizer = SlowWeightAdamW([parameter])
    optimizer.load_state_dict(native.state_dict())
    master = optimizer.master_parameters[0]
    assert optimizer.state[master]['exp_avg'].dtype == torch.float32
    torch.testing.assert_close(master, parameter.float())
