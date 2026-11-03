"""Single-forward teacher forcing and explicit over-capacity behavior."""

import pytest
import torch
from torch.nn import functional as F

from terminal_learner.config import EncodingConfig, ModelConfig
from terminal_learner.dataloader.encoding import Encoder
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.model.meta_model import action_loss, observation_loss
from terminal_learner.model.transformer import create_model


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model, fast = create_model(ModelConfig(hidden_size=8, intermediate_size=12,
        num_layers=1, num_heads=2, num_kv_heads=1, max_seq_len=128, dtype='float64'))
    encoder = Encoder(Tokenizer(), EncodingConfig(), 128)
    return model, fast, encoder


@pytest.mark.parametrize('kind', ['action', 'observation'])
def test_single_forward_matches_full_teacher_forcing(setup, kind):
    model, fast, encoder = setup
    targets, _ = encoder.target('x' * 70, kind)
    prompt = encoder.prompt('task', [], **({'action': 'ls'} if kind == 'observation' else {}))
    inputs = torch.tensor([prompt + targets[:-1]])
    logits = model(inputs, fast)[0, len(prompt) - 1:]
    expected = F.cross_entropy(logits, torch.tensor(targets), reduction='sum')
    if kind == 'observation':
        expected = expected / len(targets)
    params = tuple(model.parameters()) + tuple(fast.values())
    expected_grads = torch.autograd.grad(expected, params)
    calls = []
    projected = []
    head_hook = model.lm_head.register_forward_pre_hook(lambda module, args: projected.append(args[0].shape[1]))
    hook = model.register_forward_pre_hook(lambda model, args: calls.append(args[0].shape[1]))
    try:
        if kind == 'action':
            actual, count = action_loss(model, fast, encoder, 'task', [], 'x' * 70)
        else:
            actual, count, truncated = observation_loss(model, fast, encoder, 'task', [], 'ls', 'x' * 70)
            assert not truncated
    finally:
        hook.remove()
        head_hook.remove()
    assert calls == [len(prompt) + len(targets) - 1]
    assert projected == [len(targets)]
    assert count == 71
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(torch.autograd.grad(actual, params), expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_over_capacity_prompt_is_not_silently_cropped(setup):
    model, fast, encoder = setup
    calls = []
    hook = model.register_forward_pre_hook(lambda model, args: calls.append(args[0].shape[1]))
    try:
        with pytest.raises(ValueError, match='no text was truncated'):
            observation_loss(model, fast, encoder, 'task', [1]*80, 'ls', 'x'*120)
    finally:
        hook.remove()
    assert calls == []


@pytest.mark.parametrize('kind', ['action', 'observation'])
def test_over_capacity_targets_fail_explicitly(setup, kind):
    model, fast, encoder = setup
    with pytest.raises(ValueError, match='model context capacity'):
        if kind == 'action':
            action_loss(model, fast, encoder, 'task', [], 'x' * 180)
        else:
            observation_loss(model, fast, encoder, 'task', [], 'ls', 'x' * 180)


def test_complete_text_exceeds_old_budgets():
    encoder = Encoder(Tokenizer(), EncodingConfig(), 40960)
    task = 't' * 2000
    output = 'o' * 4000
    assert len(encoder.task_prefix(task)) == 2001
    history = encoder.initial_history(output)
    assert len(history) == 4002
    updated = encoder.history(history, 'ls', output)
    assert updated[:len(history)] == history
    assert len(updated) > 8000
    targets, truncated = encoder.target(output, 'observation')
    assert targets == encoder.tok.encode(output) + [encoder.tok.ids['obs_end']]
    assert not truncated
