from __future__ import annotations

from pathlib import Path
import json
import torch
import pytest
from terminal_learner.config import ModelConfig, TrainConfig, EncodingConfig
from terminal_learner.dataloader.encoding import Encoder
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.dataloader.trajectory import split_tasks
from terminal_learner.model.data import Episode, Turn
from terminal_learner.model.transformer import create_model
from terminal_learner.model.loop import Trainer
from terminal_learner.model.meta_model import action_loss, copy_fast, window_loss
from terminal_learner.protocol import canonical_action
from terminal_learner.session import AgentSession
from terminal_learner.checkpoint import save_checkpoint

@pytest.mark.parametrize("qk_norm", [False, True])
def test_meta_learning_and_session(qk_norm):
    """Meaningful CPU checks, including an independent meta-gradient finite difference."""
    import tempfile
    torch.set_num_threads(1)
    torch.manual_seed(17)
    tok = Tokenizer()
    model, initial = create_model(ModelConfig(hidden_size=8, intermediate_size=12,
                                              num_layers=1, num_heads=2, num_kv_heads=2,
                                              max_seq_len=1024, dtype='float64', qk_norm=qk_norm))
    enc = Encoder(tok, EncodingConfig(), 1024)
    a1 = canonical_action({'commands': [{'keystrokes': 'ls\n', 'duration': 0.1}], 'task_complete': False})
    a2 = canonical_action({'commands': [{'keystrokes': 'cd app\n', 'duration': 0.1}], 'task_complete': False})
    ep = Episode('find app', '$ ', [Turn(a1, 'app/', 2), Turn(a2, '$ ', 3)], 'synthetic', 'task-a', 0)
    cfg = TrainConfig(inner_lr=0.1, inner_clip=100, outer_lr=1e-3)

    # All command tokens are supervised even across chunks; prompt tokens are not.
    prompt = enc.prompt(ep.task, [])
    targets, _ = enc.target('ls', 'action')
    nll, n = action_loss(model, initial, enc, ep.task, [], 'ls')
    ids = torch.tensor([prompt + targets[:-1]])
    logits = model(ids, initial)[0, len(prompt) - 1:]
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor(targets), reduction='sum')
    torch.testing.assert_close(nll, expected)
    assert n == len(targets)

    # Future target/output must not alter the first action's input or loss.
    history = enc.initial_history(ep.initial_observation)
    before, _ = action_loss(model, initial, enc, ep.task, history, a1)
    changed = Episode(ep.task, ep.initial_observation, [Turn(a1, 'different future', 2)], '', '', 0)
    after, _ = action_loss(model, initial, enc, changed.task, history, a1)
    torch.testing.assert_close(before, after)

    # Compare exact meta-gradient against detached ablation, then finite differences.
    parameter = model.layers[0].self_attn.q_proj.weight
    def objective(differentiable=True):
        loss, _, _, _ = window_loss(model, copy_fast(initial), enc, ep, ep.turns,
                                    list(history), cfg, differentiable=differentiable)
        return loss
    exact = torch.autograd.grad(objective(), parameter)[0]
    detached = torch.autograd.grad(objective(False), parameter)[0]
    assert float((exact - detached).abs().max()) > 1e-9, 'missing meta-gradient path'
    position = int((exact - detached).abs().argmax())
    epsilon = 1e-5
    with torch.no_grad():
        parameter.view(-1)[position] += epsilon
    plus = float(objective().detach())
    with torch.no_grad():
        parameter.view(-1)[position] -= 2 * epsilon
    minus = float(objective().detach())
    with torch.no_grad():
        parameter.view(-1)[position] += epsilon
    torch.testing.assert_close(exact.view(-1)[position],
                               torch.tensor((plus - minus) / (2 * epsilon), dtype=torch.float64),
                               atol=1e-6, rtol=1e-3)

    # A failed episode still trains theta in the default all-actions policy.
    trainer = Trainer(model, initial, enc, cfg)
    old_theta = {k: v.detach().clone() for k, v in model.named_parameters()}
    old_initial = copy_fast(initial)
    report = trainer.train_episode(ep)
    assert report['windows'] == 1
    assert any(not torch.equal(v, old_theta[k]) for k, v in model.named_parameters())
    for k in initial:
        torch.testing.assert_close(initial[k], old_initial[k])

    # Inference changes phi only; reset restores it; cached logits match full forward.
    session = AgentSession(model, initial, enc)
    session.reset(ep.task, ep.initial_observation)
    theta = {k: v.detach().clone() for k, v in model.named_parameters()}
    session.observe(a1, 'app/')
    assert any(not torch.equal(session.fast[k], initial[k]) for k in initial)
    assert all(torch.equal(v, theta[k]) for k, v in model.named_parameters())
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        _, cache = model.forward_cached(ids[:, :3], session.fast)
        cached, _ = model.forward_cached(ids[:, 3:], session.fast, cache)
        torch.testing.assert_close(cached[:, -1], model(ids, session.fast)[:, -1])
    session.reset(ep.task)
    for k in initial:
        torch.testing.assert_close(session.fast[k], initial[k])
    session.adapt = False
    session.observe(a1, 'different output')
    for k in initial:
        torch.testing.assert_close(session.fast[k], initial[k])

    # No same-task attempts can leak across the data split.
    records = [{'task': str(i // 5)} for i in range(25)]
    all_tasks, empty_test = split_tasks(records)
    assert len(all_tasks) == 5 and empty_test == []
    train_tasks, test_tasks = split_tasks(records, test_fraction=0.2)
    assert not set(train_tasks) & set(test_tasks)

    # Checkpoint round-trip, independent frozen/adaptive sessions, and invalid JSON.
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'model.pt'
        save_checkpoint(checkpoint, model, initial, tok, enc.config, inner_lr=0.1, inner_clip=1)
        loaded = AgentSession.from_checkpoint(checkpoint)
        loaded.reset(ep.task)
        for k, v in loaded.model.state_dict().items():
            torch.testing.assert_close(v, model.state_dict()[k])
        response = loaded.act()
        assert not response['valid'] and response['action'] is None
        loaded.observe(a1, 'app/')
    print(json.dumps({'self_test': 'passed', 'checks': [
        'target-only shifted CE', 'causal supervision', 'exact vs detached meta-gradient',
        'meta-gradient finite difference', 'failed-episode action supervision',
        'theta-only optimizer', 'frozen inference theta', 'fast reset', 'KV cache parity',
        'frozen adaptation baseline', 'task-disjoint split', 'checkpoint round-trip',
        'invalid-action handling']}))
