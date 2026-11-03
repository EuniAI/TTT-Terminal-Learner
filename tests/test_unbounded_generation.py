import torch
import pytest

from terminal_learner.config import ModelConfig, EncodingConfig
from terminal_learner.model.transformer import create_model
from terminal_learner.dataloader.encoding import Encoder
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.session import AgentSession
from terminal_learner.protocol import canonical_action
from terminal_learner.checkpoint import save_checkpoint, load_checkpoint


def make_session(capacity):
    model, fast = create_model(ModelConfig(hidden_size=8, intermediate_size=12,
        num_layers=1, num_heads=2, num_kv_heads=1, max_seq_len=capacity))
    encoder = Encoder(Tokenizer(), EncodingConfig(), capacity)
    session = AgentSession(model, fast, encoder)
    session.reset('task')
    return session


def scripted(session, tokens, monkeypatch):
    values = iter(tokens)
    calls = []
    def forward(ids, fast, cache=None):
        calls.append(ids.clone())
        logits = torch.full((1, 1, session.encoder.tok.vocab_size), -100.)
        logits[0, 0, next(values)] = 100.
        return logits, []
    monkeypatch.setattr(session, '_action_forward', forward)
    return calls


def test_action_can_generate_beyond_1024_until_end(monkeypatch):
    session = make_session(4096)
    action = {'commands': [{'keystrokes': 'x'*1200, 'duration': .1}], 'task_complete': False}
    tokens = session.encoder.tok.encode(canonical_action(action))
    calls = scripted(session, tokens + [session.encoder.tok.ids['action_end']], monkeypatch)
    result = session.act()
    assert result['valid'] and result['action'] == action
    assert len(calls) == len(tokens) + 1 > 1024
    assert calls[0].tolist()[0] == session.encoder.prompt(session.task, session.history)


def test_generation_capacity_is_explicit_and_never_rebuilds_context(monkeypatch):
    session = make_session(32)
    prompt = session.encoder.prompt(session.task, session.history)
    calls = scripted(session, [ord('x')]*32, monkeypatch)
    result = session.act()
    assert not result['valid'] and not result['ended']
    assert 'context capacity' in result['error']
    assert len(result['raw']) == 32 - len(prompt)
    assert len(calls) == 32 - len(prompt)
    assert all(ids.shape[1] == 1 for ids in calls[1:])
    session.reset('t'*32)
    with pytest.raises(ValueError, match='no generation space'):
        session.act()


def test_legacy_checkpoint_budgets_are_ignored(tmp_path):
    session = make_session(4096)
    path = tmp_path / 'legacy.pt'
    save_checkpoint(path, session.model, session.initial_fast, session.encoder.tok,
                    EncodingConfig(), inner_lr=.1, inner_clip=1)
    payload = torch.load(path, weights_only=True)
    payload['encoding_config'] = dict(task_tokens=1, history_tokens=1,
                                     max_observation_tokens=1, loss_chunk_tokens=1)
    torch.save(payload, path)
    _, _, encoder, _ = load_checkpoint(path)
    assert len(encoder.task_prefix('task')) == 5
    assert len(encoder.initial_history('output')) == 8
    assert encoder.target('output', 'observation') == (encoder.tok.encode('output') + [encoder.tok.ids['obs_end']], False)
