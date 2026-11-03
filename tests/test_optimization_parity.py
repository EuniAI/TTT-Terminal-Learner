import json
import pytest
import torch

from terminal_learner.config import ModelConfig, EncodingConfig, TrainConfig
from terminal_learner.dataloader.encoding import Encoder
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.model import meta_model
from terminal_learner.model.data import Episode, Turn
from terminal_learner.model.transformer import create_model
from terminal_learner.session import AgentSession
from terminal_learner.protocol import canonical_action


def setup():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model, fast = create_model(ModelConfig(hidden_size=8, intermediate_size=12,
        num_layers=2, adaptive_layers=1, num_heads=2, num_kv_heads=1,
        max_seq_len=1024, dtype='float64', qk_norm=True))
    return model, fast, Encoder(Tokenizer(), EncodingConfig(), 1024)


@pytest.mark.parametrize('rounds', [1, 2, 3])
def test_terminal_inner_graph_optimization_preserves_meta_gradient(monkeypatch, rounds):
    model, fast, encoder = setup()
    episode = Episode('task', '$', [Turn('ls', str(i), i) for i in range(rounds)], 'trial', 'task', 0)
    original = meta_model.inner_update
    flags = []
    def capture(*args, **kwargs):
        flags.append(kwargs['create_graph'])
        return original(*args, **kwargs)
    monkeypatch.setattr(meta_model, 'inner_update', capture)
    def run():
        loss, updated, _, _ = meta_model.window_loss(model, meta_model.copy_fast(fast), encoder,
            episode, episode.turns, encoder.initial_history('$'), TrainConfig())
        grads = torch.autograd.grad(loss, tuple(model.parameters()))
        return loss.detach(), {k: v.detach() for k, v in updated.items()}, grads
    optimized = run()
    assert flags == [True] * (rounds - 1) + [False]
    def full_graph(*args, **kwargs):
        kwargs['create_graph'] = True
        return original(*args, **kwargs)
    monkeypatch.setattr(meta_model, 'inner_update', full_graph)
    reference = run()
    torch.testing.assert_close(optimized, reference)


def test_act_observe_act_shared_cache_matches_full_model(monkeypatch):
    model, fast, encoder = setup()
    session = AgentSession(model, fast, encoder)
    session.reset('find files', '$')
    action = {'commands': [{'keystrokes': 'ls\n', 'duration': 1.0}], 'task_complete': False}
    # Deliberately noncanonical formatting to exercise common-prefix rollback.
    text = json.dumps(action, indent=1)
    tokens = encoder.tok.encode(text) + [encoder.tok.ids['action_end']]
    original = session._action_forward
    seen = []
    hook = model.layers[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[1]))
    generation_calls = []
    iterator = iter(tokens)
    def checked(ids, phi, cache=None):
        if cache is None:
            generation_calls.append(0)
        logits, cache = original(ids, phi, cache)
        computed = seen[-1] if seen else 0
        if generation_calls[-1] == 0:
            generation_calls[-1] = computed
        with torch.no_grad():
            expected = model(torch.tensor([session._action_ids]), phi)[:, -1:]
        torch.testing.assert_close(logits, expected, atol=1e-10, rtol=1e-8)
        forced = torch.full_like(logits, -100)
        forced[0, 0, next(iterator)] = 100
        return forced, cache
    monkeypatch.setattr(session, '_action_forward', checked)
    assert session.act()['valid']
    reference_fast = meta_model.copy_fast(session.fast)
    loss, _, _ = meta_model.observation_loss(model, reference_fast, encoder,
        session.task, session.history, canonical_action(action), 'src tests')
    expected_fast = meta_model.inner_update(reference_fast, loss, lr=.1, clip_norm=1, create_graph=False)
    result = session.observe(action, 'src tests')
    assert result['observation_nll'] == pytest.approx(loss.item(), abs=1e-10)
    for key in expected_fast:
        torch.testing.assert_close(session.fast[key], expected_fast[key])
    iterator = iter(tokens)
    seen.clear()
    assert session.act()['valid']
    # Observe cached through the last output token; next act adds obs_end + action.
    assert generation_calls[1] == 2
    hook.remove()


def test_cache_capacity_growth_reuses_storage():
    model, fast, encoder = setup()
    session = AgentSession(model, fast, encoder)
    session._observation_prefix([1, 2])
    kv_pointer = session._observation_kv[0]['k_storage'].data_ptr()
    hidden_pointer = session._hidden_cache['hidden_storage'].data_ptr()
    for length in range(3, 17):
        session._observation_prefix(list(range(1, length + 1)))
        assert session._observation_kv[0]['k_storage'].data_ptr() == kv_pointer
        assert session._hidden_cache['hidden_storage'].data_ptr() == hidden_pointer
    session._observation_prefix(list(range(1, 18)))
    assert session._observation_kv[0]['k_storage'].shape[2] == 32
    assert session._hidden_cache['hidden_storage'].shape[1] == 32
