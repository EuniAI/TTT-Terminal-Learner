import pytest
import torch

from terminal_learner.config import ModelConfig, EncodingConfig
from terminal_learner.dataloader.encoding import Encoder
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.model.transformer import create_model
from terminal_learner.model.meta_model import observation_loss, inner_update, detach_fast, copy_fast
from terminal_learner.protocol import canonical_action
from terminal_learner.session import AgentSession


@pytest.mark.parametrize('adaptive_layers', [1, 2])
@pytest.mark.parametrize('adapt', [True, False])
def test_cached_observe_matches_full_loss_and_updates(adaptive_layers, adapt):
    torch.manual_seed(21)
    torch.set_num_threads(1)
    model, fast = create_model(ModelConfig(hidden_size=8, intermediate_size=12,
        num_layers=2, adaptive_layers=adaptive_layers, num_heads=2, num_kv_heads=1,
        max_seq_len=1024, dtype='float64', qk_norm=True))
    encoder = Encoder(Tokenizer(), EncodingConfig(), 1024)
    session = AgentSession(model, fast, encoder, adapt=adapt)
    session.reset('find files', '$')
    reference_fast = copy_fast(fast)
    theta = {k: v.clone() for k, v in model.named_parameters()}
    seen = []
    hooks = [layer.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[1]))
             for layer in model.layers[:model.split]]
    for output in ['src tests', '', 'done']:
        action = canonical_action({'commands': [{'keystrokes': 'ls\n', 'duration': 1}],
                                   'task_complete': False})
        loss, count, _ = observation_loss(model, reference_fast, encoder,
            session.task, session.history, action, output)
        if adapt:
            reference_fast = detach_fast(inner_update(reference_fast, loss, lr=.1,
                                                      clip_norm=1, create_graph=False))
        old_length = len(session._observation_ids)
        seen.clear()
        result = session.observe(action, output)
        assert result['observation_nll'] == pytest.approx(float(loss.detach()), abs=1e-11)
        assert result['tokens'] == count
        for key in reference_fast:
            torch.testing.assert_close(session.fast[key], reference_fast[key], atol=1e-11, rtol=1e-9)
        assert seen == [len(session._observation_ids) - old_length] * model.split
        assert not session._observation_hidden.requires_grad
        for entry in session._observation_kv:
            assert entry['k'].shape[-2] == len(session._observation_ids)
            assert not entry['k'].requires_grad and not entry['v'].requires_grad
    for hook in hooks:
        hook.remove()
    assert all(torch.equal(p, theta[k]) for k, p in model.named_parameters())
    # Capacity errors do not mutate history, fast weights, or cached tokens.
    old_ids, old_history = list(session._observation_ids), list(session.history)
    with pytest.raises(ValueError, match='no text was truncated'):
        session.observe(action, 'x' * 1024)
    assert session._observation_ids == old_ids and session.history == old_history
    session.reset('new task')
    assert session._observation_ids == [] and session._observation_hidden is None
    assert session._observation_kv == [{} for _ in range(model.split)]


def test_prefix_cache_rebuilds_when_tokens_change():
    model, fast = create_model(ModelConfig(hidden_size=8, intermediate_size=12,
        num_layers=2, adaptive_layers=1, num_heads=2, num_kv_heads=1,
        max_seq_len=128, dtype='float64'))
    session = AgentSession(model, fast, Encoder(Tokenizer(), EncodingConfig(), 128))
    for ids in [[1, 2, 3], [1, 2, 3, 4, 5], [1, 8], [1, 8]]:
        cached = session._observation_prefix(ids)
        tensor = torch.tensor([ids])
        with torch.no_grad():
            expected = model.prefix(tensor, torch.arange(len(ids)).unsqueeze(0))
        torch.testing.assert_close(cached, expected)
