"""CPU routing checks and real CUDA acceptance tests (no mocked CUDA kernels)."""
from dataclasses import replace

import pytest
import torch

from terminal_learner.config import ModelConfig, EncodingConfig, TrainConfig
from terminal_learner.model.attention import causal_attention
from terminal_learner.model.backends import validate_flashbob
from terminal_learner.model.transformer import create_model
from terminal_learner.model.meta_model import copy_fast, window_loss
from terminal_learner.model.data import Episode, Turn
from terminal_learner.dataloader.encoding import Encoder
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.checkpoint import save_checkpoint, load_checkpoint
from terminal_learner.protocol import canonical_action
from terminal_learner.session import AgentSession
from terminal_learner.model.meta_model import observation_loss, inner_update


def test_flashbob_requires_explicit_valid_backend():
    with pytest.raises(ValueError, match='CUDA'):
        create_model(ModelConfig(attention_backend='flashbob'))
    q = torch.zeros(1, 1, 2, 32)
    with pytest.raises(RuntimeError, match='startup validation'):
        causal_attention(q, q, q, 'flashbob')
    with pytest.raises(ValueError, match='head dimension'):
        validate_flashbob('cuda', torch.bfloat16, 8, 4, 1)


def test_flashbob_dispatch_arguments():
    calls = []
    def fake(q, k, v, **kwargs):
        calls.append(kwargs)
        return q
    q = torch.zeros(1, 4, 2, 128)
    k = torch.zeros(1, 1, 2, 128)
    assert causal_attention(q, k, k, 'flashbob', fake) is q
    assert calls == [{'is_causal': True, 'scale': 128 ** -0.5, 'window_size': 0}]


def test_checkpoint_backend_restore_and_explicit_override(tmp_path, monkeypatch):
    model, fast = create_model(ModelConfig())
    # Model shapes/weights are independent of backend; emulate a saved CUDA run.
    model.config.attention_backend = 'flashbob'
    path = tmp_path / 'flash.pt'
    save_checkpoint(path, model, fast, Tokenizer(), EncodingConfig(), inner_lr=0.1, inner_clip=1)
    with pytest.raises(ValueError, match='CUDA'):
        load_checkpoint(path)
    loaded, _, _, payload = load_checkpoint(path, attention_backend='math')
    assert loaded.config.attention_backend == 'math'
    assert payload['model_config']['attention_backend'] == 'flashbob'
    # Verify loading/configuring installs the validated function in every layer.
    from terminal_learner.model import backends
    fn = lambda q, k, v, **kwargs: causal_attention(q, k, v)
    monkeypatch.setattr(backends, 'validate_flashbob', lambda *args: fn)
    loaded, _, _, _ = load_checkpoint(path)
    assert all(layer.self_attn.flash_fn is fn for layer in loaded.layers)
    loaded.config.attention_backend = 'math'
    loaded.configure_backend()
    assert all(layer.self_attn.flash_fn is None for layer in loaded.layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('heads,kv_heads', [(4, 4), (32, 8)])
def test_cuda_flashbob_forward_and_double_backward(heads, kv_heads):
    torch.manual_seed(13)
    fn = validate_flashbob('cuda', torch.bfloat16, 128, heads, kv_heads)
    q = torch.randn(1, heads, 17, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, kv_heads, 17, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    probe = torch.randn_like(q)
    directions = [torch.randn_like(t) for t in (q, k, v)]
    results = []
    for backend in ('math', 'flashbob'):
        out = causal_attention(q, k, v, backend, fn)
        grads = torch.autograd.grad((out * probe).float().sum(), (q, k, v), create_graph=True)
        hvp = torch.autograd.grad(sum((g * d).float().sum() for g, d in zip(grads, directions)), (q, k, v))
        results.append((out, grads, hvp))
    torch.testing.assert_close(results[0][0], results[1][0], atol=0.03, rtol=0.03)
    for a, b in zip(results[0][1] + results[0][2], results[1][1] + results[1][2]):
        torch.testing.assert_close(a, b, atol=0.08, rtol=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_qwen3_meta_window_and_cache():
    torch.manual_seed(17)
    # Qwen3 head width and GQA ratio, with fewer heads/layers for bounded GPU use.
    config = ModelConfig(hidden_size=512, intermediate_size=64, num_layers=2,
                         num_heads=4, num_kv_heads=1, qk_norm=True,
                         dtype='bfloat16', max_seq_len=1024)
    reference, initial = create_model(config, 'cuda')
    flash, _ = create_model(replace(config, attention_backend='flashbob'), 'cuda')
    flash.load_state_dict(reference.state_dict())
    encoder = Encoder(Tokenizer(), EncodingConfig(), 1024)
    action = canonical_action({'commands': [], 'task_complete': False})
    episode = Episode('task', '$', [Turn(action, 'abc', 1), Turn(action, 'xyz', 2)], 'trial', 'task', 0)
    results = []
    for model in (reference, flash):
        loss, fast, _, _ = window_loss(model, copy_fast(initial), encoder, episode,
                    episode.turns, encoder.initial_history('$'), TrainConfig())
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        assert all(torch.isfinite(g).all() for g in gradients)
        results.append((loss, fast, gradients))
    torch.testing.assert_close(results[0][0], results[1][0], atol=0.03, rtol=0.03)
    for a, b in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(a, b, atol=0.003, rtol=0.1)
    for key in initial:
        torch.testing.assert_close(results[0][1][key], results[1][1][key], atol=0.003, rtol=0.1)
    ids = torch.tensor([[1, 2, 3, 4]], device='cuda')
    with torch.no_grad():
        _, cache = flash.forward_cached(ids[:, :3], initial)
        cached, _ = flash.forward_cached(ids[:, 3:], initial, cache)
        torch.testing.assert_close(cached[:, -1], flash(ids, initial)[:, -1], atol=0.04, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_shared_prefix_cache_and_observation_update():
    torch.manual_seed(23)
    model, initial = create_model(ModelConfig(hidden_size=512, intermediate_size=64,
        num_layers=2, adaptive_layers=1, num_heads=4, num_kv_heads=1,
        qk_norm=True, dtype='bfloat16', max_seq_len=512, attention_backend='flashbob'), 'cuda')
    encoder = Encoder(Tokenizer(), EncodingConfig(), 512)
    session = AgentSession(model, initial, encoder)
    session.reset('task', '$')
    action = canonical_action({'commands': [], 'task_complete': False})
    for output in ['abc', 'xyz']:
        prompt = encoder.prompt(session.task, session.history)
        ids = torch.tensor([prompt], device='cuda')
        actual, cache = session._action_forward(ids, session.fast)
        with torch.no_grad():
            torch.testing.assert_close(actual, model(ids, session.fast)[:, -1:], atol=.04, rtol=.03)
        reference = copy_fast(session.fast)
        loss, _, _ = observation_loss(model, reference, encoder, session.task, session.history, action, output)
        expected = inner_update(reference, loss, lr=.1, clip_norm=1, create_graph=False)
        result = session.observe(action, output)
        assert result['observation_nll'] == pytest.approx(loss.item(), abs=.03, rel=.03)
        for key in expected:
            torch.testing.assert_close(session.fast[key], expected[key], atol=.003, rtol=.1)
