"""CPU configuration checks and real CUDA higher-order placement acceptance tests."""
import pytest
import torch

from terminal_learner.config import ModelConfig, TrainConfig, EncodingConfig
from terminal_learner.cli import parse_args
from terminal_learner.model.transformer import TerminalTransformer, create_model
from terminal_learner.model.placement import layer_partition, validate_devices
from terminal_learner.model.loop import Trainer
from terminal_learner.model.data import Episode, Turn
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.dataloader.encoding import Encoder
from terminal_learner.checkpoint import save_checkpoint, load_checkpoint
from terminal_learner.session import AgentSession


def test_qwen_partition_without_allocating_weights():
    with torch.device('meta'):
        model = TerminalTransformer(ModelConfig(vocab_size=151936, hidden_size=4096,
            intermediate_size=12288, num_layers=36, num_heads=32, num_kv_heads=8,
            adaptive_layers=9, qk_norm=True, dtype='bfloat16'))
    counts = layer_partition(model, 4)
    assert sum(counts) == 36 and len(counts) == 4
    assert counts[-1] < counts[1]  # Fast MLPs plus the head consume static memory.
    assert layer_partition(model, 4, [9, 9, 9, 9]) == [9, 9, 9, 9]
    for counts in ([9, 9, 9], [9, 9, 9, 8], [12, 12, 12, 0]):
        with pytest.raises(ValueError):
            layer_partition(model, 4, counts)


def test_config_and_invalid_launch(monkeypatch):
    args = parse_args(['--model-parallel-devices', 'cuda:0,cuda:1',
                       '--model-parallel-layer-counts', '20,16'])
    assert args.config.run.model_parallel_layer_counts == [20, 16]
    assert parse_args(['--model-parallel-devices', 'none']).config.run.model_parallel_devices is None
    with pytest.raises(ValueError, match='distinct'):
        validate_devices(['cuda:0', 'cuda:0'])
    with pytest.raises(ValueError, match='explicit'):
        validate_devices(['cpu'])
    from terminal_learner.train import train
    monkeypatch.setenv('WORLD_SIZE', '4')
    with pytest.raises(ValueError, match='not torchrun'):
        train(model_parallel_devices=['cuda:0', 'cuda:1'])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two real CUDA GPUs')
@pytest.mark.parametrize('backend,dtype', [('math', 'float64'), ('math', 'bfloat16'), ('flashbob', 'bfloat16')])
def test_cross_device_meta_update_resume_and_cache(tmp_path, backend, dtype):
    # Three adaptive layers span BOTH stages: global clipping must cross cards.
    torch.manual_seed(42)
    config = ModelConfig(hidden_size=64, intermediate_size=80, num_layers=4,
                         num_heads=2, num_kv_heads=1, adaptive_layers=3,
                         max_seq_len=1024, dtype=dtype, attention_backend=backend)
    single, phi = create_model(config, 'cuda:0')
    split, _ = create_model(config, devices=['cuda:0', 'cuda:1'], layer_counts=[2, 2])
    split.load_state_dict(single.state_dict())
    split_phi = split.place_fast(phi)
    split.validate_fast(split_phi)
    assert {v.device.index for v in split_phi.values()} == {0, 1}
    tokenizer = Tokenizer()
    enc = Encoder(tokenizer, EncodingConfig(), 1024)
    episode = Episode('task', '$', [Turn('a', 'result', 1), Turn('b', 'done', 2)], 'trial', 'task', 0)
    cfg = TrainConfig(inner_clip=0.001, outer_lr=1e-3)
    a, b = Trainer(single, phi, enc, cfg), Trainer(split, split_phi, enc, cfg)
    wa, wb = a.compute_window(episode), b.compute_window(episode)
    tolerance = dict(atol=1e-9, rtol=1e-7) if dtype == 'float64' else dict(atol=0.003, rtol=0.05)
    for ga, gb in zip(wa.gradients, wb.gradients):
        torch.testing.assert_close(ga.cpu(), gb.cpu(), **tolerance)
    for x, y in zip(wa.stats, wb.stats):
        assert x['inner_clipped'] == y['inner_clipped']
        assert x['inner_grad_norm'] == pytest.approx(y['inner_grad_norm'], rel=0.03)
    a.finish_window(wa)
    b.finish_window(wb)
    for x, y in zip(single.parameters(), split.parameters()):
        torch.testing.assert_close(x.cpu(), y.cpu(), **tolerance)
    for master, parameter in zip(b.optimizer.master_parameters, split.parameters()):
        assert master.device == parameter.device
        assert b.optimizer.state[master]['exp_avg'].device == parameter.device
    path = tmp_path / 'agent.pt'
    save_checkpoint(path, split, split_phi, tokenizer, enc.config, inner_lr=cfg.inner_lr,
                    inner_clip=cfg.inner_clip, training={'optimizer': b.optimizer.state_dict()})
    restored, restored_phi, restored_enc, payload = load_checkpoint(
        path, devices=['cuda:0', 'cuda:1'], layer_counts=[2, 2])
    resumed = Trainer(restored, restored_phi, restored_enc, cfg)
    resumed.optimizer.load_state_dict(payload['training']['optimizer'])
    # Continue from the same adapted phi, simulating a CPU checkpoint cursor.
    state = {**b.episode_state, 'next_turn': 0,
             'fast': {k: v.detach().cpu() for k, v in b.episode_state['fast'].items()}}
    w1 = b.compute_window(episode, state=state)
    w2 = resumed.compute_window(episode, state=state)
    for g1, g2 in zip(w1.gradients, w2.gradients):
        torch.testing.assert_close(g1, g2, rtol=0, atol=0)
    session = AgentSession(restored, restored_phi, restored_enc)
    session.reset('task', '$')
    prompt, targets = restored_enc.observation_example('task', session.history, 'a', 'result')
    ids = torch.tensor([prompt + targets[:-1]], device='cuda:0')
    with torch.no_grad():
        direct = restored(ids, session.fast, logits_start=len(prompt)-1)[0]
        hidden = session._observation_prefix(ids[0].tolist())
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        cached = restored.suffix(hidden, positions, session.fast, logits_start=len(prompt)-1)[0]
    torch.testing.assert_close(direct, cached, **tolerance)
