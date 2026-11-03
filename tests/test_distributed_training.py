"""Real two-process Gloo tests, plus an optional NCCL acceptance test."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from terminal_learner.config import ModelConfig, TrainConfig
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.dataloader.trajectory import read_manifest, load_episode
from terminal_learner.dataloader.encoding import Encoder
from terminal_learner.config import EncodingConfig
from terminal_learner.distributed import DataParallel
from terminal_learner.distributed_train import epoch_order, total_updates
from terminal_learner.model.transformer import create_model
from terminal_learner.model.loop import Trainer
from terminal_learner.model.meta_model import detach_fast
from terminal_learner.schedule import make_schedule
from terminal_learner.train import train


def model_config(dtype='float64'):
    return ModelConfig(hidden_size=8, intermediate_size=12, num_layers=2,
        adaptive_layers=1, num_heads=2, num_kv_heads=1, qk_norm=True,
        max_seq_len=1024, dtype=dtype)


def dataset(path):
    records = []
    for trial, count in enumerate([5, 1, 2]):
        trajectory = {'agent': {'name': 'terminus-2', 'model_name': 'anthropic/claude-opus-4-6'},
            'steps': [{'source': 'user', 'message': f'Task Description:\ntask {trial}\nCurrent terminal state:$'}]}
        for i in range(count):
            trajectory['steps'].append({'source': 'agent', 'step_id': i + 2,
                'tool_calls': [{'function_name': 'bash_command', 'arguments':
                    {'keystrokes': 'ls' + ' ' * trial, 'duration': .1}}],
                'observation': {'results': [{'content': 'file' + str(i)}]}})
        source = path / f'trajectory-{trial}.json'
        source.write_text(json.dumps(trajectory))
        result = path / f'result-{trial}.json'
        result.write_text('{}')
        records.append({'trial': str(trial), 'task': str(trial), 'reward': 0,
                        'trajectory_file': str(source), 'result_file': str(result)})
    manifest = path / 'manifest.json'
    manifest.write_text(json.dumps({'records': records}))
    return manifest


def worker(rank, rendezvous, manifest, output, mode, dtype='float64'):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        if mode == 'reduce':
            parallel = DataParallel('cpu')
            params = (torch.nn.Parameter(torch.zeros(7, dtype=torch.float64)),
                      torch.nn.Parameter(torch.zeros(3, dtype=torch.float64)))
            gradients = tuple(torch.full_like(p, rank + 1.) for p in params)
            actual = parallel.gradients(gradients, [2, 5][rank], params, bucket_bytes=24)
            for value in actual:
                torch.testing.assert_close(value, torch.full_like(value, 12 / 7))
            actual = parallel.gradients(gradients if rank == 0 else None, 3 if rank == 0 else 0,
                                        params, bucket_bytes=24)
            for value in actual:
                torch.testing.assert_close(value, torch.ones_like(value))
            try:
                parallel.local(lambda: (_ for _ in ()).throw(ValueError('injected local failure')) if rank == 1 else None)
            except RuntimeError as error:
                assert 'rank 1' in str(error) and 'injected' in str(error)
            else:
                raise AssertionError('all ranks must see the error')
            return
        torch.manual_seed(31)
        model, fast = create_model(model_config(dtype))
        original = Trainer.finish_window
        if mode == 'stop':
            def stopping(self, *args, **kwargs):
                result = original(self, *args, **kwargs)
                if rank == 1 and self.windows_completed == 2:
                    os.kill(os.getpid(), signal.SIGTERM)
                return result
            Trainer.finish_window = stopping
        if mode == 'fail_log':
            from terminal_learner.monitoring import Tracker
            original_log = Tracker.log
            def failed_log(self, row):
                if 'action_nll' in row:
                    raise RuntimeError('injected upload failure')
                return original_log(self, row)
            Tracker.log = failed_log
        try:
            trainer = train(manifest=manifest, output=output, model=model, initial_fast=fast,
                tokenizer=Tokenizer(), config=TrainConfig(outer_lr=5e-4, outer_lr_schedule='warmup_cosine'),
                epochs=2, seed=31, resume=output if mode == 'resume' else None,
                save_every_windows=50 if mode == 'fail_log' else 1)
        except RuntimeError as error:
            if mode != 'fail_log' or 'injected upload failure' not in str(error):
                raise
            return
        if mode == 'fail_log':
            raise AssertionError('expected synchronized log failure')
        torch.save({'theta': model.state_dict() if mode != 'resume' else trainer.model.state_dict(),
                    'windows': trainer.windows_completed}, str(output) + f'.rank{rank}.check')
    finally:
        dist.destroy_process_group()


def launch(tmp_path, manifest, output, mode, dtype='float64'):
    rendezvous = str(tmp_path / ('rendezvous-' + mode + '-' + Path(output).stem))
    mp.spawn(worker, args=(rendezvous, str(manifest), str(output), mode, dtype), nprocs=2, join=True)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def serial_joint_reference(manifest):
    """Same global batches, computed sequentially, with independent trajectory phi."""
    torch.manual_seed(31)
    model, initial = create_model(model_config())
    records, _ = read_manifest(manifest)
    records.sort(key=lambda r: r['trial'])
    config = TrainConfig(outer_lr=5e-4, outer_lr_schedule='warmup_cosine')
    plan = make_schedule(config, total_updates(records, config, None, 2, 31, 2), 1)
    trainer = Trainer(model, initial, Encoder(Tokenizer(), EncodingConfig(), 1024), config, lr_plan=plan)
    for epoch in range(2):
        orders = [epoch_order(records, 31, epoch, rank, 2) for rank in range(2)]
        indices, states = [0, 0], [None, None]
        while any(indices[r] < len(orders[r]) for r in range(2)):
            work, episodes = [], []
            for rank in range(2):
                episode = load_episode(orders[rank][indices[rank]]) if indices[rank] < len(orders[rank]) else None
                episodes.append(episode)
                work.append(trainer.compute_window(episode, state=states[rank]) if episode else None)
            count = sum(w.action_tokens for w in work if w)
            gradients = [sum(w.gradients[i] * (w.action_tokens / count) for w in work if w)
                         for i, _ in enumerate(model.parameters())]
            trainer.finish_window(next(w for w in work if w), gradients=gradients)
            for rank, w in enumerate(work):
                if w:
                    if w.state['next_turn'] == len(episodes[rank].turns):
                        indices[rank] += 1
                        states[rank] = None
                    else:
                        states[rank] = {**w.state, 'fast': detach_fast(w.state['fast'])}
    return trainer


def test_two_process_token_weighting_idle_rank_and_error_consensus(tmp_path):
    launch(tmp_path, '', tmp_path / 'reduce.pt', 'reduce')


@pytest.mark.parametrize('dtype', ['float64', 'bfloat16'])
def test_distributed_matches_serial_joint_batches_and_resumes(tmp_path, dtype):
    torch.set_num_threads(1)
    manifest = dataset(tmp_path)
    full, resumed = tmp_path / 'full.pt', tmp_path / 'resumed.pt'
    launch(tmp_path, manifest, full, 'full', dtype)
    reference = serial_joint_reference(manifest)
    payload = torch.load(full, weights_only=True)
    if dtype == 'float64':
        for key, value in reference.model.state_dict().items():
            torch.testing.assert_close(payload['theta'][key], value, atol=1e-12, rtol=1e-10)
    assert payload['training']['windows_completed'] == reference.windows_completed
    rank0 = torch.load(str(full) + '.rank0.check', weights_only=True)
    rank1 = torch.load(str(full) + '.rank1.check', weights_only=True)
    assert_same(rank0, rank1)
    launch(tmp_path, manifest, resumed, 'stop', dtype)
    partial = torch.load(resumed, weights_only=True)
    assert partial['training']['windows_completed'] == 2
    directory = tmp_path / partial['training']['distributed']['rank_state_dir']
    states = [torch.load(directory / f'rank-{r:05d}.pt', weights_only=True) for r in range(2)]
    assert all(s['window'] == 2 for s in states)
    assert any(s['progress']['episode_state'] is not None for s in states)
    launch(tmp_path, manifest, resumed, 'resume', dtype)
    restored = torch.load(resumed, weights_only=True)
    assert_same(restored['theta'], payload['theta'])
    assert_same(restored['training']['optimizer'], payload['training']['optimizer'])
    assert restored['training']['windows_completed'] == reference.windows_completed
    assert len(list(tmp_path.joinpath('resumed.checkpoints').glob('*.pt'))) == 3
    directories = list(tmp_path.joinpath('resumed.rank_states').iterdir())
    assert len(directories) == 3
    assert all(len(list(p.glob('rank-*.pt'))) == 2 for p in directories)
    # Distributed checkpoints remain usable by ordinary single-device inference.
    from terminal_learner.session import AgentSession
    AgentSession.from_checkpoint(resumed)
    with pytest.raises(ValueError, match='same torchrun world size'):
        train(manifest=manifest, output=tmp_path / 'invalid.pt', resume=resumed, epochs=3)


def test_distributed_logging_failure_saves_completed_window(tmp_path):
    manifest = dataset(tmp_path)
    output = tmp_path / 'failure.pt'
    launch(tmp_path, manifest, output, 'fail_log')
    payload = torch.load(output, weights_only=True)
    assert payload['training']['windows_completed'] == 1
    directory = tmp_path / payload['training']['distributed']['rank_state_dir']
    assert len(list(directory.glob('rank-*.pt'))) == 2


def test_torchrun_entrypoint(tmp_path):
    manifest = dataset(tmp_path)
    output = tmp_path / 'torchrun.pt'
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node=2', 'train.py', '--tiny', '--manifest', str(manifest),
        '--output', str(output), '--max-seq-len', '4096', '--device', 'cpu',
        '--limit-episodes', '3', '--limit-turns', '5'],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert len([r for r in rows if r['event'] == 'dataset']) == 1
    episodes = [r['trial'] for r in rows if r['event'] == 'episode']
    assert sorted(episodes) == ['0', '1', '2']  # Neither padding nor dropped trajectories.
    assert torch.load(output, weights_only=True)['training']['distributed']['world_size'] == 2


def cuda_worker(rank, rendezvous):
    from terminal_learner.model.data import Episode, Turn
    torch.cuda.set_device(rank)
    torch.manual_seed(31)
    dist.init_process_group('nccl', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        model, initial = create_model(ModelConfig(hidden_size=128, intermediate_size=64,
            num_layers=2, adaptive_layers=1, num_heads=4, num_kv_heads=1, qk_norm=True,
            dtype='bfloat16', attention_backend='flashbob', max_seq_len=128), f'cuda:{rank}')
        trainer = Trainer(model, initial, Encoder(Tokenizer(), EncodingConfig(), 128), TrainConfig())
        parallel = DataParallel(f'cuda:{rank}')
        parallel.synchronize_initial(trainer)
        episode = Episode('task', '$', [Turn('ls', 'x' * (rank + 1), 1),
            Turn('pwd' * (rank + 1), 'done', 2)], str(rank), str(rank), 0)
        work = trainer.compute_window(episode)
        reduced = parallel.gradients(work.gradients, work.action_tokens, tuple(model.parameters()))
        trainer.finish_window(work, gradients=reduced)
        flat = torch.cat([p.detach().flatten().float() for p in model.parameters()])
        copies = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(copies, flat)
        torch.testing.assert_close(copies[0], copies[1], atol=0, rtol=0)
        assert trainer.windows_completed == 1
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='two CUDA GPUs required')
def test_two_gpu_nccl_flashbob_meta_update(tmp_path):
    mp.spawn(cuda_worker, args=(str(tmp_path / 'nccl-rendezvous'),), nprocs=2, join=True)
