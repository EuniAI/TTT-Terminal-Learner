"""Trajectory-sharded synchronous meta-training with per-rank recovery state."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time
import uuid
import torch

from .checkpoint import atomic_save, save_checkpoint
from .dataloader.trajectory import load_episode
from .distributed import distributed_context
from .model.numerics import validate_finite_tree
from .monitoring import Tracker, stop_requests
from .train import _train


def epoch_order(records, seed, epoch, rank, world_size):
    order = list(records)
    random.Random(seed + epoch).shuffle(order)
    return order[rank::world_size]


def total_updates(records, config, limit_turns, epochs, seed, world_size):
    counts = {}
    for record in records:
        turns = load_episode(record).turns
        n = len(turns[:limit_turns]) if limit_turns is not None else len(turns)
        counts[record['trial']] = (math.ceil(n / config.unroll_steps)
            if config.action_supervision == 'all' or record.get('reward') == 1 else 0)
    return sum(max(sum(counts[r['trial']] for r in epoch_order(records, seed, epoch, rank, world_size))
                   for rank in range(world_size)) for epoch in range(epochs))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def aggregate_rows(rows):
    result = {'event': 'window', 'window': rows[0]['window'], 'world_size': len(rows),
              'active_ranks': sum(r['action_tokens'] > 0 for r in rows),
              'outer_lr': rows[0]['outer_lr'], 'inner_lr': rows[0]['inner_lr'],
              'outer_grad_norm': rows[0]['outer_grad_norm'],
              'window_seconds': max(r['window_seconds'] for r in rows)}
    for kind in ('action', 'observation'):
        count = sum(r[kind + '_tokens'] for r in rows)
        result[kind + '_tokens'] = count
        result[kind + '_nll'] = (sum((r[kind + '_nll'] or 0) * r[kind + '_tokens'] for r in rows) / count
                                if count else None)
    result['target_tokens_per_second'] = (result['action_tokens'] + result['observation_tokens']) / max(result['window_seconds'], 1e-30)
    inner = [s for row in rows for s in row['steps'] if 'inner_grad_norm' in s]
    for metric in ('inner_grad_norm', 'inner_update_norm', 'inner_planned_update_norm'):
        result[metric] = sum(s[metric] for s in inner) / max(1, len(inner))
    result['inner_clip_ratio'] = sum(s['inner_clipped'] for s in inner) / max(1, len(inner))
    for metric in ('peak_allocated_bytes', 'peak_reserved_bytes'):
        values = [r[metric] for r in rows if r[metric] is not None]
        result[metric] = max(values) if values else None  # Worst rank, not sum of peaks.
    return result


def _load_rank_state(run, parallel):
    metadata = run.prior['distributed']
    relative = Path(metadata['rank_state_dir'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('invalid rank state directory')
    base = Path(run.resume).resolve().parent
    candidates = [base / relative, base.parent / relative]  # latest or retained snapshot
    directory = next((p for p in candidates if p.is_dir()), None)
    if directory is None:
        raise FileNotFoundError('rank state directory missing; copy it together with distributed checkpoint')
    state = torch.load(directory / f'rank-{parallel.rank:05d}.pt', map_location='cpu', weights_only=True)
    validate_finite_tree(state, 'rank_checkpoint')
    expected = (parallel.rank, parallel.world_size, run.prior['windows_completed'], run.completed_epochs,
                run.fingerprint)
    actual = tuple(state[k] for k in ('rank', 'world_size', 'window', 'epoch', 'data_fingerprint'))
    if actual != expected:
        raise ValueError('rank checkpoint identity/cursor mismatch')
    torch.set_rng_state(state['torch_rng'])
    random.setstate(state['python_rng'])
    if parallel.device.type == 'cuda':
        if state['cuda_rng'] is None:
            raise ValueError('CUDA resume requires CUDA rank RNG state')
        torch.cuda.set_rng_state(state['cuda_rng'], parallel.device)
    return state['progress']


def _prune_rank_states(output, active_directory):
    output = Path(output)
    history = output.parent / (output.stem + '.checkpoints')
    retained = {p.stem for p in history.glob('window-*.pt')}
    for path in active_directory.parent.iterdir():
        match = re.fullmatch(r'(window-\d{12})-[0-9a-f]{32}', path.name)
        if not match or not path.is_dir() or path.is_symlink() or path == active_directory:
            continue
        if match[1] not in retained or match[1] == active_directory.name[:19]:
            shutil.rmtree(path)


def train_distributed(**kwargs):
    with distributed_context(kwargs.get('device', 'cpu'),
                             timeout_seconds=kwargs.get('distributed_timeout_seconds', 1800)) as parallel:
        kwargs['device'] = str(parallel.device)
        run = parallel.local(lambda: _train(**kwargs, _prepare_only=True, _parallel=parallel))
        trainer = run.trainer
        signatures = parallel.objects((run.fingerprint, asdict(run.config), asdict(run.model.config),
            asdict(run.encoding_config), run.tokenizer.protocol,
            str(Path(run.output).resolve()), run.seed, run.epochs, run.limit_episodes, run.limit_turns,
            run.distributed_bucket_mb, run.save_every_windows, run.wandb_mode))
        if any(s != signatures[0] for s in signatures):
            raise ValueError('dataset, model, or training configuration differs across ranks')
        if run.prior:
            cursor = parallel.local(lambda: _load_rank_state(run, parallel))
        else:
            parallel.synchronize_initial(trainer)
            cursor = {'episode_index': 0, 'episode_state': None}
        tracker = parallel.root(lambda: Tracker(mode=run.wandb_mode, project=run.wandb_project,
            name=run.wandb_name, config={'training': asdict(run.config), 'model': asdict(run.model.config),
                                       'encoding': asdict(run.encoding_config),
                                       'world_size': parallel.world_size},
            previous=(run.prior or {}).get('wandb'), entity=run.wandb_entity,
            directory=run.wandb_dir or Path(run.output).parent, new_run=run.wandb_new_run))
        identity = parallel.objects(tracker.identity if tracker else None)[0]
        epoch = run.completed_epochs

        def checkpoint(reason, *, report=True):
            started = time.perf_counter()
            unique = parallel.objects(uuid.uuid4().hex if parallel.primary else None)[0]
            relative = Path(Path(run.output).stem + '.rank_states') / f'window-{trainer.windows_completed:012d}-{unique}'
            directory = Path(run.output).parent / relative
            def save_rank():
                state = dict(rank=parallel.rank, world_size=parallel.world_size,
                    window=trainer.windows_completed, epoch=epoch, data_fingerprint=run.fingerprint,
                    progress=cpu_tree(cursor), torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state(parallel.device) if parallel.device.type == 'cuda' else None)
                validate_finite_tree(state, 'rank_checkpoint')
                atomic_save(state, directory / f'rank-{parallel.rank:05d}.pt')
            parallel.local(save_rank)
            def save_main():
                training = dict(manifest=str(Path(run.manifest).resolve()), data_fingerprint=run.fingerprint,
                    train_tasks=run.train_tasks, test_tasks=run.test_tasks, train_config=asdict(run.config),
                    completed_epochs=epoch, seed=run.seed, limit_episodes=run.limit_episodes,
                    limit_turns=run.limit_turns, windows_completed=trainer.windows_completed,
                    outer_lr_plan=run.lr_plan, wandb=identity, optimizer=trainer.optimizer.state_dict(),
                    context_filter=run.context_filter,
                    distributed={'world_size': parallel.world_size, 'rank_state_dir': str(relative)},
                    torch_rng=torch.get_rng_state(), cuda_rng=[], progress=None)
                save_checkpoint(run.output, run.model, run.initial_fast, run.tokenizer, run.encoding_config,
                    inner_lr=run.config.inner_lr, inner_clip=run.config.inner_clip,
                    training=training, keep=run.keep_checkpoints)
                _prune_rank_states(run.output, directory)
            parallel.root(save_main)  # Publish latest only after all rank files are durable.
            seconds = time.perf_counter() - started
            row = dict(event='checkpoint', window=trainer.windows_completed, reason=reason,
                       path=str(run.output), checkpoint_seconds=seconds, checkpoint_saved=1,
                       completed_epochs=epoch, world_size=parallel.world_size)
            def report_root():
                print(json.dumps(row), flush=True)
                if report:
                    tracker.log(row)
            parallel.root(report_root)
            return seconds

        try:
            parallel.root(lambda: print(json.dumps(dict(event='dataset', retained=len(run.records),
                train_episodes=len(run.selected), world_size=parallel.world_size,
                action_supervision=run.config.action_supervision, model=run.model.config.model_id,
                max_seq_len=run.model.config.max_seq_len, protocol=run.tokenizer.protocol,
                enable_thinking=run.encoding_config.enable_thinking,
                outer_lr_plan=run.lr_plan)), flush=True))
            with stop_requests() as stop:
                if run.wandb_mode != 'disabled':
                    checkpoint('tracking_start')
                while epoch < run.epochs:
                    order = epoch_order(run.selected, run.seed, epoch, parallel.rank, parallel.world_size)
                    parallel.local(lambda: _validate_cursor(cursor, order))
                    episode = None
                    while True:
                        if parallel.any(stop['signal']):
                            checkpoint('signal')
                            return trainer
                        started = time.perf_counter()
                        def next_episode():
                            nonlocal episode
                            while cursor['episode_index'] < len(order):
                                if episode is None:
                                    episode = load_episode(order[cursor['episode_index']])
                                turns = episode.turns[:run.limit_turns] if run.limit_turns is not None else episode.turns
                                if turns and (run.config.action_supervision == 'all' or episode.reward == 1):
                                    return episode
                                cursor.update(episode_index=cursor['episode_index'] + 1, episode_state=None)
                                episode = None
                            return None
                        active = parallel.local(next_episode)
                        if not parallel.any(active is not None):
                            break
                        trainer.optimizer.zero_grad(set_to_none=True)
                        work = parallel.local(lambda: trainer.compute_window(active, state=cursor['episode_state'],
                            limit_turns=run.limit_turns) if active is not None else None)
                        gradients = parallel.gradients(work.gradients if work else None,
                            work.action_tokens if work else 0, tuple(run.model.parameters()),
                            bucket_bytes=run.distributed_bucket_mb * 1024 * 1024)
                        if work is not None:
                            work.gradients = ()  # Reduced FP32 gradients now own the update.
                        row = parallel.local(lambda: trainer.finish_window(work, gradients=gradients))
                        del gradients, work
                        episode_report = None
                        if active is not None:
                            state = trainer.episode_state
                            turns = active.turns[:run.limit_turns] if run.limit_turns is not None else active.turns
                            if state['next_turn'] == len(turns):
                                episode_report = trainer.episode_report(active)
                                cursor.update(episode_index=cursor['episode_index'] + 1, episode_state=None)
                                trainer.episode_state = None
                                episode = None
                            else:
                                cursor['episode_state'] = state
                        row['window_seconds'] = time.perf_counter() - started
                        rows = parallel.objects(row)
                        reports = parallel.objects(episode_report)
                        global_row = aggregate_rows(rows)
                        global_row['epoch'] = epoch + 1
                        saved = trainer.windows_completed % run.save_every_windows == 0
                        seconds = checkpoint('periodic', report=False) if saved else 0.0
                        global_row.update(checkpoint_seconds=seconds, checkpoint_saved=int(saved))
                        error = None
                        if parallel.primary:
                            try:
                                print(json.dumps(global_row), flush=True)
                                for rank, report in enumerate(reports):
                                    if report:
                                        print(json.dumps(dict(event='episode', epoch=epoch + 1, rank=rank, **report)), flush=True)
                                tracker.log(global_row)
                            except Exception as caught:
                                error = caught
                        if parallel.any(error is not None):
                            if not saved:
                                checkpoint('logging_failure', report=False)
                            parallel.agree(error)
                    if trainer.windows_completed == 0:
                        raise ValueError('no trainable episodes under the selected supervision policy')
                    epoch += 1
                    cursor.update(episode_index=0, episode_state=None)
                    checkpoint('epoch_end')
        finally:
            # Finish collectively for a normal exit; do not mask an earlier error.
            failed = sys.exc_info()[0] is not None
            if failed:
                if tracker:
                    tracker.close(failed=True)
            else:
                parallel.root(lambda: tracker.close())
        return trainer


def _validate_cursor(cursor, order):
    if not 0 <= cursor['episode_index'] <= len(order):
        raise ValueError('invalid distributed episode cursor')
    state = cursor['episode_state']
    if state is not None and (cursor['episode_index'] == len(order) or state['trial'] != order[cursor['episode_index']]['trial']):
        raise ValueError('distributed episode state does not match assigned trajectory')
