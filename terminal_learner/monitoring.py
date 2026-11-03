"""Optional scalar tracking and cooperative termination at window boundaries."""
from contextlib import contextmanager
import signal
import threading
import uuid
import warnings
import random
import math
from pathlib import Path
import torch


@contextmanager
def stop_requests():
    state = {'signal': None}
    previous = {}
    def request(signum, frame):
        state['signal'] = signal.Signals(signum).name
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, request)
    try:
        yield state
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class Tracker:
    def __init__(self, *, mode='disabled', project=None, name=None, config=None, previous=None,
                 entity=None, directory=None, new_run=False):
        if mode not in ('disabled', 'offline', 'online'):
            raise ValueError('wandb_mode must be disabled, offline or online')
        if type(new_run) is not bool or (new_run and mode == 'disabled'):
            raise ValueError('new_run requires enabled W&B tracking and a boolean flag')
        self.run = None
        self.identity = previous
        if mode == 'disabled':
            return  # Keep the identity through untracked continuation.
        project = project or (previous or {}).get('project')
        if not project:
            raise ValueError('wandb_project is required when tracking is enabled')
        entity = entity if entity is not None else (previous or {}).get('entity')
        if previous and not new_run:
            if project != previous['project']:
                raise ValueError('W&B project differs from checkpoint; use --wandb-new-run to fork logging')
            if entity != previous.get('entity'):
                raise ValueError('W&B entity differs from checkpoint; use --wandb-new-run to fork logging')
        resume = bool(previous and previous['mode'] == 'online' and mode == 'online' and not new_run)
        run_id = previous['id'] if resume else uuid.uuid4().hex[:16]
        payload = dict(config or {})
        if previous and not resume:
            payload['parent_run_id'] = previous['id']
        if directory is not None:
            directory = str(Path(directory).expanduser().resolve())
            Path(directory).mkdir(parents=True, exist_ok=True)
        import wandb
        python_rng = random.getstate()
        try:
            with torch.random.fork_rng():
                self.run = wandb.init(project=project, entity=entity, id=run_id, name=name,
                    mode=mode, resume='must' if resume else ('never' if mode == 'online' else None),
                    config=payload, dir=directory, save_code=False, settings=wandb.Settings(console='off'))
        finally:
            random.setstate(python_rng)
        try:
            # A recovered checkpoint can precede uploaded metrics. Keep W&B's
            # internal step monotonic and use the training window as the x axis.
            self.run.define_metric('window')
            self.run.define_metric('*', step_metric='window')
            self.identity = {'id': self.run.id, 'project': self.run.project,
                             'entity': self.run.entity, 'mode': mode}
        except BaseException:
            self.close(failed=True)
            raise

    def log(self, row):
        if self.run:
            self.run.log({k: v for k, v in row.items() if type(v) in (int, float) and math.isfinite(v)})

    def close(self, *, failed=False):
        if self.run:
            try:
                self.run.finish(exit_code=1 if failed else 0)
            except Exception:
                if not failed:
                    raise
                warnings.warn('W&B finish failed while handling an earlier error', RuntimeWarning)
