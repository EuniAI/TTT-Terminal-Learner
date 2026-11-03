"""Explicit outer-gradient synchronization, outside the higher-order graph.

DDP reducer hooks are intentionally not used: the local meta-learning graph uses
nested autograd.grad. Only fully computed theta gradients are communicated.
"""
from contextlib import contextmanager
from datetime import timedelta
import os
import torch
import torch.distributed as dist


class DataParallel:
    def __init__(self, device):
        self.device = torch.device(device)
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()

    @property
    def primary(self):
        return self.rank == 0

    def objects(self, value):
        results = [None] * self.world_size
        dist.all_gather_object(results, value)
        return results

    def any(self, value):
        flag = torch.tensor(int(bool(value)), device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def agree(self, error=None):
        """All ranks leave a failed local phase together, before the next collective."""
        if self.any(error is not None):
            errors = self.objects(None if error is None else f'{type(error).__name__}: {error}')
            details = '; '.join(f'rank {i}: {e}' for i, e in enumerate(errors) if e)
            raise RuntimeError(f'distributed operation failed: {details}') from error

    def local(self, fn):
        error, result = None, None
        try:
            result = fn()
        except Exception as caught:
            error = caught
        self.agree(error)
        return result

    def root(self, fn):
        return self.local(lambda: fn() if self.primary else None)

    @torch.no_grad()
    def synchronize_initial(self, trainer):
        signature = (trainer.model.config, [(k, tuple(v.shape)) for k, v in trainer.initial_fast.items()])
        signatures = self.objects(signature)
        if any(s != signatures[0] for s in signatures):
            raise ValueError('model configuration differs across ranks')
        for parameter in trainer.model.parameters():
            dist.broadcast(parameter, src=0)
        for value in trainer.initial_fast.values():
            dist.broadcast(value, src=0)
        # Masters were allocated during preparation, before the broadcast.
        for parameter, master in zip(trainer.model.parameters(), trainer.optimizer.master_parameters):
            if parameter is not master:
                master.copy_(parameter)

    @torch.no_grad()
    def gradients(self, local_gradients, local_tokens, parameters, bucket_bytes=25 * 1024 * 1024):
        """Return sum(token_count * mean_gradient) / global_token_count.

        An exhausted rank contributes zeros but receives the same outer update.
        FP32 reductions are used for BF16 models; float64 test models retain it.
        Communication storage is bounded, including for a large embedding matrix.
        """
        count = torch.tensor(local_tokens, dtype=torch.int64, device=self.device)
        dist.all_reduce(count)
        total = int(count.item())
        if total <= 0:
            raise ValueError('no action tokens in global window')
        dtype = torch.float64 if parameters[0].dtype == torch.float64 else torch.float32
        capacity = max(1, bucket_bytes // torch.empty((), dtype=dtype).element_size())
        capacity = min(capacity, sum(p.numel() for p in parameters))
        buffer = torch.empty(capacity, dtype=dtype, device=self.device)
        outputs = [torch.empty_like(p, dtype=dtype) for p in parameters]
        entries, used = [], 0

        def flush():
            nonlocal entries, used
            dist.all_reduce(buffer[:used])
            for index, offset, size, bucket_offset in entries:
                outputs[index].view(-1)[offset:offset + size].copy_(buffer[bucket_offset:bucket_offset + size])
            entries, used = [], 0

        weight = local_tokens / total
        for index, parameter in enumerate(parameters):
            source = local_gradients[index].reshape(-1) if local_gradients is not None else None
            offset = 0
            while offset < parameter.numel():
                size = min(capacity - used, parameter.numel() - offset)
                target = buffer[used:used + size]
                if source is None:
                    target.zero_()
                else:
                    target.copy_(source[offset:offset + size])
                    target.mul_(weight)
                entries.append((index, offset, size, used))
                used += size
                offset += size
                if used == capacity:
                    flush()
        if used:
            flush()
        return outputs


@contextmanager
def distributed_context(device, *, timeout_seconds=1800):
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError('distributed timeout must be a positive integer')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise ValueError('LOCAL_RANK does not identify an available CUDA device')
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(local_rank)
        backend = 'nccl'
    elif str(device) == 'cpu':
        backend = 'gloo'
    else:
        raise ValueError('distributed training supports CUDA/NCCL or CPU/Gloo')
    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
    elif dist.get_backend() != backend:
        raise ValueError('existing process group backend does not match training device')
    try:
        yield DataParallel(device)
    finally:
        if owned:
            dist.destroy_process_group()
