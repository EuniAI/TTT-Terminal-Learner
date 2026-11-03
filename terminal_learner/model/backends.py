"""Validated optional CUDA backend; never silently replace FlashBoB with math."""
import importlib.metadata
import json

import torch

FLASHBOB_REVISION = 'f2bd4a38a9b65bae455ef4336d7b54f0acd3caca'


def validate_flashbob(device, dtype, head_dim, heads, kv_heads):
    if torch.device(device).type != 'cuda':
        raise ValueError('FlashBoB requires CUDA; select attention_backend=math for CPU')
    if dtype != torch.bfloat16 or head_dim not in (32, 64, 128):
        raise ValueError('FlashBoB requires bfloat16 and head dimension 32/64/128 in this project')
    if kv_heads < 1 or heads % kv_heads or heads // kv_heads > 32:
        raise ValueError('FlashBoB requires an integer Q/KV head ratio <= 32')
    if not torch.cuda.is_available():
        raise RuntimeError('FlashBoB selected but CUDA is unavailable')
    version = '.'.join(torch.__version__.split('.')[:2])
    expected = {'2.8': '3.4', '2.9': '3.5'}.get(version)
    if expected is None:
        raise RuntimeError('FlashBoB requires PyTorch 2.8/Triton 3.4 or PyTorch 2.9/Triton 3.5')
    try:
        import triton
        import bob
    except ImportError as error:
        raise RuntimeError('Install requirements-training-cuda.txt for FlashBoB') from error
    if '.'.join(triton.__version__.split('.')[:2]) != expected:
        raise RuntimeError(f'PyTorch {version} requires Triton {expected} for FlashBoB')
    revisions = []
    for name in importlib.metadata.packages_distributions().get('bob', []):
        direct = importlib.metadata.distribution(name).read_text('direct_url.json')
        if direct:
            revisions.append(json.loads(direct).get('vcs_info', {}).get('commit_id'))
    if FLASHBOB_REVISION not in revisions:
        raise RuntimeError('FlashBoB commit provenance missing or mismatched; install requirements-training-cuda.txt')
    return bob.sdpa_bob


def resolve_backend(config, device):
    config.validate()
    if config.attention_backend == 'math':
        return None
    return validate_flashbob(device, config.torch_dtype, config.hidden_size // config.num_heads,
                             config.num_heads, config.num_kv_heads)
