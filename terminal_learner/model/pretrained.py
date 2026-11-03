from __future__ import annotations

import torch
from terminal_learner.config import ModelConfig
from terminal_learner.dataloader.tokenizer import Tokenizer
from .transformer import create_model
from .backends import validate_flashbob

def import_pretrained(model_id, *, revision=None, adaptive_layers=1,
                      max_seq_len=None, dtype='float32', device='cpu', attention_backend='math', devices=None, layer_counts=None):
    """Import standard Qwen2/Qwen2.5/Qwen3 or unscaled Llama, verifying base logits.

    This is a full-parameter slow-weight model, not LoRA. Unsupported architectures
    fail explicitly. The resulting checkpoint includes all weights and tokenizer.
    """
    if attention_backend not in ('math', 'flashbob'):
        raise ValueError('attention_backend must be math or flashbob')
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if attention_backend == 'flashbob':
        source_config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        validate_flashbob(devices[0] if devices else device, getattr(torch, dtype),
                          getattr(source_config, 'head_dim', source_config.hidden_size // source_config.num_attention_heads),
                          source_config.num_attention_heads, source_config.num_key_value_heads)
    reference = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=getattr(torch, dtype),
        attn_implementation='eager', trust_remote_code=False).to('cpu' if devices else device).eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, use_fast=True, trust_remote_code=False)
    c = reference.config
    if c.model_type not in ('llama', 'qwen2', 'qwen3') or c.hidden_act != 'silu':
        raise ValueError('only standard Llama and Qwen2/Qwen2.5/Qwen3 SwiGLU models are supported')
    scaling = getattr(c, 'rope_scaling', None)
    if scaling and scaling.get('rope_type', scaling.get('type', 'default')) != 'default':
        raise ValueError('scaled RoPE is not supported')
    if (getattr(c, 'mlp_bias', False)
            or (c.model_type in ('llama', 'qwen3') and getattr(c, 'attention_bias', False))
            or getattr(c, 'use_sliding_window', False)
            or 'sliding_attention' in (getattr(c, 'layer_types', None) or [])):
        raise ValueError('unsupported biased/sliding-window architecture')
    if getattr(c, 'head_dim', c.hidden_size // c.num_attention_heads) != c.hidden_size // c.num_attention_heads:
        raise ValueError('nonstandard head dimension')
    if max_seq_len is None:
        max_seq_len = c.max_position_embeddings
    if type(max_seq_len) is not int or max_seq_len < 2:
        raise ValueError('max_seq_len must be a positive capacity >= 2 or None')
    if max_seq_len > c.max_position_embeddings or not hf_tokenizer.is_fast:
        raise ValueError('context exceeds base capacity or tokenizer is not fast')
    if len(hf_tokenizer) > c.vocab_size:
        raise ValueError('tokenizer vocabulary exceeds model vocabulary')
    tok = Tokenizer(hf_tokenizer.backend_tokenizer.to_str(), c.vocab_size,
                    protocol='qwen3' if c.model_type == 'qwen3' else 'legacy',
                    chat_template=hf_tokenizer.chat_template if c.model_type == 'qwen3' else None)
    config = ModelConfig(
        vocab_size=tok.vocab_size, hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size, num_layers=c.num_hidden_layers,
        num_heads=c.num_attention_heads, num_kv_heads=c.num_key_value_heads,
        adaptive_layers=adaptive_layers, max_seq_len=max_seq_len,
        rope_theta=c.rope_theta, rms_norm_eps=c.rms_norm_eps, dtype=dtype,
        tie_embeddings=c.tie_word_embeddings, qkv_bias=c.model_type == 'qwen2',
        qk_norm=c.model_type == 'qwen3', attention_backend=attention_backend,
        model_id=model_id, revision=getattr(c, '_commit_hash', None) or revision or 'local')
    model, fast = create_model(config, device, devices=devices, layer_counts=layer_counts)
    source, consumed = reference.state_dict(), set()
    with torch.no_grad():
        for name, target in model.state_dict().items():
            if '.prime_norm.' in name:
                continue
            key = name if name.startswith('lm_head.') else 'model.' + name
            value = source[key]
            consumed.add(key)
            if name in ('embed_tokens.weight', 'lm_head.weight'):
                target[:c.vocab_size].copy_(value)
                if target.shape[0] > c.vocab_size:  # Legacy protocols only.
                    target[c.vocab_size:].copy_(value.float().mean(0).to(target))
            else:
                if target.shape != value.shape:
                    raise ValueError(f'pretrained shape mismatch: {key}')
                target.copy_(value)
        if set(source) - consumed:
            raise ValueError(f'unsupported base weights: {set(source) - consumed}')
        for text in ('pwd\n', 'ls -la\n文件.txt\n'):
            ids = torch.tensor([tok.encode(text)], device=device)
            expected = reference(ids.to(next(reference.parameters()).device), use_cache=False).logits
            actual = model(ids, fast)[..., :c.vocab_size]
            tolerance = 0.08 if dtype == 'bfloat16' else 2e-4
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=tolerance, rtol=tolerance)
    return model, fast, tok
