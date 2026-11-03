from __future__ import annotations

import math
import torch
from torch.nn import functional as F
from .checkpoint import load_checkpoint
from .protocol import canonical_action, parse_action, parse_tool_message
from .model.meta_model import copy_fast, detach_fast, finite, inner_update
from .model.cache import append_cached

class AgentSession:
    """Public inference interface: reset -> act -> execute externally -> observe.

    This instance owns its model and freezes theta. Each reset starts from the
    saved initial fast state; observations only update this session's phi.
    """
    def __init__(self, model, initial_fast, encoder, *, inner_lr=0.1, inner_clip=1.0,
                 adapt=True):
        self.model, self.encoder = model, encoder
        model.requires_grad_(False)
        model.eval()
        self.initial_fast = copy_fast(initial_fast)
        self.inner_lr, self.inner_clip, self.adapt = inner_lr, inner_clip, adapt
        self.task = None
        self._pending_action = None
        self.history = []
        self.fast = copy_fast(initial_fast)
        self._clear_observation_cache()

    def _clear_observation_cache(self):
        self._observation_ids = []
        self._observation_kv = [{} for _ in range(self.model.split)]
        self._observation_hidden = None
        self._hidden_cache = {}

    @torch.no_grad()
    def _observation_prefix(self, ids):
        """Cache only frozen, phi-independent layers; never cache the suffix.

        Keep prefix hidden states too: after each phi update the adaptive suffix
        must process the entire sequence again. Cache tensors have no graph.
        """
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('observation cache requires frozen model weights')
        previous = self._observation_ids
        common = 0
        for old, new in zip(previous, ids):
            if old != new:
                break
            common += 1
        if common < len(previous):
            # Canonical action JSON may differ from generated text. Only retain
            # the exact token prefix, including on repeated act() calls.
            for entry in self._observation_kv:
                for key in ('k', 'v'):
                    if key in entry:
                        entry[key] = entry[key][:, :, :common]
            if self._observation_hidden is not None:
                self._observation_hidden = self._observation_hidden[:, :common]
                self._hidden_cache['hidden'] = self._observation_hidden
            self._observation_ids = previous = previous[:common]
        start = len(previous)
        if start == len(ids):
            return self._observation_hidden
        device = next(self.model.parameters()).device
        new_ids = torch.tensor([ids[start:]], device=device)
        positions = torch.arange(start, len(ids), device=device).unsqueeze(0)
        x = self.model.embed_tokens(new_ids)
        # Attention replaces dictionary entries, so shallow copies keep a failed
        # forward from partially extending the committed cache.
        cache = [dict(entry) for entry in self._observation_kv]
        try:
            for i in range(self.model.split):
                x = self.model.layers[i](x, positions, kv_cache=cache[i])
            hidden = append_cached(self._hidden_cache, 'hidden', x, dim=1,
                                   limit=self.model.config.max_seq_len)
        except Exception:
            self._clear_observation_cache()
            raise
        self._observation_ids = list(ids)
        self._observation_kv = cache
        self._observation_hidden = hidden
        return hidden

    @torch.no_grad()
    def _action_forward(self, ids, fast, cache=None):
        if cache is None:
            self._action_ids = ids[0].tolist()
        else:
            self._action_ids.extend(ids[0].tolist())
        hidden = self._observation_prefix(self._action_ids)
        return self.model.suffix_cached(hidden, fast, cache)

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device='cpu', adapt=True, attention_backend=None, devices=None, layer_counts=None):
        model, fast, encoder, payload = load_checkpoint(checkpoint, device, attention_backend=attention_backend,
                                                     devices=devices, layer_counts=layer_counts)
        return cls(model, fast, encoder, inner_lr=payload['inner_lr'],
                   inner_clip=payload['inner_clip'], adapt=adapt)

    def reset(self, task, observation=''):
        if not isinstance(task, str) or not task.strip():
            raise ValueError('task must be nonempty text')
        self.task = task
        self.history = self.encoder.initial_history(observation)
        self.fast = copy_fast(self.initial_fast)
        self._pending_action = None
        self._clear_observation_cache()

    def _ready(self):
        if self.task is None:
            raise RuntimeError('call reset(task, observation) first')

    @torch.no_grad()
    def act(self, temperature=0.0):
        self._ready()
        self._pending_action = None
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError('invalid decoding settings')
        prompt = self.encoder.prompt(self.task, self.history)
        if len(prompt) >= self.model.config.max_seq_len:
            raise ValueError('complete prompt leaves no generation space within model context capacity; no text was truncated')
        ids = torch.tensor([prompt], device=next(self.model.parameters()).device)
        cache, generated, ended = None, [], False
        end = self.encoder.tok.action_end_id
        while len(prompt) + len(generated) < self.model.config.max_seq_len:
            logits, cache = self._action_forward(ids, self.fast, cache)
            scores = logits[0, -1].float().clone()
            finite([scores], 'generation logits')
            if not self.encoder.tok.native:
                for name, token in self.encoder.tok.ids.items():
                    if name != 'action_end':
                        scores[token] = -torch.inf
            token = (int(scores.argmax()) if temperature == 0
                     else int(torch.multinomial((scores / temperature).softmax(-1), 1)))
            if token == end:
                ended = True
                break
            generated.append(token)
            ids = torch.tensor([[token]], device=ids.device)
        text = self.encoder.tok.decode(generated)
        try:
            if not ended:
                raise ValueError('action reached model context capacity without an end-of-message token')
            parsed = (parse_tool_message(text) if self.encoder.tok.native else
                      {'action': parse_action(text), 'reasoning_content': None})
            self._pending_action = (canonical_action(parsed['action']), parsed['reasoning_content'])
            return {**parsed, 'raw': text, 'valid': True, 'ended': True}
        except (ValueError, TypeError) as error:
            return {'action': None, 'raw': text, 'valid': False, 'ended': ended, 'error': str(error)}

    @torch.enable_grad()
    def observe(self, action, output, *, reasoning_content=None):
        self._ready()
        if not isinstance(output, str):
            raise ValueError('output must be real terminal text')
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            raise ValueError('reasoning_content must be text or null')
        if isinstance(action, str):
            if self.encoder.tok.native and action.lstrip().startswith('<'):
                parsed = parse_tool_message(action)
                action = parsed['action']
                if reasoning_content is None:
                    reasoning_content = parsed['reasoning_content']
            else:
                action = parse_action(action)
        text = canonical_action(action)
        if reasoning_content is None and self._pending_action is not None and self._pending_action[0] == text:
            reasoning_content = self._pending_action[1]
        prompt, targets = self.encoder.observation_example(self.task, self.history, text, output,
                                                          reasoning_content=reasoning_content)
        sequence = prompt + targets
        if len(sequence) > self.model.config.max_seq_len:
            raise ValueError(f'complete prompt and targets require {len(sequence)} tokens, '
                             f'exceeding model context capacity {self.model.config.max_seq_len}; '
                             'no text was truncated')
        hidden = self._observation_prefix(sequence[:-1])
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        # Rebuild every adaptive layer with current phi, retaining its gradients.
        logits = self.model.suffix(hidden, positions, self.fast,
                                   logits_start=len(prompt) - 1)[0]
        work = logits if logits.dtype == torch.float64 else logits.float()
        count = len(targets)
        loss = F.cross_entropy(work, torch.tensor(targets, device=logits.device))
        finite([loss], 'observation loss')
        if self.adapt:
            self.fast = detach_fast(inner_update(self.fast, loss, lr=self.inner_lr,
                                                create_graph=False, clip_norm=self.inner_clip))
        self.history = self.encoder.history(self.history, text, output, reasoning_content=reasoning_content)
        self._pending_action = None
        return {'observation_nll': float(loss.detach()), 'tokens': count,
                'observation_truncated': False, 'adapted': self.adapt}
