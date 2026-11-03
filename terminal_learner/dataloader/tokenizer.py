from __future__ import annotations

import json
import re
from pathlib import Path

PROTOCOL = ("task", "action", "observation", "action_end", "obs_end")
NATIVE_TOKENS = ('<|endoftext|>', '<|im_start|>', '<|im_end|>', '<tool_call>',
                 '</tool_call>', '<tool_response>', '</tool_response>', '<think>', '</think>')

class Tokenizer:
    """Native Qwen3 serialization, with explicit legacy checkpoint support."""
    def __init__(self, backend_json=None, base_vocab=256, *, protocol='legacy', chat_template=None):
        self.backend_json, self.base_vocab = backend_json, base_vocab
        if protocol not in ('legacy', 'qwen3'):
            raise ValueError('unknown tokenizer protocol')
        self.protocol, self.chat_template = protocol, chat_template
        self.ids = dict(zip(PROTOCOL, range(base_vocab, base_vocab + len(PROTOCOL))))
        self.vocab_size = base_vocab + len(PROTOCOL)
        self.backend = None
        if backend_json is not None:
            from tokenizers import Tokenizer as Backend
            data = json.loads(backend_json)
            if protocol == 'legacy':
                data['added_tokens'] = [x for x in data.get('added_tokens', []) if not x['special']]
            self.backend = Backend.from_str(json.dumps(data))
        if self.native:
            if not isinstance(chat_template, str) or not chat_template:
                raise ValueError('Qwen3 protocol requires its official chat_template')
            if self.backend:
                self.ids = {name: self.backend.token_to_id(name) for name in NATIVE_TOKENS}
                if any(v is None or v >= base_vocab for v in self.ids.values()):
                    raise ValueError('Qwen3 native tokens are missing from the base vocabulary')
                self.vocab_size = base_vocab  # Never add rows to pretrained embeddings/head.
            else:
                # Byte-level fixture for CPU smoke tests, not a Qwen tokenizer.
                self.ids = {name: base_vocab + i for i, name in enumerate(NATIVE_TOKENS)}
                self.vocab_size = base_vocab + len(NATIVE_TOKENS)

    @property
    def native(self):
        return self.protocol == 'qwen3'

    @property
    def action_end_id(self):
        return self.ids['<|im_end|>' if self.native else 'action_end']

    @classmethod
    def native_tiny(cls):
        template = Path(__file__).with_name('templates').joinpath('qwen3.jinja').read_text()
        return cls(protocol='qwen3', chat_template=template)

    def render_chat(self, messages, *, tools, add_generation_prompt=False, enable_thinking=False):
        if not self.native:
            raise ValueError('legacy tokenizer has no native chat template')
        from transformers.utils.chat_template_utils import render_jinja_template
        rendered, _ = render_jinja_template([messages], tools=tools, chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking,
            eos_token='<|im_end|>', pad_token='<|endoftext|>', bos_token=None)
        return rendered[0]

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError('text must be a string')
        if self.native and self.backend is None:
            parts = re.split('(' + '|'.join(re.escape(x) for x in NATIVE_TOKENS) + ')', text)
            return [token for part in parts for token in
                    ([self.ids[part]] if part in self.ids else list(part.encode('utf-8')))]
        result = (self.backend.encode(text, add_special_tokens=False).ids
                  if self.backend else list(text.encode('utf-8')))
        if not self.native and any(i >= self.base_vocab for i in result):
            raise ValueError('raw text emitted a protocol ID')
        return result

    def decode(self, ids):
        if self.backend:
            return self.backend.decode(ids, skip_special_tokens=False)
        if self.native:
            names = {value: key for key, value in self.ids.items()}
            pieces, buffer = [], bytearray()
            for value in ids:
                if value in names:
                    pieces.extend((buffer.decode('utf-8', errors='replace'), names[value]))
                    buffer.clear()
                else:
                    buffer.append(value)
            pieces.append(buffer.decode('utf-8', errors='replace'))
            return ''.join(pieces)
        return bytes(ids).decode('utf-8', errors='replace')

    def state_dict(self):
        return {'backend_json': self.backend_json, 'base_vocab': self.base_vocab,
                'protocol': self.protocol, 'chat_template': self.chat_template}
