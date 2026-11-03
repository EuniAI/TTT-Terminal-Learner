"""JSONL inference service; task success is measured externally in Terminal-Bench."""
from __future__ import annotations

import argparse
import json
from .session import AgentSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    serve = commands.add_parser('serve', help='run a JSONL agent session')
    serve.add_argument('--checkpoint', required=True)
    serve.add_argument('--device', default='cpu')
    serve.add_argument('--model-parallel-devices', type=lambda s: s.split(','))
    serve.add_argument('--model-parallel-layer-counts', type=lambda s: [int(n) for n in s.split(',')])
    serve.add_argument('--attention-backend', choices=['math', 'flashbob'],
                       help='override checkpoint backend explicitly, e.g. math for CPU')
    serve.add_argument('--frozen', action='store_true', help='disable observation adaptation')
    args = parser.parse_args()
    import sys
    session = AgentSession.from_checkpoint(args.checkpoint, device=args.device, adapt=not args.frozen,
                                           attention_backend=args.attention_backend, devices=args.model_parallel_devices,
                                           layer_counts=args.model_parallel_layer_counts)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request['op']
            if operation == 'reset':
                session.reset(request['task'], request.get('observation', ''))
                result = {'ok': True}
            elif operation == 'act':
                if 'max_new_tokens' in request:
                    raise ValueError('max_new_tokens was removed; generation ends at the protocol EOS or model context capacity')
                result = session.act(temperature=request.get('temperature', 0.0))
            elif operation == 'observe':
                result = session.observe(request['action'], request['output'],
                                         reasoning_content=request.get('reasoning_content'))
            else:
                raise ValueError(f'unknown operation: {operation}')
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as error:
            print(json.dumps({'error': str(error), 'type': type(error).__name__}), flush=True)
