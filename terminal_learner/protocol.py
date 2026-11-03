from __future__ import annotations

import json
import math
import re

TERMINAL_TOOL = {'type': 'function', 'function': {
    'name': 'terminal',
    'description': 'Execute an ordered batch of terminal commands and report whether the task is complete.',
    'parameters': {'type': 'object', 'properties': {
        'commands': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'keystrokes': {'type': 'string'}, 'duration': {'type': 'number', 'minimum': 0}},
            'required': ['keystrokes', 'duration'], 'additionalProperties': False}},
        'task_complete': {'type': 'boolean'}},
        'required': ['commands', 'task_complete'], 'additionalProperties': False}}}


def assistant_tool_message(action, reasoning_content=None):
    value = parse_action(action) if isinstance(action, str) else parse_action(canonical_action(action))
    message = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'terminal', 'arguments': value}}]}
    if reasoning_content is not None:
        if not isinstance(reasoning_content, str):
            raise ValueError('reasoning_content must be text or null')
        message['reasoning_content'] = reasoning_content
    return message


def parse_tool_message(text):
    match = re.fullmatch(r'\s*(?:<think>(?P<reasoning>.*?)</think>\s*)?<tool_call>\s*(?P<call>.*?)\s*</tool_call>\s*',
                         text, flags=re.DOTALL)
    if match is None:
        raise ValueError('expected one complete terminal <tool_call>')
    call = json.loads(match['call'])
    if not isinstance(call, dict) or set(call) != {'name', 'arguments'} or call['name'] != 'terminal':
        raise ValueError('expected terminal tool with name and arguments')
    return {'action': parse_action(canonical_action(call['arguments'])),
            'reasoning_content': match['reasoning'].strip() if match['reasoning'] is not None else None}


def parse_tool_action(text):
    return parse_tool_message(text)['action']

def canonical_action(action):
    """An action is a whole Terminus batch, not an invented command/output pair."""
    if not isinstance(action, dict) or set(action) != {'commands', 'task_complete'}:
        raise ValueError('action requires exactly commands and task_complete')
    if not isinstance(action['commands'], list) or type(action['task_complete']) is not bool:
        raise ValueError('commands must be a list and task_complete must be boolean')
    commands = []
    for command in action['commands']:
        if not isinstance(command, dict) or set(command) != {'keystrokes', 'duration'}:
            raise ValueError('each command requires keystrokes and duration')
        keys, duration = command['keystrokes'], command['duration']
        if (not isinstance(keys, str) or isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration < 0):
            raise ValueError('invalid keystrokes or duration')
        commands.append({'keystrokes': keys, 'duration': float(duration)})
    return json.dumps({'commands': commands, 'task_complete': action['task_complete']},
                      ensure_ascii=False, separators=(',', ':'))


def parse_action(text):
    action = json.loads(text)
    return json.loads(canonical_action(action))
