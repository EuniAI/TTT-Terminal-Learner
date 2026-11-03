from __future__ import annotations

from terminal_learner.protocol import TERMINAL_TOOL, assistant_tool_message

class Encoder:
    def __init__(self, tokenizer, config, max_seq_len):
        config.validate(max_seq_len)
        self.tok, self.config, self.max_seq_len = tokenizer, config, max_seq_len
        if config.enable_thinking and not tokenizer.native:
            raise ValueError("thinking mode requires the native Qwen3 protocol")

    def task_prefix(self, task):
        if self.tok.native:
            return self.tok.encode(self._render(self._messages(task, [])))
        return [self.tok.ids['task']] + self.tok.encode(task)

    def history(self, previous, action, observation, *, reasoning_content=None):
        if self.tok.native:
            return previous + [self._assistant(action, reasoning_content), {'role': 'tool', 'content': observation}]
        turn = ([self.tok.ids['action']] + self.tok.encode(action)
                + [self.tok.ids['action_end'], self.tok.ids['observation']]
                + self.tok.encode(observation) + [self.tok.ids['obs_end']])
        return previous + turn

    def initial_history(self, observation):
        if self.tok.native:
            return [{'role': 'user', 'content': 'Current terminal state:\n' + observation}]
        return [self.tok.ids['observation']] + self.tok.encode(observation) + [self.tok.ids['obs_end']]

    def prompt(self, task, history, *, action=None, reasoning_content=None):
        if self.tok.native:
            if action is None:
                return self.tok.encode(self._render(self._messages(task, history), generation=True))
            prefix, _ = self._observation_parts(task, history, action, reasoning_content)
            return self.tok.encode(prefix)
        context = self.task_prefix(task) + history
        if action is None:
            return context + [self.tok.ids['action']]
        return context + [self.tok.ids['action']] + self.tok.encode(action) + [
            self.tok.ids['action_end'], self.tok.ids['observation']]

    def target(self, text, kind):
        if self.tok.native:
            if kind == 'action':
                prefix, full = self._action_texts('', [], text)
                return self.tok.encode(full[len(prefix):]), False
            _, suffix = self._observation_parts('', [], {'commands': [], 'task_complete': False})
            return self.tok.encode(text + suffix), False
        ids = self.tok.encode(text)
        ids += [self.tok.ids['obs_end' if kind == 'observation' else 'action_end']]
        return ids, False

    def _messages(self, task, history):
        return [{'role': 'system', 'content': 'You are a terminal agent. Use the terminal tool to complete the task.'},
                {'role': 'user', 'content': task}] + history

    def _render(self, messages, *, generation=False):
        return self.tok.render_chat(messages, tools=[TERMINAL_TOOL], add_generation_prompt=generation,
                                    enable_thinking=self.config.enable_thinking)

    def _assistant(self, action, reasoning_content=None):
        return assistant_tool_message(action, reasoning_content if self.config.enable_thinking else None)

    def _action_texts(self, task, history, action, reasoning_content=None):
        messages = self._messages(task, history)
        prefix = self._render(messages, generation=True)
        full = self._render(messages + [self._assistant(action, reasoning_content)])
        if not full.startswith(prefix):
            raise ValueError('chat template action prefix does not match generation prompt')
        return prefix, full

    def _observation_parts(self, task, history, action, reasoning_content=None):
        # Split around a placeholder in the last tool message, never around a
        # substring from the real terminal output. Template owns all wrappers.
        marker = '__TTT_TOOL_CONTENT_BOUNDARY_7b4f3d__'
        rendered = self._render(self._messages(task, history) + [self._assistant(action, reasoning_content),
                                {'role': 'tool', 'content': marker}])
        return rendered.rsplit(marker, 1)

    def _split_example(self, prefix, full):
        """Tokenize the entire official serialization; handle BPE boundary merges.

        A token spanning the prompt/target boundary is supervised as a target;
        no answer token is ever included in the conditioning prefix.
        """
        prefix_ids, full_ids = self.tok.encode(prefix), self.tok.encode(full)
        boundary = 0
        for a, b in zip(prefix_ids, full_ids):
            if a != b:
                break
            boundary += 1
        return full_ids[:boundary], full_ids[boundary:]

    def action_example(self, task, history, action, *, reasoning_content=None):
        if self.tok.native:
            return self._split_example(*self._action_texts(task, history, action, reasoning_content))
        return self.prompt(task, history), self.target(action, 'action')[0]

    def observation_example(self, task, history, action, output, *, reasoning_content=None):
        if self.tok.native:
            prefix, suffix = self._observation_parts(task, history, action, reasoning_content)
            return self._split_example(prefix, prefix + output + suffix)
        return self.prompt(task, history, action=action), self.target(output, 'observation')[0]
