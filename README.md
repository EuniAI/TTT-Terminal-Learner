# TTT Terminal Learner

**Act. Observe. Adapt.**

A terminal agent that updates its fast weights from real terminal feedback—even at inference time. Built on **Qwen3-8B**, with native thinking and tool calling.

```text
Task → Think & act → Execute in terminal → Learn from output → Next action
```

During training, observation loss updates fast weights in the last **9 of 36 layers**. Expert action loss then backpropagates through those updates, teaching the model how to adapt. At inference, the backbone stays frozen; only fast weights change. No expert labels needed.

- **Real feedback:** teacher forcing on actual terminal output.
- **Meta-learning:** differentiable inner SGD, outer AdamW, and FlashBoB attention.
- **Four-GPU training:** layer model parallelism, resumable checkpoints, optional W&B.
- **445 expert trajectories:** Terminus 2 + Claude Opus 4.6, including failed runs.

## Quick start

Linux + CUDA, Python 3.11/3.12. The default configuration uses four visible GPUs in **one process**.

```bash
pip install -r requirements-training-cuda.txt
python scripts/prepare_data.py

# Validate the CUDA backend and cross-device gradients.
python -m pytest tests/test_model_parallel.py tests/test_flashbob.py -q

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --config configs/experiment/terminal_ttt.yaml
```

The data script downloads only the selected expert trajectories and verifies file hashes. Configure training in [`terminal_ttt.yaml`](configs/experiment/terminal_ttt.yaml); resume with `--resume runs/agent.pt`.

## Run the agent

```bash
python inference.py serve --checkpoint runs/agent.pt \
  --model-parallel-devices cuda:0,cuda:1,cuda:2,cuda:3
```

Use the JSONL service or Python `AgentSession`: **`reset → act → execute externally → observe`**. Your terminal runner executes commands and returns their output.

**Status:** Research implementation. Full Qwen3-8B four-GPU memory usage and Terminal-Bench success rates have not yet been measured. Long-context meta-learning can still exceed GPU memory; a Harbor adapter is not included.

[Training guide](TRAINING_GUIDE.md) · [Design notes](TTT-Terminal-Agent-Design.md) *(Chinese)*

Inspired by [TTT-Terminal-World](https://github.com/EuniAI/TTT-Terminal-World).
