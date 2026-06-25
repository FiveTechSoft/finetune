# SWE-bench GRPO Training Scripts

Scripts for training Qwen3-Coder with GRPO on SWE-bench-Verified.

## Overview

This pipeline implements:
1. **SFT Phase**: Supervised Fine-Tuning on distilled trajectories (learn tool-use format)
2. **GRPO Phase**: Group Relative Policy Optimization with test execution reward (learn to resolve issues)

## Scripts

| Script | Description |
|--------|-------------|
| `grader_endpoint.py` | FastAPI endpoint for grading patches via Docker |
| `reward_functions.py` | Reward functions for GRPO training |
| `calibrate_grader.py` | Pre-training calibration and hackability tests |
| `generate_trajectories.py` | Generate SFT trajectories from base model |
| `train_sft.py` | Phase 1: SFT training |
| `train_grpo.py` | Phase 2: GRPO training |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start grader endpoint (requires Docker)
python grader_endpoint.py --port 8000

# 3. Calibrate grader
python calibrate_grader.py --grader-url http://localhost:8000

# 4. Generate SFT trajectories
python generate_trajectories.py --grader-url http://localhost:8000

# 5. Train SFT
python train_sft.py --epochs 2 --lr 2e-5

# 6. Train GRPO
python train_grpo.py --sft-checkpoint output/sft/final
```

## Configuration

### Environment Variables
- `GRADER_ENDPOINT`: URL of grader endpoint (default: http://localhost:8000)
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `PYTORCH_CUDA_ALLOC_CONF`: Memory allocation config

### Key Parameters

#### SFT
- `--lr`: Learning rate (default: 2e-5)
- `--epochs`: Number of epochs (default: 2)
- `--max-seq-length`: Max sequence length (default: 4096)

#### GRPO
- `--lr`: Learning rate (default: 2e-6)
- `--epochs`: Number of epochs (default: 2)
- `--num-generations`: GRPO generations per prompt (default: 8)

## Architecture

### Grader Endpoint
The grader runs patches in isolated Docker containers and executes repo tests.
Score components (total 1.0):
- 0.5: fail_to_pass ratio (issue resolved)
- 0.2: pass_to_pass ratio (no regressions)
- 0.2: patch applies cleanly
- 0.1: reasonable diff size

### Reward Functions
- `swe_bench_reward`: Main reward via grader endpoint
- `format_reward`: Auxiliary reward for proper tool-use format
- `combined_reward`: Weighted combination (80% test execution, 20% format)

### Anti-Reward-Hacking Guards
1. Empty patch detection
2. Test deletion detection
3. Mega-patch penalty
4. Train-val gap monitoring (>0.10 = STOP)

## Data Format

### SFT Training Data
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "instance_id": "...",
    "repo": "...",
    "grader_score": 0.85
  }
}
```

### SWE-bench Instances
```json
{
  "instance_id": "django__django-11099",
  "repo": "django/django",
  "base_commit": "0454282c39925f56fc478c0632077ab05b79a14f",
  "problem_statement": "...",
  "hints_text": "..."
}
```

## Troubleshooting

### Docker Issues
```bash
# Ensure Docker is running
sudo systemctl start docker

# Build grader image
docker build -t swe-bench-grader:latest .
```

### CUDA OOM
```bash
# Reduce batch size
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use gradient checkpointing (enabled by default)
```

### Grader Timeout
```bash
# Increase timeout
python grader_endpoint.py --timeout 1200
```

## License

Internal use only.