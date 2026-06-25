#!/usr/bin/env python3
"""
GRPO Training Script for Qwen3-Coder
Phase 2: Group Relative Policy Optimization with test execution reward.
Learns to resolve SWE-bench issues through agentic rollouts.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Environment setup ─────────────────────────────────────────────────────────
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if "CPATH" not in os.environ:
    os.environ["CPATH"] = "/usr/include/python3.12"
else:
    os.environ["CPATH"] = "/usr/include/python3.12:" + os.environ["CPATH"]

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainingArguments

# Add scripts directory to path for reward functions
sys.path.insert(0, str(Path(__file__).parent))
from reward_functions import grpo_reward_fn, swe_bench_reward, format_reward

# ── Configuration ──────────────────────────────────────────────────────────────
SFT_CHECKPOINT = Path("/home/fivetech/finetune/output/sft/final")
GRPO_DATA_PATH = Path("/home/fivetech/finetune/data/swe_bench_train.json")
OUTPUT_DIR = Path("/home/fivetech/finetune/output/grpo")
GRADER_ENDPOINT = "http://localhost:8000"
MAX_SEQ_LENGTH = 4096

# Training hyperparameters (from plan)
LEARNING_RATE = 2e-6
NUM_EPOCHS = 2
NUM_GENERATIONS = 8  # GRPO: generate N completions per prompt
MAX_EPISODE_STEPS = 6  # From plan: reduced from 10 to 6
COMPUTE_MULTIPLIER = 1.5
EVAL_INTERVAL = 5
EVAL_SAMPLES = 10
EVAL_THRESHOLD = 0.5  # Minimum reward to keep trajectory

# GRPO-specific parameters
GRPO_BETA = 0.04  # KL penalty coefficient
GRPO_EPSILON = 0.2  # PPO clipping epsilon (DAPO uses 0.2)


def load_swe_bench_data(path: Path) -> list[dict]:
    """Load SWE-bench training instances."""
    if not path.exists():
        print(f"Warning: {path} not found. Using sample instances.")
        return [
            {
                "instance_id": "django__django-11099",
                "repo": "django/django",
                "base_commit": "0454282c39925f56fc478c0632077ab05b79a14f",
                "problem_statement": "Fix URL validation...",
                "hints_text": "",
            },
            {
                "instance_id": "sympy__sympy-18057",
                "repo": "sympy/sympy",
                "base_commit": "b3ea5c42c3463e2a2c7995323cf00eff6c213cb5",
                "problem_statement": "Fix simplify()...",
                "hints_text": "",
            },
        ]
    
    with open(path) as f:
        return json.load(f)


def format_for_grpo(example: dict, tokenizer) -> dict:
    """Format example for GRPO training."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert software engineer. Analyze the issue, "
                "explore the repository, and provide a fix as a unified diff.\n"
                "Format your response as:\n"
                "1. Analysis of the issue\n"
                "2. Files to modify\n"
                "<patch>\nunified diff here\n</patch>"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Repository: {example['repo']}\n"
                f"Commit: {example['base_commit']}\n\n"
                f"Issue:\n{example.get('problem_statement', 'No description')}\n\n"
                f"Hints:\n{example.get('hints_text', 'No hints')}"
            ),
        },
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    return {
        "prompt": text,
        "instance_id": example["instance_id"],
        "repo": example["repo"],
        "base_commit": example["base_commit"],
    }


def extract_patch_from_response(response: str) -> str:
    """Extract patch from model response."""
    import re
    patch_match = re.search(r"<patch>(.*?)</patch>", response, re.DOTALL)
    if patch_match:
        return patch_match.group(1).strip()
    return ""


def grpo_reward_wrapper(
    prompts: list[str],
    completions: list[str],
    **kwargs,
) -> list[float]:
    """
    GRPO reward wrapper.
    Extracts patches from completions and calls grader.
    """
    instance_ids = kwargs.get("instance_id", [])
    repos = kwargs.get("repo", [])
    base_commits = kwargs.get("base_commit", [])
    
    scores = []
    
    for prompt, completion, instance_id, repo, base_commit in zip(
        prompts, completions, instance_ids, repos, base_commits
    ):
        patch = extract_patch_from_response(completion)
        
        result = swe_bench_reward(
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            patch=patch,
            grader_url=GRADER_ENDPOINT,
        )
        
        scores.append(result.score)
    
    return scores


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GRPO Training for Qwen3-Coder")
    parser.add_argument("--sft-checkpoint", type=Path, default=SFT_CHECKPOINT, help="SFT checkpoint path")
    parser.add_argument("--data-path", type=Path, default=GRPO_DATA_PATH, help="SWE-bench data path")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH, help="Max sequence length")
    parser.add_argument("--num-generations", type=int, default=NUM_GENERATIONS, help="GRPO: generations per prompt")
    parser.add_argument("--grader-url", default=GRADER_ENDPOINT, help="Grader endpoint URL")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    global GRADER_ENDPOINT
    GRADER_ENDPOINT = args.grader_url
    
    print("=" * 60)
    print("GRPO Training - Qwen3-Coder")
    print("=" * 60)
    
    # 1. Load model from SFT checkpoint
    print("\n1. Loading model from SFT checkpoint...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.sft_checkpoint),
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
        text_only=True,
        device_map="auto",
    )
    
    # 2. LoRA configuration (same as SFT)
    print("2. Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # 3. Load dataset
    print("3. Loading SWE-bench dataset...")
    raw_data = load_swe_bench_data(args.data_path)
    print(f"   Loaded {len(raw_data)} instances")
    
    # 4. Format for GRPO
    print("4. Formatting for GRPO...")
    
    def format_fn(example):
        return format_for_grpo(example, tokenizer)
    
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_fn, desc="Formatting")
    
    # 5. Training configuration
    print("5. Setting up GRPO training...")
    
    # GRPOConfig from trl
    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=1,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
        optim="adamw_8bit",
        
        # GRPO-specific parameters
        num_generations=args.num_generations,
        max_completion_length=args.max_seq_length,
        beta=GRPO_BETA,
        epsilon=GRPO_EPSILON,
        
        # Eval parameters
        eval_strategy="steps",
        eval_steps=EVAL_INTERVAL,
        
        # Reward function
        reward_func=grpo_reward_wrapper,
    )
    
    # 6. Create trainer
    print("6. Creating GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )
    
    # 7. Train
    print("\n7. Starting GRPO training...")
    print("=" * 60)
    
    # Workaround for pickle issues
    import torch as _torch
    _original_torch_save = _torch.save
    def _patched_torch_save(obj, *args, **kwargs):
        import pickle as _pickle
        try:
            return _original_torch_save(obj, *args, **kwargs)
        except (_pickle.PicklingError, Exception) as e:
            if "SFTConfig" in str(e) or "GRPOConfig" in str(e):
                return
            raise
    _torch.save = _patched_torch_save
    
    # Resume from checkpoint if specified
    last_checkpoint = None
    if args.resume and os.path.isdir(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        valid_checkpoints = []
        for ckpt in checkpoints:
            ckpt_path = str(args.output_dir / ckpt)
            if os.path.exists(os.path.join(ckpt_path, "optimizer.pt")):
                valid_checkpoints.append(ckpt_path)
        if valid_checkpoints:
            last_checkpoint = sorted(valid_checkpoints, key=lambda x: int(os.path.basename(x).split("-")[1]))[-1]
            print(f"   Resuming from checkpoint: {last_checkpoint}")
    
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 8. Save LoRA adapter
    print("\n8. Saving LoRA adapter...")
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    
    # 9. Export to GGUF (optional)
    print("\n9. Exporting to GGUF...")
    try:
        model.save_pretrained_gguf(
            str(args.output_dir / "gguf"),
            tokenizer,
            quantization_method="q4_k_m",
        )
    except Exception as e:
        print(f"   GGUF export failed: {e}")
        print("   Continuing without GGUF export...")
    
    print("\n" + "=" * 60)
    print("GRPO Training Complete!")
    print(f"LoRA adapter saved to: {args.output_dir / 'final'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
