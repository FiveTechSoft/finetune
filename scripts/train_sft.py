#!/usr/bin/env python3
"""
SFT Training Script for Qwen3-Coder
Phase 1: Supervised Fine-Tuning on distilled trajectories.
Learns tool-use format and repo-edit patterns.
"""

import json
import os
import sys
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
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
SFT_DATA_PATH = Path("/home/fivetech/finetune/data/sft_trajectories/sft_formatted.jsonl")
OUTPUT_DIR = Path("/home/fivetech/finetune/output/sft")
MAX_SEQ_LENGTH = 4096

# Training hyperparameters (from plan: lr=2e-5, 2 epochs)
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]


def load_sft_data(path: Path) -> list[dict]:
    """Load SFT data from JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_for_training(example: dict, tokenizer) -> dict:
    """Format example for SFT training."""
    messages = example["messages"]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    return {"text": text}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT Training for Qwen3-Coder")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Base model name")
    parser.add_argument("--data-path", type=Path, default=SFT_DATA_PATH, help="SFT data path")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH, help="Max sequence length")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SFT Training - Qwen3-Coder")
    print("=" * 60)
    
    # 1. Load model
    print("\n1. Loading model (4-bit QLoRA)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
        text_only=True,
        device_map="auto",
    )
    
    # 2. LoRA configuration
    print("2. Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # 3. Load dataset
    print("3. Loading SFT dataset...")
    raw_data = load_sft_data(args.data_path)
    print(f"   Loaded {len(raw_data)} examples")
    
    # 4. Format conversations
    print("4. Formatting conversations...")
    
    def format_fn(example):
        return format_for_training(example, tokenizer)
    
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_fn, desc="Formatting")
    
    # 5. Tokenize
    print("5. Tokenizing...")
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
    
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "messages", "metadata"],
        desc="Tokenizing",
    )
    
    total_tokens = sum(len(x) for x in dataset["input_ids"])
    print(f"   Total tokens: {total_tokens:,}")
    
    # 6. Training arguments
    print("6. Setting up training...")
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
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
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
    )
    
    # 7. Create trainer
    print("7. Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
    )
    
    # 8. Train
    print("\n8. Starting SFT training...")
    print("=" * 60)
    
    # Workaround: patch torch.save to skip pickling SFTConfig
    import torch as _torch
    _original_torch_save = _torch.save
    def _patched_torch_save(obj, *args, **kwargs):
        import pickle as _pickle
        try:
            return _original_torch_save(obj, *args, **kwargs)
        except (_pickle.PicklingError, Exception) as e:
            if "SFTConfig" in str(e):
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
    
    # 9. Save LoRA adapter
    print("\n9. Saving LoRA adapter...")
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    
    # 10. Export to GGUF (optional)
    print("\n10. Exporting to GGUF...")
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
    print("SFT Training Complete!")
    print(f"LoRA adapter saved to: {args.output_dir / 'final'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
