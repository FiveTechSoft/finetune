#!/usr/bin/env python3
"""
Harbour Fine-tuning Script for Qwen3.6-35B-A3B (MoE)
Uses Unsloth + LoRA with 4-bit QLoRA on NVIDIA GB10 (121GB unified memory)
Includes checkpoints for resume on interruption.
"""

import json
import os
import signal
import sys
import torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["CPATH"] = "/usr/include/python3.12" + (":" + os.environ["CPATH"] if "CPATH" in os.environ else "")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

INTERRUPTION_FILE = Path("/home/fivetech/finetune/output_v2/.interrupted")

# Configuration
MODEL_NAME = "/home/fivetech/finetune/models/Qwen3.6-35B-A3B"
TRAIN_FILE = Path("/home/fivetech/finetune/finetune/dataset/harbour_train.jsonl")
VAL_FILE = Path("/home/fivetech/finetune/finetune/dataset/harbour_eval.jsonl")
OUTPUT_DIR = Path("/home/fivetech/finetune/output_v2")
MAX_SEQ_LENGTH = 1024

class CheckpointCallback(TrainerCallback):
    """Guarda checkpoint en cada step para poder resumir si se interrumpe."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step > 0 and state.global_step % args.logging_steps == 0:
            INTERRUPTION_FILE.write_text(json.dumps({
                "last_step": state.global_step,
                "last_epoch": state.epoch,
                "train_loss": logs.get("train_loss"),
                "eval_loss": logs.get("eval_loss"),
            }))

    def on_save(self, args, state, control, **kwargs):
        INTERRUPTION_FILE.write_text(json.dumps({
            "last_step": state.global_step,
            "last_epoch": state.epoch,
            "checkpoint": f"checkpoint-{state.global_step}",
        }))

def handle_signal(signum, frame):
    """Guarda estado al recibir SIGTERM/SIGINT para resumir despues."""
    print(f"\n[INTERRUPTED] Signal {signum} received. Saving state...")
    INTERRUPTION_FILE.write_text(json.dumps({
        "interrupted": True,
        "signal": signum,
    }))
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

print("=" * 60)
print("Harbour v2 Fine-tuning - Qwen3.6-35B-A3B (MoE) with Unsloth + LoRA (5004 examples)")
print("=" * 60)

# 1. Load model
print("\n1. Loading model (4-bit QLoRA)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
    text_only=True,
    device_map="auto",
)

# 2. LoRA configuration
print("2. Configuring LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# 3. Load dataset
print("3. Loading dataset...")

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl(TRAIN_FILE)
val_data = load_jsonl(VAL_FILE)

print(f"   Train: {len(train_data)} entries")
print(f"   Val: {len(val_data)} entries")

# 4. Format conversations
print("4. Formatting conversations...")

def format_conversation(entry):
    messages = []
    if "system" in entry:
        messages.append({"role": "system", "content": entry["system"]})
    messages.append({"role": "user", "content": entry["instruction"]})
    messages.append({"role": "assistant", "content": entry["output"]})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_dataset = Dataset.from_list([format_conversation(e) for e in train_data])
val_dataset = Dataset.from_list([format_conversation(e) for e in val_data])

# 5. Tokenize
print("5. Tokenizing...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing train",
)
val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing val",
)

print(f"   Train tokens: {sum(len(x) for x in train_dataset['input_ids']):,}")
print(f"   Val tokens: {sum(len(x) for x in val_dataset['input_ids']):,}")

# 6. Training arguments
print("6. Setting up training...")
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=8e-5,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_steps=50,
    save_total_limit=5,
    eval_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=False,
    fp16=False,
    dataloader_num_workers=1,
    report_to="none",
    remove_unused_columns=False,
    max_grad_norm=1.0,
    optim="adamw_8bit",
    dataloader_pin_memory=False,
)

# 7. Create trainer
print("7. Creating trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    callbacks=[CheckpointCallback()],
)

# 8. Train
print("\n8. Starting training...")
print("=" * 60)

# Workaround: patch torch.save to skip pickling SFTConfig (unsloth/trl pickle incompatibility)
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

last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    valid_checkpoints = []
    for ckpt in checkpoints:
        ckpt_path = str(OUTPUT_DIR / ckpt)
        if os.path.exists(os.path.join(ckpt_path, "optimizer.pt")) and os.path.exists(os.path.join(ckpt_path, "trainer_state.json")):
            valid_checkpoints.append(ckpt_path)
    if valid_checkpoints:
        last_checkpoint = sorted(valid_checkpoints, key=lambda x: int(os.path.basename(x).split("-")[1]))[-1]
        print(f"   Resuming from checkpoint: {last_checkpoint}")
    else:
        if INTERRUPTION_FILE.exists():
            interruption = json.loads(INTERRUPTION_FILE.read_text())
            if interruption.get("checkpoint"):
                last_checkpoint = str(OUTPUT_DIR / interruption["checkpoint"])
                if os.path.isdir(last_checkpoint) and os.path.exists(os.path.join(last_checkpoint, "optimizer.pt")):
                    print(f"   Resuming from interrupted checkpoint: {last_checkpoint}")
                else:
                    last_checkpoint = None

trainer.train(resume_from_checkpoint=last_checkpoint)

# 9. Save LoRA adapter
print("\n9. Saving LoRA adapter...")
trainer.save_model(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

# 10. Export to GGUF (optional)
print("\n10. Exporting to GGUF...")
model.save_pretrained_gguf(
    str(OUTPUT_DIR / "gguf"),
    tokenizer,
    quantization_method="q4_k_m",
)

print("\n" + "=" * 60)
print("Training complete!")
print(f"LoRA adapter saved to: {OUTPUT_DIR / 'final'}")
print(f"GGUF model saved to: {OUTPUT_DIR / 'gguf'}")
print("=" * 60)
