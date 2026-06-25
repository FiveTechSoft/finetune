#!/usr/bin/env python3
"""
Harbour Fine-tuning Script for qwen3.6:35b (Qwen3.6-35B-A3B MoE)
Uses LoRA with CPU training (121GB RAM available)
"""

import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Configuration
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
TRAIN_FILE = Path("/home/fivetech/finetune/harbour_train.jsonl")
VAL_FILE = Path("/home/fivetech/finetune/harbour_val.jsonl")
OUTPUT_DIR = Path("/home/fivetech/finetune/output")
MAX_SEQ_LENGTH = 2048

print("=" * 60)
print("Harbour Fine-tuning - qwen3.6:35b (MoE) with LoRA")
print("=" * 60)

# 1. Load tokenizer
print("\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load dataset
print("2. Loading dataset...")

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

# 3. Format conversations for Qwen ChatML
print("3. Formatting conversations...")

def format_conversation(entry):
    """Convert messages to Qwen ChatML format."""
    messages = entry["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_dataset = Dataset.from_list([format_conversation(e) for e in train_data])
val_dataset = Dataset.from_list([format_conversation(e) for e in val_data])

# 4. Tokenize
print("4. Tokenizing...")

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

# 5. Load model (CPU with bfloat16 - ~70GB for 35B params, fits in 121GB)
print("5. Loading model (CPU mode, bfloat16)...")
print("   This may take a few minutes...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# 6. LoRA configuration
print("6. Configuring LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# 7. Training arguments
print("7. Setting up training...")
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_steps=50,
    save_total_limit=3,
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
    optim="adamw_torch",
    gradient_checkpointing=True,
)

# 8. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 9. Create trainer
print("8. Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# 10. Train
print("\n9. Starting training...")
print("=" * 60)

import os
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = str(OUTPUT_DIR / sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
        print(f"   Resuming from checkpoint: {last_checkpoint}")

trainer.train(resume_from_checkpoint=last_checkpoint)

# 11. Save
print("\n10. Saving model...")
trainer.save_model(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

print("\n" + "=" * 60)
print("Training complete!")
print(f"Model saved to: {OUTPUT_DIR / 'final'}")
print("=" * 60)
