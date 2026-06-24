---
language: es
task_categories:
- text-generation
tags:
- harbour
- fivewin
- fwh
- dataset
size_categories:
- 1K<n<10K
---

# Harbour/FWH Training Dataset

5,004 unique, compilable Harbour and FiveWin (FWH) examples for LLM fine-tuning.

## Structure

Each example has:
- `system`: Expert programmer prompt
- `instruction`: Task description  
- `input`: Empty
- `output`: Complete compilable code

## Format
JSONL with `instruction -> output` pairs.

## Compilation
All examples verified with Harbour v3.2.0dev compiler.
