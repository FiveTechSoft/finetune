# Harbour Fine-tuning Dataset

## Overview
This dataset contains 996 training entries extracted from:
- 1037 Harbour PRG (.prg) source files
- 143 Harbour Header (.ch) files

177 PRG files and 7 CH files were skipped due to quality issues.

## Dataset Format
The dataset is provided in JSONL format with the following structure:

### Instruction Format (harbour_train.jsonl / harbour_val.jsonl)
```json
{"instruction": "...", "input": "", "output": "..."}
```

### Full Dataset (harbour_dataset_full.jsonl)
```json
{"instruction": "...", "input": "", "output": "...", "metadata": {"file_path": "...", "language": "harbour", "category": "...", "subcategory": "..."}}
```

## Categories
- **include**: Header files with constants/macros (59 files)
- **rtl**: Harbour Runtime Library (80 files)
- **contrib**: Contribution libraries (583 files)
- **tests**: Test programs (225 files)
- **utils**: Utility programs (13 files)
- **extras**: Extra libraries (25 files)

## Cleaning Applied
- Copyright/license headers removed
- Disabled code blocks (#if 0) removed
- Excessive trailing comments removed
- Excessive blank lines removed
- Files without actual code filtered out
- Incomplete code (missing ENDCLASS, etc.) filtered out

## Usage for Fine-tuning
```bash
# Using Ollama with Modelfile
FROM qwen2.5-coder:14b

# Training command
ollama create harbour-coder -f Modelfile

# Or use with other training frameworks
# The JSONL format is compatible with:
# - OpenAI fine-tuning API
# - Hugging Face transformers
# - Axolotl
# - LLaMA-Factory
```

## File Structure
- `harbour_train.jsonl` - Training set (896 entries)
- `harbour_val.jsonl` - Validation set (100 entries)
- `harbour_dataset_full.jsonl` - Full dataset with metadata
- `dataset_stats.json` - Dataset statistics
- `generate_dataset.py` - This script

## Source
The source files are from the Harbour project (https://harbour.github.io/),
an open-source Clipper-compatible compiler.
