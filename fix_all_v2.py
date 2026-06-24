#!/usr/bin/env python3
import json, sys

examples = []
with open('/home/antonio/finetune/harbour_fwh_improved.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        examples.append(json.loads(line))

fixes = json.loads(open('/home/antonio/finetune/fixes.json', 'r', encoding='utf-8').read())

for idx_str, output in fixes.items():
    idx = int(idx_str)
    if idx < len(examples):
        examples[idx]['output'] = output

output_path = '/home/antonio/finetune/harbour_fwh_fixed.jsonl'
with open(output_path, 'w', encoding='utf-8') as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

print(f"Fixed {len(fixes)} examples, saved to {output_path}")
