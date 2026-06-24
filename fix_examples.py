#!/usr/bin/env python3
"""
Fix Harbour code examples to be syntactically correct.
"""

import json
import re

def fix_code_block(code):
    """Fix common issues in Harbour code blocks."""
    
    # Remove FiveWin/Thread includes (not available in standard Harbour)
    code = re.sub(r'#include\s+"fivewin\.ch"', '', code, flags=re.IGNORECASE)
    code = re.sub(r'#include\s+"thread\.ch"', '', code, flags=re.IGNORECASE)
    code = re.sub(r'#include\s+"fivehuff\.ch"', '', code, flags=re.IGNORECASE)
    code = re.sub(r'#include\s+"dbfddox\.ch"', '', code, flags=re.IGNORECASE)
    code = re.sub(r'#include\s+"tcpclient\.ch"', '', code, flags=re.IGNORECASE)
    
    # Remove #pragma directives
    code = re.sub(r'#pragma\s+\w+', '', code, flags=re.IGNORECASE)
    
    # Clean up multiple blank lines
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
    
    return code.strip()

def main():
    print("🔧 Arreglando ejemplos del dataset...\n")
    
    # Load dataset
    examples = []
    with open('/home/antonio/finetune/harbour_fwh_improved.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"📥 Cargados {len(examples)} ejemplos")
    
    fixed_count = 0
    fixed_examples = []
    
    for example in examples:
        output = example.get('output', '')
        original_output = output
        
        # Fix code blocks in output - improved regex
        def replace_code_block(match):
            nonlocal fixed_count
            lang = match.group(1) or ''
            code = match.group(2)
            
            fixed_code = fix_code_block(code)
            
            if fixed_code != code.strip():
                fixed_count += 1
            
            return f"```{lang}\n{fixed_code}\n```"
        
        # Fix code blocks with improved pattern
        output = re.sub(r'```(\w*)\n(.*?)```', replace_code_block, output, flags=re.DOTALL | re.IGNORECASE)
        
        example['output'] = output
        fixed_examples.append(example)
    
    print(f"🔧 Bloques arreglados: {fixed_count}")
    
    # Save fixed dataset
    output_path = '/home/antonio/finetune/harbour_fwh_fixed.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in fixed_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✅ Guardado en: {output_path}")

if __name__ == "__main__":
    main()
