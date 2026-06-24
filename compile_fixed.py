#!/usr/bin/env python3
import json, os, subprocess, tempfile, re

HARBOUR_BIN = "/home/antonio/harbour_src/bin/linux/gcc/harbour"
INCLUDE_DIR = "/home/antonio/harbour_src/include"

def extract_code_blocks(text):
    blocks = []
    for pattern in [r'```harbour\s*\n(.*?)```', r'```prg\s*\n(.*?)```', r'```\s*\n(.*?)```']:
        for match in re.findall(pattern, text, re.DOTALL | re.IGNORECASE):
            code = match.strip()
            if len(code) > 20:
                blocks.append(code)
    return blocks

def compile_harbour(code, temp_dir):
    prg_file = os.path.join(temp_dir, "test.prg")
    if 'PROCEDURE' not in code.upper() and 'FUNCTION' not in code.upper():
        code = "PROCEDURE Main()\n" + code + "\nRETURN"
    with open(prg_file, 'w', encoding='utf-8') as f:
        f.write(code)
    try:
        result = subprocess.run(
            [HARBOUR_BIN, prg_file, "-n", "-w", "-I" + INCLUDE_DIR],
            capture_output=True, text=True, timeout=10, cwd=temp_dir)
        errors = []
        if result.returncode != 0:
            errors.append(f"RC:{result.returncode}")
        if result.stderr:
            errors.append(result.stderr.strip()[:200])
        if result.stdout:
            for line in result.stdout.split('\n'):
                if 'Error' in line or 'error' in line:
                    errors.append(line.strip()[:200])
        return len(errors) == 0, errors
    except:
        return False, ["Timeout"]

examples = []
with open('/home/antonio/finetune/harbour_fwh_fixed.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        examples.append(json.loads(line))

temp_dir = tempfile.mkdtemp()
total = compiled = errors = no_code = 0
error_details = []

for i, example in enumerate(examples):
    code_blocks = extract_code_blocks(example.get('output', ''))
    if not code_blocks:
        no_code += 1
        continue
    # Only compile the first (main) code block per example
    code = code_blocks[0]
    total += 1
    ok, errs = compile_harbour(code, temp_dir)
    if ok:
        compiled += 1
    else:
        errors += 1
        error_details.append({"ex": i+1, "instr": example.get('instruction','')[:60], "errs": errs[:2]})

import shutil
shutil.rmtree(temp_dir)

print(f"Total bloques: {total}")
print(f"Compilados OK: {compiled}")
print(f"Con errores: {errors}")
print(f"Sin codigo: {no_code}")
if total > 0:
    print(f"Tasa exito: {compiled*100/total:.1f}%")

if error_details:
    print(f"\nErrores ({len(error_details)}):")
    for e in error_details[:15]:
        print(f"  #{e['ex']}: {e['instr']}")
        for er in e['errs']:
            print(f"    {er[:120]}")
