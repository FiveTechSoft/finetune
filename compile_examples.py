#!/usr/bin/env python3
"""
Compile Harbour examples from dataset to check for syntax errors.
"""

import json
import os
import subprocess
import tempfile
import re

HARBOUR_BIN = "/home/antonio/harbour_src/bin/linux/gcc/harbour"
INCLUDE_DIR = "/home/antonio/harbour_src/include"

def extract_code_blocks(text):
    """Extract Harbour code blocks from markdown text."""
    blocks = []
    
    # Find code blocks marked as harbour or generic
    patterns = [
        r'```harbour\s*\n(.*?)```',
        r'```prg\s*\n(.*?)```',
        r'```c\s*\n(.*?)```',  # For Extend API
        r'```\s*\n(.*?)```',   # Generic code blocks
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            # Clean up the code
            code = match.strip()
            if len(code) > 20:  # Skip very short blocks
                blocks.append(code)
    
    return blocks

def compile_harbour(code, temp_dir):
    """Try to compile a Harbour code snippet."""
    # Create a temporary .prg file
    prg_file = os.path.join(temp_dir, "test.prg")
    
    # Add minimal program structure if not present
    if 'PROCEDURE' not in code.upper() and 'FUNCTION' not in code.upper():
        code = "PROCEDURE Main()\n" + code + "\nRETURN"
    
    with open(prg_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    # Try to compile
    try:
        result = subprocess.run(
            [HARBOUR_BIN, prg_file, "-n", "-w", "-I" + INCLUDE_DIR],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=temp_dir
        )
        
        # Check for errors (harbour returns 0 on success)
        errors = []
        if result.returncode != 0:
            errors.append(f"Return code: {result.returncode}")
        if result.stderr:
            errors.append(result.stderr.strip())
        if result.stdout:
            # Harbour outputs warnings/errors to stdout
            for line in result.stdout.split('\n'):
                if 'Error' in line or 'error' in line or 'Warning' in line:
                    errors.append(line.strip())
        
        return len(errors) == 0, errors
        
    except subprocess.TimeoutExpired:
        return False, ["Timeout"]
    except Exception as e:
        return False, [str(e)]

def main():
    print("🔍 Compilando ejemplos del dataset Harbour/FWH...\n")
    
    # Load dataset
    examples = []
    with open('/home/antonio/finetune/harbour_fwh_improved.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"📥 Cargados {len(examples)} ejemplos\n")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    results = {
        "total": 0,
        "compiled": 0,
        "errors": 0,
        "no_code": 0,
        "error_details": []
    }
    
    for i, example in enumerate(examples):
        instruction = example.get('instruction', '')
        output = example.get('output', '')
        
        # Extract code blocks
        code_blocks = extract_code_blocks(output)
        
        if not code_blocks:
            results["no_code"] += 1
            continue
        
        for j, code in enumerate(code_blocks):
            results["total"] += 1
            success, errors = compile_harbour(code, temp_dir)
            
            if success:
                results["compiled"] += 1
            else:
                results["errors"] += 1
                results["error_details"].append({
                    "example": i + 1,
                    "instruction": instruction[:80],
                    "block": j + 1,
                    "errors": errors[:3]  # First 3 errors
                })
        
        # Progress
        if (i + 1) % 500 == 0:
            print(f"  Procesados {i + 1}/{len(examples)}...")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    # Report
    print(f"\n{'='*60}")
    print(f"📊 RESULTADOS DE COMPILACIÓN")
    print(f"{'='*60}")
    print(f"  Total bloques de código: {results['total']}")
    print(f"  ✅ Compilados correctamente: {results['compiled']}")
    print(f"  ❌ Con errores: {results['errors']}")
    print(f"  ⚠️  Sin código: {results['no_code']}")
    print(f"  Tasa de éxito: {results['compiled']*100/results['total']:.1f}%")
    
    if results['error_details']:
        print(f"\n{'='*60}")
        print(f"❌ ERRORES ENCONTRADOS (primeros 20)")
        print(f"{'='*60}")
        for err in results['error_details'][:20]:
            print(f"\n  Ejemplo #{err['example']}: {err['instruction']}...")
            print(f"  Bloque: {err['block']}")
            for e in err['errors']:
                print(f"    - {e[:100]}")
    
    # Save detailed report
    report_path = '/home/antonio/finetune/compilation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Reporte guardado en: {report_path}")

if __name__ == "__main__":
    main()
