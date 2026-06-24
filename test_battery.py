#!/usr/bin/env python3
"""
Harbour Test Battery - Generates code, compiles with harbour, evaluates with qwen3.6:35b
"""

import json
import time
import subprocess
import requests
import tempfile
import os
from pathlib import Path
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.6:35b"
HARBOUR = "/home/fivetech/harbour/bin/linux/gcc/harbour"
WORK_DIR = Path("/home/fivetech/finetune/test_output")
WORK_DIR.mkdir(exist_ok=True)

def query_ollama(prompt, system="", timeout=300):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 3000, "top_p": 0.9}
    }
    if system:
        payload["system"] = system
    try:
        start = time.time()
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        elapsed = time.time() - start
        data = r.json()
        return {
            "response": data.get("response", ""),
            "eval_count": data.get("eval_count", 0),
            "duration": elapsed,
            "tps": data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1e9, 0.001),
            "error": None
        }
    except Exception as e:
        return {"response": "", "error": str(e), "eval_count": 0, "duration": 0, "tps": 0}

def compile_harbour(code):
    """Compile code with harbour, return (success, error_msg, obj_exists)"""
    prg_file = WORK_DIR / "test.prg"
    prg_file.write_text(code)
    
    try:
        result = subprocess.run(
            [HARBOUR, str(prg_file), "-n", "-w"],
            capture_output=True, text=True, timeout=30
        )
        obj_file = WORK_DIR / "test.obj"
        success = result.returncode == 0
        obj_exists = obj_file.exists()
        error = result.stderr.strip() if result.stderr else ""
        if not success and not error:
            error = result.stdout.strip()
        return success, error, obj_exists
    except subprocess.TimeoutExpired:
        return False, "Compilation timeout", False
    except Exception as e:
        return False, str(e), False

def clean_code(response):
    """Extract code from model response, remove markdown."""
    lines = response.split('\n')
    in_code = False
    code_lines = []
    skip_explanation = True
    
    for line in lines:
        stripped = line.strip()
        
        # Skip markdown
        if stripped.startswith('```'):
            in_code = not in_code
            continue
        
        if in_code:
            code_lines.append(line)
            skip_explanation = False
        elif skip_explanation:
            # Detect start of code
            upper = stripped.upper()
            if any(upper.startswith(kw) for kw in [
                'FUNCTION', 'PROCEDURE', 'LOCAL', 'STATIC', 'PUBLIC',
                'PRIVATE', 'MEMVAR', '#DEFINE', '#INCLUDE', 'CLASS',
                'METHOD', 'RETURN', 'SET', 'REQUEST'
            ]):
                in_code = True
                code_lines.append(line)
                skip_explanation = False
    
    if not code_lines:
        # Fallback: take everything
        code_lines = response.split('\n')
    
    return '\n'.join(code_lines).strip()

# ============================================================
# TEST DEFINITIONS - Based on dataset patterns
# ============================================================

TESTS = [
    # ---- BASIC SYNTAX ----
    {
        "id": "SYNTAX_01", "category": "Basic Syntax", "name": "Variable types and declarations",
        "prompt": "Write a Harbour program that declares LOCAL variables of each type (numeric, character, logical, date, nil), prints them with ValType(), and uses proper Hungarian notation.",
        "expected_keywords": ["LOCAL", "ValType", "FUNCTION"],
        "min_lines": 8,
    },
    {
        "id": "SYNTAX_02", "category": "Basic Syntax", "name": "Preprocessor defines",
        "prompt": "Write Harbour preprocessor definitions for application constants: app name, version, max records, date format. Use #define and show conditional compilation with #ifdef.",
        "expected_keywords": ["#DEFINE", "#IFDEF", "#ENDIF"],
        "min_lines": 6,
    },
    {
        "id": "SYNTAX_03", "category": "Basic Syntax", "name": "String operations",
        "prompt": "Write a Harbour function that takes a full name string and returns initials. Use AllTrim, Upper, Left, At, SubStr, and Space functions.",
        "expected_keywords": ["FUNCTION", "AllTrim", "Upper", "Left", "At", "SubStr"],
        "min_lines": 6,
    },
    {
        "id": "SYNTAX_04", "category": "Basic Syntax", "name": "Date functions",
        "prompt": "Write a Harbour function that calculates the number of business days between two dates, excluding weekends. Use Date(), DOW(), and date arithmetic.",
        "expected_keywords": ["FUNCTION", "Date", "DOW"],
        "min_lines": 8,
    },
    {
        "id": "SYNTAX_05", "category": "Basic Syntax", "name": "Type conversion",
        "prompt": "Write Harbour code that converts between all types: Str, Val, CTOD, DTOC, ASC, Chr, Transform. Show edge cases.",
        "expected_keywords": ["Str", "Val", "CTOD", "DTOC"],
        "min_lines": 8,
    },

    # ---- CONTROL FLOW ----
    {
        "id": "CTRL_01", "category": "Control Flow", "name": "IF/ELSEIF/ENDIF",
        "prompt": "Write a Harbour function that classifies employee salary into tax brackets using IF/ELSEIF/ELSE/ENDIF. Include 5 brackets and error handling.",
        "expected_keywords": ["FUNCTION", "IF", "ELSEIF", "ELSE", "ENDIF"],
        "min_lines": 10,
    },
    {
        "id": "CTRL_02", "category": "Control Flow", "name": "DO CASE",
        "prompt": "Write a Harbour function using DO CASE to convert month number (1-12) to season name. Handle invalid input with OTHERWISE.",
        "expected_keywords": ["DO CASE", "CASE", "OTHERWISE", "ENDCASE"],
        "min_lines": 8,
    },
    {
        "id": "CTRL_03", "category": "Control Flow", "name": "FOR/NEXT loop",
        "prompt": "Write a Harbour function using FOR/NEXT to calculate the sum of all prime numbers below 100. Include STEP and EXIT.",
        "expected_keywords": ["FOR", "TO", "NEXT", "IF", "EXIT"],
        "min_lines": 10,
    },
    {
        "id": "CTRL_04", "category": "Control Flow", "name": "DO WHILE",
        "prompt": "Write a Harbour function using DO WHILE to implement the Euclidean algorithm for GCD. Include LOOP and EXIT.",
        "expected_keywords": ["DO WHILE", "ENDDO", "IF", "LOOP", "EXIT"],
        "min_lines": 6,
    },
    {
        "id": "CTRL_05", "category": "Control Flow", "name": "SCAN/ENDSCAN",
        "prompt": "Write Harbour code using SCAN/ENDSCAN to find the longest string in an array. Include NEXT clause.",
        "expected_keywords": ["SCAN", "ENDSCAN"],
        "min_lines": 6,
    },
    {
        "id": "CTRL_06", "category": "Control Flow", "name": "FOR EACH",
        "prompt": "Write Harbour code using FOR EACH to count word frequencies in a string. Use a hash for storage.",
        "expected_keywords": ["FOR EACH", "NEXT", ":="],
        "min_lines": 8,
    },

    # ---- FUNCTIONS ----
    {
        "id": "FUNC_01", "category": "Functions", "name": "Parameters and return",
        "prompt": "Write a Harbour function with default parameters, pass-by-reference using @, and return an array. Include proper Hungarian notation.",
        "expected_keywords": ["FUNCTION", "LOCAL", "RETURN"],
        "min_lines": 6,
    },
    {
        "id": "FUNC_02", "category": "Functions", "name": "Recursion",
        "prompt": "Write a recursive Harbour function for Fibonacci numbers with memoization using a hash. Include base case and error handling.",
        "expected_keywords": ["FUNCTION", "IF", "RETURN"],
        "min_lines": 8,
    },
    {
        "id": "FUNC_03", "category": "Functions", "name": "Variable scope",
        "prompt": "Write Harbour code demonstrating LOCAL, STATIC, PRIVATE, PUBLIC variables. Show scope differences with nested function calls.",
        "expected_keywords": ["LOCAL", "STATIC", "PRIVATE", "PUBLIC"],
        "min_lines": 8,
    },
    {
        "id": "FUNC_04", "category": "Functions", "name": "Code blocks",
        "prompt": "Write Harbour code using code blocks: AEval with {|x| x*2}, AScan, ASort with custom sort. Show evaluation with Eval().",
        "expected_keywords": ["AEval", "AScan", "ASort", "Eval"],
        "min_lines": 6,
    },
    {
        "id": "FUNC_05", "category": "Functions", "name": "Error handling",
        "prompt": "Write a Harbour function with BEGIN SEQUENCE/RECOVER/END SEQUENCE for file reading. Include DEFAULT and BREAK.",
        "expected_keywords": ["BEGIN SEQUENCE", "RECOVER", "END SEQUENCE"],
        "min_lines": 8,
    },

    # ---- ARRAYS ----
    {
        "id": "ARRAY_01", "category": "Arrays", "name": "Array operations",
        "prompt": "Write Harbour functions for: create 2D array, AAdd elements, ASort with custom order, AScan by value, ASize to resize. Include error handling.",
        "expected_keywords": ["ARRAY", "AAdd", "ASort", "AScan", "ASize"],
        "min_lines": 8,
    },
    {
        "id": "ARRAY_02", "category": "Arrays", "name": "Hash operations",
        "prompt": "Write Harbour code using hashes: create, add keys, iterate with FOR EACH, merge two hashes, check key existence with HB_HHasKey, convert to array.",
        "expected_keywords": [":=", "FOR EACH", "HB_HHasKey"],
        "min_lines": 8,
    },
    {
        "id": "ARRAY_03", "category": "Arrays", "name": "Sorting algorithm",
        "prompt": "Implement QuickSort in Harbour for an array of numbers. Include partition logic and proper recursion.",
        "expected_keywords": ["FUNCTION", "LOCAL", "IF", "RETURN"],
        "min_lines": 12,
    },

    # ---- OOP ----
    {
        "id": "OOP_01", "category": "OOP", "name": "Class definition",
        "prompt": "Write a Harbour class Person with DATA (name, age), METHOD (New constructor, GetName, SetAge), and CLASSDATA. Include validation in SetAge.",
        "expected_keywords": ["CLASS", "DATA", "METHOD", "RETURN"],
        "min_lines": 10,
    },
    {
        "id": "OOP_02", "category": "OOP", "name": "Inheritance",
        "prompt": "Write Harbour classes: Shape (base), Circle (derived) with area() method. Show inheritance syntax and method override.",
        "expected_keywords": ["CLASS", "METHOD", "INHERIT"],
        "min_lines": 10,
    },
    {
        "id": "OOP_03", "category": "OOP", "name": "Operator overloading",
        "prompt": "Write a Harbour class Vec2 for 2D vectors. Overload + and - operators. Include magnitude and normalize methods.",
        "expected_keywords": ["CLASS", "METHOD", "OPERATOR"],
        "min_lines": 12,
    },
    {
        "id": "OOP_04", "category": "OOP", "name": "Singleton pattern",
        "prompt": "Implement Singleton pattern in Harbour for a config manager. Ensure only one instance exists.",
        "expected_keywords": ["CLASS", "CLASSDATA", "METHOD"],
        "min_lines": 10,
    },

    # ---- DATABASE ----
    {
        "id": "DB_01", "category": "Database", "name": "Basic RDD",
        "prompt": "Write Harbour code that creates a DBF file, opens it, appends records, and closes properly. Use DBCreate and DBUseArea.",
        "expected_keywords": ["DBCreate", "DBUseArea", "DBAppend", "DBCLOSEALL"],
        "min_lines": 10,
    },
    {
        "id": "DB_02", "category": "Database", "name": "Indexing",
        "prompt": "Write Harbour code creating an index on a DBF field using RDD. Include ORDSCOPE for range queries.",
        "expected_keywords": ["ORDCREATE", "ORDSCOPE"],
        "min_lines": 8,
    },
    {
        "id": "DB_03", "category": "Database", "name": "DBEval",
        "prompt": "Write Harbour code using DBEval to process all records: count, sum field values, and mark records meeting a condition.",
        "expected_keywords": ["DBEval", "FOR", "WHILE"],
        "min_lines": 8,
    },

    # ---- FILE I/O ----
    {
        "id": "FILE_01", "category": "File I/O", "name": "Text file read/write",
        "prompt": "Write Harbour functions to read a text file line by line and write processed output. Use FCreate, FOpen, FRead, FWrite, FClose, FEof.",
        "expected_keywords": ["FCreate", "FOpen", "FRead", "FWrite", "FClose", "FEof"],
        "min_lines": 10,
    },
    {
        "id": "FILE_02", "category": "File I/O", "name": "Directory listing",
        "prompt": "Write Harbour code using Directory() to list files with a pattern, get file size and date, and process each file.",
        "expected_keywords": ["Directory", "LEN", "FOR"],
        "min_lines": 6,
    },

    # ---- COMPLEX ----
    {
        "id": "CMPX_01", "category": "Complex", "name": "CSV parser",
        "prompt": "Write a Harbour CSV parser that reads a CSV file, handles quoted fields, and returns an array of arrays. Include error handling.",
        "expected_keywords": ["FUNCTION", "LOCAL", "FClose", "FEof"],
        "min_lines": 15,
    },
    {
        "id": "CMPX_02", "category": "Complex", "name": "INI file reader",
        "prompt": "Write a Harbour INI file parser. Read sections, keys, and values into a hash. Handle comments and empty lines.",
        "expected_keywords": ["FUNCTION", "LOCAL", "HASH"],
        "min_lines": 12,
    },
    {
        "id": "CMPX_03", "category": "Complex", "name": "String template engine",
        "prompt": "Write a Harbour template engine replacing {{variable}} placeholders with hash values. Include error handling for missing keys.",
        "expected_keywords": ["FUNCTION", "LOCAL", "STRTRAN"],
        "min_lines": 8,
    },
    {
        "id": "CMPX_04", "category": "Complex", "name": "Logger",
        "prompt": "Write a Harbour logging system with DEBUG/INFO/WARN/ERROR levels, timestamp, file output, and configurable level filtering.",
        "expected_keywords": ["FUNCTION", "LOCAL", "FClose"],
        "min_lines": 12,
    },
    {
        "id": "CMPX_05", "category": "Complex", "name": "Base64 encoder",
        "prompt": "Write a Harbour Base64 encoder/decode function. Use Asc(), Chr(), and bit operations.",
        "expected_keywords": ["FUNCTION", "LOCAL", "Asc", "Chr"],
        "min_lines": 10,
    },
    {
        "id": "CMPX_06", "category": "Complex", "name": "JSON serializer",
        "prompt": "Write a Harbour function that serializes a hash to JSON string. Handle strings, numbers, booleans, arrays, and nested objects.",
        "expected_keywords": ["FUNCTION", "LOCAL", "HB_IsHash"],
        "min_lines": 15,
    },
    {
        "id": "CMPX_07", "category": "Complex", "name": "LRU Cache",
        "prompt": "Write a Harbour LRU cache class with get/set/delete, TTL expiration, and max size. Use a hash and an array for ordering.",
        "expected_keywords": ["CLASS", "DATA", "METHOD"],
        "min_lines": 15,
    },
    {
        "id": "CMPX_08", "category": "Complex", "name": "SQL-like query on arrays",
        "prompt": "Write a Harbour function that filters an array of hashes like SQL WHERE clause. Support =, <>, >, <, LIKE operators.",
        "expected_keywords": ["FUNCTION", "LOCAL", "FOR"],
        "min_lines": 12,
    },
    {
        "id": "CMPX_09", "category": "Complex", "name": "Rate limiter",
        "prompt": "Write a Harbour rate limiter class: max N requests per M seconds. Use timestamps and a queue.",
        "expected_keywords": ["CLASS", "METHOD", "LOCAL"],
        "min_lines": 12,
    },
    {
        "id": "CMPX_10", "category": "Complex", "name": "Config file writer",
        "prompt": "Write a Harbour config manager that saves/loads settings to JSON file. Include defaults, validation, and typed getters.",
        "expected_keywords": ["FUNCTION", "LOCAL", "FClose"],
        "min_lines": 12,
    },

    # ---- BUGGY CODE TO FIX ----
    {
        "id": "FIX_01", "category": "Bug Fix", "name": "Null pointer",
        "prompt": "Fix this Harbour code that crashes when array is empty:\nLOCAL a := {}\n? a[1]",
        "expected_keywords": ["IF", "LEN", "RETURN"],
        "min_lines": 3,
    },
    {
        "id": "FIX_02", "category": "Bug Fix", "name": "Wrong loop bounds",
        "prompt": "Fix this code that skips last element:\nLOCAL a := {10,20,30}\nFOR i := 1 TO LEN(a)-1\n   ? a[i]\nNEXT",
        "expected_keywords": ["FOR", "TO", "LEN"],
        "min_lines": 3,
    },
    {
        "id": "FIX_03", "category": "Bug Fix", "name": "String concat error",
        "prompt": "Fix this code that fails on nil values:\nLOCAL cName := NIL\n? 'Hello ' + cName",
        "expected_keywords": ["IF", "LOCAL", "RETURN"],
        "min_lines": 3,
    },

    # ---- HARBOUR-SPECIFIC ----
    {
        "id": "HARB_01", "category": "Harbour-Specific", "name": "HB_* functions",
        "prompt": "Write Harbour code using HB_IsString, HB_IsNumeric, HB_IsArray, HB_IsHash, HB_IsNil to validate function arguments. Include proper error messages.",
        "expected_keywords": ["HB_IsString", "HB_IsNumeric", "IF"],
        "min_lines": 6,
    },
    {
        "id": "HARB_02", "category": "Harbour-Specific", "name": "Regex",
        "prompt": "Write a Harbour function using HB_RegEx to validate email addresses. Use HB_RegExCompile and HB_RegExMatch.",
        "expected_keywords": ["HB_RegEx", "FUNCTION"],
        "min_lines": 6,
    },
    {
        "id": "HARB_03", "category": "Harbour-Specific", "name": "Serialization",
        "prompt": "Write Harbour code that serializes a hash to binary with HB_Serialize and deserializes with HB_Deserialize.",
        "expected_keywords": ["HB_Serialize", "HB_Deserialize"],
        "min_lines": 6,
    },
    {
        "id": "HARB_04", "category": "Harbour-Specific", "name": "File path operations",
        "prompt": "Write Harbour code using hb_DirBuild, hb_DirNameGet, hb_FileNameGet, hb_PathNormalize for cross-platform file handling.",
        "expected_keywords": ["hb_Dir", "hb_File", "hb_Path"],
        "min_lines": 6,
    },
]

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("HARBOUR CODE GENERATION TEST BATTERY")
    print(f"Model: {MODEL}")
    print(f"Tests: {len(TESTS)}")
    print(f"Harbour: {HARBOUR}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    SYSTEM = """You are an expert Harbour programmer. Write clean, correct, COMPILABLE Harbour code.
Use Hungarian notation: n=numeric, c=character, l=logical, a=array, o=object, d=date.
Use 3-space indentation.
Do NOT include explanations or markdown. Only raw Harbour code.
End functions with RETURN and END FUNCTION."""

    results = []
    compile_pass = 0
    compile_fail = 0
    
    for i, test in enumerate(TESTS, 1):
        print(f"\n[{i:2d}/{len(TESTS)}] {test['id']}: {test['name']}")
        
        # Query model
        result = query_ollama(test["prompt"], SYSTEM)
        
        if result["error"]:
            print(f"  MODEL ERROR: {result['error']}")
            results.append({"test": test, "model_error": result["error"], "compile": False, "compile_error": ""})
            continue
        
        # Clean response
        code = clean_code(result["response"])
        
        # Check for expected keywords
        keywords_found = [kw for kw in test["expected_keywords"] if kw.upper() in code.upper()]
        keywords_missing = [kw for kw in test["expected_keywords"] if kw.upper() not in code.upper()]
        
        # Compile
        success, error, obj = compile_harbour(code)
        
        status = "PASS" if success else "FAIL"
        if success:
            compile_pass += 1
        else:
            compile_fail += 1
        
        print(f"  Compile: {status} | Keywords: {len(keywords_found)}/{len(test['expected_keywords'])} | TPS: {result['tps']:.0f}")
        if keywords_missing:
            print(f"  Missing keywords: {', '.join(keywords_missing)}")
        if not success and error:
            # Show first error only
            first_error = error.split('\n')[0][:120]
            print(f"  Error: {first_error}")
        
        results.append({
            "test": test,
            "code": code[:3000],
            "compile_success": success,
            "compile_error": error[:500] if error else "",
            "keywords_found": keywords_found,
            "keywords_missing": keywords_missing,
            "tokens": result["eval_count"],
            "tps": result["tps"],
            "duration": result["duration"],
            "lines": code.count('\n') + 1,
        })
    
    # Summary by category
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    categories = {}
    for r in results:
        cat = r["test"]["category"]
        if cat not in categories:
            categories[cat] = {"pass": 0, "fail": 0, "total": 0}
        categories[cat]["total"] += 1
        if r.get("compile_success"):
            categories[cat]["pass"] += 1
        else:
            categories[cat]["fail"] += 1
    
    print(f"\n{'Category':<20} {'Pass':<6} {'Fail':<6} {'Rate':<8}")
    print("-" * 45)
    for cat, data in sorted(categories.items()):
        rate = data["pass"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"{cat:<20} {data['pass']:<6} {data['fail']:<6} {rate:.0f}%")
    
    print(f"\n{'TOTAL':<20} {compile_pass:<6} {compile_fail:<6} {compile_pass/len(results)*100:.0f}%")
    print(f"Total tests: {len(results)}")
    
    total_tokens = sum(r.get("tokens", 0) for r in results)
    total_time = sum(r.get("duration", 0) for r in results)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total time: {total_time:.1f}s")
    
    # Save
    output = Path("/home/fivetech/finetune/test_baseline_qwen36.json")
    with open(output, "w") as f:
        json.dump({
            "model": MODEL,
            "timestamp": datetime.now().isoformat(),
            "compile_pass": compile_pass,
            "compile_fail": compile_fail,
            "compile_rate": compile_pass / len(results) * 100,
            "categories": categories,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output}")

if __name__ == "__main__":
    main()
