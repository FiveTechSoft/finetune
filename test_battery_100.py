#!/usr/bin/env python3
"""
Harbour Test Battery - Fast version with incremental saving
"""

import json, time, subprocess, requests, sys
from pathlib import Path
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.6:35b"
HARBOUR = "/home/fivetech/harbour/bin/linux/gcc/harbour"
WORK_DIR = Path("/home/fivetech/finetune/test_output")
WORK_DIR.mkdir(exist_ok=True)
RESULTS_FILE = Path("/home/fivetech/finetune/test_baseline_100.json")

SYSTEM = """You are an expert Harbour programmer. Write clean, correct, COMPILABLE Harbour code.
Use Hungarian notation: n=numeric, c=character, l=logical, a=array, o=object, d=date.
Use 3-space indentation.
Do NOT include explanations, markdown, or #include. Only raw Harbour code.
End functions with RETURN and END FUNCTION."""

def query(prompt, timeout=180):
    payload = {"model": MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.2, "num_predict": 1500, "top_p": 0.9}}
    try:
        t0 = time.time()
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        d = r.json()
        return {"resp": d.get("response",""), "tok": d.get("eval_count",0),
                "dur": time.time()-t0, "tps": d.get("eval_count",0)/max(d.get("eval_duration",1)/1e9,.001)}
    except Exception as e:
        return {"resp":"", "tok":0, "dur":0, "tps":0, "err":str(e)}

def compile_hb(code):
    f = WORK_DIR/"test.prg"
    f.write_text(code)
    try:
        r = subprocess.run([HARBOUR, str(f), "-n", "-w"], capture_output=True, text=True, timeout=20)
        return r.returncode == 0, (r.stderr or r.stdout).strip()[:400]
    except:
        return False, "timeout"

def clean(resp):
    lines = resp.split('\n')
    in_code = False
    code = []
    for line in lines:
        s = line.strip()
        if s.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
        elif not code:
            u = s.upper()
            if any(u.startswith(k) for k in ['FUNCTION','PROCEDURE','LOCAL','STATIC','#DEFINE','CLASS','METHOD','RETURN','SET','REQUEST','MEMVAR','*']):
                code.append(line)
    return '\n'.join(code).strip() if code else resp.strip()

def save(results, meta):
    with open(RESULTS_FILE, "w") as f:
        json.dump({"model":MODEL,"ts":datetime.now().isoformat(),**meta,"results":results}, f, indent=2, ensure_ascii=False)

# 100 tests - proportional to dataset (48 Arrays, 22 OOP, 9 Other, 8 Func, 7 DB, 4 FileIO, 2 Control)
TESTS = [
    # ARRAYS (48)
    ("A01","Arrays","Create 2D array","Write a Harbour function that creates a 5x5 multiplication table as 2D array and prints it."),
    ("A02","Arrays","AAdd and resize","Write Harbour code creating empty array, adding 100 elements with AAdd, resizing to 50 with ASize."),
    ("A03","Arrays","ASort custom","Write a Harbour function sorting array of structures by numeric field using ASort with code block."),
    ("A04","Arrays","AScan code block","Write Harbour code using AScan to find first negative number in array. Return index or 0."),
    ("A05","Arrays","AFill pattern","Write Harbour code using AFill to initialize array with alternating 1 and 0 values."),
    ("A06","Arrays","AINS and ADEL","Write Harbour functions: insert element at position and delete element at position with edge cases."),
    ("A07","Arrays","ACopy array","Write Harbour code using ACopy to copy part of one array to another with offset and length."),
    ("A08","Arrays","AEval transform","Write Harbour code using AEval to double every element in array in-place."),
    ("A09","Arrays","Array join","Write Harbour function joining array elements with delimiter using loop."),
    ("A10","Arrays","String split","Write Harbour function splitting CSV string into array handling quoted values."),
    ("A11","Arrays","3D array","Write Harbour code creating and accessing 3D array, filling with sequential numbers."),
    ("A12","Arrays","Array contains","Write Harbour function checking if array contains value, return TRUE/FALSE."),
    ("A13","Arrays","Array unique","Write Harbour function removing duplicates from array returning unique values."),
    ("A14","Arrays","Array flatten","Write Harbour function flattening nested arrays into single array."),
    ("A15","Arrays","Array reverse","Write Harbour function reversing array in-place."),
    ("A16","Arrays","Array sum avg","Write Harbour functions calculating sum and average of numeric array."),
    ("A17","Arrays","Array min max","Write Harbour functions finding min and max in array with single pass."),
    ("A18","Arrays","Array filter","Write Harbour function filtering array keeping elements matching code block condition."),
    ("A19","Arrays","Array map","Write Harbour function mapping array to new array applying code block."),
    ("A20","Arrays","Array reduce","Write Harbour function reducing array to single value with accumulator."),
    ("A21","Arrays","Hash create","Write Harbour code creating hash with :=, adding keys, retrieving values."),
    ("A22","Arrays","Hash iterate","Write Harbour code iterating hash with FOR EACH printing key-value pairs."),
    ("A23","Arrays","Hash keys values","Write Harbour code extracting keys and values from hash into arrays."),
    ("A24","Arrays","Hash merge","Write Harbour function merging two hashes, second overriding on conflicts."),
    ("A25","Arrays","Hash exists","Write Harbour code checking key existence with HB_HHasKey and default value."),
    ("A26","Arrays","Hash delete","Write Harbour code deleting key from hash checking existence first."),
    ("A27","Arrays","Hash array convert","Write Harbour code converting hash to array of pairs and back."),
    ("A28","Arrays","Hash count keys","Write Harbour function counting total keys in hash."),
    ("A29","Arrays","Hash filter","Write Harbour function filtering hash keeping entries where value > threshold."),
    ("A30","Arrays","Nested hash","Write Harbour code working with nested hashes accessing deeply nested values."),
    ("A31","Arrays","FOR EACH array","Write Harbour code using FOR EACH finding longest string in array."),
    ("A32","Arrays","FOR EACH hash","Write Harbour code using FOR EACH on hash building comma-separated string."),
    ("A33","Arrays","Bubble sort","Implement bubble sort in Harbour for array of numbers with nested loops."),
    ("A34","Arrays","Selection sort","Implement selection sort in Harbour finding min and swapping."),
    ("A35","Arrays","Insertion sort","Implement insertion sort in Harbour with element shifting."),
    ("A36","Arrays","Binary search","Implement binary search in Harbour on sorted array returning index."),
    ("A37","Arrays","Array intersection","Write Harbour function returning intersection of two arrays."),
    ("A38","Arrays","Array difference","Write Harbour function returning elements in first not in second array."),
    ("A39","Arrays","Array chunk","Write Harbour function splitting array into chunks of size N."),
    ("A40","Arrays","Array zip","Write Harbour function zipping two arrays into pairs."),
    ("A41","Arrays","Array rotate","Write Harbour function rotating array left by N positions."),
    ("A42","Arrays","Array compact","Write Harbour function removing NIL elements from array."),
    ("A43","Arrays","Array distinct","Write Harbour function returning distinct elements preserving order."),
    ("A44","Arrays","Array deep copy","Write Harbour function creating deep copy of nested array."),
    ("A45","Arrays","Hash frequency","Write Harbour function counting frequency of array elements using hash."),
    ("A46","Arrays","Array group by","Write Harbour function grouping array elements by criterion."),
    ("A47","Arrays","Array windows","Write Harbour function creating sliding windows of size N."),
    ("A48","Arrays","Array cartesian","Write Harbour function computing cartesian product of two arrays."),
    # OOP (22)
    ("O01","OOP","Basic class","Write Harbour class Rectangle with DATA, METHOD New, Area, Perimeter."),
    ("O02","OOP","Inheritance","Write Harbour classes Animal base and Dog derived with Breed DATA and Speak METHOD."),
    ("O03","OOP","Polymorphism","Write Harbour classes Shape Circle Rectangle with area method each."),
    ("O04","OOP","Operator overload","Write Harbour class Complex overloading + and * operators."),
    ("O05","OOP","Singleton","Implement Singleton pattern in Harbour for Logger class GetInstance method."),
    ("O06","OOP","Observer","Implement Observer pattern in Harbour with Subject and Observer classes."),
    ("O07","OOP","Factory","Implement Factory pattern creating different shape objects based on input."),
    ("O08","OOP","Constructor","Write Harbour class with New constructor and Destroy destructor."),
    ("O09","OOP","Stack class","Write Harbour class Stack with Push Pop Peek IsEmpty methods."),
    ("O10","OOP","Dictionary","Write Harbour class Dictionary using hash with Add Get Remove Contains."),
    ("O11","OOP","Inheritance chain","Write 3-level inheritance Vehicle Car ElectricCar with overriding."),
    ("O12","OOP","Abstract class","Write abstract Drawable class with Draw method in Circle Square."),
    ("O13","OOP","ToString","Write Harbour class with ToString returning formatted representation."),
    ("O14","OOP","CompareTo","Write Harbour class with CompareTo for sorting by field."),
    ("O15","OOP","Clone","Write Harbour class with Clone method creating deep copy."),
    ("O16","OOP","Iterator","Write Harbour class implementing iteration over collection."),
    ("O17","OOP","Builder","Implement Builder pattern constructing complex HTML elements."),
    ("O18","OOP","Strategy","Implement Strategy pattern with different sorting strategies."),
    ("O19","OOP","Decorator","Implement Decorator pattern wrapping base component."),
    ("O20","OOP","Properties","Write Harbour class with property getters and setters."),
    ("O21","OOP","Static method","Write Harbour class with CLASSDATA and CLASS factory method."),
    ("O22","OOP","Composition","Write Harbour classes using composition Engine inside Car."),
    # OTHER (9)
    ("X01","Other","Preprocessor defines","Write Harbour preprocessor #define for constants and #ifdef platform detection."),
    ("X02","Other","Custom command","Write #xcommand shorthand for declaring variables with initialization."),
    ("X03","Other","HB_Is functions","Write validation using HB_IsString HB_IsNumeric HB_IsArray HB_IsNil."),
    ("X04","Other","Regex validation","Write Harbour code using HB_RegExCompile HB_RegExMatch to validate emails."),
    ("X05","Other","Serialization","Write Harbour code using HB_Serialize HB_Deserialize to save load hash."),
    ("X06","Other","File path ops","Write Harbour code using hb_DirBuild hb_FileNameGet hb_PathJoin."),
    ("X07","Other","Version check","Write Harbour code using HB_Version to detect version conditionally."),
    ("X08","Other","Translation","Write Harbour #translate directives mapping alternative syntax."),
    ("X09","Other","Conditional defines","Write Harbour code with nested ifdef ifndef else for feature toggling."),
    # FUNCTIONS (8)
    ("F01","Functions","Default params","Write Harbour function with default parameter values."),
    ("F02","Functions","Recursion","Write recursive Harbour function for factorial."),
    ("F03","Functions","Scope demo","Write Harbour code demonstrating LOCAL STATIC PRIVATE PUBLIC scope."),
    ("F04","Functions","Code block eval","Write Harbour code using Eval with code blocks and AEval."),
    ("F05","Functions","Error handling","Write Harbour function with BEGIN SEQUENCE RECOVER for safe reading."),
    ("F06","Functions","Pass by ref","Write Harbour function modifying caller variable with @."),
    ("F07","Functions","Variable args","Write Harbour function accepting variable number of arguments."),
    ("F08","Functions","Nested calls","Write Harbour code with nested function calls and scope isolation."),
    # DATABASE (7)
    ("D01","Database","Create DBF","Write Harbour code creating DBF with DBCreate specifying field types."),
    ("D02","Database","Open append","Write Harbour code opening DBF with DBUseArea appending records."),
    ("D03","Database","Indexing","Write Harbour code creating index with ORDCREATE and DBSeek."),
    ("D04","Database","DBEval sum","Write Harbour code using DBEval to sum numeric field."),
    ("D05","Database","Filter","Write Harbour code using SET FILTER TO processing filtered records."),
    ("D06","Database","Multi-area","Write Harbour code using multiple work areas with SELECT."),
    ("D07","Database","Relations","Write Harbour code setting parent-child relation DBSetRelation."),
    # FILE I/O (4)
    ("I01","File I/O","Text read write","Write Harbour functions for text file R/W using FCreate FOpen FRead FWrite FClose."),
    ("I02","File I/O","Line by line","Write Harbour code reading file line by line with FEof."),
    ("I03","File I/O","Directory list","Write Harbour code using Directory listing files with pattern."),
    ("I04","File I/O","File exists","Write Harbour code checking file existence with File function."),
    # CONTROL (2)
    ("C01","Control","Complex IF","Write Harbour function nested IF ELSEIF ELSE with AND OR conditions."),
    ("C02","Control","Nested loops","Write Harbour code nested FOR loops EXIT LOOP finding combinations."),
]

print(f"{'='*60}")
print(f"TEST BATTERY: {len(TESTS)} tests | Model: {MODEL}")
print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"{'='*60}")

results = []
pass_c = fail_c = 0

for i, (tid, cat, name, prompt) in enumerate(TESTS, 1):
    sys.stdout.write(f"\r[{i:3d}/{len(TESTS)}] {tid} {name}...")
    sys.stdout.flush()

    res = query(prompt)

    if res.get("err"):
        results.append({"id":tid,"cat":cat,"name":name,"ok":False,"err":res["err"],"code":"","tok":0,"tps":0,"dur":0,"lines":0})
        fail_c += 1
        print(f" ERR: {res['err'][:60]}")
        save(results, {"pass":pass_c,"fail":fail_c,"rate":pass_c/len(results)*100 if results else 0})
        continue

    code = clean(res["resp"])
    ok, cerr = compile_hb(code)

    if ok: pass_c += 1
    else: fail_c += 1

    err_short = cerr.split('\n')[0][:80] if cerr and not ok else ""
    print(f"\r[{i:3d}/{len(TESTS)}] {tid} {name}... {'PASS' if ok else 'FAIL'} | {code.count(chr(10))+1}L | {res['tok']}t | {res['tps']:.0f}tps | {res['dur']:.1f}s" + (f" | {err_short}" if err_short else ""))

    results.append({"id":tid,"cat":cat,"name":name,"ok":ok,"err":cerr[:400],"code":code[:2500],"tok":res["tok"],"tps":res["tps"],"dur":res["dur"],"lines":code.count('\n')+1})

    save(results, {"pass":pass_c,"fail":fail_c,"rate":pass_c/len(results)*100})

print(f"\n\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")

cats = {}
for r in results:
    c = r["cat"]
    if c not in cats: cats[c] = [0,0]
    cats[c][0 if r["ok"] else 1] += 1

print(f"\n{'Category':<12} {'Pass':>5} {'Fail':>5} {'Rate':>7}")
print("-"*32)
for c in sorted(cats):
    p,f = cats[c]
    print(f"{c:<12} {p:>5} {f:>5} {p/(p+f)*100:>6.0f}%")
print(f"\n{'TOTAL':<12} {pass_c:>5} {fail_c:>5} {pass_c/len(results)*100:>6.0f}%")

total_tok = sum(r["tok"] for r in results)
total_dur = sum(r["dur"] for r in results)
print(f"Tokens: {total_tok:,} | Time: {total_dur:.0f}s | TPS: {total_tok/max(total_dur,1):.0f}")

save(results, {"pass":pass_c,"fail":fail_c,"rate":pass_c/len(results)*100,"cats":cats,
               "total_tok":total_tok,"total_dur":total_dur})
print(f"\nSaved: {RESULTS_FILE}")
