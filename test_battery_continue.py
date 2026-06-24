#!/usr/bin/env python3
"""Continue test battery from checkpoint - tests 51-100"""

import json, time, subprocess, requests, sys
from pathlib import Path
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.6:35b"
HARBOUR = "/home/fivetech/harbour/bin/linux/gcc/harbour"
WORK_DIR = Path("/home/fivetech/finetune/test_output")
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

# Tests 70-100 (remaining after A01-A48, O01-O21)
TESTS = [
    ("O22","OOP","Composition","Write Harbour classes using composition Engine inside Car."),
    ("X01","Other","Preprocessor defines","Write Harbour preprocessor #define for constants and #ifdef platform detection."),
    ("X02","Other","Custom command","Write #xcommand shorthand for declaring variables with initialization."),
    ("X03","Other","HB_Is functions","Write validation using HB_IsString HB_IsNumeric HB_IsArray HB_IsNil."),
    ("X04","Other","Regex validation","Write Harbour code using HB_RegExCompile HB_RegExMatch to validate emails."),
    ("X05","Other","Serialization","Write Harbour code using HB_Serialize HB_Deserialize to save load hash."),
    ("X06","Other","File path ops","Write Harbour code using hb_DirBuild hb_FileNameGet hb_PathJoin."),
    ("X07","Other","Version check","Write Harbour code using HB_Version to detect version conditionally."),
    ("X08","Other","Translation","Write Harbour #translate directives mapping alternative syntax."),
    ("X09","Other","Conditional defines","Write Harbour code with nested ifdef ifndef else for feature toggling."),
    ("F01","Functions","Default params","Write Harbour function with default parameter values."),
    ("F02","Functions","Recursion","Write recursive Harbour function for factorial."),
    ("F03","Functions","Scope demo","Write Harbour code demonstrating LOCAL STATIC PRIVATE PUBLIC scope."),
    ("F04","Functions","Code block eval","Write Harbour code using Eval with code blocks and AEval."),
    ("F05","Functions","Error handling","Write Harbour function with BEGIN SEQUENCE RECOVER for safe reading."),
    ("F06","Functions","Pass by ref","Write Harbour function modifying caller variable with @."),
    ("F07","Functions","Variable args","Write Harbour function accepting variable number of arguments."),
    ("F08","Functions","Nested calls","Write Harbour code with nested function calls and scope isolation."),
    ("D01","Database","Create DBF","Write Harbour code creating DBF with DBCreate specifying field types."),
    ("D02","Database","Open append","Write Harbour code opening DBF with DBUseArea appending records."),
    ("D03","Database","Indexing","Write Harbour code creating index with ORDCREATE and DBSeek."),
    ("D04","Database","DBEval sum","Write Harbour code using DBEval to sum numeric field."),
    ("D05","Database","Filter","Write Harbour code using SET FILTER TO processing filtered records."),
    ("D06","Database","Multi-area","Write Harbour code using multiple work areas with SELECT."),
    ("D07","Database","Relations","Write Harbour code setting parent-child relation DBSetRelation."),
    ("I01","File I/O","Text read write","Write Harbour functions for text file R/W using FCreate FOpen FRead FWrite FClose."),
    ("I02","File I/O","Line by line","Write Harbour code reading file line by line with FEof."),
    ("I03","File I/O","Directory list","Write Harbour code using Directory listing files with pattern."),
    ("I04","File I/O","File exists","Write Harbour code checking file existence with File function."),
    ("C01","Control","Complex IF","Write Harbour function nested IF ELSEIF ELSE with AND OR conditions."),
    ("C02","Control","Nested loops","Write Harbour code nested FOR loops EXIT LOOP finding combinations."),
]

# Load existing results
with open(RESULTS_FILE) as f:
    data = json.load(f)
results = data["results"]
pass_c = data["pass"]
fail_c = data["fail"]

print(f"{'='*60}")
print(f"CONTINUING from test {len(results)+1}/100")
print(f"So far: {pass_c} pass, {fail_c} fail ({data['rate']:.1f}%)")
print(f"{'='*60}")

for i, (tid, cat, name, prompt) in enumerate(TESTS, len(results)+1):
    sys.stdout.write(f"\r[{i:3d}/100] {tid} {name}...")
    sys.stdout.flush()

    res = query(prompt)

    if res.get("err"):
        results.append({"id":tid,"cat":cat,"name":name,"ok":False,"err":res["err"],"code":"","tok":0,"tps":0,"dur":0,"lines":0})
        fail_c += 1
        print(f"\r[{i:3d}/100] {tid} {name}... ERR: {res['err'][:50]}")
        save(results, {"pass":pass_c,"fail":fail_c,"rate":pass_c/len(results)*100})
        continue

    code = clean(res["resp"])
    ok, cerr = compile_hb(code)

    if ok: pass_c += 1
    else: fail_c += 1

    err_short = cerr.split('\n')[0][:60] if cerr and not ok else ""
    print(f"\r[{i:3d}/100] {tid} {name}... {'PASS' if ok else 'FAIL'} | {code.count(chr(10))+1}L | {res['tok']}t | {res['tps']:.0f}tps" + (f" | {err_short}" if err_short else ""))

    results.append({"id":tid,"cat":cat,"name":name,"ok":ok,"err":cerr[:400],"code":code[:2500],"tok":res["tok"],"tps":res["tps"],"dur":res["dur"],"lines":code.count('\n')+1})

    save(results, {"pass":pass_c,"fail":fail_c,"rate":pass_c/len(results)*100})

# Final summary
print(f"\n\n{'='*60}")
print(f"FINAL RESULTS (100/100)")
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
