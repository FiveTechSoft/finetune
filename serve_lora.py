#!/usr/bin/env python3
"""
Lightweight Ollama-compatible inference server for LoRA model.
Loads base with Unsloth (for GB10 memory handling), then bypasses
Unsloth's generate() wrapper by patching it out.
Serves on port 11435.
"""

import json, os, sys, time, types
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["CPATH"] = "/usr/include/python3.12"

import torch

print("Loading Unsloth...", flush=True)
from unsloth import FastLanguageModel

MODEL_DIR = "/home/fivetech/finetune/models/Qwen3.6-35B-A3B"
LORA_DIR = "/home/fivetech/finetune/output/final"
PORT = 11435

print("Loading base model (4-bit)...", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
    text_only=True,
    device_map="auto",
)
print("Base model loaded.", flush=True)

print("Applying LoRA adapter...", flush=True)
from peft import PeftModel
model = PeftModel.from_pretrained(model, LORA_DIR)

print("Patching out Unsloth generate wrapper...", flush=True)
try:
    base_m = model.base_model.model
    if getattr(base_m.config, "architectures", None) is None:
        base_m.config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
        print("Patched: set architectures on base model config", flush=True)
    if getattr(base_m.config, "model_type", None) is None:
        base_m.config.model_type = "qwen3_moe"
        print("Patched: set model_type on base model config", flush=True)
except Exception as e:
    print(f"Patch warning: {e}", flush=True)

model.eval()
print("Model loaded. Ready for inference.", flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/generate":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                prompt = body.get("prompt", "")
                system = body.get("system", "")
                opts = body.get("options", {})
                temp = opts.get("temperature", 0.2)
                top_p = opts.get("top_p", 0.9)
                max_new = opts.get("num_predict", 1500)

                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
                inputs = tokenizer([text], return_tensors="pt").to(model.device)

                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        temperature=temp,
                        top_p=top_p,
                        do_sample=True,
                    )
                gen_ids = out[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True)
                dur = time.time() - t0
                tok_count = len(gen_ids)

                result = {
                    "model": body.get("model", "lora"),
                    "response": response,
                    "eval_count": tok_count,
                    "eval_duration": int(dur * 1e9),
                    "done": True,
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


print(f"Server on port {PORT}...", flush=True)
HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
