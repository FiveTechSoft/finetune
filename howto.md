# Howto: Fine-tuning Harbour v2 con dataset de 5,004 ejemplos

## Resumen del proyecto

- **Modelo base**: Qwen3.6-35B-A3B (MoE, 256 experts, 41 capas)
- **Hardware**: NVIDIA GB10, 121GB unified memory, CUDA 12.1
- **Dataset**: 5,004 ejemplos Harbour/FWH (vs 996 del entrenamiento anterior)
- **Método**: QLoRA con Unsloth (4-bit)
- **Tiempo estimado**: ~4-5 horas

## Datasets

| Archivo | Entradas | Descripción |
|---|---|---|
| `finetune/dataset/harbour_train.jsonl` | 4,503 | Training set |
| `finetune/dataset/harbour_eval.jsonl` | 501 | Validation set |

### Formato del dataset (nuevo)
```json
{
  "instruction": "Descripción de la tarea",
  "input": "",
  "system": "Prompt del sistema",
  "output": "Código Harbour compilable",
  "task_type": "code_generation"
}
```

### Formato anterior (referencia)
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Hiperparámetros

| Parámetro | Valor | Notas |
|---|---|---|
| Epochs | 2 | Reducido de 3 por dataset más grande |
| Learning rate | 8e-5 | Más conservador que 1e-4 anterior |
| Warmup ratio | 0.05 | Reducido de 0.1 |
| Batch size | 1 | Mantener por memoria |
| Gradient accumulation | 16 | Batch efectivo = 16 |
| Max seq length | 1024 | Suficiente para código Harbour/C |
| LoRA r | 8 | Mantener |
| LoRA alpha | 16 | Mantener (2x r) |
| LoRA dropout | 0 | Recomendado por Unsloth |
| LR scheduler | cosine | Mantener |
| Optim | adamw_8bit | Mantener |
| Save steps | 100 | Reducido de 50 |
| Eval steps | 100 | Reducido de 50 |

## Archivos de salida

```
output_v2/
├── final/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── checkpoint-100/
├── checkpoint-200/
├── checkpoint-300/
├── checkpoint-400/
├── checkpoint-500/
├── checkpoint-563/
└── gguf/
```

## Pasos de ejecución

### 1. Preparar entorno
```bash
cd /home/fivetech/finetune
source venv/bin/activate
```

### 2. Verificar dataset
```bash
wc -l finetune/dataset/harbour_train.jsonl finetune/dataset/harbour_eval.jsonl
head -1 finetune/dataset/harbour_train.jsonl | python3 -m json.tool
```

### 3. Ejecutar entrenamiento
```bash
nohup python train_unsloth.py > train_unsloth_v2.log 2>&1 &
echo $! > train.pid
```

### 4. Monitorear progreso
```bash
# Ver log en tiempo real
tail -f train_unsloth_v2.log

# Ver último checkpoint
ls -la output_v2/checkpoint-*/

# Ver GPU
nvidia-smi
```

### 5. Verificar resultados
```bash
# Ver loss final
grep "train_loss\|eval_loss" train_unsloth_v2.log | tail -5

# Ver tiempo total
grep "Training complete" train_unsloth_v2.log
```

### 6. Servir modelo para tests
```bash
python serve_lora.py &
```

### 7. Ejecutar batería de tests
```bash
python test_lora_100.py
```

## Cambios en train_unsloth.py

### Rutas (líneas 22-25)
```python
# ANTES:
TRAIN_FILE = Path("/home/fivetech/finetune/harbour_train.jsonl")
VAL_FILE = Path("/home/fivetech/finetune/harbour_val.jsonl")
OUTPUT_DIR = Path("/home/fivetech/finetune/output")

# DESPUÉS:
TRAIN_FILE = Path("/home/fivetech/finetune/finetune/dataset/harbour_train.jsonl")
VAL_FILE = Path("/home/fivetech/finetune/finetune/dataset/harbour_eval.jsonl")
OUTPUT_DIR = Path("/home/fivetech/finetune/output_v2")
```

### Formato del dataset (líneas 76-83)
```python
# ANTES:
def format_conversation(entry):
    messages = entry["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

# DESPUÉS:
def format_conversation(entry):
    messages = []
    if "system" in entry:
        messages.append({"role": "system", "content": entry["system"]})
    messages.append({"role": "user", "content": entry["instruction"]})
    messages.append({"role": "assistant", "content": entry["output"]})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}
```

### Hiperparámetros (línea 119)
```python
# ANTES:
num_train_epochs=3,
learning_rate=1e-4,
warmup_ratio=0.1,
save_steps=50,
eval_steps=50,

# DESPUÉS:
num_train_epochs=2,
learning_rate=8e-5,
warmup_ratio=0.05,
save_steps=100,
eval_steps=100,
```

## Bugs conocidos y soluciones

### 1. Python.h missing
```bash
export CPATH=/usr/include/python3.12
```

### 2. CUDA OOM con MoE
- Reducir `max_seq_length` a 1024
- Usar LoRA r=8 (no mayor)

### 3. Pickle SFTConfig
Ya incluido en el script: monkeypatch en `torch.save`

### 4. GGUF export falla con MoE 41 capas
- BF16 GGUF sí se genera
- Q4_K_M falla por bug en llama.cpp
- Solución: usar versión más nueva de llama.cpp

## Comparación con entrenamiento anterior

| Métrica | Anterior (v1) | Nuevo (v2) |
|---|---|---|
| Dataset | 996 entries | 5,004 entries |
| Epochs | 3 | 2 |
| LR | 1e-4 | 8e-5 |
| Steps | 168 | 563 |
| Tiempo | 3h 43min | ~4-5h (est.) |
| train_loss | 0.6456 | ? |
| eval_loss | 0.5957 | ? |

## Rollback

Si hay problemas, volver al modelo anterior:
```bash
# El modelo v1 sigue en output/final/
# Para usarlo:
# 1. Cambiar OUTPUT_DIR en serve_lora.py a output/final
# 2. Reiniciar serve_lora.py
```
