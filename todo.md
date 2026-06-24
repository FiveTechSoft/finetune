# TODO - Harbour Fine-tuning Session

## Estado actual (2026-06-23)

### Training COMPLETADO
- Modelo: Qwen3.6-35B-A3B (MoE, 256 experts, 41 capas)
- LoRA: r=8, alpha=16, targets: q/k/v/o_proj + gate/up/down_proj
- Dataset: 896 train, 100 val (harbour_train.jsonl / harbour_val.jsonl)
- Epochs: 3 (168 steps, batch 1, grad_accum 16)
- Resultado: train_loss=0.6456, eval_loss=0.5957
- Duración: 3h 43min en NVIDIA GB10 (121GB unified)

### Archivos generados
- `output/final/adapter_model.safetensors` (1.8GB) - LoRA adapter
- `output/final/adapter_config.json` - config del adapter
- `output/final/tokenizer.json` + `tokenizer_config.json` - tokenizer
- `output/checkpoint-50/`, `checkpoint-100/`, `checkpoint-168/` - checkpoints
- `output/gguf_gguf/Qwen3.6-35B-A3B.BF16-*.gguf` (71GB) - BF16 GGUF merge

### GGUF Export - PENDIENTE
- El export a Q4_K_M falló por bug en llama.cpp prebuilt (off-by-one en MoE 41 capas)
- Se compiló llama.cpp desde source pero persiste el bug: `Bad layer 40 for tensor blk.40.ffn_down_exps.weight`
- BF16 GGUF SÍ se generó correctamente
- **Pendiente**: usar una versión más nueva de llama.cpp o reportar el bug upstream

### Bugs corregidos durante la sesión
1. **Python.h missing**: bitsandbytes no encontraba headers de Python. Fix: `CPATH=/usr/include/python3.12`
2. **CUDA OOM**: MoE 256 experts con seq_len=2048 causaba OOM. Fix: reducir a 1024 + LoRA r=8
3. **Pickle SFTConfig**: unsloth/trl incompatibilidad al guardar checkpoints. Fix: monkeypatch en torch.save
4. **GGUF off-by-one**: llama.cpp prebuilt no soporta 41 capas en MoE. Pendiente.

### Baseline test results (modelo base, Ollama)
- Modelo: qwen3.6:35b via Ollama
- Resultado: 85/97 pass (87.6%)
- Archivo: test_baseline_100.json

### Entorno
- Python venv: /home/fivetech/finetune/venv/
- Harbour compiler: /home/fivetech/harbour/bin/linux/gcc/harbour
- Ollama: http://localhost:11434 (modelo qwen3.6:35b cargado)
- GPU: NVIDIA GB10, 121.6GB unified memory, CUDA 12.1

## PROXIMO PASO: Test battery con modelo entrenado
1. Servir el modelo Qwen3.6-35B-A3B + LoRA adapter vía HTTP API
2. Modificar test_battery_100.py para usar la API del modelo entrenado
3. Comparar resultados con baseline (87.6%)
4. Guardar resultados en test_baseline_100_trained.json

## Pendientes a futuro
- [ ] Exportar GGUF cuantizado cuando llama.cpp soporte este MoE
- [ ] Hacer merge completo (no solo LoRA) si se necesita GGUF
- [ ] Optimizar hiperparámetros para segunda ronda de training
- [ ] Evaluar con más tests o dataset más grande
