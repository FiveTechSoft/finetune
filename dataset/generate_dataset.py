#!/usr/bin/env python3
"""
Generador de dataset agentic para fine-tuning de LLMs.

Genera datos de entrenamiento para mejorar capacidades agenticas:
- Tool calling single-turn y multi-turn
- State tracking entre turnos
- Parallel function calls
- Planificación y descomposición de tareas
- Error recovery
- Abstención (cuando NO usar herramientas)

Uso:
    python generate_dataset.py --output ./dataset --samples 5000
    python generate_dataset.py --output ./dataset --samples 10000 --categories all
"""

import json
import random
import argparse
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


# =============================================================================
# Definición de herramientas disponibles
# =============================================================================

TOOL_DEFINITIONS = {
    "search_web": {
        "name": "search_web",
        "description": "Busca información en internet",
        "parameters": {
            "query": {"type": "string", "description": "Término de búsqueda"},
            "num_results": {"type": "integer", "description": "Número de resultados", "default": 5}
        }
    },
    "get_stock_price": {
        "name": "get_stock_price",
        "description": "Obtiene el precio actual de una acción",
        "parameters": {
            "symbol": {"type": "string", "description": "Símbolo de la acción (ej: AAPL, GOOGL)"}
        }
    },
    "send_email": {
        "name": "send_email",
        "description": "Envía un email",
        "parameters": {
            "to": {"type": "string", "description": "Destinatario"},
            "subject": {"type": "string", "description": "Asunto del email"},
            "body": {"type": "string", "description": "Cuerpo del email"}
        }
    },
    "create_file": {
        "name": "create_file",
        "description": "Crea un archivo en el sistema",
        "parameters": {
            "path": {"type": "string", "description": "Ruta del archivo"},
            "content": {"type": "string", "description": "Contenido del archivo"}
        }
    },
    "run_code": {
        "name": "run_code",
        "description": "Ejecuta código Python",
        "parameters": {
            "code": {"type": "string", "description": "Código Python a ejecutar"},
            "timeout": {"type": "integer", "description": "Timeout en segundos", "default": 30}
        }
    },
    "query_database": {
        "name": "query_database",
        "description": "Ejecuta una consulta SQL",
        "parameters": {
            "query": {"type": "string", "description": "Consulta SQL"},
            "database": {"type": "string", "description": "Nombre de la base de datos"}
        }
    },
    "get_weather": {
        "name": "get_weather",
        "description": "Obtiene el clima de una ciudad",
        "parameters": {
            "city": {"type": "string", "description": "Nombre de la ciudad"},
            "units": {"type": "string", "description": "Unidades (celsius/fahrenheit)", "default": "celsius"}
        }
    },
    "create_calendar_event": {
        "name": "create_calendar_event",
        "description": "Crea un evento en el calendario",
        "parameters": {
            "title": {"type": "string", "description": "Título del evento"},
            "date": {"type": "string", "description": "Fecha (YYYY-MM-DD)"},
            "time": {"type": "string", "description": "Hora (HH:MM)"},
            "duration_minutes": {"type": "integer", "description": "Duración en minutos"}
        }
    },
    "translate_text": {
        "name": "translate_text",
        "description": "Traduce texto a otro idioma",
        "parameters": {
            "text": {"type": "string", "description": "Texto a traducir"},
            "target_language": {"type": "string", "description": "Idioma destino"}
        }
    },
    "analyze_sentiment": {
        "name": "analyze_sentiment",
        "description": "Analiza el sentimiento de un texto",
        "parameters": {
            "text": {"type": "string", "description": "Texto a analizar"}
        }
    }
}


# =============================================================================
# Templates de escenarios
# =============================================================================

SINGLE_TURN_SCENARIOS = [
    {
        "user": "¿Cuál es el precio de las acciones de Apple?",
        "tools": ["get_stock_price"],
        "expected_calls": [{"function": "get_stock_price", "args": {"symbol": "AAPL"}}],
        "response_template": "El precio actual de las acciones de Apple (AAPL) es ${price}."
    },
    {
        "user": "Busca información sobre inteligencia artificial generativa",
        "tools": ["search_web"],
        "expected_calls": [{"function": "search_web", "args": {"query": "inteligencia artificial generativa"}}],
        "response_template": "Encontré los siguientes resultados sobre inteligencia artificial generativa:\n{results}"
    },
    {
        "user": "¿Qué tiempo hace en Madrid?",
        "tools": ["get_weather"],
        "expected_calls": [{"function": "get_weather", "args": {"city": "Madrid"}}],
        "response_template": "En Madrid el clima actual es: {weather}"
    },
    {
        "user": "Analiza el sentimiento de este texto: 'El producto es excelente, superó mis expectativas'",
        "tools": ["analyze_sentiment"],
        "expected_calls": [{"function": "analyze_sentiment", "args": {"text": "El producto es excelente, superó mis expectativas"}}],
        "response_template": "El sentimiento del texto es {sentiment} con una puntuación de {score}."
    },
    {
        "user": "Traduce 'Hello, how are you?' al español",
        "tools": ["translate_text"],
        "expected_calls": [{"function": "translate_text", "args": {"text": "Hello, how are you?", "target_language": "spanish"}}],
        "response_template": "La traducción es: {translation}"
    },
]

MULTI_TURN_SCENARIOS = [
    {
        "description": "Búsqueda de información + envío de email",
        "turns": [
            {"user": "Busca el precio de las acciones de Google"},
            {"assistant_calls": [{"function": "get_stock_price", "args": {"symbol": "GOOGL"}}], "tool_result": '{"price": 142.50}'},
            {"user": "Ahora envía un email a mi jefe con ese precio"},
            {"assistant_calls": [{"function": "send_email", "args": {"to": "jefe@empresa.com", "subject": "Precio GOOGL", "body": "El precio de Google (GOOGL) es $142.50"}}]}
        ],
        "state_tracking": ["GOOGL_price = 142.50"]
    },
    {
        "description": "Consulta a BD + análisis + creación de archivo",
        "turns": [
            {"user": "Consulta en la base de datos 'ventas' las ventas del último mes"},
            {"assistant_calls": [{"function": "query_database", "args": {"query": "SELECT * FROM ventas WHERE fecha >= DATE_SUB(NOW(), INTERVAL 1 MONTH)", "database": "ventas"}}], "tool_result": '{"rows": 150, "total": 45000.00}'},
            {"user": "Analiza si las ventas son positivas o negativas"},
            {"assistant_calls": [{"function": "analyze_sentiment", "args": {"text": "Ventas del último mes: 150 transacciones, total $45,000"}}], "tool_result": '{"sentiment": "positive", "score": 0.75}'},
            {"user": "Guarda un reporte con esos datos"},
            {"assistant_calls": [{"function": "create_file", "args": {"path": "reportes/ventas_mes.txt", "content": "Reporte de Ventas - Último Mes\nTransacciones: 150\nTotal: $45,000\nTendencia: Positiva"}}]}
        ],
        "state_tracking": ["ventas_count = 150", "ventas_total = 45000", "sentiment = positive"]
    },
]

PARALLEL_CALL_SCENARIOS = [
    {
        "user": "¿Qué tiempo hace en Madrid y en Barcelona?",
        "tools": ["get_weather"],
        "expected_calls": [
            {"function": "get_weather", "args": {"city": "Madrid"}},
            {"function": "get_weather", "args": {"city": "Barcelona"}}
        ],
        "response_template": "Aquí tienes el clima de ambas ciudades:\n- Madrid: {weather_madrid}\n- Barcelona: {weather_barcelona}"
    },
    {
        "user": "Dame el precio de Apple y Google",
        "tools": ["get_stock_price"],
        "expected_calls": [
            {"function": "get_stock_price", "args": {"symbol": "AAPL"}},
            {"function": "get_stock_price", "args": {"symbol": "GOOGL"}}
        ],
        "response_template": "Los precios son:\n- Apple (AAPL): ${price_aapl}\n- Google (GOOGL): ${price_googl}"
    },
    {
        "user": "Busca información sobre Python y sobre Rust",
        "tools": ["search_web"],
        "expected_calls": [
            {"function": "search_web", "args": {"query": "Python programming language"}},
            {"function": "search_web", "args": {"query": "Rust programming language"}}
        ],
        "response_template": "Aquí tienes información sobre ambos lenguajes:\n\nPython:\n{results_python}\n\nRust:\n{results_rust}"
    },
]

PLANNING_SCENARIOS = [
    {
        "user": "Crea un reporte semanal de ventas: consulta la base de datos, analiza los datos y guarda un archivo con el resumen",
        "tools": ["query_database", "analyze_sentiment", "create_file"],
        "plan": [
            "1. Consultar ventas de la semana en la BD",
            "2. Analizar tendencia de los datos",
            "3. Crear archivo con el reporte"
        ],
        "expected_sequence": [
            {"function": "query_database", "args": {"query": "SELECT * FROM ventas WHERE fecha >= DATE_SUB(NOW(), INTERVAL 1 WEEK)", "database": "ventas"}},
            {"function": "analyze_sentiment", "args": {"text": "{query_result_summary}"}},
            {"function": "create_file", "args": {"path": "reportes/semanal.txt", "content": "{report_content}"}}
        ]
    },
    {
        "user": "Investiga sobre energías renovables, traduce los hallazgos al inglés y envíame un email con el resumen",
        "tools": ["search_web", "translate_text", "send_email"],
        "plan": [
            "1. Buscar información sobre energías renovables",
            "2. Traducir los hallazgos al inglés",
            "3. Enviar email con el resumen"
        ],
        "expected_sequence": [
            {"function": "search_web", "args": {"query": "energías renovables estado actual"}},
            {"function": "translate_text", "args": {"text": "{search_summary}", "target_language": "english"}},
            {"function": "send_email", "args": {"to": "user@example.com", "subject": "Renewable Energy Research", "body": "{translated_text}"}}
        ]
    },
]

ERROR_RECOVERY_SCENARIOS = [
    {
        "description": "Error en consulta a BD - reintentar con consulta corregida",
        "turns": [
            {"user": "Consulta las ventas de enero"},
            {"assistant_calls": [{"function": "query_database", "args": {"query": "SELECT * FROM ventas WHERE mes = 'enero'", "database": "ventas"}}], "tool_result": '{"error": "Columna mes no existe. Usa columna fecha con formato YYYY-MM-DD"}'},
            {"assistant_calls": [{"function": "query_database", "args": {"query": "SELECT * FROM ventas WHERE fecha BETWEEN '2024-01-01' AND '2024-01-31'", "database": "ventas"}}], "tool_result": '{"rows": 45, "total": 12500.00}'},
            {"user": "Perfecto, ahora analiza esos datos"},
            {"assistant_calls": [{"function": "analyze_sentiment", "args": {"text": "Ventas enero 2024: 45 transacciones, total $12,500"}}]}
        ],
        "error_type": "schema_mismatch"
    },
    {
        "description": "Error de validación - argumento inválido",
        "turns": [
            {"user": "¿Qué tiempo hace en la luna?"},
            {"assistant_response": "Lo siento, no puedo consultar el clima en la luna. Solo puedo obtener información del clima para ciudades en la Tierra. ¿Te gustaría consultar el clima de alguna ciudad?"},
        ],
        "error_type": "invalid_argument"
    },
]

ABSTENTION_SCENARIOS = [
    {
        "user": "¿Qué es la fotosíntesis?",
        "tools": ["search_web", "get_stock_price", "send_email"],
        "expected_behavior": "answer_directly",
        "response": "La fotosíntesis es el proceso mediante el cual las plantas convierten la luz solar, agua y dióxido de carbono en glucosa y oxígeno. Es fundamental para la vida en la Tierra ya que produce el oxígeno que respiramos y es la base de las cadenas alimenticias."
    },
    {
        "user": "Explica qué es una función recursiva en programación",
        "tools": ["run_code", "query_database"],
        "expected_behavior": "answer_directly",
        "response": "Una función recursiva es una función que se llama a sí misma para resolver un problema. Tiene dos componentes esenciales: el caso base (que detiene la recursión) y el caso recursivo (que divide el problema en subproblemas más pequeños). Por ejemplo, el factorial de n se define como n! = n * (n-1)! con caso base 0! = 1."
    },
    {
        "user": "¿Quién pintó la Mona Lisa?",
        "tools": ["search_web", "get_weather"],
        "expected_behavior": "answer_directly",
        "response": "La Mona Lisa fue pintada por Leonardo da Vinci entre 1503 y 1519. Se encuentra expuesta en el Museo del Louvre en París, Francia."
    },
]


# =============================================================================
# Generador de datos
# =============================================================================

class AgenticDatasetGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def _make_tool_call_message(self, function_name: str, args: dict, result: str = None) -> dict:
        """Genera un mensaje con tool_call."""
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{hashlib.md5(json.dumps(args).encode()).hexdigest()[:8]}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(args)
                }
            }]
        }
        return msg
    
    def _make_tool_result_message(self, result: str) -> dict:
        """Genera un mensaje de resultado de herramienta."""
        return {
            "role": "tool",
            "content": result
        }
    
    def generate_single_turn(self) -> list:
        """Genera ejemplos de tool calling single-turn."""
        examples = []
        for scenario in SINGLE_TURN_SCENARIOS:
            messages = [
                {"role": "system", "content": self._get_system_prompt(scenario["tools"])},
                {"role": "user", "content": scenario["user"]}
            ]
            for call in scenario["expected_calls"]:
                args = call["args"]
                # Agregar variaciones
                varied_args = self._vary_args(args)
                messages.append(self._make_tool_call_message(call["function"], varied_args))
                # Simular resultado
                result = self._simulate_result(call["function"], varied_args)
                messages.append(self._make_tool_result_message(result))
            
            # Respuesta final
            messages.append({
                "role": "assistant",
                "content": scenario["response_template"].format(price="185.50", results="...", weather="Soleado 22°C", sentiment="positivo", score="0.85", translation="Hola, ¿cómo estás?")
            })
            examples.append({"messages": messages})
        
        return examples
    
    def generate_multi_turn(self) -> list:
        """Genera ejemplos de multi-turn con state tracking."""
        examples = []
        for scenario in MULTI_TURN_SCENARIOS:
            messages = [
                {"role": "system", "content": self._get_system_prompt(["get_stock_price", "send_email", "query_database", "analyze_sentiment", "create_file"])}
            ]
            for turn in scenario["turns"]:
                if "user" in turn:
                    messages.append({"role": "user", "content": turn["user"]})
                if "assistant_calls" in turn:
                    for call in turn["assistant_calls"]:
                        messages.append(self._make_tool_call_message(call["function"], call["args"]))
                    if "tool_result" in turn:
                        messages.append(self._make_tool_result_message(turn["tool_result"]))
            examples.append({"messages": messages})
        
        return examples
    
    def generate_parallel(self) -> list:
        """Genera ejemplos de llamadas paralelas."""
        examples = []
        for scenario in PARALLEL_CALL_SCENARIOS:
            messages = [
                {"role": "system", "content": self._get_system_prompt(scenario["tools"])},
                {"role": "user", "content": scenario["user"]}
            ]
            # Agregar todas las llamadas en un solo mensaje
            tool_calls = []
            for call in scenario["expected_calls"]:
                tool_calls.append({
                    "id": f"call_{hashlib.md5(json.dumps(call['args']).encode()).hexdigest()[:8]}",
                    "type": "function",
                    "function": {
                        "name": call["function"],
                        "arguments": json.dumps(call["args"])
                    }
                })
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls
            })
            # Resultados
            for call in scenario["expected_calls"]:
                result = self._simulate_result(call["function"], call["args"])
                messages.append(self._make_tool_result_message(result))
            # Respuesta final
            messages.append({
                "role": "assistant",
                "content": scenario["response_template"].format(
                    weather_madrid="Soleado 25°C", weather_barcelona="Nublado 20°C",
                    price_aapl="185.50", price_googl="142.30",
                    results_python="Python es un lenguaje interpretado...", results_rust="Rust es un lenguaje compilado..."
                )
            })
            examples.append({"messages": messages})
        
        return examples
    
    def generate_planning(self) -> list:
        """Genera ejemplos de planificación."""
        examples = []
        for scenario in PLANNING_SCENARIOS:
            messages = [
                {"role": "system", "content": self._get_system_prompt(scenario["tools"])},
                {"role": "user", "content": scenario["user"]}
            ]
            # Primero, mostrar el plan
            plan_text = "Voy a dividir esta tarea en los siguientes pasos:\n"
            for step in scenario["plan"]:
                plan_text += f"{step}\n"
            plan_text += "\nEmpezando:\n"
            
            # Ejecutar cada paso
            for i, step in enumerate(scenario["expected_sequence"]):
                if i == 0:
                    messages.append({"role": "assistant", "content": plan_text})
                messages.append(self._make_tool_call_message(step["function"], step["args"]))
                result = self._simulate_result(step["function"], step["args"])
                messages.append(self._make_tool_result_message(result))
            
            # Respuesta final
            messages.append({
                "role": "assistant",
                "content": "He completado la tarea. He creado el reporte con la información solicitada."
            })
            examples.append({"messages": messages})
        
        return examples
    
    def generate_error_recovery(self) -> list:
        """Genera ejemplos de manejo de errores."""
        examples = []
        for scenario in ERROR_RECOVERY_SCENARIOS:
            messages = [
                {"role": "system", "content": self._get_system_prompt(["query_database", "analyze_sentiment", "get_weather"])}
            ]
            for turn in scenario["turns"]:
                if "user" in turn:
                    messages.append({"role": "user", "content": turn["user"]})
                if "assistant_calls" in turn:
                    for call in turn["assistant_calls"]:
                        messages.append(self._make_tool_call_message(call["function"], call["args"]))
                    if "tool_result" in turn:
                        messages.append(self._make_tool_result_message(turn["tool_result"]))
                if "assistant_response" in turn:
                    messages.append({"role": "assistant", "content": turn["assistant_response"]})
            examples.append({"messages": messages})
        
        return examples
    
    def generate_abstention(self) -> list:
        """Genera ejemplos donde NO se debe usar herramientas."""
        examples = []
        for scenario in ABSTENTION_SCENARIOS:
            messages = [
                {"role": "system", "content": self._get_system_prompt(scenario["tools"])},
                {"role": "user", "content": scenario["user"]},
                {"role": "assistant", "content": scenario["response"]}
            ]
            examples.append({"messages": messages})
        
        return examples
    
    def _get_system_prompt(self, tools: list) -> str:
        """Genera el system prompt con las herramientas disponibles."""
        tool_defs = []
        for t in tools:
            if t in TOOL_DEFINITIONS:
                tool_defs.append(json.dumps(TOOL_DEFINITIONS[t], ensure_ascii=False))
        
        return f"""Eres un asistente inteligente con acceso a las siguientes herramientas:

{chr(10).join(tool_defs)}

Reglas importantes:
1. Usa herramientas SOLO cuando sea necesario para responder la pregunta
2. Si puedes responder directamente, no uses herramientas
3. Para llamadas paralelas, incluye todas en un solo mensaje
4. Mantén el contexto entre turnos - recuerda los resultados anteriores
5. Si una herramienta falla, intenta corregir o explica el problema al usuario
6. Siempre verifica que los argumentos sean válidos antes de llamar una función"""
    
    def _vary_args(self, args: dict) -> dict:
        """Añade variaciones argumentos para diversidad."""
        varied = args.copy()
        # Variaciones de símbolos de acciones
        stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]
        if "symbol" in varied and varied["symbol"] in ["AAPL", "GOOGL"]:
            varied["symbol"] = self.rng.choice(stock_symbols)
        # Variaciones de ciudades
        cities = ["Madrid", "Barcelona", "Lima", "Buenos Aires", "Ciudad de México", "Bogotá", "Santiago"]
        if "city" in varied:
            varied["city"] = self.rng.choice(cities)
        return varied
    
    def _simulate_result(self, function_name: str, args: dict) -> str:
        """Simula el resultado de una llamada a función."""
        if function_name == "get_stock_price":
            price = round(self.rng.uniform(50, 300), 2)
            return json.dumps({"symbol": args.get("symbol", "AAPL"), "price": price, "currency": "USD"})
        elif function_name == "get_weather":
            temp = self.rng.randint(15, 35)
            conditions = ["Soleado", "Nublado", "Lluvioso", "Parcialmente nublado"]
            return json.dumps({"city": args.get("city", "Madrid"), "temperature": temp, "condition": self.rng.choice(conditions), "units": "celsius"})
        elif function_name == "search_web":
            return json.dumps({"results": [{"title": "Resultado 1", "snippet": "Información relevante..."}, {"title": "Resultado 2", "snippet": "Más información..."}]})
        elif function_name == "send_email":
            return json.dumps({"status": "sent", "message_id": f"msg_{self.rng.randint(1000, 9999)}"})
        elif function_name == "create_file":
            return json.dumps({"status": "created", "path": args.get("path", "file.txt")})
        elif function_name == "run_code":
            return json.dumps({"stdout": "Resultado de la ejecución", "exit_code": 0})
        elif function_name == "query_database":
            rows = self.rng.randint(10, 200)
            total = round(self.rng.uniform(1000, 50000), 2)
            return json.dumps({"rows": rows, "total": total})
        elif function_name == "translate_text":
            return json.dumps({"translation": "Traducción simulada del texto", "source": "auto", "target": args.get("target_language", "spanish")})
        elif function_name == "analyze_sentiment":
            sentiments = ["positive", "negative", "neutral"]
            return json.dumps({"sentiment": self.rng.choice(sentiments), "score": round(self.rng.uniform(0.3, 0.95), 2)})
        return json.dumps({"result": "ok"})
    
    def generate_all(self, num_samples: int, categories: list = None) -> list:
        """Genera el dataset completo."""
        if categories is None:
            categories = ["single_turn", "multi_turn", "parallel", "planning", "error_recovery", "abstention"]
        
        all_examples = []
        
        # Distribuir muestras entre categorías
        weights = {
            "single_turn": 0.30,
            "multi_turn": 0.25,
            "parallel": 0.15,
            "planning": 0.15,
            "error_recovery": 0.10,
            "abstention": 0.05
        }
        
        for cat in categories:
            if cat not in weights:
                continue
            
            n = int(num_samples * weights[cat])
            
            if cat == "single_turn":
                base = self.generate_single_turn()
            elif cat == "multi_turn":
                base = self.generate_multi_turn()
            elif cat == "parallel":
                base = self.generate_parallel()
            elif cat == "planning":
                base = self.generate_planning()
            elif cat == "error_recovery":
                base = self.generate_error_recovery()
            elif cat == "abstention":
                base = self.generate_abstention()
            else:
                continue
            
            # Repetir y variar para alcanzar el número deseado
            while len(base) < n:
                base.extend(base[:min(10, n - len(base))])
            
            all_examples.extend(base[:n])
        
        self.rng.shuffle(all_examples)
        return all_examples


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Genera dataset agentic para fine-tuning")
    parser.add_argument("--output", type=str, default="./dataset", help="Directorio de salida")
    parser.add_argument("--samples", type=int, default=5000, help="Número de muestras a generar")
    parser.add_argument("--split", type=float, default=0.9, help="Proporción train/eval")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidad")
    parser.add_argument("--categories", nargs="+", default=None,
                       choices=["single_turn", "multi_turn", "parallel", "planning", "error_recovery", "abstention", "all"],
                       help="Categorías a generar")
    args = parser.parse_args()
    
    if args.categories and "all" in args.categories:
        args.categories = None
    
    print(f"Generando {args.samples} muestras del dataset agentic...")
    generator = AgenticDatasetGenerator(seed=args.seed)
    dataset = generator.generate_all(args.samples, args.categories)
    
    # Split train/eval
    split_idx = int(len(dataset) * args.split)
    train_data = dataset[:split_idx]
    eval_data = dataset[split_idx:]
    
    # Guardar
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(eval_path, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Dataset generado:")
    print(f"  Train: {len(train_data)} muestras -> {train_path}")
    print(f"  Eval:  {len(eval_data)} muestras -> {eval_path}")
    print(f"  Total: {len(dataset)} muestras")
    
    # Estadísticas
    print(f"\nDistribución por categoría:")
    cat_counts = {}
    for item in dataset:
        # Contar turnos como proxy de categoría
        num_turns = len(item["messages"])
        if num_turns <= 4:
            cat = "single_turn"
        elif num_turns <= 8:
            cat = "multi_turn"
        else:
            cat = "complex"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
