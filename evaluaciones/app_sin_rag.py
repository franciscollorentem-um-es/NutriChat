"""
app_sin_rag.py — Versión SIN RAG del asistente NutriChat (CLI).

Esta versión usa SOLO el LLM (Gemini) sin acceso a la base de datos de recetas.
El modelo responde con su conocimiento general, sin buscar en ChromaDB.

Se usa para comparar manualmente con la versión CON RAG (app_nutrichat.py):
  - CON RAG:  python app_nutrichat.py      (busca recetas reales en la BD)
  - SIN RAG:  python app_sin_rag.py    (responde de memoria, sin BD)

Así se pueden hacer las mismas preguntas en ambas versiones y comparar
la calidad de las respuestas.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import (
    GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    LLM_MAX_RETRIES, LLM_RETRY_WAIT_SECONDS,
)

# Máximo de mensajes en el historial
MAX_HISTORY_MESSAGES = 30

# System prompt SIN mencionar herramientas ni base de datos.
# El modelo responde solo con su conocimiento general.
SYSTEM_PROMPT_SIN_RAG = """Eres NutriChat, un asistente nutricional experto y amable.
Tienes conocimientos generales sobre recetas y nutrición.

INSTRUCCIONES:
- Responde SIEMPRE en el idioma del usuario.
- Cuando el usuario pida recetas, proporciona recetas basándote en tu conocimiento general.
- Incluye información nutricional aproximada (calorías, proteínas, grasas, carbohidratos,
  fibra, azúcar, sodio) para cada receta.
- Presenta cada receta con:
  • Nombre de la receta
  • Calorías y macronutrientes principales
  • Ingredientes
  • Pasos de preparación resumidos
  • Dieta(s) compatibles
- Si el usuario especifica restricciones dietéticas (vegetariano, keto, sin gluten, etc.),
  respétalas.
- Para comparaciones, usa un formato lado a lado o tabla.
- Si no puedes dar una receta exacta, sugiere alternativas.
- Sé conciso y útil.

RESTRICCIONES:
- NO des consejos médicos ni diagnósticos.
"""


def extract_text(content) -> str:
    """
    Extrae texto limpio del contenido de un mensaje del LLM.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def truncate_history(history: list, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    """
    Mantiene solo los últimos N mensajes para no exceder el contexto del LLM.
    """
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


def invoke_with_retry(llm, messages: list) -> any:
    """
    Invoca al LLM con reintentos automáticos ante errores de cuota (429)."""
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < LLM_MAX_RETRIES:
                    wait = LLM_RETRY_WAIT_SECONDS * attempt
                    print(f"\n  Cuota excedida. Reintentando en {wait}s... (intento {attempt}/{LLM_MAX_RETRIES})")
                    time.sleep(wait)
                    continue
            raise
    raise RuntimeError("Se agotaron los reintentos.")


def main():
    print("=" * 55)
    print(" 💪 NUTRICHAT ASSISTANT — VERSIÓN SIN RAG 🍏")
    print("=" * 55)
    print("⚠️  Esta versión NO usa la base de datos de recetas.")
    print("    El modelo responde solo con su conocimiento general.")

    if not GOOGLE_API_KEY:
        print("\nERROR: No se encontró GOOGLE_API_KEY en el archivo .env")
        print("Crea un archivo .env con: GOOGLE_API_KEY=tu_clave_aquí")
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )

    # El historial empieza con el system prompt
    chat_history: list = [SystemMessage(content=SYSTEM_PROMPT_SIN_RAG)]

    print("\nNutriChat (Sin RAG) está listo. Escribe 'salir' para terminar.\n")

    while True:
        try:
            user_input = input("Tú: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("salir", "exit", "quit"):
                print("\nNutriChat: ¡Hasta luego! Que tengas un día saludable.")
                break

            chat_history.append(HumanMessage(content=user_input))
            chat_history = truncate_history(chat_history)

            start = time.time()
            response = invoke_with_retry(llm, chat_history)
            elapsed = time.time() - start

            clean_text = extract_text(response.content)
            print(f"\nNutriChat (Sin RAG): {clean_text}")
            print(f"  [Tiempo: {elapsed:.2f}s]\n")

            chat_history.append(AIMessage(content=clean_text))

        except KeyboardInterrupt:
            print("\n\nNutriChat: ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}")
            print("Inténtalo de nuevo con otra consulta.\n")


if __name__ == "__main__":
    main()
