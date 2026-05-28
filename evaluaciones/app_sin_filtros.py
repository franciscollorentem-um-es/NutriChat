"""
app_sin_filtros.py — Versión con solo búsqueda semántica del asistente NutriChat (CLI).

Esta versión usa RAG pero SIN filtros de metadatos. La herramienta de búsqueda
solo usa la query semántica, ignora los filtros de calorías, dieta, proteína, etc.

Comparación:
  - CON FILTROS (híbrida):   python app_nutrichat.py     (semántica + filtros)
  - SIN FILTROS (semántica): python app_sin_filtros.py   (solo semántica)

Así se puede ver que sin filtros, el modelo puede devolver recetas que no cumplen
las restricciones del usuario (ej: pide vegetariano y le sale pollo).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from config import (
    GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT,
    LLM_MAX_RETRIES, LLM_RETRY_WAIT_SECONDS,
)
from nutri_vectordb import NutriVectorDB

MAX_HISTORY_MESSAGES = 30

# Singleton de la BD
_db: Optional[NutriVectorDB] = None

def get_db() -> NutriVectorDB:
    global _db
    if _db is None:
        _db = NutriVectorDB()
    return _db


# Herramienta SIN FILTROS: solo búsqueda semántica
class RecipeSearchInputSinFiltros(BaseModel):
    """
    Esquema simplificado: solo query, sin filtros.
    """
    query: str = Field(
        description=(
            "El concepto principal de la comida que el usuario desea. "
            "Tradúcelo al inglés para mejorar la búsqueda semántica."
        )
    )
    max_calories: Optional[float] = Field(default=None, description="Ignorado en esta versión.")
    diet: Optional[str] = Field(default=None, description="Ignorado en esta versión.")
    min_protein: Optional[float] = Field(default=None, description="Ignorado en esta versión.")
    max_sugar: Optional[float] = Field(default=None, description="Ignorado en esta versión.")
    min_fiber: Optional[float] = Field(default=None, description="Ignorado en esta versión.")


@tool(args_schema=RecipeSearchInputSinFiltros)
def buscar_recetas_sin_filtros(
    query: str,
    max_calories: Optional[float] = None,
    diet: Optional[str] = None,
    min_protein: Optional[float] = None,
    max_sugar: Optional[float] = None,
    min_fiber: Optional[float] = None,
) -> str:
    """
    Busca recetas en la base de datos NutriCuisine usando SOLO búsqueda semántica.
    Los filtros de metadatos (calorías, dieta, etc.) se IGNORAN.
    """
    db = get_db()

    # Se llama a search_recipes pero SIN pasar ningún filtro
    resultados = db.search_recipes(
        query=query,
        limit=5,
    )

    if not resultados:
        return (
            f"[SIN RESULTADOS] No se encontraron recetas para '{query}'. "
            "Sugiere al usuario ampliar sus criterios de búsqueda."
        )

    texto = f"[RESULTADOS — SOLO SEMÁNTICA, SIN FILTROS] Se encontraron {len(resultados)} receta(s):\n\n"

    for i, r in enumerate(resultados, 1):
        meta = r["metadata"]
        score = 1 - r["distance"]

        texto += f"━━━ RECETA {i} (relevancia: {score:.0%}) ━━━\n"
        texto += f"Título: {meta.get('title', 'Sin título')}\n"
        texto += f"Calorías: {meta.get('calories', 0):.0f} kcal\n"
        texto += f"Proteínas: {meta.get('protein', 0):.1f}g | "
        texto += f"Carbohidratos: {meta.get('carbs', 0):.1f}g | "
        texto += f"Grasas: {meta.get('fat', 0):.1f}g\n"
        texto += f"Fibra: {meta.get('fiber', 0):.1f}g | "
        texto += f"Azúcar: {meta.get('sugar', 0):.1f}g | "
        texto += f"Sodio: {meta.get('sodium', 0):.0f}mg\n"
        texto += f"Dieta(s): {meta.get('diets', 'No especificada')}\n"
        texto += f"Raciones: {meta.get('servings', 'No especificado')}\n"
        texto += f"Contenido:\n{r['content']}\n\n"

    return texto


# Funciones auxiliares
def extract_text(content) -> str:
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
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


def invoke_with_retry(agent, messages: list) -> dict:
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return agent.invoke({"messages": messages})
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
    print("=" * 60)
    print(" 💪 NUTRICHAT ASSISTANT — SIN FILTROS (solo semántica) 🍏")
    print("=" * 60)
    print("⚠️  Esta versión usa RAG pero SIN filtros de metadatos.")
    print("    La búsqueda es solo por similaridad semántica.")

    if not GOOGLE_API_KEY:
        print("\nERROR: No se encontró GOOGLE_API_KEY en .env")
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )
    # Se usa la herramienta SIN FILTROS en vez de la normal
    agent = create_agent(llm, [buscar_recetas_sin_filtros], system_prompt=SYSTEM_PROMPT)
    chat_history: list = []

    print(f"\nRecetas en la BD: {get_db().count:,}")
    print("NutriChat (Sin filtros) está listo. Escribe 'salir' para terminar.\n")

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
            response = invoke_with_retry(agent, chat_history)
            elapsed = time.time() - start

            ai_message = response["messages"][-1]
            clean_text = extract_text(ai_message.content)
            print(f"\nNutriChat (Sin filtros): {clean_text}")
            print(f"  [Tiempo: {elapsed:.2f}s | Modo: solo semántica]\n")

            chat_history.append(AIMessage(content=clean_text))

        except KeyboardInterrupt:
            print("\n\nNutriChat: ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}")
            print("Inténtalo de nuevo con otra consulta.\n")


if __name__ == "__main__":
    main()
