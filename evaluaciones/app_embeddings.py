"""
app_embeddings.py — Comparacion de modelos de embeddings del asistente NutriChat (CLI).

Permite elegir entre dos modelos de embeddings alternativos al multilingue
por defecto (paraphrase-multilingual-MiniLM-L12-v2) que usa app_nutrichat.py:

  1. all-MiniLM-L6-v2      (384 dimensiones, solo ingles)
  2. all-mpnet-base-v2      (768 dimensiones, solo ingles)

Al ejecutar el script se muestra un menu para seleccionar el modelo.
Asi se pueden comparar tiempos de respuesta y calidad de resultados
usando las mismas preguntas en todas las versiones.

Comparacion completa:
  - MULTILINGUE (384d): python app_nutrichat.py        (paraphrase-multilingual-MiniLM-L12-v2)
  - SOLO INGLES (384d): python app_embeddings.py → 1   (all-MiniLM-L6-v2)
  - MPNET (768d):       python app_embeddings.py → 2   (all-mpnet-base-v2)

Ejecutar:
  python app_embeddings.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import warnings
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from sentence_transformers import SentenceTransformer

from config import (
    GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT,
    LLM_MAX_RETRIES, LLM_RETRY_WAIT_SECONDS,
)
from nutri_vectordb import NutriVectorDB

MAX_HISTORY_MESSAGES = 30

# Modelos de embeddings disponibles
MODELOS_EMBEDDINGS = {
    "1": {
        "nombre": "all-MiniLM-L6-v2",
        "modelo": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensiones": 384,
        "coleccion": "nutricuisine", 
    },
    "2": {
        "nombre": "all-mpnet-base-v2",
        "modelo": "sentence-transformers/all-mpnet-base-v2",
        "dimensiones": 768,
        "coleccion": "nutricuisine_mpnet", 
    },
}

# Singleton de la BD
_db: Optional[NutriVectorDB] = None
_modelo_elegido: dict = {}


def get_db() -> NutriVectorDB:
    """
    Singleton de la BD con el modelo de embeddings seleccionado.
    """
    global _db
    if _db is None:
        _db = NutriVectorDB(collection_name=_modelo_elegido["coleccion"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _db._embedding_model = SentenceTransformer(_modelo_elegido["modelo"])
    return _db


# Esquema y herramienta
class RecipeSearchInput(BaseModel):
    """Esquema que el LLM rellena para buscar recetas."""
    query: str = Field(
        description=(
            "El concepto principal de la comida que el usuario desea. "
            "Traducelo al ingles para mejorar la busqueda semantica."
        )
    )
    max_calories: Optional[float] = Field(default=None, description="Calorias maximas.")
    diet: Optional[str] = Field(default=None, description="Dieta en ingles: 'Vegetarian', 'Vegan', 'Keto', etc.")
    max_fat: Optional[float] = Field(default=None, description="Gramos maximos de grasa total.")
    max_saturated_fat: Optional[float] = Field(default=None, description="Gramos maximos de grasa saturada.")
    max_carbs: Optional[float] = Field(default=None, description="Gramos maximos de carbohidratos.")
    min_carbs: Optional[float] = Field(default=None, description="Gramos minimos de carbohidratos.")
    min_protein: Optional[float] = Field(default=None, description="Gramos minimos de proteina.")
    max_sugar: Optional[float] = Field(default=None, description="Gramos maximos de azucar.")
    min_fiber: Optional[float] = Field(default=None, description="Gramos minimos de fibra.")
    max_sodium: Optional[float] = Field(default=None, description="Gramos maximos de sodio.")


@tool(args_schema=RecipeSearchInput)
def buscar_recetas_embeddings(
    query: str,
    max_calories: Optional[float] = None,
    diet: Optional[str] = None,
    max_fat: Optional[float] = None,
    max_saturated_fat: Optional[float] = None,
    max_carbs: Optional[float] = None,
    min_carbs: Optional[float] = None,
    min_protein: Optional[float] = None,
    max_sugar: Optional[float] = None,
    min_fiber: Optional[float] = None,
    max_sodium: Optional[float] = None,
) -> str:
    """
    Busca recetas en la base de datos NutriChat usando el modelo de embeddings
    alternativo seleccionado. Combina busqueda semantica con filtros.
    """
    db = get_db()

    resultados = db.search_recipes(
        query=query,
        max_calories=max_calories,
        diet=diet,
        max_fat=max_fat,
        max_saturated_fat=max_saturated_fat,
        max_carbs=max_carbs,
        min_carbs=min_carbs,
        min_protein=min_protein,
        max_sugar=max_sugar,
        min_fiber=min_fiber,
        max_sodium=max_sodium,
        limit=5,
    )

    if not resultados:
        return (
            f"[SIN RESULTADOS] No se encontraron recetas para '{query}'. "
            "Sugiere al usuario ampliar sus criterios de busqueda."
        )

    nombre_modelo = _modelo_elegido["nombre"]
    dims = _modelo_elegido["dimensiones"]
    texto = f"[RESULTADOS — Embeddings: {nombre_modelo} ({dims}d)] Se encontraron {len(resultados)} receta(s):\n\n"

    for i, r in enumerate(resultados, 1):
        meta = r["metadata"]
        score = 1 - r["distance"]

        texto += f"━━━ RECETA {i} (relevancia: {score:.0%}) ━━━\n"
        texto += f"Titulo: {meta.get('title', 'Sin titulo')}\n"
        texto += f"Calorias: {meta.get('calories', 0):.0f} kcal\n"
        texto += f"Proteinas: {meta.get('protein', 0):.1f}g | "
        texto += f"Carbohidratos: {meta.get('carbs', 0):.1f}g | "
        texto += f"Grasas: {meta.get('fat', 0):.1f}g (saturadas: {meta.get('saturatedFat', 0):.1f}g)\n"
        texto += f"Fibra: {meta.get('fiber', 0):.1f}g | "
        texto += f"Azucar: {meta.get('sugar', 0):.1f}g | "
        texto += f"Sodio: {meta.get('sodium', 0):.0f}mg\n"
        texto += f"Dieta(s): {meta.get('diets', 'No especificada')}\n"
        texto += f"Raciones: {meta.get('servings', 'No especificado')}\n"
        texto += f"Contenido:\n{r['content']}\n\n"

    return texto


def _ingest_with_model(db: NutriVectorDB):
    """
    Ingesta automatica del dataset en la coleccion del modelo seleccionado.
    """
    import os
    import pandas as pd
    import ast

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Subimos un nivel: evaluaciones/ → raiz del proyecto
    csv_path = os.path.join(script_dir, "..", "dataset", "datos_limpios.csv")

    if not os.path.exists(csv_path):
        print(f"  ERROR: No se encontro {csv_path}")
        print("  Ejecuta primero: python preprocesamiento.py && python ingest_data.py")
        return

    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(csv_path, "r", encoding=encoding) as f:
                first_line = f.readline()
            sep = ";" if first_line.count(";") > first_line.count(",") else ","
            df = pd.read_csv(csv_path, sep=sep, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if "serve" in df.columns:
        df.rename(columns={"serve": "servings"}, inplace=True)

    for col in ["ingredients", "directions"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: ast.literal_eval(str(v)) if pd.notna(v) and str(v).startswith("[") else [str(v)] if pd.notna(v) else []
            )

    recipes = df.to_dict(orient="records")
    nombre = _modelo_elegido["nombre"]
    print(f"  Ingestando {len(recipes)} recetas con {nombre}...")
    start = time.time()
    db.process_and_ingest(recipes)
    elapsed = time.time() - start
    print(f"  Ingesta completada en {elapsed:.1f}s")


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


def seleccionar_modelo() -> dict:
    """
    Muestra un menu para que el usuario elija el modelo de embeddings.
    """
    print("\nSelecciona el modelo de embeddings alternativo:\n")
    print("  1. all-MiniLM-L6-v2      (384 dimensiones, solo ingles)")
    print("  2. all-mpnet-base-v2      (768 dimensiones, solo ingles)")
    print()

    while True:
        opcion = input("Opcion [1/2]: ").strip()
        if opcion in MODELOS_EMBEDDINGS:
            return MODELOS_EMBEDDINGS[opcion]
        print("  Opcion no valida. Introduce 1 o 2.")


def main():
    global _modelo_elegido

    print("=" * 65)
    print("  NUTRICHAT — COMPARACION DE MODELOS DE EMBEDDINGS")
    print("=" * 65)

    if not GOOGLE_API_KEY:
        print("\nERROR: No se encontro GOOGLE_API_KEY en .env")
        sys.exit(1)

    # Seleccion del modelo
    _modelo_elegido = seleccionar_modelo()
    nombre = _modelo_elegido["nombre"]
    dims = _modelo_elegido["dimensiones"]

    print(f"\n  Modelo seleccionado: {nombre} ({dims}d)")
    print(f"  Cargando modelo de embeddings...")
    start = time.time()
    db = get_db()
    elapsed = time.time() - start
    print(f"  Modelo cargado en {elapsed:.1f}s | Recetas en BD: {db.count:,}")

    # Si la coleccion esta vacia (solo posible con MPNet que usa coleccion separada)
    if db.count == 0:
        print("\n  La coleccion esta vacia. Ingestando recetas automaticamente...")
        _ingest_with_model(db)
        print(f"  Recetas en BD despues de ingesta: {db.count:,}")

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )
    agent = create_react_agent(llm, [buscar_recetas_embeddings], prompt=SYSTEM_PROMPT)
    chat_history: list = []

    print(f"\nNutriChat ({nombre}) esta listo. Escribe 'salir' para terminar.\n")

    while True:
        try:
            user_input = input("Tu: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("salir", "exit", "quit"):
                print("\nNutriChat: Hasta luego. Que tengas un dia saludable.")
                break

            chat_history.append(HumanMessage(content=user_input))
            chat_history = truncate_history(chat_history)

            start = time.time()
            response = invoke_with_retry(agent, chat_history)
            elapsed = time.time() - start

            ai_message = response["messages"][-1]
            clean_text = extract_text(ai_message.content)
            print(f"\nNutriChat ({nombre}): {clean_text}")
            print(f"  [Tiempo: {elapsed:.2f}s | Modelo embeddings: {nombre} ({dims}d)]\n")

            chat_history.append(AIMessage(content=clean_text))

        except KeyboardInterrupt:
            print("\n\nNutriChat: Hasta luego!")
            break
        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}")
            print("Intentalo de nuevo con otra consulta.\n")


if __name__ == "__main__":
    main()
