"""
nutri_vectordb.py — Capa de acceso a la base de datos vectorial de recetas.

Funciones de la clase NutriVectorDB:
  1. Conectar con ChromaDB de forma persistente.
  2. Ingestar recetas (text + metadatos + embeddings) por lotes.
  3. Realizar búsquedas híbridas: similitud semántica + filtros de metadatos
     (calorías, dietas, macronutrientes).

"""

import os
import math
import logging
import warnings
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    DEFAULT_SEARCH_LIMIT,
    INGESTION_BATCH_SIZE,
)

# Silenciar los mensajes informativos del modelo de embeddings 
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"




class NutriVectorDB:
    """
    Clase de acceso a la base de datos vectorial de recetas.
    Proporciona métodos para ingestar recetas y realizar búsquedas híbridas con filtros nutricionales.
    """
    
    
    # Constructor de la clase
    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION,
        persist_directory: str = CHROMA_PERSIST_DIR,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Se crea la conexión persistente a ChromaDB.
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Recetas con metadatos nutricionales",
                "hnsw:space": "cosine",
            },
        )
        # Se crea un atributo para el modelo de embeddings, que se cargará de forma lazy
        self._embedding_model: Optional[SentenceTransformer] = None

    # Propiedades
    @property
    def embedding_model(self) -> SentenceTransformer:
        """
        Carga el modelo de embeddings solo la primera vez que se usa.
        Si el modelo ya está cargado, lo devuelve. Si no, lo carga y luego lo devuelve.
        """
        if self._embedding_model is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    @property
    def count(self) -> int:
        """
        Devuelve el número de recetas almacenadas en la colección.
        """
        return self.collection.count()

    # Utilidades
    @staticmethod
    def _safe_float(value: Any) -> float:
        """
        Convierte a float de forma segura; devuelve 0.0 ante NaN, None o cadenas vacías.
        """
        if value is None:
            return 0.0
        try:
            f = float(value)
            return 0.0 if math.isnan(f) else f
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _safe_str(value: Any) -> str:
        """
        Convierte a string seguro para metadatos de ChromaDB.
        """
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    # Ingesta
    def process_and_ingest(self, recipes: List[Dict[str, Any]]) -> None:
        """
        Procesa una lista de recetas crudas y las inserta en ChromaDB por lotes.

        Cada receta debe ser un dict con claves como:
          title, ingredients, directions, calories, fatContent, carbohydrateContent,
          proteinContent, fiberContent, sugarContent, saturatedFatContent,
          sodiumContent, cholesterolContent, servings, diets, etc.
        """
        print(f"Procesando {len(recipes)} recetas para ingesta...")

        # Se preparan las listas de documentos, metadatos e IDs para la ingesta masiva
        documents: List[str] = [] 
        metadatas: List[Dict[str, Any]] = []  
        ids: List[str] = []   
        seen_ids: Dict[str, int] = {} 

        for recipe in tqdm(recipes, desc="Preparando datos"):
            ingredients = recipe.get("ingredients", [])
            directions = recipe.get("directions", [])

            ingredients_text = (
                ", ".join(ingredients) if isinstance(ingredients, list) else str(ingredients)
            )
            directions_text = (
                " ".join(directions) if isinstance(directions, list) else str(directions)
            )

            semantic_text = (
                f"Title: {recipe.get('title', 'Unknown')}\n"
                f"Ingredients: {ingredients_text}\n"
                f"Directions: {directions_text}"
            )

            metadata = {
                "title": self._safe_str(recipe.get("title", "Desconocido")),
                "calories": self._safe_float(recipe.get("calories")),
                "fat": self._safe_float(recipe.get("fatContent")),
                "saturatedFat": self._safe_float(recipe.get("saturatedFatContent")),
                "carbs": self._safe_float(recipe.get("carbohydrateContent")),
                "sugar": self._safe_float(recipe.get("sugarContent")),
                "fiber": self._safe_float(recipe.get("fiberContent")),
                "protein": self._safe_float(recipe.get("proteinContent")),
                "sodium": self._safe_float(recipe.get("sodiumContent")),
                "cholesterol": self._safe_float(recipe.get("cholesterolContent")),
                "servings": self._safe_str(recipe.get("servings", "")),
                "diets": self._safe_str(recipe.get("diets", [])),
            }

            base_id = (
                str(recipe.get("title", "receta"))
                .lower()
                .replace(" ", "_")
                .replace(",", "")[:50]
            )
            # Si el ID base ya se ha visto, se añade un sufijo numérico para hacerlo único
            if base_id in seen_ids:
                seen_ids[base_id] += 1
                safe_id = f"{base_id}_{seen_ids[base_id]}"
            else:
                seen_ids[base_id] = 1
                safe_id = base_id
            # Se añaden el texto semántico, los metadatos y el ID a las listas para la ingesta masiva
            documents.append(semantic_text)
            metadatas.append(metadata)
            ids.append(safe_id)

        # Se inserta en ChromaDB por lotes
        total_batches = math.ceil(len(ids) / INGESTION_BATCH_SIZE)
        print(f"\nInsertando {len(ids)} registros en {total_batches} lotes...")

        # Se procesa cada lote.
        for i in range(0, len(ids), INGESTION_BATCH_SIZE):
            end = min(i + INGESTION_BATCH_SIZE, len(ids))
            batch_num = i // INGESTION_BATCH_SIZE + 1
            print(f"  Lote {batch_num}/{total_batches} (registros {i}–{end - 1})")

            # Se generan los embeddings para el texto semántico del lote actual
            batch_embeddings = self.embedding_model.encode(
                documents[i:end], show_progress_bar=False
            ).tolist()

            # Se realiza el upsert en ChromaDB con los datos del lote actual
            self.collection.upsert(
                ids=ids[i:end],
                embeddings=batch_embeddings,
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        print(f"Ingesta completada: {len(ids)} recetas en la colección '{self.collection_name}'.")

    # Búsqueda
    def _build_where_clause(self,
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
    ) -> Optional[Dict[str, Any]]:
        """
        Construye dinámicamente una cláusula `where` para ChromaDB a partir de
        los filtros proporcionados. Devuelve None si no hay filtros activos.

        Ejemplo de salida para max_calories=400 y diet='Vegetarian':
          {"$and": [
              {"calories": {"$lte": 400}},
              {"diets": {"$eq": "Vegetarian"}}
          ]}
        """
        conditions: List[Dict[str, Any]] = []

        # Se construye la cláusula de filtros según los parámetros proporcionados. Solo se añaden los filtros que no son None
        if max_calories is not None:
            conditions.append({"calories": {"$lte": max_calories}})
        if diet is not None:
            # Se utiliza $contains para dietas compuestas: "Vegetarian, Gluten-Free"
            conditions.append({"diets": {"$contains": diet}})
        if max_fat is not None:
            conditions.append({"fat": {"$lte": max_fat}})
        if max_saturated_fat is not None:
            conditions.append({"saturatedFat": {"$lte": max_saturated_fat}})
        if max_carbs is not None:
            conditions.append({"carbs": {"$lte": max_carbs}})
        if min_carbs is not None:
            conditions.append({"carbs": {"$gte": min_carbs}})
        if min_protein is not None:
            conditions.append({"protein": {"$gte": min_protein}})
        if max_sugar is not None:
            conditions.append({"sugar": {"$lte": max_sugar}})
        if min_fiber is not None:
            conditions.append({"fiber": {"$gte": min_fiber}})
        if max_sodium is not None:
            conditions.append({"sodium": {"$lte": max_sodium}})

        # Si no hay condiciones, se devuelve None
        # Si solo hay una condición, se devuelve directamente.
        # Si hay varias, se combinan con $and
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def search_recipes(
        self,
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
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda híbrida: similitud semántica + filtros de metadatos.

        Parámetros:
          query             — texto libre del usuario (se embede para búsqueda semántica).
          max_calories      — calorías máximas (inclusive).
          diet              — etiqueta de dieta: 'Vegetarian', 'Keto', 'Paleo', etc.
          max_fat           — gramos de grasa total máximos.
          max_saturated_fat — gramos de grasa saturada máximos.
          max_carbs         — gramos de carbohidratos máximos.
          min_carbs         — gramos de carbohidratos mínimos.
          min_protein       — gramos mínimos de proteína.
          max_sugar         — gramos de azúcar máximos.
          min_fiber         — gramos mínimos de fibra.
          max_sodium        — gramos de sodio máximos.
          limit             — número de resultados a devolver (default: 5).

        Devuelve:
          Lista de dicts con keys: id, distance, metadata, content.
        """
        # Se construye la cláusula de filtros a partir de los parámetros proporcionados
        where_clause = self._build_where_clause(
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
        )


        # Se genera el embedding de la consulta
        query_embedding = self.embedding_model.encode(
            [query], show_progress_bar=False
        ).tolist()

        # Primero se intenta con todos los filtros primero
        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=limit,
                where=where_clause,
            )
        except Exception as e:
            print(f"[Advertencia] Filtro ChromaDB falló ({e}). Reintentando sin filtro de dieta...")
            results = {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

        # Fallback automático
        needs_manual_diet_filter = False
        if diet and (not results["ids"] or not results["ids"][0]):
            # Fallback silencioso: $contains no devolvió resultados, se filtra en Python
            fallback_clause = self._build_where_clause(
                max_calories=max_calories,
                max_fat=max_fat,
                max_saturated_fat=max_saturated_fat,
                max_carbs=max_carbs,
                min_carbs=min_carbs,
                min_protein=min_protein,
                max_sugar=max_sugar,
                min_fiber=min_fiber,
                max_sodium=max_sodium,
            )
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=limit * 3,  
                where=fallback_clause,
            )
            needs_manual_diet_filter = True

        if not results["ids"] or not results["ids"][0]:
            return []

        formatted: List[Dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]

            # Filtro manual de dieta cuando el fallback lo requiere
            if needs_manual_diet_filter and diet:
                if diet.lower() not in meta.get("diets", "").lower():
                    continue

            formatted.append(
                {
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": meta,
                    "content": results["documents"][0][i],
                }
            )

        return formatted[:limit]
