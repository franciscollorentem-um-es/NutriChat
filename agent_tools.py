"""
agent_tools.py — Definición de herramientas que el agente LLM puede invocar.
Cada herramienta es una función decorada con @tool que especifica su esquema de entrada.
Define la función buscar_recetas, que el agente usará para consultar la base de datos vectorial.
"""

from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.tools import tool

from nutri_vectordb import NutriVectorDB

# Instancia global con lazy-loading
_db: Optional[NutriVectorDB] = None


def get_db() -> NutriVectorDB:
    """
    Singleton de la base de datos vectorial.
    """
    global _db
    if _db is None:
        _db = NutriVectorDB()
    return _db


# Esquema Pydantic: define qué parámetros puede extraer el LLM
class RecipeSearchInput(BaseModel):
    """
    Esquema que el LLM rellena para buscar recetas. Las descripciones
    son esenciales: el LLM las lee para decidir qué valores extraer.
    """

    query: str = Field(
        description=(
            "El concepto principal de la comida que el usuario desea. "
            "Ejemplos: 'pasta con queso', 'postre de chocolate', 'cena ligera', "
            "'ensalada mediterránea'. Tradúcelo al inglés si es posible para "
            "mejorar la búsqueda semántica."
        )
    )
    max_calories: Optional[float] = Field(
        default=None,
        description=(
            "Límite máximo de calorías. Extrae el número si el usuario dice "
            "'menos de 400 kcal' → 400.0, 'bajo en calorías' → 300.0. "
            "Si no menciona calorías, déjalo vacío (null)."
        ),
    )
    diet: Optional[str] = Field(
        default=None,
        description=(
            "Restricción dietética traducida a su etiqueta estándar EN INGLÉS. "
            "Mapeo: 'vegano/a' → 'Vegan', 'vegetariano/a' → 'Vegetarian', "
            "'cetogénica/keto' → 'Keto', 'paleo' → 'Paleo', "
            "'sin gluten' → 'Gluten-Free', 'sin lactosa' → 'Dairy-Free'. "
            "Si no menciona dieta, déjalo vacío (null)."
        ),
    )
    max_fat: Optional[float] = Field(
        default=None,
        description=(
            "Gramos máximos de grasa total. Extraer si el usuario dice "
            "'bajo en grasa' → 15.0, 'menos de 20g de grasa' → 20.0. "
            "Si no menciona grasa, déjalo vacío (null)."
        ),
    )
    max_saturated_fat: Optional[float] = Field(
        default=None,
        description=(
            "Gramos máximos de grasa saturada. Extraer si el usuario dice "
            "'bajo en grasas saturadas' → 5.0, 'poca grasa saturada' → 3.0. "
            "Si no menciona grasa saturada, déjalo vacío (null)."
        ),
    )
    max_carbs: Optional[float] = Field(
        default=None,
        description=(
            "Gramos máximos de carbohidratos. Extraer si el usuario dice "
            "'bajo en carbohidratos' → 30.0, 'menos de 50g de carbos' → 50.0. "
            "Si no menciona carbohidratos máximos, déjalo vacío (null)."
        ),
    )
    min_carbs: Optional[float] = Field(
        default=None,
        description=(
            "Gramos mínimos de carbohidratos. Extraer si el usuario dice "
            "'alto en carbohidratos' → 50.0, 'al menos 60g de carbos' → 60.0. "
            "Si no menciona carbohidratos mínimos, déjalo vacío (null)."
        ),
    )
    min_protein: Optional[float] = Field(
        default=None,
        description=(
            "Gramos mínimos de proteína por ración. Extraer si el usuario dice "
            "'alta en proteínas' → 20.0, 'al menos 30g de proteína' → 30.0. "
            "Si no menciona proteína, déjalo vacío (null)."
        ),
    )
    max_sugar: Optional[float] = Field(
        default=None,
        description=(
            "Gramos máximos de azúcar. Extraer si el usuario dice "
            "'bajo en azúcar' → 10.0, 'sin azúcar añadida' → 5.0. "
            "Si no menciona azúcar, déjalo vacío (null)."
        ),
    )
    min_fiber: Optional[float] = Field(
        default=None,
        description=(
            "Gramos mínimos de fibra. Extraer si el usuario dice "
            "'rico en fibra' → 5.0, 'alta fibra' → 8.0. "
            "Si no menciona fibra, déjalo vacío (null)."
        ),
    )
    max_sodium: Optional[float] = Field(
        default=None,
        description=(
            "Gramos máximos de sodio. Extraer si el usuario dice "
            "'bajo en sodio' → 1.0, 'bajo en sal' → 0.5. "
            "Si no menciona sodio ni sal, déjalo vacío (null)."
        ),
    )


# Herramienta LangChain
@tool(args_schema=RecipeSearchInput)
def buscar_recetas(
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
    Busca recetas en la base de datos NutriCuisine combinando búsqueda semántica
    con filtros nutricionales. Devuelve información detallada de las recetas
    encontradas para que el asistente genere su respuesta.
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
        filtros_usados = []
        if max_calories:
            filtros_usados.append(f"máx. {max_calories} kcal")
        if diet:
            filtros_usados.append(f"dieta: {diet}")
        if max_fat:
            filtros_usados.append(f"máx. {max_fat}g grasa")
        if max_saturated_fat:
            filtros_usados.append(f"máx. {max_saturated_fat}g grasa saturada")
        if max_carbs:
            filtros_usados.append(f"máx. {max_carbs}g carbohidratos")
        if min_carbs:
            filtros_usados.append(f"mín. {min_carbs}g carbohidratos")
        if min_protein:
            filtros_usados.append(f"mín. {min_protein}g proteína")
        if max_sugar:
            filtros_usados.append(f"máx. {max_sugar}g azúcar")
        if min_fiber:
            filtros_usados.append(f"mín. {min_fiber}g fibra")
        if max_sodium:
            filtros_usados.append(f"máx. {max_sodium}g sodio")

        filtros_str = ", ".join(filtros_usados) if filtros_usados else "ninguno"
        return (
            f"[SIN RESULTADOS] No se encontraron recetas para la consulta '{query}' "
            f"con filtros: {filtros_str}. "
            "Sugiere al usuario ampliar o modificar sus criterios de búsqueda."
        )

    texto = f"[RESULTADOS] Se encontraron {len(resultados)} receta(s) en la base de datos:\n\n"

    for i, r in enumerate(resultados, 1):
        meta = r["metadata"]
        # Se convierte distancia a similaridad (0–1)
        score = 1 - r["distance"]  

        texto += f"━━━ RECETA {i} (relevancia: {score:.0%}) ━━━\n"
        texto += f"Título: {meta.get('title', 'Sin título')}\n"
        texto += f"Calorías: {meta.get('calories', 0):.0f} kcal\n"
        texto += f"Proteínas: {meta.get('protein', 0):.1f}g | "
        texto += f"Carbohidratos: {meta.get('carbs', 0):.1f}g | "
        texto += f"Grasas: {meta.get('fat', 0):.1f}g (saturadas: {meta.get('saturatedFat', 0):.1f}g)\n"
        texto += f"Fibra: {meta.get('fiber', 0):.1f}g | "
        texto += f"Azúcar: {meta.get('sugar', 0):.1f}g | "
        texto += f"Sodio: {meta.get('sodium', 0):.0f}g\n"
        texto += f"Dieta(s): {meta.get('diets', 'No especificada')}\n"
        texto += f"Raciones: {meta.get('servings', 'No especificado')}\n"
        texto += f"Contenido:\n{r['content']}\n\n"

    return texto
