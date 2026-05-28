"""
benchmark.py — Evaluacion formal del sistema de busqueda de NutriChat.

Ejecuta un conjunto de consultas de referencia contra ChromaDB y calcula
metricas de relevancia estandar:
  - nDCG@k  (normalized Discounted Cumulative Gain)
  - MRR     (Mean Reciprocal Rank)

Cada consulta tiene una lista de recetas relevantes esperadas (ground truth)
definidas manualmente, lo que permite medir objetivamente la calidad del
ranking que produce el sistema.

Ejecutar:
  python benchmark.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import time
import warnings
from typing import List, Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from nutri_vectordb import NutriVectorDB

# Directorio donde se guardan las graficas
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# Consultas de referencia (ground truth)
BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    {
        "description": "Busqueda basica: ensalada de pollo",
        "query": "chicken salad",
        "filters": {},
        "relevant_titles": ["chicken salad", "chicken caesar", "grilled chicken"],
    },
    {
        "description": "Busqueda con filtro calorico: postre bajo en calorias",
        "query": "chocolate dessert",
        "filters": {"max_calories": 300},
        "relevant_titles": ["chocolate", "brownie", "mousse", "cake", "cocoa"],
    },
    {
        "description": "Busqueda con filtro de dieta: receta vegetariana",
        "query": "vegetarian pasta",
        "filters": {"diet": "Vegetarian"},
        "relevant_titles": ["pasta", "spaghetti", "penne", "noodle", "macaroni"],
    },
    {
        "description": "Busqueda con filtro de proteina: alto en proteina",
        "query": "high protein meal",
        "filters": {"min_protein": 30},
        "relevant_titles": ["chicken", "beef", "steak", "protein", "turkey", "salmon", "tuna"],
    },
    {
        "description": "Busqueda especifica: sopa de tomate",
        "query": "tomato soup",
        "filters": {},
        "relevant_titles": ["tomato soup", "tomato bisque", "tomato"],
    },
    {
        "description": "Busqueda con filtro de azucar: desayuno bajo en azucar",
        "query": "healthy breakfast",
        "filters": {"max_sugar": 10},
        "relevant_titles": ["breakfast", "oatmeal", "egg", "omelet", "granola", "pancake"],
    },
    {
        "description": "Busqueda con filtro de grasa: cena ligera",
        "query": "light dinner",
        "filters": {"max_fat": 15},
        "relevant_titles": ["salad", "soup", "light", "grilled", "steamed", "fish"],
    },
    {
        "description": "Busqueda con filtro calorico: cena baja en calorias",
        "query": "light chicken dinner",
        "filters": {"max_calories": 500},
        "relevant_titles": ["chicken", "light", "grilled", "roast", "baked"],
    },
    {
        "description": "Busqueda de receta especifica: smoothie de frutas",
        "query": "fruit smoothie",
        "filters": {},
        "relevant_titles": ["smoothie", "shake", "fruit", "berry", "banana"],
    },
    {
        "description": "Busqueda con filtro de fibra: receta rica en fibra",
        "query": "high fiber recipe",
        "filters": {"min_fiber": 8},
        "relevant_titles": ["bean", "lentil", "quinoa", "oat", "fiber", "whole grain",
                            "chickpea", "parsnip", "sweet potato", "broccoli", "pea",
                            "vegetable", "roast", "galette", "celeriac"],
    },
    {
        "description": "Busqueda con dieta vegana",
        "query": "vegan lunch",
        "filters": {"diet": "Vegan"},
        "relevant_titles": ["vegan", "tofu", "bean", "lentil", "vegetable", "chickpea", "quinoa"],
    },
    {
        "description": "Busqueda especifica: pizza",
        "query": "pizza",
        "filters": {},
        "relevant_titles": ["pizza"],
    },
    {
        "description": "Busqueda con filtro de sodio: bajo en sodio",
        "query": "low sodium meal",
        "filters": {"max_sodium": 200},
        "relevant_titles": ["salad", "fruit", "vegetable", "rice", "grilled"],
    },
    {
        "description": "Busqueda combinada: pasta baja en calorias",
        "query": "pasta",
        "filters": {"max_calories": 500},
        "relevant_titles": ["pasta", "spaghetti", "penne", "noodle", "linguine", "macaroni"],
    },
    {
        "description": "Busqueda sin gluten",
        "query": "gluten free snack",
        "filters": {"diet": "Gluten-Free"},
        "relevant_titles": ["gluten free", "rice", "nut", "fruit", "chip", "popcorn"],
    },
]


# Metricas de evaluacion
def normalize(text: str) -> str:
    """
    Normaliza texto para comparacion: minusculas, guiones a espacios.
    """
    return text.lower().replace("-", " ").replace("–", " ")


def is_relevant(title: str, relevant_titles: List[str]) -> bool:
    """
    Comprueba si un titulo de receta coincide con alguno de los relevantes esperados.
    """
    title_norm = normalize(title)
    return any(normalize(rt) in title_norm for rt in relevant_titles)


def reciprocal_rank(results: List[Dict], relevant_titles: List[str]) -> float:
    """
    Reciprocal Rank: 1/posicion del primer resultado relevante.
    Devuelve 0 si ningun resultado es relevante.
    """
    for i, r in enumerate(results):
        title = r["metadata"].get("title", "")
        if is_relevant(title, relevant_titles):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(results: List[Dict], relevant_titles: List[str], k: int = 5) -> float:
    """
    normalized Discounted Cumulative Gain a profundidad k.

    Cada resultado relevante tiene relevancia=1, no relevante=0.
    DCG = sum( rel_i / log2(i+2) )  para i=0..k-1
    IDCG = DCG ideal (todos los relevantes al principio)
    nDCG = DCG / IDCG
    """
    results_k = results[:k]

    # Relevancia binaria de cada resultado
    gains = []
    for r in results_k:
        title = r["metadata"].get("title", "")
        gains.append(1.0 if is_relevant(title, relevant_titles) else 0.0)

    # DCG real
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    # IDCG (caso ideal: todos los relevantes primero)
    num_relevant = sum(gains)
    ideal_gains = [1.0] * int(num_relevant) + [0.0] * (len(gains) - int(num_relevant))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# Ejecucion del benchmark
def run_benchmark(k: int = 5, verbose: bool = True) -> Dict[str, float]:
    """
    Ejecuta todas las consultas de referencia y calcula nDCG@k y MRR.

    Parametros:
      k       — profundidad para nDCG (default: 5)
      verbose — si True, imprime detalle de cada consulta

    Devuelve:
      dict con 'mean_ndcg', 'mean_mrr' y detalle por consulta
    """
    print("=" * 65)
    print(" BENCHMARK DE EVALUACION — NutriChat")
    print("=" * 65)
    print(f"  Consultas de referencia: {len(BENCHMARK_QUERIES)}")
    print(f"  Profundidad (k): {k}")
    print()

    # Conectar a ChromaDB
    print("  Conectando a ChromaDB...")
    db = NutriVectorDB()
    print(f"  Recetas en BD: {db.count:,}")
    print()

    all_ndcg = []
    all_rr = []
    query_details = []

    for idx, bq in enumerate(BENCHMARK_QUERIES, 1):
        query = bq["query"]
        filters = bq["filters"]
        relevant = bq["relevant_titles"]
        desc = bq["description"]

        # Ejecutar busqueda
        start = time.time()
        results = db.search_recipes(
            query=query,
            max_calories=filters.get("max_calories"),
            diet=filters.get("diet"),
            max_fat=filters.get("max_fat"),
            max_saturated_fat=filters.get("max_saturated_fat"),
            max_carbs=filters.get("max_carbs"),
            min_carbs=filters.get("min_carbs"),
            min_protein=filters.get("min_protein"),
            max_sugar=filters.get("max_sugar"),
            min_fiber=filters.get("min_fiber"),
            max_sodium=filters.get("max_sodium"),
            limit=k,
        )
        elapsed = time.time() - start

        # Calcular metricas
        rr = reciprocal_rank(results, relevant)
        ndcg = ndcg_at_k(results, relevant, k)

        all_rr.append(rr)
        all_ndcg.append(ndcg)

        # Detalle de resultados
        detail = {
            "query": query,
            "description": desc,
            "filters": filters,
            "ndcg": ndcg,
            "rr": rr,
            "time_ms": elapsed * 1000,
            "num_results": len(results),
            "results": [],
        }

        for r in results:
            title = r["metadata"].get("title", "Unknown")
            rel = is_relevant(title, relevant)
            dist = r["distance"]
            detail["results"].append({
                "title": title,
                "relevant": rel,
                "distance": dist,
            })

        query_details.append(detail)

        # Imprimir detalle si verbose
        if verbose:
            print(f"  [{idx:02d}/{len(BENCHMARK_QUERIES):02d}] {desc}")
            print(f"       Query: \"{query}\"  |  Filtros: {filters if filters else 'ninguno'}")
            print(f"       nDCG@{k}: {ndcg:.3f}  |  RR: {rr:.3f}  |  Tiempo: {elapsed*1000:.0f}ms")

            for j, r in enumerate(results):
                title = r["metadata"].get("title", "Unknown")
                rel = is_relevant(title, relevant)
                dist = r["distance"]
                marca = "V" if rel else "X"
                print(f"         {j+1}. [{marca}] {title[:55]:<55} (dist: {dist:.2f})")

            print()

    # Resumen global
    mean_ndcg = sum(all_ndcg) / len(all_ndcg) if all_ndcg else 0
    mean_mrr = sum(all_rr) / len(all_rr) if all_rr else 0

    print("=" * 65)
    print(" RESULTADOS GLOBALES")
    print("=" * 65)
    print(f"  nDCG@{k} medio: {mean_ndcg:.4f}")
    print(f"  MRR medio:      {mean_mrr:.4f}")
    print()
    print(f"  Interpretacion:")
    print(f"    nDCG@{k} = 1.0 significa que todos los resultados relevantes")
    print(f"    aparecen en las primeras {k} posiciones en orden perfecto.")
    print(f"    MRR = 1.0 significa que el primer resultado siempre es relevante.")
    print()

    # Tabla resumen por consulta
    print("  Detalle por consulta:")
    print(f"  {'#':<4} {'Query':<28} {'nDCG@'+str(k):<10} {'RR':<8} {'Tiempo':<10}")
    print("  " + "-" * 62)
    for i, d in enumerate(query_details, 1):
        q_short = d["query"][:26]
        print(f"  {i:<4} {q_short:<28} {d['ndcg']:<10.3f} {d['rr']:<8.3f} {d['time_ms']:<8.0f}ms")

    print("  " + "-" * 62)
    print(f"  {'MEDIA':<32} {mean_ndcg:<10.4f} {mean_mrr:<8.4f}")
    print()

    return {
        "mean_ndcg": mean_ndcg,
        "mean_mrr": mean_mrr,
        "k": k,
        "num_queries": len(BENCHMARK_QUERIES),
        "details": query_details,
    }


def generate_charts(results: Dict[str, Any]) -> None:
    """
    Genera graficas PNG listas para copiar en la memoria del TFG.
    Se guardan en el mismo directorio que benchmark.py.
    """
    details = results["details"]
    k = results["k"]

    queries = [d["query"] for d in details]
    ndcgs = [d["ndcg"] for d in details]
    rrs = [d["rr"] for d in details]

    # Etiquetas cortas para el eje X
    labels = [f"Q{i+1}" for i in range(len(queries))]

    # Grafica de barras agrupadas nDCG y MRR por consulta
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, ndcgs, width, label=f"nDCG@{k}", color="#2E75B6", edgecolor="white")
    bars2 = ax.bar(x + width/2, rrs, width, label="MRR", color="#E07B39", edgecolor="white")

    ax.set_xlabel("Consulta", fontsize=11)
    ax.set_ylabel("Puntuacion", fontsize=11)
    ax.set_title(f"nDCG@{k} y MRR por consulta de referencia", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(y=results["mean_ndcg"], color="#2E75B6", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=results["mean_mrr"], color="#E07B39", linestyle="--", alpha=0.5, linewidth=1)

    # Etiquetas de valor sobre cada barra
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7, color="#2E75B6")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7, color="#E07B39")

    # Leyenda de consultas debajo de la grafica
    legend_text = "  |  ".join([f"Q{i+1}: {q}" for i, q in enumerate(queries)])
    fig.text(0.5, -0.02, legend_text, ha="center", fontsize=7, style="italic", color="gray",
             wrap=True)

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "benchmark_barras.png")
    fig.savefig(path1, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Grafica guardada: {path1}")


if __name__ == "__main__":
    results = run_benchmark(k=5, verbose=True)
    print("\n  Generando graficas para la memoria...")
    generate_charts(results)
