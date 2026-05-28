"""
preprocesamiento.py — Limpieza y preparación del dataset NutriCuisine.

Este script transforma el dataset original (diet_type_recipes.csv) en un CSV
limpio (datos_limpios.csv) listo para ser ingerido en ChromaDB.
Solo se ejecuta una vez.

Fuente del dataset:
  NutriCuisine database — https://github.com/NutriCuisine/database
"""

import argparse
import ast
import os
import re
import sys
import numpy as np
import pandas as pd

# Directorio del dataset (carpeta dataset dentro del proyecto)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_DIR = os.path.join(_SCRIPT_DIR,  "dataset")
DEFAULT_INPUT = os.path.join(_DATASET_DIR, "diet_type_recipes.csv")
DEFAULT_OUTPUT = os.path.join(_DATASET_DIR, "datos_limpios.csv")


# Parsing de la columna 'nutrition'
def parse_nutrition(texto_nutricion):
    """
    Convierte la columna 'nutrition' del dataset original a un diccionario plano.

    El campo puede venir como:
      - Un diccionario: {'calories': '405 calories', ...}
      - Una lista de diccionarios: [{'calories': '405 calories'}, {'fatContent': '21 grams fat'}]
      - Un string representando cualquiera de los anteriores.

    Devuelve un dict con las claves unificadas, o NaN si no se puede parsear.
    Ejemplo: "['calories': '405 calories', 'fatContent': '21 grams fat']" → {'calories': '405 calories', 'fatContent': '21 grams fat'}
    """
    try:
        data = ast.literal_eval(str(texto_nutricion))

        if isinstance(data, list):
            merged = {}
            for item in data:
                if isinstance(item, dict):
                    merged.update(item)
            return merged
        elif isinstance(data, dict):
            return data
        else:
            return np.nan
    except (ValueError, SyntaxError, TypeError):
        return np.nan


def extract_number(value):
    """
    Extrae el valor numérico de strings como '405 calories', '21 grams fat'.
    Devuelve float o NaN.
    Ejemplo: '405 calories' → 405.0
    """
    if pd.isna(value) or value is None:
        return np.nan
    match = re.search(r"(\d+\.?\d*)", str(value))
    if match is None:
        return np.nan
    return float(match.group(1))


# Limpieza de la columna 'diets'
def clean_diets(value):
    """
    Convierte la columna 'diets' de string de lista Python a string separado por comas.
    Ejemplo: "['Vegetarian', 'Gluten-Free']" → "Vegetarian, Gluten-Free"
    """
    if pd.isna(value) or value is None:
        return ""
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return ", ".join(str(d).strip() for d in parsed)
        return str(parsed)
    except (ValueError, SyntaxError):
        return str(value)


# Pipeline principal
def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    """Ejecuta todo el pipeline de preprocesamiento."""

    print(f"Cargando dataset original: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el fichero '{input_path}'.")
        sys.exit(1)
    print(f"  Filas originales: {len(df)}")
    print(f"  Columnas: {list(df.columns)}")

    # Eliminar columna 'link' 
    if "link" in df.columns:
        df = df.drop(columns=["link"])
        print("  Columna 'link' eliminada.")

    # Parsear 'nutrition' → columnas individuales
    if "nutrition" in df.columns:
        print("  Parseando columna 'nutrition'...")
        nutrition_dicts = df["nutrition"].apply(parse_nutrition)
        nutrition_df = pd.json_normalize(nutrition_dicts)
        df = pd.concat([df.drop(columns=["nutrition"]), nutrition_df], axis=1)

    # Seleccionar y renombrar columnas 
    columnas_deseadas = [
        "title", "directions", "ingredients", "diets", "serve",
        "calories", "fatContent", "saturatedFatContent",
        "carbohydrateContent", "sugarContent", "fiberContent",
        "proteinContent", "sodiumContent",
    ]
    columnas_presentes = [c for c in columnas_deseadas if c in df.columns]
    df = df[columnas_presentes]

    columnas_numericas = [
        "calories", "fatContent", "saturatedFatContent", "carbohydrateContent",
        "sugarContent", "fiberContent", "proteinContent", "sodiumContent",
    ]
    print("  Extrayendo valores numéricos de columnas nutricionales...")
    for col in columnas_numericas:
        if col in df.columns:
            df[col] = df[col].apply(extract_number)

    # Limpiar columna 'diets' 
    if "diets" in df.columns:
        print("  Limpiando columna 'diets'...")
        df["diets"] = df["diets"].apply(clean_diets)

    # Rellenar NaN numéricos con 0 
    for col in columnas_numericas:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Eliminar recetas con campos esenciales vacíos
    n_antes_vacios = len(df)
    if "title" in df.columns:
        df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    if "ingredients" in df.columns:
        df = df[df["ingredients"].notna() & (df["ingredients"].str.strip() != "")]
    if "directions" in df.columns:
        df = df[df["directions"].notna() & (df["directions"].str.strip() != "")]
    n_vacios = n_antes_vacios - len(df)
    if n_vacios > 0:
        print(f"  Recetas eliminadas por campos vacíos (título/ingredientes/instrucciones): {n_vacios}")

    # Validaciones lógicas
    n_antes = len(df)

    # Calorías y porciones deben ser positivas
    df = df[df["calories"] > 0]
    if "serve" in df.columns:
        df = df[df["serve"] >= 1]

    # Grasa saturada no puede ser mayor que grasa total
    if "saturatedFatContent" in df.columns and "fatContent" in df.columns:
        df = df[df["saturatedFatContent"] <= df["fatContent"]]

    # Azúcar no puede ser mayor que carbohidratos totales
    if "sugarContent" in df.columns and "carbohydrateContent" in df.columns:
        df = df[df["sugarContent"] <= df["carbohydrateContent"]]

    # Eliminar valores fuera de rangos razonables
    filtros_rango = {
        "serve": ("<=", 30),
        "calories": ("<", 2000),
        "fatContent": ("<=", 100),
        "carbohydrateContent": ("<=", 250),
        "proteinContent": ("<=", 100),
        "sodiumContent": ("<=", 2),  
    }

    for col, (op, val) in filtros_rango.items():
        if col in df.columns:
            if op == "<=":
                df = df[df[col] <= val]
            elif op == "<":
                df = df[df[col] < val]

    n_despues = len(df)
    print(f"  Filas eliminadas por validación: {n_antes - n_despues}")
    print(f"  Filas finales: {n_despues}")

    # Exportar a CSV limpio 
    df.to_csv(output_path, index=False, sep=";", encoding="utf-8")
    print(f"\nDataset limpio guardado en: {output_path}")

    # Resumen del dataset limpio
    print("\n" + "=" * 50)
    print("RESUMEN DEL DATASET LIMPIO")
    print("=" * 50)
    print(f"  Recetas: {len(df)}")
    print(f"  Columnas: {list(df.columns)}")
    print(f"\n  Estadísticas nutricionales:")
    for col in columnas_numericas:
        if col in df.columns:
            print(f"    {col}: media={df[col].mean():.1f}, min={df[col].min():.1f}, max={df[col].max():.1f}")
    print(f"\n  Dietas más frecuentes:")
    if "diets" in df.columns:
        all_diets = df["diets"].str.split(", ").explode()
        top_diets = all_diets.value_counts().head(5)
        for diet, count in top_diets.items():
            print(f"    {diet}: {count}")
    print(f"\n  Nulos restantes:")
    nulls = df.isnull().sum()
    nulls_nonzero = nulls[nulls > 0]
    if len(nulls_nonzero) == 0:
        print("    Ninguno")
    else:
        for col, n in nulls_nonzero.items():
            print(f"    {col}: {n}")

    return df


# Punto de entrada
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesamiento del dataset NutriCuisine"
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help="Ruta al CSV original (diet_type_recipes.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help="Ruta de salida del CSV limpio",
    )
    args = parser.parse_args()

    preprocess(args.input, args.output)
