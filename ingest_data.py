"""
ingest_data.py — Fichero de ingesta del dataset en ChromaDB.

Esta ingesta sigue estos pasos:
  1. Carga el dataset limpio de NutriCuisine desde el CSV.
  2. Transforma los datos a un formato adecuado para ChromaDB.
  3. Invoca a NutriVectorDB para generar los embeddings y almacenarlos en ChromaDB.


Solo se ejecuta una vez, después de haber preprocesado los datos con preprocesamiento.py.

"""

import argparse
import ast
import json
import os
import shutil
import sys
import time
import pandas as pd
from config import CHROMA_PERSIST_DIR
from nutri_vectordb import NutriVectorDB

# Ruta por defecto al CSV limpio generado por preprocesamiento.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILE = os.path.join(_SCRIPT_DIR, "dataset", "datos_limpios.csv")


def parse_list_column(value):
    """
    Intenta parsear un string que representa una lista Python.
    Si el valor ya es una lista, lo devuelve tal cual.
    Si el valor es NaN, None o un string vacío, devuelve una lista vacía.
    Si el valor es un string que representa una lista (ej: "['sal', 'pimienta']"), lo convierte a una lista real.
    Si el valor es cualquier otro string, lo devuelve como una lista con un solo elemento.
    """
    if isinstance(value, list):
        return value
    if pd.isna(value) or value is None or value == "":
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except (ValueError, SyntaxError):
        return [str(value)]


def load_csv(filepath: str) -> list:
    """
    Carga el CSV de NutriCuisine y lo convierte a lista de dicts.
    Detecta automáticamente el separador (coma o punto y coma) y el encoding (utf-8, latin-1, cp1252).
    """
    print(f"Cargando CSV: {filepath}")

    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                first_line = f.readline()
            sep = ";" if first_line.count(";") > first_line.count(",") else ","
            df = pd.read_csv(filepath, sep=sep, encoding=encoding)
            print(f"  Encoding detectado: {encoding} | Separador: '{sep}'")
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        raise ValueError(f"No se pudo leer el archivo {filepath} con ningún encoding conocido.")

    print(f"  Filas: {len(df)}, Columnas: {list(df.columns)}")

    # Se renombran columnas que difieren entre el CSV y lo que espera NutriVectorDB
    rename_map = {"serve": "servings"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    list_columns = ["ingredients", "directions"]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_column)

    # Se convierte el DataFrame a una lista de diccionarios
    recipes = df.to_dict(orient="records")
    print(f"  Recetas procesadas: {len(recipes)}")
    return recipes

# Se crea una función para cargar datos desde un JSON
def load_json(filepath: str) -> list:
    """
    Carga un JSON de otra fuente.
    """
    print(f"Cargando JSON: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        recipes = data
    elif isinstance(data, dict) and "recipes" in data:
        recipes = data["recipes"]
    else:
        raise ValueError("Formato JSON no reconocido. Se espera una lista o {recipes: [...]}")

    print(f"  Recetas cargadas: {len(recipes)}")
    return recipes


def main():
    parser = argparse.ArgumentParser(
        description="Ingesta del dataset NutriCuisine en ChromaDB"
    )
    parser.add_argument(
        "--file", "-f",
        default=DEFAULT_DATA_FILE,
        help="Ruta al archivo de datos (CSV o JSON). Por defecto: ./dataset/datos_limpios.csv",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Eliminar la base de datos existente antes de ingestar",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: No se encontró el archivo '{args.file}'")
        sys.exit(1)

    # Se elimina la BD si se pide
    if args.reset and os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Eliminando base de datos existente en '{CHROMA_PERSIST_DIR}'...")
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print("Base de datos eliminada.")

    # Se cargan los datos
    ext = os.path.splitext(args.file)[1].lower()
    if ext == ".csv":
        recipes = load_csv(args.file)
    elif ext == ".json":
        recipes = load_json(args.file)
    else:
        print(f"Error: Formato '{ext}' no soportado. Usa CSV o JSON.")
        sys.exit(1)

    if not recipes:
        print("Error: No se encontraron recetas en el archivo.")
        sys.exit(1)


    # Se realiza la ingesta en ChromaDB
    print("\n" + "=" * 50)
    db = NutriVectorDB()
    print(f"Recetas en BD antes de ingesta: {db.count}")

    start = time.time()
    db.process_and_ingest(recipes)
    elapsed = time.time() - start

    print(f"\nRecetas en BD después de ingesta: {db.count}")
    print(f"Tiempo total: {elapsed:.1f} segundos")
    print("=" * 50)


if __name__ == "__main__":
    main()
