# NutriChat

Asistente nutricional inteligente basado en Retrieval-Augmented Generation (RAG).

NutriChat permite buscar recetas saludables mediante lenguaje natural, aplicando filtros nutricionales automáticos, y recibir recomendaciones personalizadas basadas en una base de datos de mas de 17.000 recetas reales.

## Tecnologías

- **LLM**: Google Gemini 2.5 Flash (API gratuita)
- **Base de datos vectorial**: ChromaDB con persistencia local
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2 (multilingue, 384 dimensiones)
- **Orquestacion**: LangChain con patrón ReAct
- **Interfaz web**: Streamlit
- **Esquemas**: Pydantic (10 filtros nutricionales)

## Características principales  

- Búsqueda semántica + filtros nutricionales (calorías, grasas, proteínas, carbohidratos, fibra, azúcar, sodio, dieta)
- Búsqueda cross-lingual (consultas en español sobre recetas en ingles)
- Reconocimiento de imágenes de platos (multimodal)
- Historial conversacional
- Mecanismo de fallback para filtros de ChromaDB
- Reintentos automáticos ante errores de cuota de la API

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/[usuario]/nutrichat.git
cd nutrichat
```

2. Crear y activar un entorno virtual:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar la API key:
# Crear (basándose en .env.example) .env y poner la GOOGLE_API_KEY


5. Preprocesar el dataset:
```bash
python preprocesamiento.py
```

6. Ingestar las recetas en ChromaDB:
```bash
python ingest_data.py
```

## Uso

### Aplicación CLI
```bash
python app_nutrichat.py
```

### Interfaz web (Streamlit)
```bash
streamlit run app_streamlit.py
```

## Evaluación

La carpeta `evaluaciones/` contiene scripts para comparar distintas configuraciones del sistema. Todos se ejecutan desde esta carpeta y acceden a los modulos principales del proyecto automaticamente.

- **app_sin_rag.py** — Sin RAG (solo LLM) vs con RAG
- **app_sin_filtros.py** — Solo busqueda semantica vs busqueda hibrida (semantica + filtros)
- **app_prompt_ingles.py** — System prompt en ingles vs en espanol
- **app_embeddings.py** — Modelos de embeddings alternativos (all-MiniLM-L6-v2, all-mpnet-base-v2)
- **benchmark.py** — Evaluacion formal con metricas nDCG@5 y MRR sobre 15 consultas de referencia

Ejemplo de uso:
```bash
cd evaluaciones
python app_sin_rag.py
python benchmark.py
```

## Estructura del proyecto

```
nutrichat/
├── config.py                    # Configuracion centralizada
├── preprocesamiento.py          # Pipeline de limpieza del dataset
├── ingest_data.py               # Pipeline ETL (CSV -> ChromaDB)
├── nutri_vectordb.py            # Capa de acceso a ChromaDB
├── agent_tools.py               # Herramienta de busqueda + esquema Pydantic
├── app_nutrichat.py             # Aplicacion CLI principal
├── app_streamlit.py             # Interfaz web con Streamlit
├── requirements.txt             # Dependencias
├── .env.example                 # Ejemplo del archivo .env
├── .gitignore                   # Reglas de exclusión de archivos para Git
├── README.md                    # Documentacion del proyecto
├── dataset/
│   └── diet_type_recipes.csv    # Dataset original
└── evaluaciones/
    ├── app_sin_rag.py           # Comparacion: sin RAG vs con RAG
    ├── app_sin_filtros.py       # Comparacion: sin filtros vs con filtros
    ├── app_prompt_ingles.py     # Comparacion: prompt ingles vs espanol
    ├── app_embeddings.py        # Comparacion: modelos de embeddings
    └── benchmark.py             # Evaluacion sistema de recuperación (nDCG@5, MRR)
```

## Autor

Francisco Llorente Meroño - Estudiante del Grado en Ciencia e Ingenieria de Datos, Universidad de Murcia

