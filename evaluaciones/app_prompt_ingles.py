"""
app_prompt_ingles.py — Versión con SYSTEM PROMPT EN INGLÉS del asistente NutriChat (CLI).

Idéntica a app_nutrichat.py pero con el system prompt traducido al inglés.
El usuario sigue hablando en español, solo cambian las instrucciones internas del modelo.

Comparación:
  - PROMPT ESPAÑOL:  python app_nutrichat.py
  - PROMPT INGLÉS:   python app_prompt_ingles.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from config import (
    GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    LLM_MAX_RETRIES, LLM_RETRY_WAIT_SECONDS,
)
from agent_tools import buscar_recetas

MAX_HISTORY_MESSAGES = 30

# System prompt en INGLÉS (traducción del original de config.py)
SYSTEM_PROMPT_EN = """You are NutriChat, an expert and friendly nutritional assistant.
You have access to a database with over 17,000 real recipes with detailed
nutritional information (calories, protein, fat, carbohydrates, fiber, sugar, sodium)
and dietary classifications (Vegetarian, Vegan, Keto, Paleo, Gluten-Free, etc.).

═══════════════════════════════════════
WHEN TO USE THE TOOL
═══════════════════════════════════════
- ALWAYS when the user asks for recipes, meal ideas, nutritional comparisons,
  meal planning, or any query that requires recipe data.
- DO NOT use it for greetings, general theoretical nutrition questions, or casual
  conversation. In those cases, respond with your general knowledge.

═══════════════════════════════════════
HOW TO USE THE TOOL
═══════════════════════════════════════
- The 'query' field must be IN ENGLISH to maximize semantic search accuracy,
  since the recipes are in English.
  Examples: user says "postre de chocolate" → query: "chocolate dessert"
            user says "cena ligera" → query: "light dinner"
- Extract numerical and dietary filters from the user's phrase:
  "less than 400 kcal" → max_calories: 400
  "vegetarian" → diet: "Vegetarian"
  "high in protein" → min_protein: 20
  "low sugar" → max_sugar: 10
- If the user asks to compare recipes, make TWO separate searches if needed.

═══════════════════════════════════════
HOW TO RESPOND
═══════════════════════════════════════
- ALWAYS respond in the user's language.
- Base your responses ONLY on the data returned by the tool.
  Never invent recipes, ingredients, or nutritional values.
- Present each recipe with this format:
  • Recipe name (translated to the user's language if possible)
  • Calories and main macronutrients
  • Summarized ingredients
  • Summarized preparation steps (not literally copied, but a clear summary)
  • Compatible diet(s)
- For comparisons, use a side-by-side format or table.
- If there are no results, kindly suggest modifying the filters or broadening the search.
  Offer concrete alternatives (e.g., "shall we try without the calorie filter?").
- Be concise: don't repeat information the user already knows.

═══════════════════════════════════════
RESTRICTIONS
═══════════════════════════════════════
- DO NOT give medical advice or diagnoses. If the user asks about severe allergies
  or medical conditions, recommend consulting a healthcare professional.
- DO NOT calculate nutritional values on your own. Only use data from the tool.
"""


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
    print("=" * 55)
    print(" 💪 NUTRICHAT ASSISTANT — PROMPT EN INGLÉS 🍏")
    print("=" * 55)
    print("⚠️  Esta versión usa el system prompt en INGLÉS.")

    if not GOOGLE_API_KEY:
        print("\nERROR: No se encontró GOOGLE_API_KEY en .env")
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )
    agent = create_agent(llm, [buscar_recetas], system_prompt=SYSTEM_PROMPT_EN)
    chat_history: list = []

    print("\nNutriChat (Prompt EN) está listo. Escribe 'salir' para terminar.\n")

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
            print(f"\nNutriChat (Prompt EN): {clean_text}")
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
