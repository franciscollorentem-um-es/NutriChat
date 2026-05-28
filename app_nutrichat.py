"""
app_nutrichat.py — Punto de entrada CLI del asistente NutriChat.
Contiene un bucle de conversación donde el usuario puede hacer preguntas sobre recetas,
nutrición, comparaciones, etc. El agente LLM responde usando la base de datos de recetas
y la herramienta de búsqueda definida en agent_tools.py.
"""

import sys
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from config import (
    GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT,
    LLM_MAX_RETRIES, LLM_RETRY_WAIT_SECONDS,
)
from agent_tools import buscar_recetas

# Máximo de mensajes en el historial para no exceder la ventana de contexto
MAX_HISTORY_MESSAGES = 30


# Funciones auxiliares
def extract_text(content) -> str:
    """
    Extrae texto limpio del contenido de un mensaje del LLM.
    Gemini puede devolver un string directamente o una lista de bloques
    como [{'type': 'text', 'text': '...', 'extras': {...}}].
    Esta función maneja ambos casos y devuelve siempre un string limpio.
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


def create_nutri_agent():
    """
    Crea el agente con Gemini y la herramienta de búsqueda de recetas.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )
    return create_agent(llm, [buscar_recetas], system_prompt=SYSTEM_PROMPT)


def truncate_history(history: list, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    """
    Mantiene solo los últimos N mensajes (en nuestro caso 30) para no exceder el contexto del LLM.
    """
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


def invoke_with_retry(agent, messages: list) -> dict:
    """
    Invoca al agente con reintentos automáticos ante errores de cuota.
    """
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return agent.invoke({"messages": messages})
        except Exception as e:
            error_str = str(e)
            # Error de cuota
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < LLM_MAX_RETRIES:
                    wait = LLM_RETRY_WAIT_SECONDS * attempt
                    print(f"\n  Cuota excedida. Reintentando en {wait}s... (intento {attempt}/{LLM_MAX_RETRIES})")
                    time.sleep(wait)
                    continue
            # Si no es un error de cuota o se agotaron los reintentos, se lanza la excepción.
            raise
    raise RuntimeError("Se agotaron los reintentos.")

# Función principal del asistente NutriChat
def main():
    print("=" * 50)
    print(" 💪 BIENVENIDO A NUTRICHAT 🍏 ")
    print("=" * 50)

    if not GOOGLE_API_KEY:
        print("ERROR: No se encontró GOOGLE_API_KEY en el archivo .env")
        print("Crea un archivo .env con: GOOGLE_API_KEY=tu_clave_aquí")
        sys.exit(1)

    agent = create_nutri_agent()
    chat_history: list = []

    print("\nNutriChat está listo. Puedes preguntar por recetas, filtrar por dietas, macronutrientes y mucho más.")
    print("Escribe 'salir' para terminar.\n")

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
            print(f"\nNutriChat: {clean_text}")
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
