"""
app_streamlit.py — Interfaz web del asistente NutriChat con Streamlit.

Características:
  - Chat conversacional con historial.
  - Barra lateral con filtros rápidos (calorías, dieta, proteína).
  - Reconocimiento de imágenes de platos para buscar recetas.
  - Visualización de macronutrientes con gráfico de barras.
  - Estadísticas de la base de datos.

"""

import base64
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT
from agent_tools import buscar_recetas


def extract_text(content) -> str:
    """
    Extrae texto limpio del contenido de un mensaje del LLM.
    Gemini puede devolver un string directamente o una lista de bloques.
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

# Configuración de la página
st.set_page_config(
    page_title="NutriChat",
    page_icon="🥗",
    layout="wide",
)


# Inicialización del agente
@st.cache_resource
def init_agent():
    """
    Crea el agente ReAct una sola vez y lo reutiliza entre reruns.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )
    return create_agent(llm, [buscar_recetas], system_prompt=SYSTEM_PROMPT)


# Estado de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Barra lateral
with st.sidebar:
    st.subheader("Filtros rápidos")
    st.caption("Se añaden automáticamente a tu consulta.")

    # Mapeo de español a inglés para los filtros de dieta
    DIETAS_ES = {
        "Ninguna": None,
        "Vegetariana": "Vegetarian",
        "Vegana": "Vegan",
        "Keto": "Keto",
        "Paleo": "Paleo",
        "Sin gluten": "Gluten-Free",
        "Sin lácteos": "Dairy-Free",
    }
    quick_diet_es = st.selectbox("Dietas principales", list(DIETAS_ES.keys()))
    quick_diet = DIETAS_ES[quick_diet_es]
    quick_max_cal = st.slider("Calorías máximas", 0, 1500, 0, step=50,
                               help="0 = sin límite")
    quick_min_protein = st.slider("Proteína mínima (g)", 0, 80, 0, step=5,
                                   help="0 = sin mínimo")

    st.markdown("---")

    # Subida de imágenes de platos
    st.subheader("📷 Reconocimiento de platos")
    st.caption("Sube una foto y NutriChat identificará la receta o alimento.")
    uploaded_image = st.file_uploader(
        "Sube una imagen",
        type=["jpg", "jpeg", "png", "webp"],
        help="Formatos aceptados: JPG, JPEG, PNG, WEBP",
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Imagen cargada", use_container_width=True)

    st.markdown("---")

    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()



# Área principal de chat
st.title("NutriChat")
st.caption("Hazme cualquier consulta sobre nutrición, recetas, comparaciones y más :)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Si el mensaje contiene una imagen, la mostramos junto al texto
        if "image" in msg:
            st.image(msg["image"], caption="Imagen enviada", width=300)
        st.markdown(msg["content"])

# Input del usuario
if prompt := st.chat_input("Escribe tu consulta sobre recetas..."):

    # Se añaden al prompt los filtros rápidos si están activos
    enrichment_parts = []
    if quick_diet is not None:
        enrichment_parts.append(f"dieta: {quick_diet}")
    if quick_max_cal > 0:
        enrichment_parts.append(f"máximo {quick_max_cal} calorías")
    if quick_min_protein > 0:
        enrichment_parts.append(f"mínimo {quick_min_protein}g de proteína")

    if enrichment_parts:
        enriched_prompt = f"{prompt} (Filtros adicionales: {', '.join(enrichment_parts)})"
    else:
        enriched_prompt = prompt

    has_image = uploaded_image is not None
    image_bytes = None

    if has_image:
        image_bytes = uploaded_image.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        # Se obtiene el tipo MIME de la imagen
        mime_type = uploaded_image.type or "image/jpeg"

        image_instruction = (
            "El usuario ha enviado una foto de un plato. "
            "Identifica qué plato o alimento aparece en la imagen y busca "
            "la receta más parecida en la base de datos."
        )
        if prompt.strip():
            image_instruction += f" El usuario también dice: '{enriched_prompt}'"
        elif enrichment_parts:
            image_instruction += f" (Filtros adicionales: {', '.join(enrichment_parts)})"

        # Mensaje multimodal: imagen + texto 
        human_message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                },
                {"type": "text", "text": image_instruction},
            ]
        )
    else:
        human_message = HumanMessage(content=enriched_prompt)

    # Mostrar mensaje del usuario
    user_display = {"role": "user", "content": prompt if prompt.strip() else "📷 Imagen enviada"}
    if has_image:
        user_display["image"] = image_bytes
    st.session_state.messages.append(user_display)

    with st.chat_message("user"):
        if has_image:
            st.image(image_bytes, caption="Imagen enviada", width=300)
        st.markdown(prompt if prompt.strip() else "📷 *Imagen enviada para identificar*")

    # Ejecutar agente
    with st.chat_message("assistant"):
        spinner_text = "Analizando imagen y buscando recetas..." if has_image else "Buscando recetas..."
        with st.spinner(spinner_text):
            try:
                st.session_state.chat_history.append(human_message)

                if len(st.session_state.chat_history) > 30:
                    st.session_state.chat_history = st.session_state.chat_history[-30:]

                response = init_agent().invoke(
                    {"messages": st.session_state.chat_history}
                )

                ai_content = extract_text(response["messages"][-1].content)
                st.markdown(ai_content)

                st.session_state.chat_history.append(
                    AIMessage(content=ai_content)
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_content}
                )

            except Exception as e:
                error_msg = f"Error: {type(e).__name__} — {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
