import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import time

# --- CONFIGURACIÓN DE RUTAS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.data_utils import limpiar_texto_medico
from src.manchester import calcular_prioridad
from src.derivacion import calcular_derivacion

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="TrIAje 593",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Fondo general más suave */
    .stApp {
        background-color: #f8fafc;
    }
    /* Estilo de la tarjeta de métricas */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    /* Botones personalizados */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
    }
    /* Inputs más limpios */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #ccc;
    }
    </style>
""", unsafe_allow_html=True)

# --- CARGA DE MODELOS ---
@st.cache_resource
def load_models():
    try:
        if not os.path.exists(config.MODEL_SVM_PATH) or not os.path.exists(config.LABEL_ENCODER_PATH):
            return None, None
        with open(config.MODEL_SVM_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(config.LABEL_ENCODER_PATH, 'rb') as f:
            le = pickle.load(f)
        return model, le
    except Exception as e:
        st.error(f"Error técnico: {e}")
        return None, None


model, le = load_models()

# --- INTERFAZ ---

# 1. SIDEBAR (Barra Lateral)
with st.sidebar:
    st.title("🏥 TrIAje 593")
    st.markdown("Sistema de Clasificación Médica")
    st.divider()

    st.info(
        "**Instrucciones:**\nDescribe los síntomas del paciente con el mayor detalle posible para obtener una predicción precisa.")

    st.divider()
    # Estado del sistema (Indicador visual)
    if model:
        st.success("● Sistema En Línea")
    else:
        st.error("● Sistema Desconectado")
        st.caption("No se encontraron los modelos en `models/`")

# 2. PANEL PRINCIPAL
st.title("Asistente de Triaje Inteligente")
st.markdown("Identificación automática de especialidades y nivel de urgencia médica basada en síntomas.")

# Si no hay modelo, detenemos la app visualmente
if not model:
    st.warning(
        "⚠️ **Atención:** Debes entrenar el modelo antes de usar la app. Ejecuta `python src/train.py` en tu terminal.")
    st.stop()

# Área de entrada de texto
col_input, col_help = st.columns([3, 2])

with col_input:
    texto_input = st.text_area(
        "📝 Descripción del Caso",
        placeholder="Ejemplo: Paciente masculino de 45 años que acude por dolor torácico opresivo irradiado a brazo izquierdo, acompañado de sudoración fría...",
        height=100,
        width=570
    )

    # Botones de acción
    col_btn_1, col_btn_2 = st.columns([2, 4])
    with col_btn_1:
        analizar = st.button("🔍 Analizar", type="primary", use_container_width=True)
    with col_btn_2:
        if st.button("Borrar", type="secondary"):
            texto_input = ""

with col_help:
    st.markdown("#### ❓ ¿Cómo describir los síntomas?")
    st.markdown("""
    - Sé lo más detallado posible.
    - Incluye duración, intensidad y factores asociados.
    - Ejemplos:
        - "Dolor abdominal intenso desde hace 2 horas, náuseas y vómitos."
        - "Fiebre alta de 39°C, tos seca y dificultad para respirar."
    """)

# Lógica de Análisis
if analizar and texto_input:
    if len(texto_input) < 10:
        st.warning("⚠️ La descripción es demasiado breve para un diagnóstico fiable.")
    else:
        # Procesamiento
        with st.spinner('Analizando terminología clínica...'):
            # 1. Limpiar
            texto_limpio = limpiar_texto_medico(texto_input)
            time.sleep(0.5)  # Pequeña pausa para UX

            # 2. Predecir
            pred_probs = model.predict_proba([texto_limpio])[0]
            max_idx = np.argmax(pred_probs)
            confidence = pred_probs[max_idx]
            especialidad_pred = le.inverse_transform([max_idx])[0]

        # --- SECCIÓN DE RESULTADOS ---
        st.divider()
        st.subheader("📋 Resultados del Análisis")

        # Columnas para métricas
        col_res_1, col_res_3 = st.columns([2, 2])

        with col_res_1:
            # Tarjeta de Diagnóstico
            if confidence > 0.8:
                st.success(f"### {especialidad_pred}")
                st.caption("Nivel de certeza: Alto")
            elif confidence > 0.5:
                st.warning(f"### {especialidad_pred}")
                st.caption("Nivel de certeza: Medio (Revisar)")
            else:
                st.error(f"### {especialidad_pred}")
                st.caption("Nivel de certeza: Bajo (Requiere valoración humana)")

        with col_res_3:
            # 3. Cálculo de Prioridad (Manchester)
            triaje = calcular_prioridad(texto_input)  # Usamos texto original para Manchester

            # Creamos un contenedor con el color del nivel
            st.markdown(f"""
                <div style="
                    background-color: {triaje['color']};
                    padding: 20px;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin-bottom: 20px;">
                    <h2 style="color: white; margin:0;">NIVEL {triaje['nivel']}: {triaje['nombre']}</h2>
                    <p style="margin:0; font-size: 1.2rem;">⏱️ Tiempo de espera objetivo: <strong>{triaje['tiempo']}</strong></p>
                </div>
            """, unsafe_allow_html=True)

        st.divider()
        # Calculo de Derivación
        derivacion = calcular_derivacion(triaje['nivel'], especialidad_pred)

        # --- TARJETA DE DERIVACIÓN ---
        st.subheader("🗺️ Ruta de Derivación Sugerida")

        with st.container(border=True):
            col_icon, col_text = st.columns([1, 5])

            with col_icon:
                # Icono grande centrado
                st.markdown(f"<h1 style='text-align: center;'>{derivacion['icono']}</h1>", unsafe_allow_html=True)

            with col_text:
                st.markdown(f"### {derivacion['tipo']}")
                st.markdown(f"**ACCIÓN:** {derivacion['accion']}")
                st.info(derivacion['mensaje'])

        st.divider()
        # Gráfico de barras simple con las top 3 probabilidades
        st.subheader("Otras posibilidades")
        top3_idx = np.argsort(pred_probs)[-3:][::-1]

        # Preparamos datos para gráfico
        chart_data = pd.DataFrame({
            "Especialidad": le.inverse_transform(top3_idx),
            "Probabilidad": pred_probs[top3_idx]
        })

        st.bar_chart(chart_data, x="Especialidad", y="Probabilidad", color="#008080")

elif analizar and not texto_input:
    st.error("Por favor ingresa una descripción para comenzar.")