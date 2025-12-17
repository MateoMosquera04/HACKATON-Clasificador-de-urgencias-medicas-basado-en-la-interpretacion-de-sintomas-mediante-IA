import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import time
from pathlib import Path
from audio_recorder_streamlit import audio_recorder

# --- CONFIGURACIÓN DE RUTAS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.data_utils import limpiar_texto_medico
from src.manchester import calcular_prioridad
from src.derivacion import calcular_derivacion
from src.voice_recognition import transcribe_audio, append_text

# --- FUNCIONES AUXILIARES DE TEMPLATES ---
TEMPLATES_DIR = Path(__file__).parent / "templates"
ASSETS_DIR = Path(__file__).parent / "assets"

def load_template(filename: str) -> str:
    """Carga un template HTML/CSS."""
    try:
        with open(TEMPLATES_DIR / filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"<!-- Template {filename} no encontrado -->"

def load_css() -> str:
    """Carga los estilos CSS."""
    return f"<style>\n{load_template('styles.css')}\n</style>"

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="TrIAje 593",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar estilos CSS personalizados
st.markdown(load_css(), unsafe_allow_html=True)

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
    st.image(str(ASSETS_DIR / "logo.png"), width=100)
    st.title("Bienvenido")
    st.markdown("Sistema de Clasificación Médica")
    st.divider()

    st.info("**Instrucciones:**\nDescribe los síntomas del paciente con el mayor detalle posible para obtener una predicción precisa.")

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
        "**Atención:** Debes entrenar el modelo antes de usar la app. Ejecuta `python src/train.py` en tu terminal.")
    st.stop()

# Área de entrada de texto
col_input, col_help = st.columns([3, 2])

with col_input:
    # Estado para el texto
    if 'texto_completo' not in st.session_state:
        st.session_state.texto_completo = ""
    
    # Label con micrófono integrado
    col_label, col_mic = st.columns([10, 1])
    with col_label:
        st.markdown("#### Descripción del Caso")
    with col_mic:
        # Componente de grabación de audio compacto
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="2x"
        )
    
    # Procesar el audio cuando esté disponible
    if audio_bytes:
        # Convertir audio a texto usando el módulo de voice_recognition
        with st.spinner("Transcribiendo audio..."):
            success, texto_transcrito, error_msg = transcribe_audio(audio_bytes)
            
            if success:
                # Agregar el texto transcrito usando la función del módulo
                st.session_state.texto_completo = append_text(
                    st.session_state.texto_completo, 
                    texto_transcrito
                )
                st.success(f"Transcrito correctamente")
            else:
                # Mostrar el error apropiado
                if "no se pudo entender" in error_msg.lower():
                    st.warning(f"{error_msg}")
                else:
                    st.error(f"{error_msg}")
    
    # Área de texto editable
    texto_input = st.text_area(
        "Escribe o dicta los síntomas del paciente",
        value=st.session_state.texto_completo,
        placeholder="Ejemplo: Paciente con dolor abdominal intenso desde hace 2 horas, náuseas y vómitos...",
        height=150,
        label_visibility="collapsed"
    )
    
    # Actualizar el estado si se edita manualmente
    st.session_state.texto_completo = texto_input

    # Botones de acción
    col_btn_1, col_btn_2 = st.columns([2, 4])
    with col_btn_1:
        analizar = st.button("Analizar", type="primary", use_container_width=True)
    with col_btn_2:
        if st.button("Limpiar todo", type="secondary", use_container_width=True):
            st.session_state.texto_completo = ""
            st.rerun()
            st.rerun()

with col_help:
    st.markdown("#### ❓ ¿Cómo describir los síntomas?")
    st.markdown("""
    **Escribiendo:**
    - Sé lo más detallado posible
    - Incluye duración, intensidad y factores asociados
    
    **🎤 Dictando por voz:**
    1. Haz clic en el ícono del micrófono
    2. Habla claramente describiendo los síntomas
    3. El audio se detendrá automáticamente
    4. El texto se transcribirá automáticamente
    
    **Ejemplos:**
    - "Dolor abdominal intenso desde hace 2 horas, náuseas y vómitos"
    - "Fiebre alta de 39°C, tos seca y dificultad para respirar"
    """)
    
    st.info("💡 **Tip**: Puedes combinar dictado y escritura. El audio se convierte a texto que puedes editar.")

# Lógica de Análisis
if analizar and texto_input:
    if len(texto_input) < 10:
        st.warning("La descripción es demasiado breve para un diagnóstico fiable.")
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
        st.subheader(" Resultados del Análisis")

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
            triaje = calcular_prioridad(texto_input)

            # Renderizar tarjeta de triaje desde template
            template = load_template("triaje_card.html")
            html = template.format(
                nivel=triaje['nivel'],
                nombre=triaje['nombre'],
                color=triaje['color'],
                tiempo=triaje['tiempo']
            )
            st.markdown(html, unsafe_allow_html=True)

        st.divider()
        # Calculo de Derivación
        derivacion = calcular_derivacion(triaje['nivel'], especialidad_pred)

        # --- TARJETA DE DERIVACIÓN ---
        st.subheader("Ruta de Derivación Sugerida")

        with st.container(border=True):
            col_icon, col_text = st.columns([1, 5])

            with col_icon:
                # Icono grande centrado desde template
                icon_template = load_template("icon_centered.html")
                st.markdown(icon_template.format(icon=derivacion['icono'], size="3rem"), unsafe_allow_html=True)

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