import os
import sys
import pickle
import numpy as np

# Truco para importar módulos hermanos si se ejecuta como script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.data_utils import limpiar_texto_medico

def load_artifacts():
    """Carga el modelo y el codificador de etiquetas."""
    if not os.path.exists(config.MODEL_SVM_PATH) or not os.path.exists(config.LABEL_ENCODER_PATH):
        raise FileNotFoundError("❌ No se encuentran los modelos. Ejecuta 'python src/train.py' primero.")
    
    print("⏳ Cargando cerebro (modelo)...")
    with open(config.MODEL_SVM_PATH, 'rb') as f:
        model = pickle.load(f)
        
    with open(config.LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
        
    return model, le

def predict_single(text, model, le):
    """
    Realiza una predicción para un solo texto.
    Retorna: (Especialidad, Confianza, Texto_Procesado)
    """
    # 1. Limpieza usando función centralizada
    text_clean = limpiar_texto_medico(text)
    
    if not text_clean or len(text_clean) < 3:
        return None, 0.0, text_clean

    # 2. Predicción
    # Nota: Como es un pipeline, le pasamos el texto directo (en una lista)
    pred_probs = model.predict_proba([text_clean])
    
    # 3. Obtener la clase con mayor probabilidad
    max_idx = np.argmax(pred_probs)
    confidence = pred_probs[0][max_idx]
    
    # 4. Decodificar el número a nombre (0 -> 'CARDIOLOGÍA')
    specialty = le.inverse_transform([max_idx])[0]
    
    return specialty, confidence, text_clean

def interactive_mode():
    """Bucle infinito para probar frases en la consola."""
    try:
        model, le = load_artifacts()
        print("\n" + "="*50)
        print("🤖 SISTEMA DE TRIAJE INTELIGENTE (Modo Consola)")
        print("Escribe los síntomas del paciente (o 'salir' para terminar).")
        print("="*50 + "\n")
        
        while True:
            user_input = input("\n👤 Describe el caso: ")
            
            if user_input.lower() in ['salir', 'exit', 'q']:
                print("👋 ¡Hasta luego!")
                break
            
            specialty, conf, clean_text = predict_single(user_input, model, le)
            
            if specialty:
                print(f"⚙️ Procesado: '{clean_text}'")
                print(f"🏥 Especialidad: {specialty}")
                print(f"📊 Confianza: {conf:.2%}")
            else:
                print("⚠️ Texto insuficiente o no válido. Intenta ser más descriptivo.")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_mode()