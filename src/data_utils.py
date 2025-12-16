import spacy
import re
import sys
import os

# Intentamos importar la configuración.
try:
    from src.config import STOPWORDS_EXCEPTIONS
except ImportError:
    # Fallback por si se ejecuta como script suelto
    STOPWORDS_EXCEPTIONS = {
        'no', 'sin', 'ni', 'nunca', 'jamás', 'tampoco', 'nada', 'poco', 'apenas'
    }

# Variable global para el modelo (Patrón Singleton para no cargarlo mil veces)
_nlp_model = None

def load_spacy_model():
    """
    Carga el modelo de Spacy en memoria si no está cargado aún.
    Configura las excepciones de stopwords (negaciones).
    """
    global _nlp_model
    if _nlp_model is None:
        try:
            # print("⏳ Cargando modelo Spacy 'es_core_news_sm'...")
            _nlp_model = spacy.load("es_core_news_sm")
            
            # --- CONFIGURACIÓN CRÍTICA ---
            # Evitamos que Spacy elimine palabras como 'no', 'sin', 'nunca'
            for palabra in STOPWORDS_EXCEPTIONS:
                _nlp_model.vocab[palabra].is_stop = False
            
            # print("✅ Modelo Spacy cargado y configurado.")
        except OSError:
            print("❌ ERROR CRÍTICO: Modelo de Spacy no encontrado.")
            print("Por favor ejecuta en tu terminal: python -m spacy download es_core_news_sm")
            sys.exit(1)
    return _nlp_model

def limpiar_texto_medico(texto):
    """
    Función maestra de limpieza para texto clínico.
    Úsala para entrenar y para predecir.
    
    Pasos:
    1. Minúsculas.
    2. Eliminación de caracteres no alfabéticos (dejando tildes y ñ).
    3. Tokenización y Lematización (Spacy).
    4. Filtrado de stopwords (respetando negaciones).
    """
    if not isinstance(texto, str):
        return ""
    
    # Cargar modelo (solo lo hace la primera vez)
    nlp = load_spacy_model()
    
    # 1. Limpieza básica con Regex
    # Mantenemos letras (a-z), vocales con tilde y espacios. Borramos números y símbolos.
    texto = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', ' ', texto)
    
    # 2. Procesamiento con Spacy
    # disable=['parser', 'ner'] acelera la carga ya que no necesitamos análisis sintáctico profundo aquí
    doc = nlp(texto.lower())
    tokens_limpios = []
    
    for token in doc:
        # Filtros:
        # - No es puntuación
        # - No es stopword (las negaciones ya no son stopwords gracias a la config)
        # - Longitud mayor a 1 (evita letras sueltas como "y", "o", "a")
        if not token.is_punct and not token.is_stop and len(token.text) > 1:
            tokens_limpios.append(token.lemma_)
            
    return " ".join(tokens_limpios)

# Bloque de prueba
if __name__ == "__main__":
    texto_prueba = "El paciente NO presenta fiebre, SIN dolor de cabeza."
    print(f"Original: {texto_prueba}")
    print(f"Procesado: {limpiar_texto_medico(texto_prueba)}")