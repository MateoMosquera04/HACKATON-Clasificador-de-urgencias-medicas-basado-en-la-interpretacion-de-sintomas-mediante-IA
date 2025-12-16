import os

# ==========================================
# 1. RUTAS DEL SISTEMA (PATHS)
# ==========================================
# Obtenemos la ruta absoluta donde está este archivo (src/) y subimos un nivel
# para encontrar la raíz del proyecto.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Carpetas principales
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Archivos de Datos
# Archivo original en inglés
MTSAMPLES_FILE = os.path.join(RAW_DATA_DIR, 'mtsamples.csv')
# Archivo intermedio traducido (para no gastar tiempo re-traduciendo)
MTSAMPLES_TRANSLATED_FILE = os.path.join(RAW_DATA_DIR, 'mtsamples_translated.csv')
# Archivo unificado (CodiEsp + MTSamples)
UNIFIED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'datos_triaje_unificados.csv')
# Archivo final limpio listo para entrenar (NLP procesado)
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'datos_nlp_procesados.csv')

# Archivos de Modelos (Artefactos)
# El modelo SVM entrenado (Pipeline)
MODEL_SVM_PATH = os.path.join(MODELS_DIR, 'modelo_triaje_svm.pkl')
# El diccionario que traduce números a especialidades (0 -> Cardiología)
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder_final.pkl')

# ==========================================
# 2. HIPERPARÁMETROS Y CONSTANTES
# ==========================================
# Semilla para reproducibilidad (que siempre salga el mismo resultado)
RANDOM_STATE = 42

# Tamaño del set de prueba (20%)
TEST_SIZE = 0.2

# Configuración de Procesamiento de Texto (NLP)
# Palabras que NO debemos borrar porque cambian el sentido clínico
STOPWORDS_EXCEPTIONS = {
    'no', 'sin', 'ni', 'nunca', 'jamás', 'tampoco', 'nada', 'poco', 'apenas'
}

# Configuración del Vectorizador (TF-IDF)
VOCAB_SIZE = None       # Número máximo de palabras/bigramas a aprender
NGRAM_RANGE = (1, 2)    # Usar palabras sueltas y pares de palabras
MIN_DF = 3              # Ignorar palabras que aparezcan en menos de 3 documentos