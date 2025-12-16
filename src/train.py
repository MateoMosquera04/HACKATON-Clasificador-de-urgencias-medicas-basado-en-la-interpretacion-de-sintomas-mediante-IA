import pandas as pd
import numpy as np
import pickle
import os
import sys

# Truco para permitir importaciones relativas si se ejecuta esto como script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Importamos nuestra configuración y utilidades
from src import config
from src.data_utils import limpiar_texto_medico

def unificar_categorias(especialidad):
    """
    Aplica la misma Ingeniería de Etiquetas que en los notebooks.
    Unifica Traumatología y agrupa clases minoritarias.
    """
    especialidad = str(especialidad).upper().strip()
    
    # 1. Fusión de Traumatología
    if 'TRAUMATOLOGÍA' in especialidad: 
        return 'TRAUMATOLOGÍA/MUSCULAR'
    
    # 2. Agrupar clases muy pequeñas para evitar ruido (Precisión 0.00)
    clases_minoritarias = [ ]
    
    if especialidad in clases_minoritarias:
        return 'MEDICINA GENERAL'
        
    return especialidad

def train():
    print("🚀 Iniciando proceso de entrenamiento automatizado...")
    
    # 1. Cargar Datos Procesados
    if not os.path.exists(config.PROCESSED_DATA_FILE):
        print(f"❌ Error: No se encuentra el archivo {config.PROCESSED_DATA_FILE}")
        print("Ejecuta primero los notebooks de obtención y preprocesamiento.")
        return

    df = pd.read_csv(config.PROCESSED_DATA_FILE)
    print(f"📄 Datos cargados: {len(df)} registros.")

    # 2. Refinamiento de Etiquetas (Label Engineering)
    print("🔧 Refinando y unificando etiquetas...")
    df['especialidad_final'] = df['especialidad'].apply(unificar_categorias)
    
    # 3. Preparación de X e y
    X = df['sintomas_procesados'].astype(str) # Texto limpio
    y_labels = df['especialidad_final']       # Etiquetas texto

    # Codificar etiquetas a números
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    
    # Guardar LabelEncoder (CRÍTICO para la App)
    with open(config.LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"💾 LabelEncoder actualizado y guardado en {config.LABEL_ENCODER_PATH}")
    
    # 4. Split (Train/Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    # 5. Construcción del Pipeline (Vectorizador + Modelo SVM)
    # Usamos los parámetros de config.py para mantener consistencia
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=config.NGRAM_RANGE,
            min_df=config.MIN_DF,
            max_features=config.VOCAB_SIZE,
            strip_accents='unicode'
        )),
        ('svm', SVC(
            C=10, 
            kernel='linear', 
            class_weight='balanced', 
            probability=True,  # Necesario para mostrar % de confianza
            random_state=config.RANDOM_STATE
        ))
    ])

    # 6. Entrenamiento
    print("🧠 Entrenando modelo SVM (esto puede tardar unos segundos)...")
    pipeline.fit(X_train, y_train)
    
    # 7. Evaluación rápida
    print("📊 Evaluando modelo...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🏆 Precisión en Test: {acc*100:.2f}%")
    
    # 8. Guardar Modelo Final
    with open(config.MODEL_SVM_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"✅ Modelo guardado exitosamente en: {config.MODEL_SVM_PATH}")

if __name__ == "__main__":
    train()