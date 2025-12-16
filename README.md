# 🏥 SIC: Clasificador de Urgencias Médicas con IA

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Status](https://img.shields.io/badge/Status-En_Desarrollo-green)
![NLP](https://img.shields.io/badge/NLP-Spacy-yellow)
![Model](https://img.shields.io/badge/Model-SVM-orange)

# TrIAje 593

---

**TriAje 593** es una herramienta de **triaje médico automatizado** que utiliza Procesamiento de Lenguaje Natural (NLP) e Inteligencia Artificial para analizar descripciones de síntomas y predecir la especialidad médica adecuada (Cardiología, Traumatología, Neurología, etc.).

Este proyecto busca reducir la saturación en los servicios de urgencias agilizando la derivación de pacientes mediante un modelo de Machine Learning robusto.

---

## 🚀 Características Clave

* **🧠 IA Híbrida y Robusta:** Utiliza un modelo **SVM (Support Vector Machine)** optimizado con TF-IDF y Bigramas, superando en precisión a redes neuronales simples en datasets de tamaño medio.
* **🗣️ NLP Médico Avanzado:**
    * Manejo inteligente de **negaciones** ("No tiene fiebre", "Sin dolor") para evitar falsos positivos.
    * Lematización y limpieza de ruido clínico.
* **🌍 Data Augmentation:** Estrategia de traducción automática (Inglés -> Español) integrando el dataset **MTSamples** para enriquecer las clases minoritarias del dataset original **CodiEsp**.
* **🏗️ Ingeniería de Etiquetas:** Agrupación inteligente de especialidades confusas (ej: unificación de Traumatología) para maximizar la fiabilidad clínica (>80% de precisión).

---

## 📂 Estructura del Proyecto

El proyecto sigue una arquitectura modular para facilitar la escalabilidad y el mantenimiento:

```text
SIC-Clasificador-Urgencias/
├── app/
│   ├── app.py/                     # Ejecucion de interfaz web
├── data/                           # Almacenamiento de datos
│   ├── raw/                        # Datos crudos (CodiEsp, MTSamples original)
│   └── processed/                  # Datos limpios y unificados listos para el modelo
│   └── external/                   # Graficos para reportes estadisticos, etc
│
├── models/                         # Artefactos del modelo
│   ├── modelo_triaje_svm.pickle    # El cerebro (Pipeline entrenado)
│   └── label_encoder_final.pickle  # Diccionario de traducción (Número -> Especialidad)
│
├── notebooks/                      # Laboratorio de experimentación
│   ├── 1.0-obtencion-datos.ipynb   # Descarga, traducción y unificación
│   ├── 2.0-preprocesamiento.ipynb  # Limpieza NLP y codificación
│   └── 3.0-entrenamiento.ipynb     # Entrenamiento, evaluación y análisis de errores
│
├── src/                            # Código Fuente (Producción)
│   ├── config.py                   # Configuración centralizada (Rutas, Hiperparámetros)
│   ├── data_utils.py               # Funciones de limpieza y carga de Spacy
│   ├── train.py                    # Script de re-entrenamiento automatizado
│   └── predict.py                  # Script para probar el modelo en consola
│
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Documentación

```

---

## 📝 Creditos
Desarrolladorres:
- Axel Steven Anzules V.
- Stefany Michelle Perachimba P.
- Mateo Steven Mosquera A.
- Cristian Stiven Pusda H.

---

## Datos: 
Basado en el corpus CodiEsp (Plan de Impulso de las Tecnologías del Lenguaje) y MTSamples.

---

## 🛠️ Instalación y Configuración
Sigue estos pasos para ejecutar el proyecto en tu máquina local:

1. Clonar el repositorio
   ```
   git clone [https://github.com/tu-usuario/sic-clasificador-urgencias.git](https://github.com/tu-usuario/sic-clasificador-urgencias.git)
   cd sic-clasificador-urgencias
   ```
2. Crear un entorno virtual (Recomendado)
   ```
    python -m venv venv

   - En Windows:
   venv\Scripts\activate

   - En Mac/Linux:
   source venv/bin/activate
   ```
3. Instalar dependencias
   ```
   pip install -r requirements.txt
   ```
4. Descargar el modelo de lenguaje (Spacy)
   ```
   python -m spacy download es_core_news_sm
   ```



