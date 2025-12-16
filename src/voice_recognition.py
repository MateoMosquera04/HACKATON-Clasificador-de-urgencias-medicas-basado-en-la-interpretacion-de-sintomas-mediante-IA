"""
Módulo de reconocimiento de voz para el sistema de triaje médico.
Maneja la transcripción de audio a texto usando Google Speech Recognition.
"""

import speech_recognition as sr
import os
from typing import Optional, Tuple

try:
    from src.config import (
        VOICE_LANGUAGE, 
        VOICE_ENERGY_THRESHOLD, 
        VOICE_DYNAMIC_THRESHOLD,
        VOICE_AMBIENT_DURATION
    )
except ImportError:
    # Valores por defecto si no se puede importar config
    VOICE_LANGUAGE = "es-ES"
    VOICE_ENERGY_THRESHOLD = 4000
    VOICE_DYNAMIC_THRESHOLD = True
    VOICE_AMBIENT_DURATION = 0.5


def transcribe_audio(audio_bytes: bytes, language: str = None) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Transcribe audio a texto usando Google Speech Recognition.
    
    Args:
        audio_bytes: Bytes del archivo de audio WAV
        language: Código de idioma para la transcripción (default: desde config)
    
    Returns:
        Tuple[bool, Optional[str], Optional[str]]: 
            - success: True si la transcripción fue exitosa
            - text: Texto transcrito (None si falla)
            - error_message: Mensaje de error (None si éxito)
    """
    if language is None:
        language = VOICE_LANGUAGE
    
    temp_file = "temp_audio.wav"
    
    try:
        # Guardar el audio temporalmente
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)
        
        # Inicializar el reconocedor
        recognizer = sr.Recognizer()
        
        # Configuración para mejorar la precisión
        recognizer.energy_threshold = VOICE_ENERGY_THRESHOLD
        recognizer.dynamic_energy_threshold = VOICE_DYNAMIC_THRESHOLD
        
        # Cargar y procesar el audio
        with sr.AudioFile(temp_file) as source:
            # Ajustar al ruido ambiental
            recognizer.adjust_for_ambient_noise(source, duration=VOICE_AMBIENT_DURATION)
            # Grabar el audio
            audio_data = recognizer.record(source)
        
        # Intentar transcribir
        try:
            texto_transcrito = recognizer.recognize_google(audio_data, language=language)
            cleanup_temp_file(temp_file)
            return True, texto_transcrito, None
            
        except sr.UnknownValueError:
            cleanup_temp_file(temp_file)
            return False, None, "No se pudo entender el audio. Intenta hablar más claro."
            
        except sr.RequestError as e:
            cleanup_temp_file(temp_file)
            return False, None, f"Error del servicio de reconocimiento: {str(e)}"
    
    except Exception as e:
        cleanup_temp_file(temp_file)
        return False, None, f"Error procesando audio: {str(e)}"


def cleanup_temp_file(filepath: str) -> None:
    """
    Elimina un archivo temporal de forma segura.
    
    Args:
        filepath: Ruta del archivo a eliminar
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass  # Ignorar errores al eliminar archivos temporales


def append_text(current_text: str, new_text: str) -> str:
    """
    Agrega nuevo texto transcrito al texto existente.
    
    Args:
        current_text: Texto actual
        new_text: Nuevo texto a agregar
    
    Returns:
        str: Texto combinado con espaciado correcto
    """
    if not current_text or current_text.strip() == "":
        return new_text
    return f"{current_text} {new_text}"
