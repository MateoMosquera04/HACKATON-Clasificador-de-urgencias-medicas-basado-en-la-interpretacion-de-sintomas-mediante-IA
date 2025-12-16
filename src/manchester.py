import re
from src.data_utils import limpiar_texto_medico

# Definición del Sistema de Triaje Manchester (Adaptado a Texto)
# Formato: (Nivel, Color, Tiempo de Espera, Palabras Clave)
# El orden importa: Se evalúa de mayor gravedad (1) a menor (5).

MANCHESTER_RULES = [
    {
        "nivel": 1,
        "nombre": "EMERGENCIA (Resucitación)",
        "color": "#FF0000", # Rojo
        "tiempo": "Inmediato (0 min)",
        "keywords": [
            "paro", "no respira", "inconsciente", "no responde", "convulsion", 
            "sangrado masivo", "hemorragia severa", "azul", "cianosis", "fria",
            "trauma craneal grave", "electrocutado", "caida altura"
        ]
    },
    {
        "nivel": 2,
        "nombre": "MUY URGENTE",
        "color": "#FF8C00", # Naranja
        "tiempo": "10-15 min",
        "keywords": [
            "dolor toracico", "dolor pecho", "infarto", "asfixia", "ahogo", 
            "dificultad respiratoria", "disnea", "quemadura", "fractura expuesta",
            "sanguinolenta", "vomito sangre", "amputacion", "alteracion mental",
            "agresivo", "desorientado", "dolor severo", "muy fuerte", "insoportable"
        ]
    },
    {
        "nivel": 3,
        "nombre": "URGENTE",
        "color": "#FFD700", # Amarillo
        "tiempo": "60 min",
        "keywords": [
            "dolor abdominal", "dolor moderado", "fiebre alta", "mas de 38", 
            "vomito", "diarrea", "deshidratacion", "herida", "corte", 
            "golpe", "trauma", "asma", "crisis", "sangrado"
        ]
    },
    {
        "nivel": 4,
        "nombre": "ESTÁNDAR (Poco Urgente)",
        "color": "#32CD32", # Verde
        "tiempo": "2 horas",
        "keywords": [
            "dolor leve", "molestia", "fiebre", "gripe", "tos", "dolor garganta",
            "cuerpo cortado", "infeccion urinaria", "ardor", "ojo rojo", 
            "alergia", "sarpullido", "ronchas", "esguince", "torcedura"
        ]
    },
    {
        "nivel": 5,
        "nombre": "NO URGENTE",
        "color": "#1E90FF", # Azul
        "tiempo": "4 horas",
        "keywords": [
            # Si no cae en los anteriores, caerá aquí por defecto, 
            # pero ponemos keywords por si acaso.
            "revision", "chequeo", "resultados", "certificado", "consulta",
            "cronico", "hace meses", "receta", "medicacion"
        ]
    }
]

def calcular_prioridad(texto):
    """
    Analiza el texto y determina el nivel de triaje Manchester.
    Retorna un diccionario con la información del nivel.
    """
    # Usamos la misma limpieza para normalizar (quitar tildes, minúsculas)
    # pero NO quitamos stopwords aquí porque "no respira" es clave.
    texto_lower = texto.lower()
    
    # 1. Búsqueda secuencial (De Rojo a Verde)
    for regla in MANCHESTER_RULES:
        for palabra in regla["keywords"]:
            # Buscamos la palabra clave en el texto (con bordes de palabra para exactitud)
            # Ej: que no detecte "paro" dentro de "disparo" si no queremos.
            # Por simplicidad, usamos 'in' directo que es robusto para frases.
            if palabra in texto_lower:
                return regla
                
    # 2. Si no se encuentra nada grave, se asume Nivel 4 o 5
    # Por seguridad, si hay síntomas no clasificados, mejor Amarillo/Verde que Azul.
    # Retornamos Nivel 4 (Verde) por defecto si hay síntomas.
    return MANCHESTER_RULES[3] # Verde