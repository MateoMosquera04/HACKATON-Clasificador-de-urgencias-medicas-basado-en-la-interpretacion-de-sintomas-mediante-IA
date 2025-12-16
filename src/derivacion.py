def calcular_derivacion(nivel_manchester, especialidad_predicha):
    """
    Determina el lugar de atención adecuado según la gravedad y especialidad.
    Retorna un diccionario con la recomendación visual.
    """
    
    # LÓGICA DE DERIVACIÓN (IESS / MSP Ecuador)
    
    if nivel_manchester <= 2:
        # ROJO o NARANJA -> Hospital de Tercer Nivel
        return {
            "tipo": "HOSPITAL DE ESPECIALIDADES / TERCER NIVEL",
            "accion": "🚨 ACUDIR A EMERGENCIAS INMEDIATAMENTE",
            "icono": "🏥",
            "color_box": "#d32f2f", # Rojo oscuro para alerta
            "mensaje": (
                "La condición del paciente pone en riesgo su vida o función vital. "
                "No requiere cita previa. Ingrese directamente por el área de Emergencias (Shock Room)."
            )
        }
        
    elif nivel_manchester == 3:
        # AMARILLO -> Hospital General o Tipo C
        return {
            "tipo": "HOSPITAL GENERAL / CENTRO DE SALUD TIPO C",
            "accion": "⚠️ ACUDIR A URGENCIAS",
            "icono": "🚑",
            "color_box": "#fbc02d", # Amarillo oscuro
            "mensaje": (
                "Requiere atención médica pronta para evitar complicaciones. "
                "Acuda al servicio de urgencias de su hospital de zona o Materno-Infantil."
            )
        }
        
    else:
        # VERDE o AZUL -> Primer Nivel de Atención
        return {
            "tipo": "CENTRO DE SALUD (TIPO A/B) / DISPENSARIO",
            "accion": "📅 AGENDAR CITA (CONSULTA EXTERNA)",
            "icono": "👨‍⚕️",
            "color_box": "#388e3c", # Verde
            "mensaje": (
                f"No es una emergencia vital. Debe agendar una cita médica para **{especialidad_predicha}** "
                "o Medicina General en su dispensario más cercano (IESS/MSP). "
                "No sature las urgencias hospitalarias."
            )
        }