import os
import sys
import subprocess
from app.settings import MODEL_PATH, ENCODER_PATH, DATASET_INTELIGENTE_PATH

def ejecutar_script(nombre_script):
    """Ejecuta un archivo .py secundario de forma segura en la terminal actual."""
    print(f"\n⚙️ [PIPELINE] Ejecutando {nombre_script}...")
    try:
        # Ejecuta el script usando el mismo intérprete de Python activo (entorno virtual)
        resultado = subprocess.run([sys.executable, nombre_script], check=True)
        if resultado.returncode == 0:
            print(f"✅ [PIPELINE] {nombre_script} finalizó con éxito.")
    except subprocess.CalledProcessError as e:
        print(f"❌ [ERROR CRÍTICO] Falló la ejecución de {nombre_script}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("="*80)
    print("   🌐 INICIALIZADOR INTELIGENTE DEL SISTEMA DE LA LICORERÍA ")
    print("="*80)
    
    # 1. VERIFICACIÓN DE ARTEFACTOS Y DATOS EXISTENTES
    artefactos_existen = os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)
    data_procesada_existe = os.path.exists(DATASET_INTELIGENTE_PATH)
    
    if not artefactos_existen or not data_procesada_existe:
        print("⚠️  [ALERTA] No se detectaron los modelos entrenados o los datos limpios.")
        print("🚀 [AUTOMATIZACIÓN] Iniciando reconstrucción completa del Pipeline de ML...\n")
        
        # Paso 1: Crear dataset inteligente de calendario y feriados de Bolivia
        ejecutar_script("procesamiento.py")
        
        # Paso 2: Calcular las métricas iniciales de la Línea Base Tradicional
        ejecutar_script("linea_base.py")
        
        # Paso 3: Separar los datos cronológicamente (2022-2025 Train | 2026 Test)
        ejecutar_script("separacion_temporal.py")
        
        # Paso 4: Entrenar la regresión Ridge y el XGBoost, y exportar los artefactos .pkl
        ejecutar_script("entrenamiento.py")
        
        print("\n🎉 [PIPELINE] ¡Pipeline End-to-End completado con éxito!")
        print("📦 Todos los datos y archivos .pkl han sido regenerados de forma correcta.\n")
    else:
        print("✨ [INFO] Modelos precalculados y datos listos detectados en 'artifacts/'.")
        print("⚡ Saltando etapa de entrenamiento para un encendido ultra veloz.\n")
    
    # 2. ENCENDER EL SERVIDOR WEB (FASTAPI)
    print("="*80)
    print("🚀 Levantando el Servidor Backend con FastAPI...")
    print("📱 Accede a la documentación interactiva en: http://127.0.0")
    print("="*80 + "\n")
    
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
