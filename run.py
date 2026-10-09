import os
import subprocess
import sys

from app.settings import DATASET_PATH, ENCODER_PATH, METRICS_PATH, MODEL_PATH

# Pipeline steps, in order: (script, description shown to the user)
PIPELINE = [
    ("build_catalog.py", "Validar el catálogo (data/Catalogo.xlsx)"),
    ("generate_data.py", "Generar las ventas sintéticas (semilla fija)"),
    ("preprocess.py", "Agregar las variables de calendario de Tarija"),
    ("split_data.py", "Separar los datos por fecha (2023-2025 | 2026)"),
    ("select_model.py", "Comparar modelos en validación (2025)"),
    ("train.py", "Prueba final en 2026 y entrenamiento del modelo final"),
]


def run_script(script: str, description: str):
    """Run one pipeline script with the same Python interpreter (virtual environment)."""
    print(f"\n[PIPELINE] {description}: {script}")
    try:
        # "-u" shows the output of the script immediately, so the progress is visible
        subprocess.run([sys.executable, "-u", script], check=True)
        print(f"[PIPELINE] {script} finalizó con éxito.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Falló la ejecución de {script}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 80)
    print(" SISTEMA DE PRONÓSTICO DE VENTAS - LICORERÍA")
    print("=" * 80)

    # 1. Check whether the model and the processed data already exist
    artifacts_exist = all(os.path.exists(p) for p in [MODEL_PATH, ENCODER_PATH, METRICS_PATH, DATASET_PATH])

    # "python run.py --reconstruir" rebuilds everything even if the files exist
    # (useful after changing the catalog or the generator rules)
    force_rebuild = "--reconstruir" in sys.argv

    if force_rebuild or not artifacts_exist:
        if force_rebuild:
            print("Reconstrucción solicitada con --reconstruir.")
        else:
            print("No se encontraron el modelo o los datos procesados.")
        print("[PIPELINE] Ejecutando el proceso completo...\n")
        for script, description in PIPELINE:
            run_script(script, description)
        print("\n[PIPELINE] ¡Proceso completo terminado con éxito!\n")
    else:
        print("[INFO] Modelo y datos ya existen: se omite el entrenamiento.")
        print("Si cambiaste el catálogo o el generador, ejecuta: python run.py --reconstruir\n")

    # 2. Start the web server (FastAPI)
    print("=" * 80)
    print("Iniciando el servidor...")
    print("Aplicación:    http://127.0.0.1:8000/")
    print("Documentación: http://127.0.0.1:8000/docs")
    print("=" * 80 + "\n")

    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
