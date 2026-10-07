import os
from pathlib import Path

# 1. Raíz del proyecto (ruta absoluta)
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Carpetas críticas
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# 3. Crear carpetas si no existen
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 4. Archivos del modelo y los datos
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "modelo_licoreria.pkl")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "codificador_productos.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metricas.json")
DATASET_INTELIGENTE_PATH = os.path.join(DATA_DIR, "dataset_licoreria_inteligente.csv")

# 5. Pronóstico a futuro: máximo de días después del último dato real
HORIZONTE_MAX_DIAS = 366

# 6. Margen de ganancia por unidad (Bs) según la categoría REAL del dataset.
#    ⚠️ Son valores de ejemplo: reemplázalos con (precio de venta - costo) reales.
#    Las categorías que no estén aquí usan MARGEN_POR_DEFECTO.
MARGEN_POR_DEFECTO = 5.0
MARGEN_POR_CATEGORIA = {
    "VINOS": 25.0,
    "WISKY": 45.0,
    "AMARULA": 20.0,
    "SODA": 3.0,
    "AGUA": 1.5,
}