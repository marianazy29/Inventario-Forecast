import os
from pathlib import Path

# 1. Buscamos la raíz del proyecto y la convertimos en una ruta absoluta real en el disco duro
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Construimos las rutas absolutas para las carpetas críticas
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# 3. Forzamos la creación física de las carpetas si es que no existen en la computadora
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 4. Mapas exactos de archivos para el modelo y los datos
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "modelo_xgboost_licoreria.pkl")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "codificador_productos.pkl")
DATASET_INTELIGENTE_PATH = os.path.join(DATA_DIR, "dataset_licoreria_inteligente.csv")
