import os

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBRegressor

from app.settings import DATA_DIR

# Este script SOLO lee train_datos.csv (2023-2025). El archivo de prueba (2026) no se abre.
FECHA_INICIO_VALIDACION = "2025-01-01"

# 1. Cargar datos y crear el lag (igual que antes)
df = pd.read_csv(os.path.join(DATA_DIR, "train_datos.csv"), parse_dates=["Fecha"])
df = df.sort_values(["ProductoId", "Fecha"]).reset_index(drop=True)
df["Venta_Semana_Anterior"] = df.groupby("ProductoId")["Cantidad_Vendida"].shift(7)
df = df.dropna(subset=["Venta_Semana_Anterior"]).copy()
df["Producto_Codificado"] = LabelEncoder().fit_transform(df["ProductoId"])

# 2. Separación temporal: entrenamiento (2023-2024) y validación (2025)
entrenamiento = df[df["Fecha"] < FECHA_INICIO_VALIDACION]
validacion = df[df["Fecha"] >= FECHA_INICIO_VALIDACION]
y_entrenamiento = entrenamiento["Cantidad_Vendida"]
y_validacion = validacion["Cantidad_Vendida"]

print(f"Entrenamiento: {entrenamiento['Fecha'].min().date()} a {entrenamiento['Fecha'].max().date()} ({len(entrenamiento)} filas)")
print(f"Validación:    {validacion['Fecha'].min().date()} a {validacion['Fecha'].max().date()} ({len(validacion)} filas)")

sin_lag = ["Producto_Codificado", "Mes", "Día_Semana", "Es_Fin_De_Semana", "Es_Feriado", "Es_Evento_Festivo"]
con_lag = sin_lag + ["Venta_Semana_Anterior"]


def crear_ridge():
    preprocesador = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), ["Producto_Codificado", "Mes", "Día_Semana"]),
        remainder="passthrough",
    )
    return make_pipeline(preprocesador, Ridge(alpha=1.0))


# 3. Lista de candidatos: (nombre, función que crea el modelo, variables que usa)
candidatos = [
    ("Ridge sin lag", crear_ridge, sin_lag),
    ("Ridge con lag", crear_ridge, con_lag),
    ("XGBoost original (150 árboles, prof. 6)",
     lambda: XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42), con_lag),
    ("XGBoost más grande (400 árboles, prof. 8)",
     lambda: XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=8, random_state=42), con_lag),
]

# 4. Evaluar cada candidato en validación
resultados = []

mae_base = mean_absolute_error(y_validacion, validacion["Venta_Semana_Anterior"])
resultados.append(("Línea base (semana anterior)", mae_base))

for nombre, crear_modelo, columnas in candidatos:
    modelo = crear_modelo().fit(entrenamiento[columnas], y_entrenamiento)
    pred = np.clip(modelo.predict(validacion[columnas]), 0, None).round()
    resultados.append((nombre, mean_absolute_error(y_validacion, pred)))

# 5. Mostrar la tabla ordenada de mejor a peor
print("\nRESULTADOS EN VALIDACIÓN (2025)")
print("-" * 60)
for nombre, mae in sorted(resultados, key=lambda r: r[1]):
    print(f"{nombre:<45} MAE = {mae:.2f}")