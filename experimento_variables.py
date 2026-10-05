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

FECHA_CORTE = "2026-01-01"

# 1. Mismos datos y mismo preprocesamiento que entrenamiento.py
df_train = pd.read_csv(os.path.join(DATA_DIR, "train_datos.csv"), parse_dates=["Fecha"])
df_test = pd.read_csv(os.path.join(DATA_DIR, "test_datos.csv"), parse_dates=["Fecha"])
df = pd.concat([df_train, df_test]).sort_values(["ProductoId", "Fecha"]).reset_index(drop=True)
df["Venta_Semana_Anterior"] = df.groupby("ProductoId")["Cantidad_Vendida"].shift(7)
df = df.dropna(subset=["Venta_Semana_Anterior"]).copy()
df["Producto_Codificado"] = LabelEncoder().fit_transform(df["ProductoId"])

train = df[df["Fecha"] < FECHA_CORTE]
test = df[df["Fecha"] >= FECHA_CORTE]

# 2. Los dos conjuntos de variables que queremos comparar
sin_lag = ["Producto_Codificado", "Mes", "Día_Semana", "Es_Fin_De_Semana", "Es_Feriado", "Es_Evento_Festivo"]
con_lag = sin_lag + ["Venta_Semana_Anterior"]


# 3. Los dos modelos (Ridge con one-hot, igual que en el paso 1)
def crear_ridge():
    preprocesador = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), ["Producto_Codificado", "Mes", "Día_Semana"]),
        remainder="passthrough",
    )
    return make_pipeline(preprocesador, Ridge(alpha=1.0))


def crear_xgboost():
    return XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42)


# 4. Probar las 4 combinaciones: cada modelo, con y sin la venta de hace 7 días
print("\nModelo     Variables   MAE")
print("-" * 30)
for nombre, crear_modelo in [("Ridge", crear_ridge), ("XGBoost", crear_xgboost)]:
    for etiqueta, columnas in [("sin lag", sin_lag), ("con lag", con_lag)]:
        modelo = crear_modelo().fit(train[columnas], train["Cantidad_Vendida"])
        pred = np.clip(modelo.predict(test[columnas]), 0, None).round()
        mae = mean_absolute_error(test["Cantidad_Vendida"], pred)
        print(f"{nombre:<10} {etiqueta:<11} {mae:.2f}")