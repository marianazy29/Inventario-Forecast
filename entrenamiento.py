import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBRegressor

from app.settings import DATA_DIR, ENCODER_PATH, METRICS_PATH, MODEL_PATH

FECHA_CORTE = "2026-01-01"  # mismo corte que separacion_temporal.py

# 1. CARGAR TRAIN + TEST
df_train = pd.read_csv(os.path.join(DATA_DIR, "train_datos.csv"), parse_dates=["Fecha"])
df_test = pd.read_csv(os.path.join(DATA_DIR, "test_datos.csv"), parse_dates=["Fecha"])
df = pd.concat([df_train, df_test]).sort_values(["ProductoId", "Fecha"]).reset_index(drop=True)

# 2. LAG DE 7 DÍAS sobre la serie COMPLETA de cada producto
#    (así los primeros 7 días de 2026 usan las ventas reales de fines de 2025 y no se pierden)
df["Venta_Semana_Anterior"] = df.groupby("ProductoId")["Cantidad_Vendida"].shift(7)
df = df.dropna(subset=["Venta_Semana_Anterior"]).copy()

# 3. CODIFICAR PRODUCTOS (con TODOS los productos: ya no falla si uno solo aparece en test)
le = LabelEncoder()
df["Producto_Codificado"] = le.fit_transform(df["ProductoId"])

columnas_calendario = ["Producto_Codificado", "Mes", "Día_Semana", "Es_Fin_De_Semana", "Es_Feriado", "Es_Evento_Festivo"]
columnas_modelo = columnas_calendario + ["Venta_Semana_Anterior"]

train = df[df["Fecha"] < FECHA_CORTE]
test = df[df["Fecha"] >= FECHA_CORTE]
y_train, y_test = train["Cantidad_Vendida"], test["Cantidad_Vendida"]

PARAMS_XGB = dict(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42)

# 4. EVALUACIÓN HONESTA: entrenar con el pasado (<2026) y examinar en 2026
print("--- Entrenando XGBoost (evaluación 2022/23-2025 -> 2026) ---")
modelo_eval = XGBRegressor(**PARAMS_XGB).fit(train[columnas_modelo], y_train)
pred_xgb = np.clip(modelo_eval.predict(test[columnas_modelo]), 0, None).round()

columnas_categoricas = ["Producto_Codificado", "Mes", "Día_Semana"]

preprocesador = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
    remainder="passthrough",
)

ridge = make_pipeline(preprocesador, Ridge(alpha=1.0))
ridge.fit(train[columnas_calendario], y_train)
pred_ridge = np.clip(ridge.predict(test[columnas_calendario]), 0, None).round()

pred_base = test["Venta_Semana_Anterior"]

mae_base = mean_absolute_error(y_test, pred_base)
mae_ridge = mean_absolute_error(y_test, pred_ridge)
mae_xgb = mean_absolute_error(y_test, pred_xgb)

metricas = {
    "periodo_prueba": f"{test['Fecha'].min().date()} a {test['Fecha'].max().date()}",
    "filas_prueba": int(len(test)),
    "linea_base": {"mae": round(mae_base, 2), "mase": 1.0},
    "ridge": {"mae": round(mae_ridge, 2), "mase": round(mae_ridge / mae_base, 4)},
    "xgboost": {"mae": round(mae_xgb, 2), "mase": round(mae_xgb / mae_base, 4)},
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metricas, f, ensure_ascii=False, indent=2)

print("=" * 65)
print(f" MAE Línea Base (semana anterior): {mae_base:.2f}")
print(f" MAE Ridge (solo calendario):      {mae_ridge:.2f}  (relativo {mae_ridge / mae_base:.4f})")
print(f" MAE XGBoost:                      {mae_xgb:.2f}  (relativo {mae_xgb / mae_base:.4f})")
print("=" * 65)

# 5. MODELO FINAL PARA PRODUCCIÓN: se reentrena con TODOS los datos (incluye 2026),
#    para que las predicciones futuras conozcan el comportamiento más reciente.
print("--- Reentrenando el modelo final con todos los datos ---")
modelo_final = XGBRegressor(**PARAMS_XGB).fit(df[columnas_modelo], df["Cantidad_Vendida"])

joblib.dump(modelo_final, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"📦 Modelo:  {MODEL_PATH}")
print(f"📦 Encoder: {ENCODER_PATH}")
print(f"📊 Métricas: {METRICS_PATH}")
