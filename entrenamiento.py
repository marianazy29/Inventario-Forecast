import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
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
df["Venta_Semana_Anterior"] = df.groupby("ProductoId")["Cantidad_Vendida"].shift(7)
df = df.dropna(subset=["Venta_Semana_Anterior"]).copy()

# 3. CODIFICAR PRODUCTOS
le = LabelEncoder()
df["Producto_Codificado"] = le.fit_transform(df["ProductoId"])

columnas_calendario = ["Producto_Codificado", "Mes", "Día_Semana", "Es_Fin_De_Semana", "Es_Feriado", "Es_Evento_Festivo"]
columnas_modelo = columnas_calendario + ["Venta_Semana_Anterior"]

train = df[df["Fecha"] < FECHA_CORTE]  # 2023-2025 (entrenamiento + validación)
test = df[df["Fecha"] >= FECHA_CORTE]  # 2026 (examen final)
y_train, y_test = train["Cantidad_Vendida"], test["Cantidad_Vendida"]


def crear_ridge():
    preprocesador = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), ["Producto_Codificado", "Mes", "Día_Semana"]),
        remainder="passthrough",
    )
    return make_pipeline(preprocesador, Ridge(alpha=1.0))


def crear_xgboost():
    return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=8, random_state=42)


def calcular_escala_mase(datos_entrenamiento):
    """Error de la línea base ingenua (semana anterior) DENTRO del entrenamiento, por producto."""
    error_ingenuo = (datos_entrenamiento["Cantidad_Vendida"] - datos_entrenamiento["Venta_Semana_Anterior"]).abs()
    return error_ingenuo.groupby(datos_entrenamiento["ProductoId"]).mean()


def calcular_mase(datos_evaluacion, prediccion, escala):
    """MASE de cada producto (su MAE dividido por su escala) y luego el promedio de todos."""
    error = (datos_evaluacion["Cantidad_Vendida"] - prediccion).abs()
    mae_por_producto = error.groupby(datos_evaluacion["ProductoId"]).mean()
    mase_por_producto = mae_por_producto / escala.loc[mae_por_producto.index]
    return mase_por_producto.mean()


# 4. EXAMEN FINAL: los modelos y las métricas ya fueron elegidos en seleccion_modelo.py.
#    Métrica principal: MASE (cada producto pesa igual). Complementaria: MAE (unidades).
print("--- Evaluación final en 2026 ---")
escala = calcular_escala_mase(train)  # vara del MASE: solo con datos de entrenamiento

# Modelo elegido: XGBoost con la mejor configuración encontrada en validación
modelo_xgb = crear_xgboost().fit(train[columnas_modelo], y_train)
pred_xgb = np.clip(modelo_xgb.predict(test[columnas_modelo]), 0, None).round()

# Ridge se mantiene solo como referencia
modelo_ridge = crear_ridge().fit(train[columnas_modelo], y_train)
pred_ridge = np.clip(modelo_ridge.predict(test[columnas_modelo]), 0, None).round()

pred_base = test["Venta_Semana_Anterior"]

mae_base = mean_absolute_error(y_test, pred_base)


def resumir(prediccion):
    """Las tres cifras que reportamos para cada modelo."""
    mae = mean_absolute_error(y_test, prediccion)
    return {
        "mae": round(mae, 2),
        "mae_relativo": round(mae / mae_base, 4),
        "mase": round(calcular_mase(test, prediccion, escala), 4),
    }


metricas = {
    "periodo_prueba": f"{test['Fecha'].min().date()} a {test['Fecha'].max().date()}",
    "filas_prueba": int(len(test)),
    "modelo_elegido": "xgboost",
    "metrica_principal": "mase",
    "linea_base": resumir(pred_base),
    "ridge": resumir(pred_ridge),
    "xgboost": resumir(pred_xgb),
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metricas, f, ensure_ascii=False, indent=2)

print("=" * 70)
print(f"{'Modelo':<30} {'MAE':>8} {'MAE relativo':>14} {'MASE':>10}")
for nombre, clave in [("Línea base", "linea_base"), ("XGBoost (ELEGIDO)", "xgboost"), ("Ridge con lag (referencia)", "ridge")]:
    m = metricas[clave]
    print(f"{nombre:<30} {m['mae']:>8.2f} {m['mae_relativo']:>14.4f} {m['mase']:>10.4f}")
print("=" * 70)

# 5. MODELO FINAL PARA PRODUCCIÓN: XGBoost, reentrenado con TODOS los datos (2023-2026)
print("--- Reentrenando el modelo elegido con todos los datos ---")
modelo_final = crear_xgboost().fit(df[columnas_modelo], df["Cantidad_Vendida"])

joblib.dump(modelo_final, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"📦 Modelo:  {MODEL_PATH}")
print(f"📦 Encoder: {ENCODER_PATH}")
print(f"📊 Métricas: {METRICS_PATH}")