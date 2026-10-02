import os
import pandas as pd
import numpy as np
import joblib  # Necesario para exportar los artefactos finales
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
# Importamos las variables centralizadas desde la configuración de la app
from app.settings import DATA_DIR, MODEL_PATH, ENCODER_PATH

# 1. CARGAR LOS BLOQUES TEMPORALES SEPARADOS CRONOLÓGICAMENTE
df_train = pd.read_csv(os.path.join(DATA_DIR, "train_datos.csv"))
df_test = pd.read_csv(os.path.join(DATA_DIR, "test_datos.csv"))

# Aseguramos el orden cronológico estricto por producto para las variables del pasado
df_train = df_train.sort_values(by=['ProductoId', 'Fecha']).reset_index(drop=True)
df_test = df_test.sort_values(by=['ProductoId', 'Fecha']).reset_index(drop=True)

# 2. INGENIERÍA DE CARACTERÍSTICAS (Pistas del pasado)
df_train['Venta_Semana_Anterior'] = df_train.groupby('ProductoId')['Cantidad_Vendida'].shift(7)
df_test['Venta_Semana_Anterior'] = df_test.groupby('ProductoId')['Cantidad_Vendida'].shift(7)

# Definimos la predicción de la Línea Base 1 (Semana anterior)
df_test['Prediccion_LB_Tradicional'] = df_test['Venta_Semana_Anterior']

# Eliminamos registros vacíos (NaN) provocados por los desplazamientos en el tiempo
df_train = df_train.dropna(subset=['Venta_Semana_Anterior']).copy()
df_test = df_test.dropna(subset=['Venta_Semana_Anterior', 'Prediccion_LB_Tradicional']).copy()

# 3. CODIFICAR LA IDENTIDAD DE LOS PRODUCTOS (De texto a número)
le = LabelEncoder()
df_train['Producto_Codificado'] = le.fit_transform(df_train['ProductoId'])
df_test['Producto_Codificado'] = le.transform(df_test['ProductoId'])

# =====================================================================
# 📊 EVALUACIÓN 1: REGRESIÓN REGULARIZADA RIDGE (Escalón Inicial)
# =====================================================================
# Solo usa identidad y variables de calendario (tal como pidió tu docente)
columnas_calendario = ['Producto_Codificado', 'Mes', 'Día_Semana', 'Es_Fin_De_Semana', 'Es_Feriado', 'Es_Evento_Festivo']

X_train_ridge = df_train[columnas_calendario]
y_train_ridge = df_train['Cantidad_Vendida']
X_test_ridge = df_test[columnas_calendario]

modelo_ridge = Ridge(alpha=1.0, random_state=42)
modelo_ridge.fit(X_train_ridge, y_train_ridge)

pred_ridge = modelo_ridge.predict(X_test_ridge)
df_test['Prediccion_LB_Ridge'] = np.clip(pred_ridge, 0, None).round().astype(int)

# =====================================================================
# 🚀 EVALUACIÓN 2: XGBOOST REGRESSOR (Tu Modelo Avanzado)
# =====================================================================
# Usa el calendario Y ADEMÁS la memoria del pasado (Venta_Semana_Anterior)
columnas_xgboost = columnas_calendario + ['Venta_Semana_Anterior']

X_train_xgb = df_train[columnas_xgboost]
y_train_xgb = df_train['Cantidad_Vendida']
X_test_xgb = df_test[columnas_xgboost]
y_test = df_test['Cantidad_Vendida'] # El examen real para todos

modelo_xgb = XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42)
modelo_xgb.fit(X_train_xgb, y_train_xgb)

pred_xgb = modelo_xgb.predict(X_test_xgb)
df_test['Prediccion_XGBoost'] = np.clip(pred_xgb, 0, None).round().astype(int)

# =====================================================================
# 🏁 CÁLCULO DE MÉTRICAS COMPLEMENTARIAS REALES (MAE Y MASE)
# =====================================================================
mae_tradicional = mean_absolute_error(y_test, df_test['Prediccion_LB_Tradicional'])
mae_ridge = mean_absolute_error(y_test, df_test['Prediccion_LB_Ridge'])
mae_xgboost = mean_absolute_error(y_test, df_test['Prediccion_XGBoost'])

# MASE de cada enfoque matemático (utilizando el error tradicional como denominador base)
mase_ridge = mae_ridge / mae_tradicional
mase_xgboost = mae_xgboost / mae_tradicional

# =====================================================================
# 📦 EXPORTAR ARTEFACTOS DE FORMA SEGURA EN LAS RUTAS ABSOLUTAS
# =====================================================================
joblib.dump(modelo_xgb, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)

# =====================================================================
# 📺 REPORTE CONSOLIDADO FINAL EN CONSOLA (Año de Prueba 2026)
# =====================================================================
print("\n" + "="*85)
print("     🏁 TABLA COMPARATIVA FINAL DE ENFOQUES PREDICTIVOS (CONJUNTO PRUEBA 2026) ")
print("="*85)
print(f" 📑 1. Línea Base Ingenua (Semana Anterior): MAE = {mae_tradicional:.2f} unidades | MASE = 1.0000")
print(f" 📑 2. Regresión Regularizada (Ridge):       MAE = {mae_ridge:.2f} unidades | MASE = {mase_ridge:.4f}")
print(f" 🚀 3. Modelo Inteligente Definitivo (XGBoost): MAE = {mae_xgboost:.2f} unidades | MASE = {mase_xgboost:.4f}")
print("-"*85)

# Análisis rápido de ganancia
if mae_xgboost < mae_ridge and mae_xgboost < mae_tradicional:
    print(f" 🎉 ¡Conclusión validada! Tu XGBoost es el ganador absoluto.")
    print(f" Superó la propuesta lineal del docente en {mae_ridge - mae_xgboost:.2f} botellas de error promedio.")
print(f"📦 Artefactos de inferencia exportados correctamente a la carpeta 'artifacts/'.")
print("="*85 + "\n")
