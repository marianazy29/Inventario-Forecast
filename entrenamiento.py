import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
# Importamos la configuración centralizada de la aplicación
from app.settings import DATA_DIR, MODEL_PATH, ENCODER_PATH

# 1. CARGAR LOS BLOQUES TEMPORALES DESDE LA CARPETA DATA
df_train = pd.read_csv(os.path.join(DATA_DIR, "train_datos.csv"))
df_test = pd.read_csv(os.path.join(DATA_DIR, "test_datos.csv"))

# Ordenamos cronológicamente por producto para asegurar la consistencia del pasado
df_train = df_train.sort_values(by=['ProductoId', 'Fecha']).reset_index(drop=True)
df_test = df_test.sort_values(by=['ProductoId', 'Fecha']).reset_index(drop=True)

# 2. INGENIERÍA DE CARACTERÍSTICAS: Crear la pista del pasado (Lag de 7 días)
df_train['Venta_Semana_Anterior'] = df_train.groupby('ProductoId')['Cantidad_Vendida'].shift(7)
df_test['Venta_Semana_Anterior'] = df_test.groupby('ProductoId')['Cantidad_Vendida'].shift(7)

# La regla simple de la semana anterior (para calcular el MASE al final)
df_test['Prediccion_Linea_Base'] = df_test['Venta_Semana_Anterior']

# Eliminamos filas iniciales que queden vacías (NaN) por el desplazamiento temporal
df_train = df_train.dropna(subset=['Venta_Semana_Anterior']).copy()
df_test = df_test.dropna(subset=['Venta_Semana_Anterior', 'Prediccion_Linea_Base']).copy()

# 3. CODIFICAR EL ID DEL PRODUCTO (De texto a número)
le = LabelEncoder()
# Entrenamos el codificador con el pasado y transformamos ambos bloques
df_train['Producto_Codificado'] = le.fit_transform(df_train['ProductoId'])
df_test['Producto_Codificado'] = le.transform(df_test['ProductoId'])

# 4. SELECCIONAR NUEVAS COLUMNAS (¡Ahora con identidad y pasado!)
columnas_modelo = ['Producto_Codificado', 'Mes', 'Día_Semana', 'Es_Fin_De_Semana', 'Es_Feriado', 'Es_Evento_Festivo', 'Venta_Semana_Anterior']

X_train = df_train[columnas_modelo]
y_train = df_train['Cantidad_Vendida']

X_test = df_test[columnas_modelo]
y_test = df_test['Cantidad_Vendida']

# 5. ENTRENAMIENTO DEL MODELO INTELIGENTE
print("--- Iniciando Entrenamiento de XGBoost con Identidad de Producto ---")
modelo_inteligente = XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42)
modelo_inteligente.fit(X_train, y_train)
print("¡Modelo entrenado con éxito!\n")

# 6. PREDICCIÓN EN EL CONJUNTO DE PRUEBA (Año 2026)
df_test['Prediccion_ML'] = modelo_inteligente.predict(X_test)
df_test['Prediccion_ML'] = df_test['Prediccion_ML'].clip(lower=0).round().astype(int)

# 7. EVALUACIÓN DE LA COMPETENCIA (MAE y MASE)
mae_linea_base = mean_absolute_error(y_test, df_test['Prediccion_Linea_Base'])
mae_xgboost = mean_absolute_error(y_test, df_test['Prediccion_ML'])

# MASE Global: Error del modelo inteligente dividido entre el error del rival tradicional
mase_global = mae_xgboost / mae_linea_base

print("="*65)
print("       📊 REPORTE FINAL DE LA COMPETENCIA EN CONSOLA (2026) ")
print("="*65)
print(f" 📉 MAE Línea Base Tradicional:    {mae_linea_base:.2f} unidades.")
print(f" 🚀 MAE Modelo XGBoost Inteligente: {mae_xgboost:.2f} unidades.")
print("-"*65)
print(f" 🎯 MASE GLOBAL DEL MODELO (Escalado): {mase_global:.4f}")
print("-"*65)

if mase_global < 1.0:
    print(f" 🎉 ¡ÉXITO TOTAL! Tu IA le ganó a la Línea Base.")
    print(f" El MASE es menor a 1. Tu modelo es un {((1 - mase_global)*100):.1f}% más preciso que la regla tradicional.")
else:
    print(" ⚠️ La Línea Base sigue resistiendo. Necesitamos ajustar hiperparámetros.")
print("="*65 + "\n")

# 8. EXPORTAR ARTEFACTOS PARA INFERENCIA EN LAS RUTAS CONFIGURADAS (¡CORREGIDO!)
joblib.dump(modelo_inteligente, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)

print("="*65)
print(" 📦 ARTEFACTOS DE INFERENCIA GUARDADOS EXITOSAMENTE ")
print(f" - Cerebro del modelo guardado en: {MODEL_PATH}")
print(f" - Traductor de ID guardado en:    {ENCODER_PATH}")
print("="*65 + "\n")
