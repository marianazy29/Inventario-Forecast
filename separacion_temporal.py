import os
import pandas as pd
# Importamos la ruta centralizada desde la configuración de la app
from app.settings import DATA_DIR

# 1. Cargamos el dataset inteligente completo desde la carpeta DATA
ruta_entrada = os.path.join(DATA_DIR, "dataset_licoreria_inteligente.csv")
df = pd.read_csv(ruta_entrada)
df['Fecha'] = pd.to_datetime(df['Fecha'])

# 2. SEPARACIÓN CRONOLÓGICA AJUSTADA (Respetando el orden temporal)
# Bloque de Entrenamiento: Datos desde 2022 hasta el 31 de Diciembre de 2025
df_train = df[df['Fecha'] <= '2025-12-31'].copy()

# Bloque de Prueba Final: Todo el año 2026 (El presente/futuro en evaluación)
df_test = df[df['Fecha'] >= '2026-01-01'].copy()

# 3. REPORTE EN CONSOLA
print("\n" + "="*70)
print("   REPORTE ACTUALIZADO DE VALIDACIÓN TEMPORAL (2022 - 2026) ")
print("="*70)
print(f" Inicio del Dataset: {df['Fecha'].min().strftime('%Y-%m-%d')}")
print(f" Fin del Dataset:    {df['Fecha'].max().strftime('%Y-%m-%d')}")
print("-"*70)
print(f" 📂 CONJUNTO DE ENTRENAMIENTO (Train):")
print(f"   - Período:  Desde {df_train['Fecha'].min().strftime('%Y-%m-%d')} hasta {df_train['Fecha'].max().strftime('%Y-%m-%d')}")
print(f"   - Tamaño:   {len(df_train)} filas")
print("-"*70)
print(f" 🎯 CONJUNTO DE PRUEBA (Test - Año 2026):")
print(f"   - Período:  Desde {df_test['Fecha'].min().strftime('%Y-%m-%d')} hasta {df_test['Fecha'].max().strftime('%Y-%m-%d')}")
print(f"   - Tamaño:   {len(df_test)} filas")
print("="*70)

# 4. GUARDAR LOS NUEVOS BLOQUES EXCLUSIVAMENTE EN LA CARPETA DATA
ruta_train = os.path.join(DATA_DIR, "train_datos.csv")
ruta_test = os.path.join(DATA_DIR, "test_datos.csv")

df_train.to_csv(ruta_train, index=False)
df_test.to_csv(ruta_test, index=False)

print("\n¡Nuevos bloques guardados con éxito en la carpeta 'data/' para el modelado!")
