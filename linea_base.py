import os
import pandas as pd
import numpy as np
# Importamos las rutas centralizadas desde la configuración de la app
from app.settings import ARTIFACTS_DIR, DATA_DIR

# 1. Cargamos el dataset inteligente directo desde la carpeta DATA
ruta_entrada = os.path.join(DATA_DIR, "dataset_licoreria_inteligente.csv")
df = pd.read_csv(ruta_entrada)
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Nos aseguramos de ordenar cronológicamente por producto
df = df.sort_values(by=['ProductoId', 'Fecha']).reset_index(drop=True)

# 2. CALCULAR EL RIVAL PRINCIPAL: Mismo día de la semana anterior (7 días atrás)
df['Prediccion_Linea_Base'] = df.groupby('ProductoId')['Cantidad_Vendida'].shift(7)

# 3. LÍNEA BASE INGENUA ABSOLUTA (Denominador del MASE)
# Es la diferencia diaria consecutiva (lo vendido hoy vs ayer)
df['Error_Iniciatico_Ayer'] = df.groupby('ProductoId')['Cantidad_Vendida'].shift(1)

# Eliminamos filas con valores vacíos originados por los desplazamientos temporales (NaN)
df_evaluacion = df.dropna(subset=['Prediccion_Linea_Base', 'Error_Iniciatico_Ayer']).copy()

# 4. CÁLCULO DE ERRORES ABSOLUTOS
df_evaluacion['Error_Abs_Linea_Base'] = np.abs(df_evaluacion['Cantidad_Vendida'] - df_evaluacion['Prediccion_Linea_Base'])
df_evaluacion['Error_Abs_Ayer'] = np.abs(df_evaluacion['Cantidad_Vendida'] - df_evaluacion['Error_Iniciatico_Ayer'])

# 5. OBTENER VOLUMEN, MAE Y MASE POR PRODUCTO DIRECTO EN CONSOLA
print("\n" + "="*95)
print("   VISUALIZACIÓN DE ESCALAS DE VENTAS VS MÉTRICAS DE LÍNEA BASE (MAE Y MASE) ")
print("="*95)

productos_metricas = []

for prod_id, group in df_evaluacion.groupby('ProductoId'):
    volumen_total = group['Cantidad_Vendida'].sum()
    venta_promedio = group['Cantidad_Vendida'].mean()
    mae_prod_lb = group['Error_Abs_Linea_Base'].mean()
    mae_prod_ayer = group['Error_Abs_Ayer'].mean()
    
    # El MASE es la división de ambos errores promedio
    mase_prod = mae_prod_lb / (mae_prod_ayer if mae_prod_ayer > 0 else 1)
    
    nombre_prod = group['NombreProducto'].iloc[0]
    
    productos_metricas.append({
        "ProductoId": prod_id,
        "NombreProducto": nombre_prod,
        "Volumen_Total_Vendido": int(volumen_total),
        "Venta_Promedio_Diaria": round(venta_promedio, 2),
        "MAE_Linea_Base": round(mae_prod_lb, 2),
        "MASE_Escalado": round(mase_prod, 4)
    })

df_reporte = pd.DataFrame(productos_metricas)

# Forzamos a pandas a mostrar todas las columnas limpias en consola
with pd.option_context('display.max_columns', None, 'display.width', 1000):
    print(df_reporte.head(15))

# 6. EXPORTAR EL REPORTE LIMPIO DIRECTO A LA CARPETA ARTIFACTS
ruta_reporte = os.path.join(ARTIFACTS_DIR, "reporte_escalas_mase.csv")
df_reporte.to_csv(ruta_reporte, index=False)

# CALCULO DE MÉTRICAS GLOBALES FINALES
mae_global = df_evaluacion['Error_Abs_Linea_Base'].mean()

# El MASE Global correcto es el promedio de los MASE individuales de cada producto
mase_global_promedio = df_reporte['MASE_Escalado'].mean()

print("="*95)
print(f" 📉 MAE GLOBAL DE LÍNEA BASE TRADICIONAL:         {mae_global:.2f} unidades.")
print(f" 🎯 MASE GLOBAL PROMEDIO DE LA TIENDA (Escalado): {mase_global_promedio:.4f}")
print("="*95 + "\n")
