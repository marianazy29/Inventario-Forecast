import pandas as pd
import holidays
import os
# Importamos la ruta centralizada desde tu configuración estructurada
from app.settings import DATA_DIR

# 1. CARGAR TU ARCHIVO EXCEL LOCAL
nombre_archivo = "df_diario.xlsx"  
df = pd.read_excel(nombre_archivo)

print("¡Archivo cargado con éxito en tu máquina local!")
print(f"Columnas detectadas inicialmente: {list(df.columns)}")

# 1.5 CORREGIR CÓDIGOS DUPLICADOS: un mismo ProductoId con dos nombres distintos
#     (ej. CVZ-001 = PACEÑA LATA GRANDE y PACEÑA BOTELLA LITRO) mezcla las ventas
#     de dos productos y rompe el lag de 7 días. Al segundo nombre se le asigna un código nuevo.
nombres_por_id = df.groupby('ProductoId')['NombreProducto'].nunique()
for pid in nombres_por_id[nombres_por_id > 1].index:
    prefijo = pid.split('-')[0]
    max_num = max(int(x.split('-')[1]) for x in df['ProductoId'].unique() if x.startswith(prefijo + '-'))
    nombres = sorted(df.loc[df['ProductoId'] == pid, 'NombreProducto'].unique())
    for i, nombre in enumerate(nombres[1:], start=1):
        nuevo_id = f"{prefijo}-{max_num + i:03d}"
        df.loc[(df['ProductoId'] == pid) & (df['NombreProducto'] == nombre), 'ProductoId'] = nuevo_id
        print(f"⚠️ Código duplicado {pid}: '{nombre}' ahora es {nuevo_id}")

# 2. CONVERTIR COLUMNA FECHA
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

# 3. EXTRAER VARIABLES DE CALENDARIO BÁSICAS
df['Año'] = df['Fecha'].dt.year
df['Mes'] = df['Fecha'].dt.month
df['Día_Semana'] = df['Fecha'].dt.dayofweek  # Lunes=0, Domingo=6
df['Es_Fin_De_Semana'] = df['Fecha'].dt.dayofweek.isin([4, 5, 6]).astype(int) # Viernes, Sábado y Domingo

# 4. AGREGAR FERIADOS OFICIALES DE BOLIVIA (Con filtro estricto anti-nulos)
años_sucios = df['Año'].unique()
años_presentes = [int(a) for a in años_sucios if pd.notna(a)]
feriados_bo = holidays.Bolivia(years=años_presentes)
df['Es_Feriado'] = df['Fecha'].apply(lambda x: 1 if x in feriados_bo else 0)

# 5. AGREGAR FESTIVIDADES COMERCIALES CLAVE
festividades_comerciales = {
    (5, 27),   # Día de la Madre
    (6, 23),   # Noche de San Juan
    (9, 21),   # Día de la Primavera / Amor
    (11, 1),   # Todos Santos
    (12, 24),  # Nochebuena
    (12, 31)   # Fin de Año
}
df['Es_Evento_Festivo'] = df['Fecha'].apply(lambda x: 1 if (x.month, x.day) in festividades_comerciales else 0)

# 6. GUARDAR EL DATASET PROCESADO EN LA CARPETA DATA (Ruta Relativa Segura)
ruta_guardado = os.path.join(DATA_DIR, "dataset_licoreria_inteligente.csv")
df.to_csv(ruta_guardado, index=False)

print("\n--- ¡PROCESO COMPLETADO CON ÉXITO! ---")
print(f"Total de registros procesados: {len(df)}")
print(f"📍 Archivo guardado de forma ordenada en: {ruta_guardado}")
print("\nMuestra de las primeras 5 filas con la nueva inteligencia temporal:")
print(df.head())
