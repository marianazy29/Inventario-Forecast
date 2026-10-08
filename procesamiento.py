import os

import pandas as pd

from app.calendario import variables_calendario
from app.settings import DATA_DIR

# 1. CARGAR LAS VENTAS GENERADAS POR generar_datos.py
df = pd.read_csv(os.path.join(DATA_DIR, "ventas_sinteticas.csv"), parse_dates=["Fecha"])
print(f"Archivo cargado: {len(df)} filas")

# 2. VARIABLES DE CALENDARIO DE TARIJA (definidas en app/calendario.py):
#    Año, Mes, Día_Semana, Es_Fin_De_Semana, Es_Feriado, Es_Carnaval, Es_Evento_Festivo
calendario = variables_calendario(df["Fecha"])
df = pd.concat([df.reset_index(drop=True), calendario], axis=1)

# 3. GUARDAR EL DATASET PROCESADO
ruta_guardado = os.path.join(DATA_DIR, "dataset_licoreria_inteligente.csv")
df.to_csv(ruta_guardado, index=False)
print(f"Dataset procesado guardado en: {ruta_guardado}")
print(f"Días de Carnaval: {df.loc[df['Es_Carnaval'] == 1, 'Fecha'].dt.date.nunique()}")
print(f"Días festivos: {df.loc[df['Es_Evento_Festivo'] == 1, 'Fecha'].dt.date.nunique()}")
print(f"Feriados: {df.loc[df['Es_Feriado'] == 1, 'Fecha'].dt.date.nunique()}")