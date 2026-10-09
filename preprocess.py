import pandas as pd

from app.settings import DATASET_PATH, SALES_PATH
from app.tarija_calendar import calendar_features

# 1. Load the sales created by generate_data.py
df = pd.read_csv(SALES_PATH, parse_dates=["date"])
print(f"Archivo cargado: {len(df)} filas")

# 2. Tarija calendar features (defined in app/tarija_calendar.py):
#    year, month, day_of_week, is_weekend, is_holiday, is_carnival, is_festive_event
calendar = calendar_features(df["date"])
df = pd.concat([df.reset_index(drop=True), calendar], axis=1)

# 3. Save the processed dataset
df.to_csv(DATASET_PATH, index=False)
print(f"Dataset procesado guardado en: {DATASET_PATH}")
print(f"Días de Carnaval: {df.loc[df['is_carnival'] == 1, 'date'].dt.date.nunique()}")
print(f"Días festivos: {df.loc[df['is_festive_event'] == 1, 'date'].dt.date.nunique()}")
print(f"Feriados: {df.loc[df['is_holiday'] == 1, 'date'].dt.date.nunique()}")
