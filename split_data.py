import pandas as pd

from app.settings import DATASET_PATH, TEST_PATH, TEST_START, TRAIN_PATH

# Chronological split (respects time order): the model is always evaluated
# on dates that come AFTER the ones it learned from.
df = pd.read_csv(DATASET_PATH, parse_dates=["date"])

train = df[df["date"] < TEST_START]   # 2023-2025 (training + validation)
test = df[df["date"] >= TEST_START]   # 2026 (final test)

train.to_csv(TRAIN_PATH, index=False)
test.to_csv(TEST_PATH, index=False)

print("=" * 70)
print("   SEPARACIÓN TEMPORAL DE LOS DATOS")
print("=" * 70)
print(f" Entrenamiento: {train['date'].min().date()} a {train['date'].max().date()} ({len(train)} filas)")
print(f" Prueba final:  {test['date'].min().date()} a {test['date'].max().date()} ({len(test)} filas)")
print("=" * 70)
