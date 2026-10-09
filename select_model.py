import pandas as pd
from sklearn.metrics import mean_absolute_error

from app.modeling import (
    add_history_features, compute_mase, create_ridge, create_xgboost, encode_products, mase_scale, predict_units,
)
from app.settings import CALENDAR_FEATURES, MODEL_FEATURES, TRAIN_PATH, VALIDATION_START

# MODEL SELECTION ON VALIDATION (2025).
# This script ONLY reads train.csv (2023-2025). The test file (2026) is never opened,
# so every decision made here is free of information from the final test.

# 1. Load data, add the sales history features and encode products
df = add_history_features(pd.read_csv(TRAIN_PATH, parse_dates=["date"]))
df, _ = encode_products(df)

# 2. Temporal split: training (2023-2024) and validation (2025)
train = df[df["date"] < VALIDATION_START]
validation = df[df["date"] >= VALIDATION_START]
y_train = train["quantity_sold"]
y_validation = validation["quantity_sold"]

print(f"Entrenamiento: {train['date'].min().date()} a {train['date'].max().date()} ({len(train)} filas)")
print(f"Validación:    {validation['date'].min().date()} a {validation['date'].max().date()} ({len(validation)} filas)")

scale = mase_scale(train)  # MASE denominator: computed ONCE, with training data only

# 3. Candidates: (display name, function that creates the model, features it uses)
LAST_WEEK_ONLY = CALENDAR_FEATURES + ["sales_last_week"]
candidates = [
    ("Ridge sin historial", create_ridge, CALENDAR_FEATURES),
    ("Ridge con historial", create_ridge, MODEL_FEATURES),
    ("XGBoost original (150 árboles, prof. 6)",
     lambda: create_xgboost(n_estimators=150, learning_rate=0.08, max_depth=6), MODEL_FEATURES),
    ("XGBoost grande, solo semana anterior", create_xgboost, LAST_WEEK_ONLY),
    ("XGBoost grande con historial (400, prof. 8)", create_xgboost, MODEL_FEATURES),
]

# 4. Evaluate every candidate on validation with both metrics
baseline_prediction = validation["sales_last_week"]
results = [(
    "Línea base (semana anterior)",
    mean_absolute_error(y_validation, baseline_prediction),
    compute_mase(validation, baseline_prediction, scale),
)]

print("\nEntrenando los modelos candidatos (puede tardar unos minutos)...")
for name, build_model, features in candidates:
    print(f"  - {name}...", flush=True)
    model = build_model().fit(train[features], y_train)
    prediction = predict_units(model, validation[features])
    results.append((
        name,
        mean_absolute_error(y_validation, prediction),
        compute_mase(validation, prediction, scale),
    ))

# 5. Show the table sorted by MASE (primary metric)
print("\nRESULTADOS EN VALIDACIÓN (2025)")
print("-" * 70)
print(f"{'Modelo':<45} {'MAE':>8} {'MASE':>10}")
for name, mae, mase in sorted(results, key=lambda r: r[2]):
    print(f"{name:<45} {mae:>8.2f} {mase:>10.4f}")
