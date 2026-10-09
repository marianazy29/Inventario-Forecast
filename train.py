import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from app.modeling import (
    add_history_features, apply_interval, compute_mase, create_ridge, create_xgboost, encode_products,
    fit_interval, mase_scale, predict_units, weekly_totals,
)
from app.settings import (
    ENCODER_PATH, INTERVAL_BINS, INTERVAL_HIGH_QUANTILE, INTERVAL_LOW_QUANTILE, INTERVAL_PATH, METRICS_PATH,
    MODEL_FEATURES, MODEL_PATH, TEST_PATH, TEST_START, TRAIN_PATH, VALIDATION_START, WEEKLY_INTERVAL_BINS,
)

# FINAL TEST (2026) AND PRODUCTION MODEL.
# The model and the metrics were already chosen in select_model.py (validation 2025).
# Here NO decisions are made: the test is measured only once.
# Primary metric: MASE (every product weighs the same). Secondary: MAE (units).

# 1. Load train + test. The history features are computed over the FULL series of
#    each product, so the first days of 2026 use the real sales of the end of 2025.
df = pd.concat([
    pd.read_csv(TRAIN_PATH, parse_dates=["date"]),
    pd.read_csv(TEST_PATH, parse_dates=["date"]),
])
df = add_history_features(df)
df, encoder = encode_products(df)

train = df[df["date"] < TEST_START]   # 2023-2025
test = df[df["date"] >= TEST_START]   # 2026
y_train, y_test = train["quantity_sold"], test["quantity_sold"]

# 2. Prediction interval, learned on VALIDATION (2025) with a model that did not see 2025.
#    The test (2026) is not used to build it, only to check it afterwards.
print("--- Calculando el rango de confianza con los datos de validación (2025) ---")
print("  - Entrenando XGBoost con 2023-2024...", flush=True)
before_validation = train[train["date"] < VALIDATION_START]
validation = train[train["date"] >= VALIDATION_START]
calibration_model = create_xgboost().fit(before_validation[MODEL_FEATURES], before_validation["quantity_sold"])
calibration_prediction = calibration_model.predict(validation[MODEL_FEATURES]).clip(min=0)
interval_table = fit_interval(
    validation["quantity_sold"], calibration_prediction, INTERVAL_BINS, INTERVAL_LOW_QUANTILE, INTERVAL_HIGH_QUANTILE,
)
# Same idea for 7-day totals: good and bad days compensate, so the weekly range is narrower
validation_weeks = weekly_totals(validation, calibration_prediction)
weekly_interval_table = fit_interval(
    validation_weeks["actual"], validation_weeks["predicted"], WEEKLY_INTERVAL_BINS,
    INTERVAL_LOW_QUANTILE, INTERVAL_HIGH_QUANTILE,
)

print("--- Evaluación final en 2026 ---")
scale = mase_scale(train)  # MASE denominator: training data only

# 3. Chosen model: XGBoost with the best configuration found in validation
print("  - Entrenando XGBoost...", flush=True)
xgb_model = create_xgboost().fit(train[MODEL_FEATURES], y_train)
xgb_prediction = predict_units(xgb_model, test[MODEL_FEATURES])

# Ridge is kept only as a reference
print("  - Entrenando Ridge...", flush=True)
ridge_model = create_ridge().fit(train[MODEL_FEATURES], y_train)
ridge_prediction = predict_units(ridge_model, test[MODEL_FEATURES])

baseline_prediction = test["sales_last_week"]
baseline_mae = mean_absolute_error(y_test, baseline_prediction)

# How often the real sales fell inside the range (should be close to 80%)
xgb_raw_prediction = xgb_model.predict(test[MODEL_FEATURES]).clip(min=0)
low, high = apply_interval(xgb_raw_prediction, interval_table)
interval_coverage = float(np.mean((y_test.to_numpy() >= low) & (y_test.to_numpy() <= high)))

# Weekly results: error of the 7-day totals and coverage of the weekly range
test_weeks = weekly_totals(test, xgb_raw_prediction)
baseline_weeks = weekly_totals(test, baseline_prediction)
week_low, week_high = apply_interval(test_weeks["predicted"], weekly_interval_table)
weekly = {
    "weeks": int(len(test_weeks)),
    "xgboost_error_pct": round(100 * (test_weeks["predicted"] - test_weeks["actual"]).abs().sum() / test_weeks["actual"].sum(), 1),
    "baseline_error_pct": round(100 * (baseline_weeks["predicted"] - baseline_weeks["actual"]).abs().sum() / baseline_weeks["actual"].sum(), 1),
    "interval_coverage": round(float(np.mean((test_weeks["actual"] >= week_low) & (test_weeks["actual"] <= week_high))), 4),
}


def summarize(prediction) -> dict:
    """The three numbers reported for each model."""
    mae = mean_absolute_error(y_test, prediction)
    return {
        "mae": round(mae, 2),
        "relative_mae": round(mae / baseline_mae, 4),
        "mase": round(compute_mase(test, prediction, scale), 4),
    }


metrics = {
    "test_period": f"{test['date'].min().date()} a {test['date'].max().date()}",
    "test_rows": int(len(test)),
    "chosen_model": "xgboost",
    "primary_metric": "mase",
    "baseline": summarize(baseline_prediction),
    "ridge": summarize(ridge_prediction),
    "xgboost": summarize(xgb_prediction),
    "interval_target": round(INTERVAL_HIGH_QUANTILE - INTERVAL_LOW_QUANTILE, 2),
    "interval_coverage": round(interval_coverage, 4),
    "weekly": weekly,
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
with open(INTERVAL_PATH, "w", encoding="utf-8") as f:
    json.dump({"daily": interval_table, "weekly": weekly_interval_table}, f, ensure_ascii=False, indent=2)

print("=" * 70)
print(f"{'Modelo':<30} {'MAE':>8} {'MAE relativo':>14} {'MASE':>10}")
for label, key in [("Línea base", "baseline"), ("XGBoost (ELEGIDO)", "xgboost"), ("Ridge con historial (ref.)", "ridge")]:
    m = metrics[key]
    print(f"{label:<30} {m['mae']:>8.2f} {m['relative_mae']:>14.4f} {m['mase']:>10.4f}")
print("-" * 70)
print(f"Rango de confianza del {metrics['interval_target']:.0%}: la venta real cayó dentro "
      f"en el {interval_coverage:.1%} de los casos de 2026.")
print(f"Pronóstico semanal (total de 7 días por producto): error del {weekly['xgboost_error_pct']} % "
      f"(línea base: {weekly['baseline_error_pct']} %); el rango semanal acertó en el {weekly['interval_coverage']:.1%} de las semanas.")
print("=" * 70)

# 4. Production model: XGBoost retrained with ALL the data (2023-2026)
print("--- Reentrenando el modelo elegido con todos los datos ---", flush=True)
final_model = create_xgboost().fit(df[MODEL_FEATURES], df["quantity_sold"])

joblib.dump(final_model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)
print(f"Modelo:  {MODEL_PATH}")
print(f"Codificador: {ENCODER_PATH}")
print(f"Métricas: {METRICS_PATH}")
print(f"Rango de confianza: {INTERVAL_PATH}")
