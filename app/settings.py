import os
from pathlib import Path

# Project root (absolute path)
BASE_DIR = Path(__file__).resolve().parent.parent

# Main folders
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Data files (in pipeline order)
CATALOG_XLSX_PATH = os.path.join(DATA_DIR, "Catalogo.xlsx")      # input edited by hand
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.csv")             # validated catalog
SALES_PATH = os.path.join(DATA_DIR, "synthetic_sales.csv")       # generated sales
BASE_SALES_PATH = os.path.join(DATA_DIR, "base_sales.csv")       # base sales assigned to each product
DATASET_PATH = os.path.join(DATA_DIR, "sales_dataset.csv")       # sales + calendar features
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")                 # 2023-2025
TEST_PATH = os.path.join(DATA_DIR, "test.csv")                   # 2026

# Model artifacts
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "product_encoder.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
ERRORS_PATH = os.path.join(ARTIFACTS_DIR, "errors_by_product.csv")
INTERVAL_PATH = os.path.join(ARTIFACTS_DIR, "prediction_interval.json")

# Temporal split
VALIDATION_START = "2025-01-01"  # validation: 2025
TEST_START = "2026-01-01"        # final test: 2026

# Model features
CALENDAR_FEATURES = [
    "product_code", "month", "day_of_week", "is_weekend",
    "is_holiday", "is_carnival", "is_festive_event",
]
# Recent sales history of each product (only data from 7 or more days before, so it
# is also available when forecasting future dates)
HISTORY_FEATURES = [
    "sales_last_week",   # sales of the same product 7 days before
    "sales_avg_4w",      # average daily sales over the 28 days that end 7 days before
]
HISTORY_DAYS = 34        # days of history needed to compute HISTORY_FEATURES
MODEL_FEATURES = CALENDAR_FEATURES + HISTORY_FEATURES
CATEGORICAL_FEATURES = ["product_code", "month", "day_of_week"]  # one-hot encoded for Ridge

# Forecast horizon: maximum days after the last real data point
MAX_HORIZON_DAYS = 366

# Profit margin per unit (Bs) by category.
DEFAULT_MARGIN = 5.0
MARGIN_BY_CATEGORY = {
    "VINOS": 25.0,
    "WHISKY": 45.0,
    "AMARULA": 20.0,
    "SODA": 3.0,
    "AGUA": 1.5,
}

# Prediction interval: range that should contain the real sales 80% of the time
INTERVAL_LOW_QUANTILE = 0.10
INTERVAL_HIGH_QUANTILE = 0.90
INTERVAL_BINS = [0, 5, 15, 40, float("inf")]  # groups of predictions by size (units)
WEEKLY_INTERVAL_BINS = [0, 35, 105, 280, float("inf")]  # same groups for 7-day totals

# Weekly forecast: number of days shown from the consulted date
FORECAST_DAYS = 7
