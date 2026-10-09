import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBRegressor

from app.settings import CATEGORICAL_FEATURES

# =====================================================================
# SHARED MODELING FUNCTIONS
# Used by select_model.py, train.py and analyze_errors.py, so that all
# of them prepare the data and build the models in exactly the same way.
# =====================================================================


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the recent sales history of each product and drop the first days, which have no history.

    - sales_last_week: sales of the same product 7 days before.
    - sales_avg_4w: average daily sales over the 28 days that end 7 days before.
      It tells the model the recent sales LEVEL, so it can follow the growth trend.
    Both use only data from 7 or more days before, so they can also be computed
    when forecasting future dates.
    """
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    sales = df.groupby("product_id")["quantity_sold"]
    df["sales_last_week"] = sales.shift(7)
    df["sales_avg_4w"] = sales.transform(lambda s: s.shift(7).rolling(28).mean())
    return df.dropna(subset=["sales_last_week", "sales_avg_4w"]).copy()


def encode_products(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Turn product_id (text) into product_code (number)."""
    encoder = LabelEncoder()
    df["product_code"] = encoder.fit_transform(df["product_id"])
    return df, encoder


def create_ridge():
    """Ridge with one-hot encoding of product, month and day of week."""
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        remainder="passthrough",
    )
    return make_pipeline(preprocessor, Ridge(alpha=1.0))


def create_xgboost(n_estimators=400, learning_rate=0.05, max_depth=8):
    """XGBoost. Defaults are the best configuration found in validation (2025)."""
    return XGBRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42,
    )


def predict_units(model, X: pd.DataFrame) -> np.ndarray:
    """Predictions as whole, non-negative units."""
    return np.clip(model.predict(X), 0, None).round()


def mase_scale(train: pd.DataFrame) -> pd.Series:
    """MASE denominator: error of the naive forecast (last week) on TRAINING data, per product."""
    naive_error = (train["quantity_sold"] - train["sales_last_week"]).abs()
    return naive_error.groupby(train["product_id"]).mean()


def compute_mase(data: pd.DataFrame, prediction, scale: pd.Series) -> float:
    """MASE of each product (its MAE divided by its scale), averaged over all products."""
    error = (data["quantity_sold"] - prediction).abs()
    mae_by_product = error.groupby(data["product_id"]).mean()
    return (mae_by_product / scale.loc[mae_by_product.index]).mean()


def fit_interval(actual: pd.Series, prediction, bins, low_q: float, high_q: float) -> list[dict]:
    """Learn how far real sales usually are from the prediction, for each size of prediction.

    For each group of predictions (for example, 15 to 40 units) it stores the ratio
    real / predicted below which 10% and 90% of the cases fall. Multiplying a new
    prediction by those ratios gives a range that contains the real sales ~80% of the time.
    """
    prediction = np.asarray(prediction, dtype=float)
    base = np.maximum(prediction, 1.0)
    ratio = np.asarray(actual, dtype=float) / base
    groups = pd.cut(prediction, bins, right=False)
    table = []
    for group in groups.categories:
        in_group = groups == group
        table.append({
            "from": float(group.left),
            "to": float(group.right),
            "low_ratio": round(float(np.quantile(ratio[in_group], low_q)), 4),
            "high_ratio": round(float(np.quantile(ratio[in_group], high_q)), 4),
        })
    return table


def apply_interval(prediction, table: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Turn predictions into (low, high) ranges using the table from fit_interval."""
    prediction = np.atleast_1d(np.asarray(prediction, dtype=float))
    base = np.maximum(prediction, 1.0)
    low = np.empty_like(prediction)
    high = np.empty_like(prediction)
    for row in table:
        in_group = (prediction >= row["from"]) & (prediction < row["to"])
        low[in_group] = base[in_group] * row["low_ratio"]
        high[in_group] = base[in_group] * row["high_ratio"]
    return np.round(low), np.round(high)


def weekly_totals(data: pd.DataFrame, prediction) -> pd.DataFrame:
    """Add up real and predicted sales per product and week (Monday to Sunday).

    Only complete weeks (7 days) are kept, so all totals are comparable.
    """
    weekly = pd.DataFrame({
        "product_id": data["product_id"].to_numpy(),
        "week": data["date"].dt.to_period("W").to_numpy(),
        "actual": data["quantity_sold"].to_numpy(),
        "predicted": np.asarray(prediction, dtype=float),
        "days": 1,
    }).groupby(["product_id", "week"]).sum()
    return weekly[weekly["days"] == 7]
