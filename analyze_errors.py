import numpy as np
import pandas as pd

from app.modeling import add_history_features, create_xgboost, encode_products, mase_scale, predict_units
from app.settings import ERRORS_PATH, MODEL_FEATURES, TEST_PATH, TEST_START, TRAIN_PATH

# ERROR ANALYSIS of the chosen model (XGBoost) on the final test (2026).
# errors are only looked at here, to understand them. The model is not changed based on what we see, because that would be tuning the model on the test.

DAY_NAMES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# Spanish labels for the printed tables (the person reading the output speaks Spanish)
DISPLAY_LABELS = {
    "rows": "Filas", "avg_sales": "Venta_Promedio", "mae": "MAE", "bias": "Sesgo", "error_pct": "Error_%",
    "volume_group": "Grupo_Volumen", "category": "Categoría", "day_name": "Día", "day_type": "Tipo_Día",
}

# 1. Same data, model and configuration as train.py
df = pd.concat([
    pd.read_csv(TRAIN_PATH, parse_dates=["date"]),
    pd.read_csv(TEST_PATH, parse_dates=["date"]),
])
df = add_history_features(df)
df, _ = encode_products(df)

train = df[df["date"] < TEST_START]
test = df[df["date"] >= TEST_START].copy()

model = create_xgboost().fit(train[MODEL_FEATURES], train["quantity_sold"])
test["prediction"] = predict_units(model, test[MODEL_FEATURES])

# 2. Error columns
#    error = prediction - actual. Positive: the model overshot. Negative: it fell short.
test["error"] = test["prediction"] - test["quantity_sold"]
test["abs_error"] = test["error"].abs()

scale = mase_scale(train)

# Volume group: each product is classified by its average sales in training
avg_sales_by_product = train.groupby("product_id")["quantity_sold"].mean()
volume_group = pd.qcut(avg_sales_by_product, 3, labels=["1. Venta baja", "2. Venta media", "3. Venta alta"])
test["volume_group"] = test["product_id"].map(volume_group)

test["day_type"] = np.select(
    [test["is_carnival"] == 1, test["is_festive_event"] == 1, test["is_holiday"] == 1],
    ["Carnaval", "Evento festivo", "Feriado"],
    default="Día normal",
)
test["day_name"] = test["day_of_week"].map(dict(enumerate(DAY_NAMES)))


def error_summary(column: str) -> pd.DataFrame:
    """Error table grouped by one column."""
    grouped = test.groupby(column, observed=True)
    table = grouped.agg(
        rows=("error", "size"),
        avg_sales=("quantity_sold", "mean"),
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
    )
    # Error as a percentage of what was sold
    table["error_pct"] = 100 * grouped["abs_error"].sum() / grouped["quantity_sold"].sum()
    return table.round(2)


def show(table: pd.DataFrame) -> str:
    """Table with Spanish column names, ready to print."""
    table = table.rename(columns=DISPLAY_LABELS)
    table.index.name = DISPLAY_LABELS.get(table.index.name, table.index.name)
    return table.to_string()


# 3. Errors by product (to find the worst ones)
by_product = test.groupby(["product_id", "product_name", "category"]).agg(
    avg_sales=("quantity_sold", "mean"),
    mae=("abs_error", "mean"),
    bias=("error", "mean"),
).reset_index()
by_product["mase"] = by_product["mae"] / by_product["product_id"].map(scale)
by_product = by_product.sort_values("mase", ascending=False).round(3)

# 4. Show the results
print("\n=== ERROR POR GRUPO DE VOLUMEN ===")
print(show(error_summary("volume_group")))
print("\n=== ERROR POR CATEGORÍA (ordenado por Error_%) ===")
print(show(error_summary("category").sort_values("error_pct", ascending=False)))
print("\n=== ERROR POR DÍA DE LA SEMANA ===")
print(show(error_summary("day_name").reindex(DAY_NAMES)))
print("\n=== ERROR POR TIPO DE DÍA ===")
print(show(error_summary("day_type")))
print("\n=== LOS 10 PRODUCTOS CON PEOR MASE ===")
print(by_product.head(10).to_string(index=False))
print("\n=== LOS 10 PRODUCTOS CON MEJOR MASE ===")
print(by_product.tail(10).to_string(index=False))

# 5. Save the per-product detail for the report
by_product.to_csv(ERRORS_PATH, index=False)
print(f"\nDetalle por producto guardado en: {ERRORS_PATH}")
