import json
import os
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.modeling import apply_interval
from app.schemas import PredictionRequest, RecommendationRequest, WeekForecastRequest
from app.settings import (
    DATASET_PATH,
    DEFAULT_MARGIN,
    ENCODER_PATH,
    FORECAST_DAYS,
    FRONTEND_DIR,
    HISTORY_DAYS,
    INTERVAL_PATH,
    MARGIN_BY_CATEGORY,
    MAX_HORIZON_DAYS,
    METRICS_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
)
from app.tarija_calendar import calendar_features

app = FastAPI(
    title="Licorería - Pronóstico de ventas y recomendaciones",
    description="API que pronostica la demanda diaria por producto y recomienda productos a promocionar.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# LOAD ONCE AT STARTUP
# =====================================================================
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise RuntimeError("No se encontraron los .pkl en 'artifacts/'. Ejecuta: python run.py")
if not os.path.exists(DATASET_PATH):
    raise RuntimeError("No existe data/sales_dataset.csv. Ejecuta: python run.py")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# Prediction interval tables (learned in train.py), one for single days and one for
# 7-day totals. Without them, only the point prediction is shown.
_INTERVALS = {}
if os.path.exists(INTERVAL_PATH):
    with open(INTERVAL_PATH, encoding="utf-8") as f:
        _INTERVALS = json.load(f)

_sales_history = pd.read_csv(
    DATASET_PATH,
    usecols=["date", "product_id", "product_name", "category", "quantity_sold"],
    parse_dates=["date"],
)

_CATALOG = (
    _sales_history.drop_duplicates("product_id")[["product_id", "product_name", "category"]]
    .sort_values("product_name")
    .reset_index(drop=True)
)
_PRODUCT_CODE = {pid: int(code) for pid, code in zip(encoder.classes_, encoder.transform(encoder.classes_))}

# Date x product matrix (days without records = 0 sales)
_SALES = (
    _sales_history.pivot_table(index="date", columns="product_id", values="quantity_sold", aggfunc="sum")
    .asfreq("D")
    .fillna(0.0)
)
_LAST_DATE = _SALES.index[-1]


def _normalize(text) -> str:
    """Lowercase, without accents or symbols: "JACK DANIEL'S" -> 'jack daniel s'."""
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


_CATALOG["search_name"] = _CATALOG["product_name"].map(_normalize)
_CATALOG["search_compact"] = _CATALOG["search_name"].str.replace(" ", "", regex=False)
_CATALOG["search_id"] = _CATALOG["product_id"].str.lower()
_CATALOG["search_category"] = _CATALOG["category"].map(_normalize)


# =====================================================================
# HELPERS
# =====================================================================
def _product_dict(row) -> dict:
    return {"product_id": row["product_id"], "name": row["product_name"], "category": row["category"]}


def _search(query: str, limit: int = 10) -> list[dict]:
    """Search by name (or code / category). Every word of the query must appear."""
    normalized = _normalize(query)
    tokens = normalized.split()
    if not tokens:
        return []
    matches = []
    for row in _CATALOG.itertuples(index=False):
        searchable = f"{row.search_name} {row.search_id} {row.search_category}"
        if all(t in searchable or t in row.search_compact for t in tokens):
            if row.search_name == normalized:
                priority = 0
            elif row.search_name.startswith(normalized):
                priority = 1
            elif any(word.startswith(tokens[0]) for word in row.search_name.split()):
                priority = 2
            else:
                priority = 3
            matches.append((priority, row.product_name, row))
    matches.sort(key=lambda m: (m[0], m[1]))
    return [
        {"product_id": r.product_id, "name": r.product_name, "category": r.category}
        for _, _, r in matches[:limit]
    ]


def _resolve_product(product_id: str | None, name: str | None) -> str:
    """Return a valid product_id from the code or the name."""
    pid = (product_id or "").strip().upper()
    if pid and pid in _PRODUCT_CODE:
        return pid
    text = (name or product_id or "").strip()
    exact = _CATALOG[_CATALOG["search_name"] == _normalize(text)]
    if len(exact) == 1:
        return exact.iloc[0]["product_id"]
    found = _search(text, limit=5)
    if len(found) == 1:
        return found[0]["product_id"]
    if not found:
        raise HTTPException(404, f"No se encontró ningún producto parecido a '{text}'.")
    options = "; ".join(f"{p['name']} ({p['product_id']})" for p in found)
    raise HTTPException(409, f"'{text}' es ambiguo. ¿Quisiste decir: {options}?")


def _parse_date(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).normalize()


def _calendar(date: pd.Timestamp) -> dict:
    """Calendar features for one date (same rules as training, see app/tarija_calendar.py)."""
    row = calendar_features(pd.Series([date])).iloc[0]
    return {column: int(row[column]) for column in MODEL_FEATURES if column in row.index}


def _predict(date: pd.Timestamp, codes: np.ndarray, recent_sales: list, manual_lag: float | None = None) -> np.ndarray:
    """One model step for several products at once. Returns floats >= 0.

    recent_sales: list with the sales of the previous days (the last element is the day
    before `date`); each element is an array with one value per product.
    """
    sales_last_week = recent_sales[-7]                                    # 7 days before
    sales_avg_4w = np.mean(np.array(recent_sales[-HISTORY_DAYS:-6]), axis=0)  # days -34 to -7
    if manual_lag is not None:
        sales_last_week = np.full(len(codes), float(manual_lag))
    X = pd.DataFrame({
        "product_code": codes,
        **_calendar(date),
        "sales_last_week": sales_last_week,
        "sales_avg_4w": sales_avg_4w,
    })[MODEL_FEATURES]
    return np.clip(model.predict(X), 0, None)


def _recent_sales_until(date: pd.Timestamp, ids: list[str], codes: np.ndarray, last_day: pd.Timestamp):
    """
    Return (sales of the days before `date`, source of that history).
    - `date` inside the history: the real sales of the previous weeks.
    - future `date`: real sales up to the last real date, followed by a recursive
      day-by-day forecast that is used as if it were real sales.
    `last_day` is the last day that will be forecast; it is only used to check the horizon.
    """
    first_date_allowed = _SALES.index[0] + pd.Timedelta(days=HISTORY_DAYS)
    if date < first_date_allowed:
        raise HTTPException(400, f"No hay suficiente historial antes de esa fecha. Consulta desde {first_date_allowed.date()}.")
    if (last_day - _LAST_DATE).days > MAX_HORIZON_DAYS:
        # Latest date that can be consulted (for a week, its last day must fit in the horizon)
        limit = (_LAST_DATE + pd.Timedelta(days=MAX_HORIZON_DAYS) - (last_day - date)).date()
        raise HTTPException(400, f"Fecha demasiado lejana. El último dato real es {_LAST_DATE.date()}; consulta hasta {limit}.")

    history = _SALES[ids]
    if date <= _LAST_DATE + pd.Timedelta(days=1):
        start = date - pd.Timedelta(days=HISTORY_DAYS)
        return list(history.loc[start: date - pd.Timedelta(days=1)].to_numpy(float)), "history"

    recent_sales = list(history.iloc[-HISTORY_DAYS:].to_numpy(float))  # last real days
    for k in range(1, (date - _LAST_DATE).days):  # forecast every day up to the day before `date`
        recent_sales.append(_predict(_LAST_DATE + pd.Timedelta(days=k), codes, recent_sales))
    return recent_sales, "recursive_forecast"


def _forecast(date: pd.Timestamp, ids: list[str], manual_lag: float | None = None):
    """Return (prediction per product for one day, source of the history, days ahead)."""
    codes = np.array([_PRODUCT_CODE[i] for i in ids])
    recent_sales, source = _recent_sales_until(date, ids, codes, last_day=date)
    prediction = _predict(date, codes, recent_sales, manual_lag)
    if manual_lag is not None:
        source = "manual"
    return prediction, source, max((date - _LAST_DATE).days, 0)


def _forecast_week(start: pd.Timestamp, product_id: str):
    """Forecast FORECAST_DAYS consecutive days for one product, starting at `start`.

    Inside one week every prediction only needs sales from 7 or more days before,
    which are all known (or already forecast) before `start`.
    """
    codes = np.array([_PRODUCT_CODE[product_id]])
    last_day = start + pd.Timedelta(days=FORECAST_DAYS - 1)
    recent_sales, source = _recent_sales_until(start, [product_id], codes, last_day)
    predictions = []
    for k in range(FORECAST_DAYS):
        prediction = _predict(start + pd.Timedelta(days=k), codes, recent_sales)
        recent_sales.append(prediction)
        predictions.append(float(prediction[0]))
    return predictions, source


def _margin(category: str) -> float:
    return MARGIN_BY_CATEGORY.get(category, DEFAULT_MARGIN)


def _warning(days_ahead: int):
    if days_ahead > 28:
        return f"La fecha está {days_ahead} días después del último dato real; a mayor horizonte, menor precisión."
    return None


# =====================================================================
# ENDPOINTS
# =====================================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "products": len(_CATALOG),
        "last_date_with_data": str(_LAST_DATE.date()),
    }


@app.get("/products")
def search_products(q: str = Query("", description="Texto a buscar en el nombre"), limit: int = Query(10, ge=1, le=50)):
    """Autocomplete: search products by name (ignoring accents and case)."""
    if not q.strip():
        return {"results": [_product_dict(row) for _, row in _CATALOG.head(limit).iterrows()]}
    return {"results": _search(q, limit)}


@app.get("/metrics")
def get_metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(404, "Aún no hay métricas. Ejecuta: python run.py")
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict")
def predict_demand(request: PredictionRequest):
    date = _parse_date(request.date)
    pid = _resolve_product(request.product_id, request.name)
    row = _CATALOG[_CATALOG["product_id"] == pid].iloc[0]

    prediction, lag_source, days_ahead = _forecast(date, [pid], request.sales_last_week)

    response = {
        "status": "success",
        "product": _product_dict(row),
        "date": request.date,
        "predicted_demand": int(round(prediction[0])),
        "predicted_low": None,
        "predicted_high": None,
        "lag_source": lag_source,
        "days_ahead": days_ahead,
        "avg_daily_sales_28d": round(float(_SALES[pid].iloc[-28:].mean()), 1),
        "warning": _warning(days_ahead),
    }
    if "daily" in _INTERVALS:
        low, high = apply_interval(prediction, _INTERVALS["daily"])
        response["predicted_low"], response["predicted_high"] = int(low[0]), int(high[0])
    if date in _SALES.index:
        response["actual_sales"] = int(_SALES.at[date, pid])
    return response


@app.post("/forecast-week")
def forecast_week(request: WeekForecastRequest):
    """Daily forecast for the 7 days that start at the given date, plus the weekly total and its range."""
    start = _parse_date(request.date)
    pid = _resolve_product(request.product_id, request.name)
    row = _CATALOG[_CATALOG["product_id"] == pid].iloc[0]

    predictions, source = _forecast_week(start, pid)
    days = []
    for k, prediction in enumerate(predictions):
        day = start + pd.Timedelta(days=k)
        item = {"date": str(day.date()), "day_of_week": int(day.dayofweek), "predicted": int(round(prediction))}
        if day in _SALES.index:
            item["actual"] = int(_SALES.at[day, pid])
        days.append(item)

    week_total = sum(predictions)
    response = {
        "status": "success",
        "product": _product_dict(row),
        "start_date": str(start.date()),
        "end_date": days[-1]["date"],
        "days": days,
        "week_total": int(round(week_total)),
        "week_low": None,
        "week_high": None,
        "lag_source": source,
        "days_ahead": max((start - _LAST_DATE).days, 0),
        "warning": _warning((start + pd.Timedelta(days=FORECAST_DAYS - 1) - _LAST_DATE).days),
    }
    if "weekly" in _INTERVALS:
        low, high = apply_interval(week_total, _INTERVALS["weekly"])
        response["week_low"], response["week_high"] = int(low[0]), int(high[0])
    if all("actual" in d for d in days):
        response["actual_total"] = sum(d["actual"] for d in days)
    return response


@app.post("/recommend")
def recommend_products(request: RecommendationRequest):
    date = _parse_date(request.date)
    ids = list(_CATALOG["product_id"])
    prediction, lag_source, days_ahead = _forecast(date, ids)

    recent_avg = _SALES[ids].iloc[-28:].mean().to_numpy(float)
    ranking = _CATALOG[["product_id", "product_name", "category"]].copy()
    ranking["expected_demand"] = np.round(prediction).astype(int)
    ranking["change_vs_recent_pct"] = np.where(
        recent_avg > 0, (prediction - recent_avg) / np.where(recent_avg > 0, recent_avg, 1) * 100, 0.0
    ).round(1)
    ranking["unit_margin_bs"] = ranking["category"].map(_margin)
    ranking["estimated_profit_bs"] = (ranking["expected_demand"] * ranking["unit_margin_bs"]).round(2)

    top = ranking.sort_values("estimated_profit_bs", ascending=False).head(request.top)
    top = top.rename(columns={"product_name": "name"})

    return {
        "status": "success",
        "date": request.date,
        "lag_source": lag_source,
        "days_ahead": days_ahead,
        "warning": _warning(days_ahead),
        "top_recommendations": top.to_dict(orient="records"),
    }


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
