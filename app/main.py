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

from app.calendario import variables_calendario
from app.schemas import ConsultaPrediccion, ConsultaRecomendacion
from app.settings import (
    DATASET_INTELIGENTE_PATH,
    ENCODER_PATH,
    FRONTEND_DIR,
    HORIZONTE_MAX_DIAS,
    MARGEN_POR_CATEGORIA,
    MARGEN_POR_DEFECTO,
    METRICS_PATH,
    MODEL_PATH,
)

app = FastAPI(
    title="Sistema Inteligente de Inventario y Ventas - Licorería",
    description="API integrada para predicciones de stock y analítica prescriptiva de recomendaciones.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# CARGA ÚNICA AL ARRANCAR (antes se leía el CSV de 27 MB en cada /recommend)
# =====================================================================
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise RuntimeError("No se encontraron los .pkl en 'artifacts/'. Ejecuta: python entrenamiento.py")
if not os.path.exists(DATASET_INTELIGENTE_PATH):
    raise RuntimeError("No existe data/dataset_licoreria_inteligente.csv. Ejecuta: python procesamiento.py")

modelo_xgb = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

COLUMNAS = [
    "Producto_Codificado", "Mes", "Día_Semana", "Es_Fin_De_Semana",
    "Es_Feriado", "Es_Carnaval", "Es_Evento_Festivo", "Venta_Semana_Anterior",
]

_df = pd.read_csv(
    DATASET_INTELIGENTE_PATH,
    usecols=["Fecha", "ProductoId", "NombreProducto", "Categoría", "Cantidad_Vendida"],
    parse_dates=["Fecha"],
)

# Cada código debe pertenecer a UN solo producto. Si no, las ventas se mezclan.
_nombres_por_id = _df.groupby("ProductoId")["NombreProducto"].nunique()
if (_nombres_por_id > 1).any():
    raise RuntimeError(
        f"Códigos duplicados con nombres distintos: {list(_nombres_por_id[_nombres_por_id > 1].index)}. "
        "Vuelve a ejecutar: python procesamiento.py, separacion_temporal.py y entrenamiento.py"
    )

_CATALOGO = (
    _df.drop_duplicates("ProductoId")[["ProductoId", "NombreProducto", "Categoría"]]
    .sort_values("NombreProducto")
    .reset_index(drop=True)
)
_CODIGO = {pid: int(code) for pid, code in zip(encoder.classes_, encoder.transform(encoder.classes_))}

# Matriz fecha x producto (días sin registro = 0 ventas)
_VENTAS = (
    _df.pivot_table(index="Fecha", columns="ProductoId", values="Cantidad_Vendida", aggfunc="sum")
    .asfreq("D")
    .fillna(0.0)
)
_ULTIMA_FECHA = _VENTAS.index[-1]


def _norm(texto) -> str:
    """minúsculas, sin tildes ni signos: 'JACK DANIEL`S' -> 'jack daniel s'"""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


_CATALOGO["busq_n"] = _CATALOGO["NombreProducto"].map(_norm)
_CATALOGO["busq_compacto"] = _CATALOGO["busq_n"].str.replace(" ", "", regex=False)
_CATALOGO["busq_id"] = _CATALOGO["ProductoId"].str.lower()
_CATALOGO["busq_cat"] = _CATALOGO["Categoría"].map(_norm)


# =====================================================================
# UTILIDADES
# =====================================================================
def _producto_dict(fila) -> dict:
    return {"producto_id": fila["ProductoId"], "nombre": fila["NombreProducto"], "categoria": fila["Categoría"]}


def _buscar(q: str, limite: int = 10) -> list[dict]:
    """Busca por nombre (o por código / categoría). Todas las palabras deben aparecer."""
    consulta = _norm(q)
    tokens = consulta.split()
    if not tokens:
        return []
    resultados = []
    for fila in _CATALOGO.itertuples(index=False):
        campo = f"{fila.busq_n} {fila.busq_id} {fila.busq_cat}"
        if all(t in campo or t in fila.busq_compacto for t in tokens):
            if fila.busq_n == consulta:
                prioridad = 0
            elif fila.busq_n.startswith(consulta):
                prioridad = 1
            elif any(p.startswith(tokens[0]) for p in fila.busq_n.split()):
                prioridad = 2
            else:
                prioridad = 3
            resultados.append((prioridad, fila.NombreProducto, fila))
    resultados.sort(key=lambda r: (r[0], r[1]))
    return [
        {"producto_id": f.ProductoId, "nombre": f.NombreProducto, "categoria": f.Categoría}
        for _, _, f in resultados[:limite]
    ]


def _resolver_producto(producto_id: str | None, nombre: str | None) -> str:
    """Devuelve un ProductoId válido a partir del código o del nombre."""
    pid = (producto_id or "").strip().upper()
    if pid and pid in _CODIGO:
        return pid
    texto = (nombre or producto_id or "").strip()
    n = _norm(texto)
    exacto = _CATALOGO[_CATALOGO["busq_n"] == n]
    if len(exacto) == 1:
        return exacto.iloc[0]["ProductoId"]
    encontrados = _buscar(texto, limite=5)
    if len(encontrados) == 1:
        return encontrados[0]["producto_id"]
    if not encontrados:
        raise HTTPException(404, f"No se encontró ningún producto parecido a '{texto}'.")
    opciones = "; ".join(f"{p['nombre']} ({p['producto_id']})" for p in encontrados)
    raise HTTPException(409, f"'{texto}' es ambiguo. ¿Quisiste decir: {opciones}?")


def _parsear_fecha(texto: str) -> pd.Timestamp:
    return pd.Timestamp(texto).normalize()


def _calendario(fecha: pd.Timestamp) -> dict:
    """Calendar features for one date (same rules as training, see app/calendario.py)."""
    fila = variables_calendario(pd.Series([fecha])).iloc[0]
    return {columna: int(fila[columna]) for columna in COLUMNAS if columna in fila.index}


def _predecir(fecha: pd.Timestamp, codigos: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Un paso del modelo para varios productos a la vez. Devuelve float >= 0."""
    X = pd.DataFrame({"Producto_Codificado": codigos, **_calendario(fecha), "Venta_Semana_Anterior": lags})[COLUMNAS]
    return np.clip(modelo_xgb.predict(X), 0, None)


def _pronosticar(fecha: pd.Timestamp, ids: list[str], lag_manual: float | None = None):
    """
    Devuelve (predicción por producto, fuente del lag, días hacia el futuro).
    - lag manual: se usa tal cual.
    - fecha dentro del historial: lag real de hace 7 días.
    - fecha futura: pronóstico recursivo día a día desde el último dato real,
      alimentando cada predicción como el "lag 7" del día siguiente.
    """
    codigos = np.array([_CODIGO[i] for i in ids])
    dias_futuro = (fecha - _ULTIMA_FECHA).days

    if lag_manual is not None:
        lags = np.full(len(ids), float(lag_manual))
        return _predecir(fecha, codigos, lags), "manual", max(dias_futuro, 0)

    sub = _VENTAS[ids]

    if dias_futuro <= 0:
        ref = fecha - pd.Timedelta(days=7)
        if ref >= _VENTAS.index[0]:
            return _predecir(fecha, codigos, sub.loc[ref].to_numpy(float)), "historial real", 0
        return _predecir(fecha, codigos, sub.mean().to_numpy(float)), "promedio histórico", 0

    if dias_futuro > HORIZONTE_MAX_DIAS:
        limite = (_ULTIMA_FECHA + pd.Timedelta(days=HORIZONTE_MAX_DIAS)).date()
        raise HTTPException(400, f"Fecha demasiado lejana. El último dato real es {_ULTIMA_FECHA.date()}; consulta hasta {limite}.")

    cola = list(sub.iloc[-7:].to_numpy(float))  # ventas de los últimos 7 días reales
    pred = None
    for k in range(1, dias_futuro + 1):
        pred = _predecir(_ULTIMA_FECHA + pd.Timedelta(days=k), codigos, cola[-7])
        cola.append(pred)
    return pred, "pronóstico recursivo", dias_futuro


def _margen(categoria: str) -> float:
    return MARGEN_POR_CATEGORIA.get(categoria, MARGEN_POR_DEFECTO)


def _aviso(dias_futuro: int):
    if dias_futuro > 28:
        return f"La fecha está {dias_futuro} días después del último dato real; a mayor horizonte, menor precisión."
    return None


# =====================================================================
# ENDPOINTS
# =====================================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "productos": len(_CATALOGO),
        "ultima_fecha_con_datos": str(_ULTIMA_FECHA.date()),
    }


@app.get("/productos")
def buscar_productos(q: str = Query("", description="Texto a buscar en el nombre"), limite: int = Query(10, ge=1, le=50)):
    """Autocompletado: busca productos por nombre (sin importar tildes ni mayúsculas)."""
    if not q.strip():
        return {"resultados": [_producto_dict(f) for _, f in _CATALOGO.head(limite).iterrows()]}
    return {"resultados": _buscar(q, limite)}


@app.get("/metrics")
def metricas():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(404, "Aún no hay métricas. Ejecuta: python entrenamiento.py")
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict")
def predecir_demanda(consulta: ConsultaPrediccion):
    fecha = _parsear_fecha(consulta.fecha)
    pid = _resolver_producto(consulta.producto_id, consulta.nombre)
    fila = _CATALOGO[_CATALOGO["ProductoId"] == pid].iloc[0]

    pred, fuente, dias_futuro = _pronosticar(fecha, [pid], consulta.venta_semana_anterior)

    respuesta = {
        "status": "success",
        "producto": _producto_dict(fila),
        "fecha": consulta.fecha,
        "demanda_predicha": int(round(pred[0])),
        "fuente_lag": fuente,
        "dias_hacia_el_futuro": dias_futuro,
        "promedio_diario_28d": round(float(_VENTAS[pid].iloc[-28:].mean()), 1),
        "aviso": _aviso(dias_futuro),
    }
    if fecha in _VENTAS.index:
        respuesta["venta_real"] = int(_VENTAS.at[fecha, pid])
    return respuesta


@app.post("/recommend")
def recomendar_productos(consulta: ConsultaRecomendacion):
    fecha = _parsear_fecha(consulta.fecha)
    ids = list(_CATALOGO["ProductoId"])
    pred, fuente, dias_futuro = _pronosticar(fecha, ids)

    base_28d = _VENTAS[ids].iloc[-28:].mean().to_numpy(float)
    df_rec = _CATALOGO[["ProductoId", "NombreProducto", "Categoría"]].copy()
    df_rec["demanda_esperada"] = np.round(pred).astype(int)
    df_rec["incremento_estacional"] = np.where(base_28d > 0, (pred - base_28d) / np.where(base_28d > 0, base_28d, 1) * 100, 0.0).round(1)
    df_rec["margen_unitario_bs"] = df_rec["Categoría"].map(_margen)
    df_rec["ganancia_estimada_bs"] = (df_rec["demanda_esperada"] * df_rec["margen_unitario_bs"]).round(2)

    top = df_rec.sort_values("ganancia_estimada_bs", ascending=False).head(consulta.top)
    top = top.rename(columns={"ProductoId": "producto_id", "NombreProducto": "nombre", "Categoría": "categoria"})

    return {
        "status": "success",
        "fecha": consulta.fecha,
        "fuente_lag": fuente,
        "dias_hacia_el_futuro": dias_futuro,
        "aviso": _aviso(dias_futuro),
        "top_recomendaciones": top.to_dict(orient="records"),
    }


# El frontend se sirve desde la misma API: abre http://127.0.0.1:8000/
# (debe ir al final para no tapar los endpoints anteriores)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")