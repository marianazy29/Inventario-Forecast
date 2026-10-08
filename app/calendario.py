from datetime import timedelta
from functools import lru_cache

import holidays
import pandas as pd

# =====================================================================
# CALENDARIO DE TARIJA
# Única fuente de verdad sobre qué días son especiales. La usan el generador
# de datos, el procesamiento y la API, para que los tres marquen exactamente
# los mismos días.
# =====================================================================

SUBDIVISION = "T"  # Departamento de Tarija (incluye el feriado del 15 de abril)

# Fechas festivas comerciales de fecha fija: (mes, día)
FESTIVIDADES_FIJAS = {
    (5, 27): "Día de la Madre",
    (6, 23): "Noche de San Juan",
    (8, 16): "San Roque",
    (9, 21): "Día de la Primavera y del Estudiante",
    (11, 1): "Todos Santos",
    (12, 24): "Nochebuena",
    (12, 31): "Fin de Año",
}


@lru_cache(maxsize=None)
def feriados(anios: tuple) -> holidays.HolidayBase:
    """Feriados nacionales de Bolivia más los departamentales de Tarija."""
    return holidays.Bolivia(years=list(anios), subdiv=SUBDIVISION)


@lru_cache(maxsize=None)
def dias_carnaval(anios: tuple) -> frozenset:
    """Lunes y martes de Carnaval (su fecha cambia cada año)."""
    return frozenset(d for d, nombre in feriados(anios).items() if "Carnaval" in nombre)


@lru_cache(maxsize=None)
def jueves_de_comadres(anios: tuple) -> frozenset:
    """Jueves anterior al Carnaval: 4 días antes del lunes de Carnaval."""
    lunes = [d for d in dias_carnaval(anios) if d.weekday() == 0]
    return frozenset(d - timedelta(days=4) for d in lunes)


def variables_calendario(fechas: pd.Series) -> pd.DataFrame:
    """Calcula las variables de calendario para una serie de fechas."""
    fechas = pd.to_datetime(fechas).reset_index(drop=True)
    anios = tuple(sorted(int(a) for a in fechas.dt.year.unique()))
    dias = fechas.dt.date

    es_festividad_fija = [(f.month, f.day) in FESTIVIDADES_FIJAS for f in fechas]
    es_comadres = dias.isin(jueves_de_comadres(anios))

    return pd.DataFrame({
        "Año": fechas.dt.year,
        "Mes": fechas.dt.month,
        "Día_Semana": fechas.dt.dayofweek,                                 # Lunes=0, Domingo=6
        "Es_Fin_De_Semana": fechas.dt.dayofweek.isin([4, 5, 6]).astype(int),  # Viernes a domingo
        "Es_Feriado": dias.isin(feriados(anios)).astype(int),
        "Es_Carnaval": dias.isin(dias_carnaval(anios)).astype(int),
        "Es_Evento_Festivo": (pd.Series(es_festividad_fija) | es_comadres).astype(int),
    })