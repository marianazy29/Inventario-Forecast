from datetime import timedelta
from functools import lru_cache

import holidays
import pandas as pd
from dateutil.easter import easter

# =====================================================================
# TARIJA CALENDAR
# Single source of truth for special days. Used by the data generator,
# the preprocessing step and the API, so all three mark the same days.
# =====================================================================

SUBDIVISION = "T"  # Tarija department (adds the April 15 holiday)

# Fixed-date commercial festivities: (month, day)
FIXED_FESTIVITIES = {
    (5, 27): "Día de la Madre",
    (6, 23): "Noche de San Juan",
    (9, 21): "Día de la Primavera y del Estudiante",
    (11, 1): "Todos Santos",
    (12, 24): "Nochebuena",
    (12, 31): "Fin de Año",
}


@lru_cache(maxsize=None)
def get_holidays(years: tuple) -> holidays.HolidayBase:
    """National holidays of Bolivia plus Tarija department holidays.

    language="es" is set explicitly: otherwise the library uses the language of the
    computer, and holiday names would change from one computer to another.
    """
    return holidays.Bolivia(years=list(years), subdiv=SUBDIVISION, language="es")


@lru_cache(maxsize=None)
def carnival_days(years: tuple) -> frozenset:
    """Carnival Monday and Tuesday (the date changes every year).

    They are computed from Easter Sunday (Monday = 48 days before, Tuesday = 47 days
    before), so they do not depend on the language of the holiday names.
    """
    days = set()
    for year in years:
        easter_sunday = easter(year)
        days.add(easter_sunday - timedelta(days=48))
        days.add(easter_sunday - timedelta(days=47))
    return frozenset(days)


@lru_cache(maxsize=None)
def comadres_thursdays(years: tuple) -> frozenset:
    """Jueves de Comadres: the Thursday before Carnival (4 days before Carnival Monday)."""
    mondays = [day for day in carnival_days(years) if day.weekday() == 0]
    return frozenset(day - timedelta(days=4) for day in mondays)


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Build the calendar features for a series of dates."""
    dates = pd.to_datetime(dates).reset_index(drop=True)
    years = tuple(sorted(int(y) for y in dates.dt.year.unique()))
    days = dates.dt.date

    is_fixed_festivity = pd.Series([(d.month, d.day) in FIXED_FESTIVITIES for d in dates])
    is_comadres = days.isin(comadres_thursdays(years))

    return pd.DataFrame({
        "year": dates.dt.year,
        "month": dates.dt.month,
        "day_of_week": dates.dt.dayofweek,                               # Monday=0, Sunday=6
        "is_weekend": dates.dt.dayofweek.isin([4, 5, 6]).astype(int),    # Friday to Sunday
        "is_holiday": days.isin(get_holidays(years)).astype(int),
        "is_carnival": days.isin(carnival_days(years)).astype(int),
        "is_festive_event": (is_fixed_festivity | is_comadres).astype(int),
    })
