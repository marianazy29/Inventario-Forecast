import numpy as np
import pandas as pd
from scipy.stats import gamma, poisson

from app.settings import BASE_SALES_PATH, CATALOG_PATH, SALES_PATH
from app.tarija_calendar import calendar_features

# =====================================================================
# SYNTHETIC SALES GENERATOR - LIQUOR STORE UNAS FRÍAS IN TARIJA
# Each row = how many units of one product were sold on one day.
# Only code, name and category are taken from the catalog. All quantities  are generated here, from the rules defined below.
# The same seed always generates exactly the same data.
# =====================================================================

SEED = 2026
START_DATE = "2023-01-01"
END_DATE = "2026-09-30"

# For each category: (behavior group, minimum daily sales, maximum daily sales).
# Each product gets a random base sales value inside that range (units per day).
CATEGORIES = {
    # Party alcohol: peaks on weekends, at Carnival and in December
    "CERVEZA":        ("party_alcohol", 15, 60),
    "SINGANI":        ("party_alcohol", 6, 20),    # local product of Tarija
    "FERNET":         ("party_alcohol", 3, 10),
    "RON":            ("party_alcohol", 2, 12),
    "VODKA":          ("party_alcohol", 1, 5),
    "TEQUILA":        ("party_alcohol", 1, 4),
    "GIN":            ("party_alcohol", 1, 4),
    "JÄGERMEISTER":   ("party_alcohol", 1, 4),
    "LICOR":          ("party_alcohol", 1, 5),
    "APERITIVO":      ("party_alcohol", 1, 5),
    "CHERRYS":        ("party_alcohol", 1, 3),
    "HIELO":          ("party_alcohol", 10, 25),
    # Table alcohol: rises on weekends and strongly in December (gifts and dinners)
    "VINOS":          ("table_alcohol", 4, 15),    # local product of Tarija
    "VINO ESPUMANTE": ("table_alcohol", 1, 4),
    "SIDRA":          ("table_alcohol", 2, 8),
    "WHISKY":         ("table_alcohol", 1, 5),
    "AMARULA":        ("table_alcohol", 1, 6),
    # Soft and cold drinks: rise in hot months
    "SODA":           ("drinks", 8, 40),
    "AGUA":           ("drinks", 8, 35),
    "JUGO":           ("drinks", 4, 15),
    "ENERGIZANTE":    ("drinks", 5, 20),
    "BEBIDAS":        ("drinks", 5, 20),
    "HELADOS":        ("drinks", 5, 30),
    "FREEZEE":        ("drinks", 4, 12),
    # Store products: stable sales during the week
    "CIGARROS":       ("store", 5, 20),
    "GOLOSINAS":      ("store", 8, 35),
    "SNACKS":         ("store", 6, 25),
    "CHOCOLATES":     ("store", 4, 15),
    "GALLETAS":       ("store", 4, 12),
    "COCA MACHUCADA": ("store", 2, 5),
    "ENCENDEDORES":   ("store", 2, 6),
    "TARJETAS":       ("store", 1, 3),
    "ANTIÁCIDOS":     ("store", 1, 2),
    "BOTELLAS":       ("store", 1, 3),
    "OTROS":          ("store", 1, 4),
}
DEFAULT_CATEGORY = ("store", 2, 8)  # for new categories not listed above

# Trend: sales grow 6% per year
ANNUAL_GROWTH = 0.06

# Day-of-week factor (Monday to Sunday) for each group
DAY_FACTOR = {
    "party_alcohol": [0.60, 0.65, 0.70, 0.85, 1.40, 1.70, 1.10],
    "table_alcohol": [0.80, 0.80, 0.85, 0.90, 1.15, 1.35, 1.15],
    "drinks":        [0.90, 0.90, 0.95, 0.95, 1.05, 1.15, 1.10],
    "store":         [1.00, 1.00, 1.00, 1.00, 1.05, 1.05, 0.90],
}

# Month factor (January to December). Tarija valley climate: hot summer and
# cool winter. December increases alcohol sales. Carnival is not here because  its date changes every year: CARNIVAL_FACTOR handles it.
MONTH_FACTOR = {
    "party_alcohol": [1.00, 1.00, 0.95, 0.90, 0.95, 0.95, 0.90, 1.00, 1.00, 1.00, 1.05, 1.30],
    "table_alcohol": [0.95, 1.00, 0.90, 0.90, 1.05, 0.95, 0.95, 0.95, 0.95, 1.00, 1.05, 1.50],
    "drinks":        [1.15, 1.15, 1.05, 0.95, 0.85, 0.80, 0.80, 0.90, 1.00, 1.10, 1.15, 1.20],
    "store":         [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.05],
}

# Multipliers on special days (the dates are defined in app/tarija_calendar.py)
# National holidays of Bolivia and Tarija holidays (except Carnival)
HOLIDAY_FACTOR = {"party_alcohol": 1.50, "table_alcohol": 1.30, "drinks": 1.20, "store": 1.05}
# Carnaval chapaco: the biggest party of the year (replaces HOLIDAY_FACTOR)
CARNIVAL_FACTOR = {"party_alcohol": 2.20, "table_alcohol": 1.40, "drinks": 1.40, "store": 1.10}
# Festive dates: Mother's Day, San Juan, Jueves de Comadres, etc.
FESTIVE_FACTOR = {"party_alcohol": 1.80, "table_alcohol": 1.90, "drinks": 1.30, "store": 1.10}

# Noise: the lower this number, the more irregular the daily sales
DISPERSION = 6.0


def assign_base_sales(catalog: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Each product gets a random base sales value inside the range of its category."""
    rules = catalog["category"].map(lambda c: CATEGORIES.get(c, DEFAULT_CATEGORY))
    minimums = np.array([r[1] for r in rules])
    maximums = np.array([r[2] for r in rules])
    catalog["group"] = [r[0] for r in rules]
    catalog["base_sales"] = rng.uniform(minimums, maximums).round(2)
    return catalog


def main():
    rng = np.random.default_rng(SEED)

    catalog = pd.read_csv(CATALOG_PATH)
    catalog = catalog.sort_values("product_id").reset_index(drop=True)  # fixed order = repeatable results
    catalog = assign_base_sales(catalog, rng)

    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    # 1. Table with every date x product combination
    df = pd.MultiIndex.from_product(
        [dates, catalog["product_id"]], names=["date", "product_id"]
    ).to_frame(index=False)
    df = df.merge(catalog, on="product_id")
    group = df["group"]

    # 2. Calendar factors for each row
    years_elapsed = (df["date"] - dates[0]).dt.days / 365.25
    trend = (1 + ANNUAL_GROWTH) ** years_elapsed

    calendar = calendar_features(df["date"])
    day_factor = np.array([DAY_FACTOR[g][d] for g, d in zip(group, calendar["day_of_week"])])
    # On Carnival the party rules over the weekday: the day-of-week factor is not applied
    day_factor = np.where(calendar["is_carnival"] == 1, 1.0, day_factor)
    month_factor = np.array([MONTH_FACTOR[g][m - 1] for g, m in zip(group, calendar["month"])])

    # Carnival uses its own factor; the other holidays use the holiday factor
    holiday_factor = np.select(
        [calendar["is_carnival"] == 1, calendar["is_holiday"] == 1],
        [group.map(CARNIVAL_FACTOR), group.map(HOLIDAY_FACTOR)],
        default=1.0,
    )
    festive_factor = np.where(calendar["is_festive_event"] == 1, group.map(FESTIVE_FACTOR), 1.0)

    # 3. Expected sales = base x trend x day x month x holiday/Carnival x festive date
    expected_sales = df["base_sales"] * trend * day_factor * month_factor * holiday_factor * festive_factor

    # 4. Observed sales = expected sales + randomness (Gamma-Poisson distribution):
    #    whole numbers, never negative, with realistic day-to-day variation.
    #    Each row gets its OWN two random numbers ("its own dice"), drawn before
    #    using the rules. So changing a rule only changes the affected days, and
    #    tiny rounding differences between computers do not shift the other days.
    day_dice = rng.random(len(df))     # how good or bad the day is
    count_dice = rng.random(len(df))   # how many units are finally sold
    daily_mean = gamma.ppf(day_dice, a=DISPERSION, scale=expected_sales / DISPERSION)
    df["quantity_sold"] = poisson.ppf(count_dice, np.maximum(daily_mean, 1e-9)).clip(min=0).astype(int)

    # 5. Save the sales and, separately, the base sales of each product (for documentation)
    df = df[["date", "product_id", "product_name", "category", "quantity_sold"]]
    df.to_csv(SALES_PATH, index=False)
    catalog.to_csv(BASE_SALES_PATH, index=False)

    print(f"Semilla: {SEED}")
    print(f"Filas generadas: {len(df)} ({len(dates)} días x {len(catalog)} productos)")
    print(f"Período: {START_DATE} a {END_DATE}")
    print(f"Venta promedio por producto y día: {df['quantity_sold'].mean():.2f}")
    print(f"Días con venta cero: {100 * (df['quantity_sold'] == 0).mean():.1f}%")
    print(f"Ventas guardadas en: {SALES_PATH}")
    print(f"Venta base por producto guardada en: {BASE_SALES_PATH}")


if __name__ == "__main__":
    main()
