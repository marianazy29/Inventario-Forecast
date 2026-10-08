import os

import numpy as np
import pandas as pd

from app.calendario import variables_calendario
from app.settings import DATA_DIR

# =====================================================================
# GENERADOR DE DATOS SINTÉTICOS DE VENTAS - LICORERÍA EN TARIJA
# Cada fila = cuántas unidades de un producto se vendieron en un día.
# Del catálogo solo se usan código, nombre y categoría. Todas las cantidades
# se generan aquí, a partir de las reglas definidas abajo.
# Con la misma SEMILLA siempre se generan exactamente los mismos datos.
# =====================================================================

SEMILLA = 2026
FECHA_INICIO = "2023-01-01"
FECHA_FIN = "2026-09-30"

# --- Reglas del generador (todas son SUPUESTOS, no datos reales) ---

# Para cada categoría: (grupo de comportamiento, venta diaria mínima, venta diaria máxima).
# Cada producto recibe al azar una venta base dentro de ese rango (en unidades por día).
CATEGORIAS = {
    # Alcohol de fiesta: se dispara el fin de semana, en Carnaval y en diciembre
    "CERVEZA":        ("alcohol_fiesta", 15, 60),
    "SINGANI":        ("alcohol_fiesta", 6, 20),   # producto local de Tarija
    "FERNET":         ("alcohol_fiesta", 3, 10),
    "RON":            ("alcohol_fiesta", 2, 12),
    "VODKA":          ("alcohol_fiesta", 1, 5),
    "TEQUILA":        ("alcohol_fiesta", 1, 4),
    "GIN":            ("alcohol_fiesta", 1, 4),
    "JÄGERMEISTER":   ("alcohol_fiesta", 1, 4),
    "LICOR":          ("alcohol_fiesta", 1, 5),
    "APERITIVO":      ("alcohol_fiesta", 1, 5),
    "CHERRYS":        ("alcohol_fiesta", 1, 3),
    "HIELO":          ("alcohol_fiesta", 10, 25),
    # Alcohol de mesa: sube el fin de semana y mucho en diciembre (regalos y cenas)
    "VINOS":          ("alcohol_mesa", 4, 15),     # producto local de Tarija
    "VINO ESPUMANTE": ("alcohol_mesa", 1, 4),
    "SIDRA":          ("alcohol_mesa", 2, 8),
    "WHISKY":         ("alcohol_mesa", 1, 5),
    "AMARULA":        ("alcohol_mesa", 1, 6),
    # Bebidas sin alcohol y frías: suben en los meses de calor
    "SODA":           ("bebidas", 8, 40),
    "AGUA":           ("bebidas", 8, 35),
    "JUGO":           ("bebidas", 4, 15),
    "ENERGIZANTE":    ("bebidas", 5, 20),
    "BEBIDAS":        ("bebidas", 5, 20),
    "HELADOS":        ("bebidas", 5, 30),
    "FREEZEE":        ("bebidas", 4, 12),
    # Productos de tienda: venta estable toda la semana
    "CIGARROS":       ("tienda", 5, 20),
    "GOLOSINAS":      ("tienda", 8, 35),
    "SNACKS":         ("tienda", 6, 25),
    "CHOCOLATES":     ("tienda", 4, 15),
    "GALLETAS":       ("tienda", 4, 12),
    "COCA MACHUCADA": ("tienda", 2, 5),
    "ENCENDEDORES":   ("tienda", 2, 6),
    "TARJETAS":       ("tienda", 1, 3),
    "ANTIÁCIDOS":     ("tienda", 1, 2),
    "BOTELLAS":       ("tienda", 1, 3),
    "OTROS":          ("tienda", 1, 4),
}
CATEGORIA_POR_DEFECTO = ("tienda", 2, 8)  # para categorías nuevas que no estén en la lista

# Tendencia: las ventas crecen un 6% por año
CRECIMIENTO_ANUAL = 0.06

# Factor por día de la semana (lunes ... domingo) para cada grupo
FACTOR_DIA = {
    "alcohol_fiesta": [0.60, 0.65, 0.70, 0.85, 1.40, 1.70, 1.10],
    "alcohol_mesa":   [0.80, 0.80, 0.85, 0.90, 1.15, 1.35, 1.15],
    "bebidas":        [0.90, 0.90, 0.95, 0.95, 1.05, 1.15, 1.10],
    "tienda":         [1.00, 1.00, 1.00, 1.00, 1.05, 1.05, 0.90],
}

# Factor por mes (enero ... diciembre). Clima de valle de Tarija: verano caluroso
# e invierno fresco. Diciembre aumenta el alcohol. El Carnaval NO va aquí porque
# cambia de fecha cada año: lo maneja FACTOR_CARNAVAL.
FACTOR_MES = {
    "alcohol_fiesta": [1.00, 1.00, 0.95, 0.90, 0.95, 0.95, 0.90, 1.00, 1.00, 1.00, 1.05, 1.30],
    "alcohol_mesa":   [0.95, 1.00, 0.90, 0.90, 1.05, 0.95, 0.95, 0.95, 0.95, 1.00, 1.05, 1.50],
    "bebidas":        [1.15, 1.15, 1.05, 0.95, 0.85, 0.80, 0.80, 0.90, 1.00, 1.10, 1.15, 1.20],
    "tienda":         [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.05],
}

# Multiplicadores en días especiales (las fechas están definidas en app/calendario.py)
# Feriados nacionales de Bolivia y departamentales de Tarija (excepto Carnaval)
FACTOR_FERIADO = {"alcohol_fiesta": 1.50, "alcohol_mesa": 1.30, "bebidas": 1.20, "tienda": 1.05}
# Carnaval chapaco: la fiesta más grande del año (reemplaza a FACTOR_FERIADO)
FACTOR_CARNAVAL = {"alcohol_fiesta": 2.20, "alcohol_mesa": 1.40, "bebidas": 1.40, "tienda": 1.10}
# Fechas festivas: Día de la Madre, San Juan, San Roque, Jueves de Comadres, etc.
FACTOR_EVENTO = {"alcohol_fiesta": 1.80, "alcohol_mesa": 1.90, "bebidas": 1.30, "tienda": 1.10}

# Ruido: cuanto MENOR este número, más irregulares son las ventas diarias
DISPERSION = 6.0


def asignar_venta_base(catalogo, rng):
    """Cada producto recibe al azar una venta base dentro del rango de su categoría."""
    reglas = catalogo["Categoría"].map(lambda c: CATEGORIAS.get(c, CATEGORIA_POR_DEFECTO))
    minimos = np.array([r[1] for r in reglas])
    maximos = np.array([r[2] for r in reglas])
    catalogo["Grupo"] = [r[0] for r in reglas]
    catalogo["Venta_Base"] = rng.uniform(minimos, maximos).round(2)
    return catalogo


def main():
    rng = np.random.default_rng(SEMILLA)

    catalogo = pd.read_csv(os.path.join(DATA_DIR, "catalogo_productos.csv"))
    catalogo = catalogo.sort_values("ProductoId").reset_index(drop=True)  # orden fijo = resultados repetibles
    catalogo = asignar_venta_base(catalogo, rng)

    fechas = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")

    # 1. Tabla con todas las combinaciones fecha x producto
    df = pd.MultiIndex.from_product(
        [fechas, catalogo["ProductoId"]], names=["Fecha", "ProductoId"]
    ).to_frame(index=False)
    df = df.merge(catalogo, on="ProductoId")
    grupo = df["Grupo"]

    # 2. Factores de calendario para cada fila
    anios = (df["Fecha"] - fechas[0]).dt.days / 365.25
    tendencia = (1 + CRECIMIENTO_ANUAL) ** anios

    calendario = variables_calendario(df["Fecha"])
    factor_dia = np.array([FACTOR_DIA[g][d] for g, d in zip(grupo, calendario["Día_Semana"])])
    factor_mes = np.array([FACTOR_MES[g][m - 1] for g, m in zip(grupo, calendario["Mes"])])

    # En Carnaval se usa su propio factor; en los demás feriados, el factor de feriado
    factor_feriado = np.select(
        [calendario["Es_Carnaval"] == 1, calendario["Es_Feriado"] == 1],
        [grupo.map(FACTOR_CARNAVAL), grupo.map(FACTOR_FERIADO)],
        default=1.0,
    )
    factor_evento = np.where(calendario["Es_Evento_Festivo"] == 1, grupo.map(FACTOR_EVENTO), 1.0)

    # 3. Venta esperada = base x tendencia x día de semana x mes x feriado/Carnaval x evento
    venta_esperada = df["Venta_Base"] * tendencia * factor_dia * factor_mes * factor_feriado * factor_evento

    # 4. Venta real = venta esperada + azar (distribución Gamma-Poisson):
    #    produce números enteros, nunca negativos, y con variación realista
    media_diaria = rng.gamma(shape=DISPERSION, scale=venta_esperada / DISPERSION)
    df["Cantidad_Vendida"] = rng.poisson(media_diaria)

    # 5. Guardar las ventas y, aparte, la venta base asignada a cada producto (para documentar)
    df = df[["Fecha", "ProductoId", "NombreProducto", "Categoría", "Cantidad_Vendida"]]
    ruta_ventas = os.path.join(DATA_DIR, "ventas_sinteticas.csv")
    df.to_csv(ruta_ventas, index=False)
    ruta_bases = os.path.join(DATA_DIR, "venta_base_generada.csv")
    catalogo.to_csv(ruta_bases, index=False)

    print(f"Semilla: {SEMILLA}")
    print(f"Filas generadas: {len(df)} ({len(fechas)} días x {len(catalogo)} productos)")
    print(f"Período: {FECHA_INICIO} a {FECHA_FIN}")
    print(f"Venta promedio por producto y día: {df['Cantidad_Vendida'].mean():.2f}")
    print(f"Días con venta cero: {100 * (df['Cantidad_Vendida'] == 0).mean():.1f}%")
    print(f"Ventas guardadas en: {ruta_ventas}")
    print(f"Venta base por producto guardada en: {ruta_bases}")


if __name__ == "__main__":
    main()