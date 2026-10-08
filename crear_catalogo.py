import os

import pandas as pd

from app.settings import DATA_DIR

# Lee el catálogo (data/Catalogo.xlsx, primera hoja), lo valida y lo guarda en
# data/catalogo_productos.csv, que es lo que usa generar_datos.py.
# El catálogo NO tiene cantidades vendidas: las ventas se generan desde cero.
# Este script NO corrige el catálogo: si encuentra un problema, se detiene y
# avisa, para que se corrija directamente en el Excel.

ruta_excel = os.path.join(DATA_DIR, "Catalogo.xlsx")
catalogo = pd.read_excel(ruta_excel, usecols=["Codigo", "Nombre", "Categoria"])

# 1. Renombrar columnas a los nombres que usa el resto del proyecto
catalogo = catalogo.rename(columns={
    "Codigo": "ProductoId",
    "Nombre": "NombreProducto",
    "Categoria": "Categoría",
})

# 2. Quitar espacios sobrantes al inicio o al final (invisibles en Excel)
for columna in ["ProductoId", "NombreProducto", "Categoría"]:
    catalogo[columna] = catalogo[columna].astype(str).str.strip()

# 3. Validaciones: si alguna falla, el programa se detiene y explica qué corregir en el Excel
assert catalogo.notna().all().all(), "Hay celdas vacías en el catálogo"
assert catalogo["ProductoId"].is_unique, "Hay códigos de producto repetidos"
assert catalogo["NombreProducto"].is_unique, "Hay nombres de producto repetidos"
assert catalogo["ProductoId"].str.match(r"^[A-Z]+-\d{3}$").all(), "Hay códigos que no tienen el formato ABC-001"
prefijos = catalogo["ProductoId"].str.split("-").str[0]
assert (prefijos.groupby(catalogo["Categoría"]).nunique() == 1).all(), "Alguna categoría usa más de un prefijo de código"

# 4. Guardar
catalogo = catalogo.sort_values("ProductoId").reset_index(drop=True)
ruta = os.path.join(DATA_DIR, "catalogo_productos.csv")
catalogo.to_csv(ruta, index=False)
print(f"Catálogo válido: {len(catalogo)} productos, {catalogo['Categoría'].nunique()} categorías -> {ruta}")