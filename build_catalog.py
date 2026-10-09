import pandas as pd

from app.settings import CATALOG_PATH, CATALOG_XLSX_PATH

# Reads the product catalog (data/Catalogo.xlsx, first sheet), validates it  and saves it as data/catalog.csv, which is used by generate_data.py.
# The catalog has NO sales quantities: sales are generated from scratch.
# This script does not fix the catalog: if it finds a problem, it stops and explains what to fix directly in the Excel file.

catalog = pd.read_excel(CATALOG_XLSX_PATH, usecols=["Codigo", "Nombre", "Categoria"])

# 1. Rename the Excel columns to the names used in the code
catalog = catalog.rename(columns={
    "Codigo": "product_id",
    "Nombre": "product_name",
    "Categoria": "category",
})

# 2. Remove leading/trailing spaces (invisible in Excel)
for column in ["product_id", "product_name", "category"]:
    catalog[column] = catalog[column].astype(str).str.strip()

# 3. Validations: if one fails, the program stops and says what to fix in the Excel
assert catalog.notna().all().all(), "Hay celdas vacías en el catálogo"
assert catalog["product_id"].is_unique, "Hay códigos de producto repetidos"
assert catalog["product_name"].is_unique, "Hay nombres de producto repetidos"
assert catalog["product_id"].str.match(r"^[A-Z]+-\d{3}$").all(), "Hay códigos que no tienen el formato ABC-001"
prefixes = catalog["product_id"].str.split("-").str[0]
assert (prefixes.groupby(catalog["category"]).nunique() == 1).all(), "Alguna categoría usa más de un prefijo de código"

# 4. Save
catalog = catalog.sort_values("product_id").reset_index(drop=True)
catalog.to_csv(CATALOG_PATH, index=False)
print(f"Catálogo válido: {len(catalog)} productos, {catalog['category'].nunique()} categorías -> {CATALOG_PATH}")
