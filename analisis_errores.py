import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from app.settings import ARTIFACTS_DIR, DATA_DIR

# Análisis de errores del modelo elegido (XGBoost) en la prueba final (2026).
# IMPORTANTE: aquí solo MIRAMOS los errores para entenderlos. No se cambia el modelo
# con lo que veamos, porque eso sería ajustar el modelo con la prueba.

FECHA_CORTE = "2026-01-01"
NOMBRES_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# 1. Mismos datos, mismo modelo y misma configuración que entrenamiento.py
df_train = pd.read_csv(os.path.join(DATA_DIR, "train_datos.csv"), parse_dates=["Fecha"])
df_test = pd.read_csv(os.path.join(DATA_DIR, "test_datos.csv"), parse_dates=["Fecha"])
df = pd.concat([df_train, df_test]).sort_values(["ProductoId", "Fecha"]).reset_index(drop=True)
df["Venta_Semana_Anterior"] = df.groupby("ProductoId")["Cantidad_Vendida"].shift(7)
df = df.dropna(subset=["Venta_Semana_Anterior"]).copy()
df["Producto_Codificado"] = LabelEncoder().fit_transform(df["ProductoId"])

columnas_modelo = ["Producto_Codificado", "Mes", "Día_Semana", "Es_Fin_De_Semana",
                   "Es_Feriado", "Es_Evento_Festivo", "Venta_Semana_Anterior"]

train = df[df["Fecha"] < FECHA_CORTE]
test = df[df["Fecha"] >= FECHA_CORTE].copy()

modelo = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=8, random_state=42)
modelo.fit(train[columnas_modelo], train["Cantidad_Vendida"])
test["Prediccion"] = np.clip(modelo.predict(test[columnas_modelo]), 0, None).round()

# 2. Columnas de error
#    Error = predicción - real. Positivo: el modelo se pasó. Negativo: se quedó corto.
test["Error"] = test["Prediccion"] - test["Cantidad_Vendida"]
test["Error_Abs"] = test["Error"].abs()

# Escala del MASE por producto (solo con entrenamiento), para comparar productos de distinto tamaño
escala = (train["Cantidad_Vendida"] - train["Venta_Semana_Anterior"]).abs().groupby(train["ProductoId"]).mean()

# Grupo de volumen: cada producto se clasifica según su venta promedio en entrenamiento
venta_promedio = train.groupby("ProductoId")["Cantidad_Vendida"].mean()
grupo_volumen = pd.qcut(venta_promedio, 3, labels=["1. Venta baja", "2. Venta media", "3. Venta alta"])
test["Grupo_Volumen"] = test["ProductoId"].map(grupo_volumen)

test["Tipo_Dia"] = np.select(
    [test["Es_Evento_Festivo"] == 1, test["Es_Feriado"] == 1],
    ["Evento festivo", "Feriado"],
    default="Día normal",
)
test["Dia"] = test["Día_Semana"].map(dict(enumerate(NOMBRES_DIAS)))


def resumen(columna):
    """Tabla de errores agrupada por una columna."""
    tabla = test.groupby(columna, observed=True).agg(
        Filas=("Error", "size"),
        Venta_Promedio=("Cantidad_Vendida", "mean"),
        MAE=("Error_Abs", "mean"),
        Sesgo=("Error", "mean"),
    )
    # Error porcentual: qué fracción de lo vendido representa el error
    tabla["Error_%"] = 100 * test.groupby(columna, observed=True)["Error_Abs"].sum() / test.groupby(columna, observed=True)["Cantidad_Vendida"].sum()
    return tabla.round(2)


# 3. Errores por producto (para encontrar los peores)
por_producto = test.groupby(["ProductoId", "NombreProducto", "Categoría"]).agg(
    Venta_Promedio=("Cantidad_Vendida", "mean"),
    MAE=("Error_Abs", "mean"),
    Sesgo=("Error", "mean"),
).reset_index()
por_producto["MASE"] = por_producto["MAE"] / por_producto["ProductoId"].map(escala)
por_producto = por_producto.sort_values("MASE", ascending=False).round(3)

# 4. Mostrar resultados
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

print("\n=== ERROR POR GRUPO DE VOLUMEN ===")
print(resumen("Grupo_Volumen"))
print("\n=== ERROR POR CATEGORÍA (ordenado por Error_%) ===")
print(resumen("Categoría").sort_values("Error_%", ascending=False))
print("\n=== ERROR POR DÍA DE LA SEMANA ===")
print(resumen("Dia").reindex(NOMBRES_DIAS))
print("\n=== ERROR POR TIPO DE DÍA ===")
print(resumen("Tipo_Dia"))
print("\n=== LOS 10 PRODUCTOS CON PEOR MASE ===")
print(por_producto.head(10).to_string(index=False))
print("\n=== LOS 10 PRODUCTOS CON MEJOR MASE ===")
print(por_producto.tail(10).to_string(index=False))

# 5. Guardar el detalle por producto para el informe
ruta = os.path.join(ARTIFACTS_DIR, "errores_por_producto.csv")
por_producto.to_csv(ruta, index=False)
print(f"\n📊 Detalle por producto guardado en: {ruta}")