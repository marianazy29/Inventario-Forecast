# Documentación del dataset

## 1. Resumen

| Aspecto | Valor |
|---|---|
| Tipo de datos | **Sintéticos** (generados por código, con semilla fija) |
| Unidad de observación | Un producto en un día: unidades vendidas de ese producto ese día |
| Período | 1 de enero de 2023 a 30 de septiembre de 2026 (1369 días) |
| Productos | 302, en 35 categorías |
| Filas | 413438 (1369 días × 302 productos) |
| Valores faltantes | Ninguno |
| Venta diaria por producto | Promedio 15,9 unidades; mediana 11; máximo 350 |
| Días con venta cero | 2,6 % de las filas |
| Semilla | `SEED = 2026` (en `generate_data.py`) |

## 2. Origen

El dataset tiene dos partes con orígenes distintos:

**Catálogo de productos (`data/Catalogo.xlsx`).** Lista de 302 productos reales que vende la licorería UNAS FRÍAS en Tarija, cada producto con su código, nombre comercial y categoría. Se revisó manualmente para corregir errores de escritura en nombres y categorías. No contiene cantidades vendidas.

**Ventas diarias (`data/synthetic_sales.csv`).** Son **100 % sintéticas**: las genera `generate_data.py` a partir del catálogo y de un conjunto de reglas definidas (punto 4). No provienen de registros reales de ventas.

Una versión anterior del proyecto usaba ventas generadas por una herramienta de IA conversacional. Se descartaron porque no tenían código generador ni semilla y, por lo tanto, no eran reproducibles.

## 3. Cómo reproducirlo

Con el mismo catálogo y la misma semilla se obtienen exactamente los mismos datos:

```
python build_catalog.py       # valida Catalogo.xlsx -> data/catalog.csv
python generate_data.py       # genera data/synthetic_sales.csv
```

O todo el proceso de una vez: `python run.py --reconstruir`.

`build_catalog.py` no modifica el catálogo: verifica que no haya celdas vacías, códigos o nombres repetidos, códigos con formato distinto. Si algo falla, se detiene e indica qué se debe corregir en el Excel.

## 4. Cómo se generan las ventas

Para cada producto y cada día se calcula una **venta esperada**:

```
venta esperada = venta base × tendencia × factor día × factor mes × factor feriado/Carnaval × factor fecha festiva
```

y luego se le agrega azar para obtener la **venta observada**. Todas las reglas están al inicio de `generate_data.py`; las fechas especiales, en `app/tarija_calendar.py`.

### 4.1 Grupos de comportamiento

Cada categoría pertenece a un grupo que define cómo reacciona al calendario:

| Grupo | Categorías | Productos |
|---|---|---|
| Alcohol de fiesta | Cerveza, singani, fernet, ron, vodka, tequila, gin, Jägermeister, licor, aperitivo, cherrys, hielo | 75 |
| Alcohol de mesa | Vinos, vino espumante, sidra, whisky, amarula | 71 |
| Bebidas | Soda, agua, jugo, energizante, bebidas, helados, freezee | 80 |
| Tienda | Cigarros, golosinas, snacks, chocolates, galletas, encendedores, tarjetas, etc. | 76 |

### 4.2 Venta base

Unidades que vende un producto en un día normal. Cada categoría tiene un rango (por ejemplo, cerveza de 15 a 60; whisky de 1 a 5) y cada producto recibe al azar un valor dentro de ese rango. Vinos (4 a 15) y singani (6 a 20) tienen rangos más altos por ser productos locales de Tarija. La venta base asignada a cada producto se guarda en `data/base_sales.csv`.

### 4.3 Día de la semana

| Grupo | Lun | Mar | Mié | Jue | Vie | Sáb | Dom |
|---|---|---|---|---|---|---|---|
| Alcohol de fiesta | 0.60 | 0.65 | 0.70 | 0.85 | 1.40 | 1.70 | 1.10 |
| Alcohol de mesa | 0.80 | 0.80 | 0.85 | 0.90 | 1.15 | 1.35 | 1.15 |
| Bebidas | 0.90 | 0.90 | 0.95 | 0.95 | 1.05 | 1.15 | 1.10 |
| Tienda | 1.00 | 1.00 | 1.00 | 1.00 | 1.05 | 1.05 | 0.90 |

Cada grupo tiene un patrón semanal distinto, lo que crea una **interacción entre producto y día**. En los días de Carnaval este factor **no se aplica**: la fiesta manda sobre el día de la semana, así que un lunes de Carnaval vende como un día de fiesta y no como un lunes.

### 4.4 Mes

Las bebidas suben en verano (hasta ×1.20 en diciembre) y bajan en invierno (×0.80 en junio y julio). El alcohol de mesa tiene su pico en diciembre (×1.50) y sube en mayo por el Día de la Madre. El alcohol de fiesta sube en diciembre (×1.30). El Carnaval no se modela por mes porque su fecha cambia cada año.

### 4.5 Días especiales

| Grupo | Feriado | Carnaval | Fecha festiva |
|---|---|---|---|
| Alcohol de fiesta | ×1.50 | ×2.20 | ×1.80 |
| Alcohol de mesa | ×1.30 | ×1.40 | ×1.90 |
| Bebidas | ×1.20 | ×1.40 | ×1.30 |
| Tienda | ×1.05 | ×1.10 | ×1.10 |

- **Feriados:** nacionales de Bolivia y departamentales de Tarija (15 de abril), según la librería `holidays`. 53 días en el período.
- **Carnaval:** lunes y martes de Carnaval, con fecha variable. Reemplaza al factor de feriado y al del día de la semana. 8 días en el período.
- **Fechas festivas:** Día de la Madre (27/5), San Juan (23/6), Día de la Primavera y del Estudiante (21/9), Todos Santos (1/11), Nochebuena (24/12), Fin de Año (31/12) y Jueves de Comadres (jueves anterior al Carnaval). 25 días en el período.

### 4.6 Tendencia y azar

- **Tendencia:** las ventas crecen un 6 % anual (`ANNUAL_GROWTH = 0.06`).
- **Azar:** la venta observada se obtiene con una distribución Gamma-Poisson (`DISPERSION = 6.0`), que produce enteros no negativos con variación realista. Para un producto con venta esperada de 40 unidades, el 80 % de los días vende entre 20 y 63.
- **Un dado por día:** cada fila (producto y día) recibe sus propios dos números aleatorios antes de aplicar las reglas: uno decide qué tan bueno es el día y otro cuántas unidades se venden. Así, si se cambia una regla (por ejemplo, una fecha festiva), solo cambian las ventas de los días afectados, y las pequeñas diferencias de redondeo entre computadoras no alteran el resto de los datos.

## 5. Variables

Archivo final: `data/sales_dataset.csv` (lo genera `preprocess.py`).

| Variable | Tipo | Descripción |
|---|---|---|
| `date` | fecha | Día de la venta |
| `product_id` | texto | Código del producto (ej. `CVZ-001`) |
| `product_name` | texto | Nombre comercial |
| `category` | texto | Categoría del producto |
| `quantity_sold` | entero | **Variable objetivo**: unidades vendidas ese día |
| `year`, `month` | entero | Año y mes (1 a 12) |
| `day_of_week` | entero | 0 = lunes … 6 = domingo |
| `is_weekend` | 0/1 | Viernes, sábado o domingo |
| `is_holiday` | 0/1 | Feriado nacional o de Tarija (incluye Carnaval) |
| `is_carnival` | 0/1 | Lunes o martes de Carnaval |
| `is_festive_event` | 0/1 | Fecha festiva comercial |

Durante el entrenamiento se calculan además dos variables con el historial reciente de cada producto (en `app/modeling.py`):

- `sales_last_week`: la venta del mismo producto 7 días antes.
- `sales_avg_4w`: la venta diaria promedio de las 4 semanas que terminan 7 días antes. Le indica al modelo el **nivel reciente** de ventas, lo que le permite seguir la tendencia de crecimiento.

Ambas usan solo datos de hace 7 días o más, para poder calcularlas también al pronosticar fechas futuras. Las primeras 34 fechas de cada producto no tienen historial suficiente y se descartan.

## 6. División de los datos

La división respeta el orden temporal para no usar información del futuro:

| Conjunto | Período | Filas | Uso |
|---|---|---|---|
| Entrenamiento | febrero de 2023 a 2024 | 210.494 | Ajustar los modelos candidatos |
| Validación | 2025 | 110.230 | Comparar modelos y elegir el mejor (`select_model.py`) |
| Prueba | enero a septiembre de 2026 | 82.446 | Evaluación final, una sola vez (`train.py`) |

Para la prueba final, el modelo elegido se reentrena con 2023 a 2025. Para la API se reentrena con todos los datos.

El **rango de confianza** que muestra la página se calcula con los errores del modelo en validación (2025), sin usar la prueba; la prueba solo se usa después para comprobar cuántas veces la venta real cayó dentro del rango.

## 7. Limitaciones

- **Los patrones fueron definidos por la autora.** El modelo aprende las reglas escritas en el generador; los resultados demuestran que el sistema funciona, pero no garantizan el mismo rendimiento con ventas reales, que deberán usarse para validarlo.
- **Supuestos simplificados.** No hay quiebres de stock, promociones, cambios de precio, productos nuevos o descontinuados, ni días de cierre. Todos los productos existen durante todo el período.
- **Efectos fijos.** Un mismo grupo reacciona igual cada año; en la realidad el efecto de una fecha puede cambiar.
- **Pocos ejemplos de días especiales.** Solo hay 8 días de Carnaval y unos 4 por cada fecha festiva, lo que limita lo que un modelo puede aprender de ellos.
- **Mucha variación diaria.** Las ventas de un producto en un día concreto varían mucho por azar, así que ningún modelo puede acertarlas con exactitud; por eso cada pronóstico se acompaña de un rango. Al sumar por semana, el azar se compensa y el error relativo es mucho menor.
- **Error irreducible.** Por el azar del generador, ni siquiera un modelo que conociera las reglas exactas podría predecir sin error: en la prueba de 2026, ese modelo ideal tendría un MAE de aproximadamente 6,2 unidades.