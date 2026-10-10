# Documentación del dataset

## 1. Resumen

| Tipo de datos | **Sintéticos** (generados por código, con semilla fija) |
| Unidad de observación | Un producto en un día: unidades vendidas de ese producto ese día |
| Período | 1 de enero de 2023 a 30 de septiembre de 2026 (1369 días) |
| Productos | 302, en 35 categorías |
| Filas | 413438 (1369 días × 302 productos) |
| Valores faltantes | Ninguno |
| Días con venta cero | 2.6 % de las filas |
| Semilla | `SEMILLA = 2026` (en `generar_datos.py`) |

## 2. Origen

El dataset tiene dos partes con orígenes distintos:

**Catálogo de productos (`data/Catalogo.xlsx`).** Lista de 302 productos reales que vende la licorería UNAS FRÍAS, en la ciudad de Tarija - Bolivia, con su código, nombre comercial y categoría. Fue revisado manualmente para corregir errores de escritura en nombres y categorías. No contiene cantidades vendidas.

**Ventas diarias (`data/ventas_sinteticas.csv`).** Son **100 % sintéticas**: las genera `generar_datos.py` a partir del catálogo y de un conjunto de reglas definidas (punto 4). No provienen de registros reales de ventas.

En una versión anterior del proyecto usaba ventas generadas por una herramienta de IA conversacional. Se descartaron porque no tenían código generador ni semilla y, por lo tanto, no eran reproducibles.

## 3. Cómo ejecutar

Con el mismo catálogo y la misma semilla se obtienen exactamente los mismos datos:

```
python crear_catalogo.py      # valida Catalogo.xlsx -> data/catalogo_productos.csv
python generar_datos.py       # genera data/ventas_sinteticas.csv
```

O todo el proceso de una vez: `python run.py --reconstruir`.

`crear_catalogo.py` no modifica el catálogo: verifica que no haya celdas vacías, códigos o nombres repetidos o códigos con formato distinto. Si algo falla, se detiene e indica que se debe corregir en el Excel.

## 4. Cómo se generan las ventas

Para cada producto y cada día se calcula una **venta esperada**:

```
venta esperada = venta base × tendencia × factor día × factor mes × factor feriado/Carnaval × factor fecha festiva
```

y luego se le agrega azar para obtener la **venta observada**. Todas las reglas están al inicio de `generar_datos.py`; las fechas especiales, en `app/calendario.py`.

### 4.1 Grupos de comportamiento

Cada categoría pertenece a un grupo que define cómo reacciona al calendario:

| Grupo | Categorías | Productos |
|---|---|---|
| Alcohol de fiesta | Cerveza, singani, fernet, ron, vodka, tequila, gin, Jägermeister, licor, aperitivo, cherrys, hielo | 75 |
| Alcohol de mesa | Vinos, vino espumante, sidra, whisky, amarula | 71 |
| Bebidas | Soda, agua, jugo, energizante, bebidas, helados, freezee | 80 |
| Tienda | Cigarros, golosinas, snacks, chocolates, galletas, encendedores, tarjetas, etc. | 76 |

### 4.2 Venta base

Unidades que vende un producto en un día normal. Cada categoría tiene un rango (por ejemplo, cerveza de 15 a 60; whisky de 1 a 5) y cada producto recibe al azar un valor dentro de ese rango. Vinos (4 a 15) y singani (6 a 20) tienen rangos más altos por ser productos locales de Tarija. La venta base asignada a cada producto se guarda en `data/venta_base_generada.csv`.

### 4.3 Día de la semana

| Grupo | Lun | Mar | Mié | Jue | Vie | Sáb | Dom |
|---|---|---|---|---|---|---|---|
| Alcohol de fiesta | 0.60 | 0.65 | 0.70 | 0.85 | 1.40 | 1.70 | 1.10 |
| Alcohol de mesa | 0.80 | 0.80 | 0.85 | 0.90 | 1.15 | 1.35 | 1.15 |
| Bebidas | 0.90 | 0.90 | 0.95 | 0.95 | 1.05 | 1.15 | 1.10 |
| Tienda | 1.00 | 1.00 | 1.00 | 1.00 | 1.05 | 1.05 | 0.90 |

Cada grupo tiene un patrón semanal distinto. lo que crea una **interacción entre producto y día** que un modelo aditivo no puede representar.

### 4.4 Mes

Clima de valle de Tarija: las bebidas suben en verano (hasta ×1.20 en diciembre) y bajan en invierno (×0.80 en junio y julio). El alcohol de mesa tiene su pico en diciembre (×1.50) y sube en mayo por el Día de la Madre. El alcohol de fiesta sube en diciembre (×1.30). El Carnaval no se modela por mes porque su fecha cambia cada año.

### 4.5 Días especiales

| Grupo | Feriado | Carnaval | Fecha festiva |
|---|---|---|---|
| Alcohol de fiesta | ×1.50 | ×2.20 | ×1.80 |
| Alcohol de mesa | ×1.30 | ×1.40 | ×1.90 |
| Bebidas | ×1.20 | ×1.40 | ×1.30 |
| Tienda | ×1.05 | ×1.10 | ×1.10 |

- **Feriados:** nacionales de Bolivia y departamentales de Tarija (15 de abril), según la librería `holidays`. 53 días en el período.
- **Carnaval:** lunes y martes de Carnaval, con fecha variable. Reemplaza al factor de feriado. 8 días en el período.
- **Fechas festivas:** Día de la Madre (27/5), San Juan (23/6), San Roque (16/8), Día de la Primavera y del Estudiante (21/9), Todos Santos (1/11), Nochebuena (24/12), Fin de Año (31/12) y Jueves de Comadres (jueves anterior al Carnaval). 29 días en el período.

### 4.6 Tendencia y azar

- **Tendencia:** las ventas crecen un 6 % anual (`CRECIMIENTO_ANUAL = 0.06`).
- **Azar:** la venta observada se obtiene con una distribución Gamma-Poisson (`DISPERSION = 6.0`), que produce enteros no negativos con variación realista. Para un producto con venta esperada de 40 unidades, el 80 % de los días vende entre 20 y 63.

## 5. Variables

Archivo final: `data/dataset_licoreria_inteligente.csv` (lo genera `procesamiento.py`).

| Variable | Tipo | Descripción |
|---|---|---|
| `Fecha` | fecha | Día de la venta |
| `ProductoId` | texto | Código del producto (ej. `CVZ-001`) |
| `NombreProducto` | texto | Nombre comercial |
| `Categoría` | texto | Categoría del producto |
| `Cantidad_Vendida` | entero | **Variable objetivo**: unidades vendidas ese día |
| `Año`, `Mes` | entero | Año y mes (1 a 12) |
| `Día_Semana` | entero | 0 = lunes … 6 = domingo |
| `Es_Fin_De_Semana` | 0/1 | Viernes, sábado o domingo |
| `Es_Feriado` | 0/1 | Feriado nacional o de Tarija (incluye Carnaval) |
| `Es_Carnaval` | 0/1 | Lunes o martes de Carnaval |
| `Es_Evento_Festivo` | 0/1 | Fecha festiva comercial |

Durante el entrenamiento se calcula además `Venta_Semana_Anterior`: la venta del mismo producto 7 días antes. Las primeras 7 fechas de cada producto no tienen este valor y se descartan.

## 6. División de los datos

La división respeta el orden temporal para no usar información del futuro:

| Conjunto | Período | Filas | Uso |
|---|---|---|---|
| Entrenamiento | 2023 a 2024 | 220762 | Ajustar los modelos candidatos |
| Validación | 2025 | 110230 | Comparar modelos y elegir el mejor (`seleccion_modelo.py`) |
| Prueba | enero a septiembre de 2026 | 82446 | Evaluación final, una sola vez (`entrenamiento.py`) |

Para la prueba final, el modelo elegido se reentrena con 2023 a 2025. Para la API se reentrena con todos los datos.

## 7. Limitaciones

1. **Las reglas se difinieron por mi persona.** El modelo aprendió las reglas que se escribieron en el generador (por ejemplo, que el sábado se vende más cerveza). Esto demuestra que el modelo sabe aprender patrones del calendario, es importante aclarar que estas reglas se establecieron de acuerdo a la experiencia que se pudo rescatar del dueño de la licorería. 

2. **Los datos son más simples que la realidad.** El generador no incluye situaciones comunes en una tienda real: que se acabe un producto, promociones (por ejemplo, un 2x1), cambios de precio, productos nuevos o que se dejan de vender, ni días en que la tienda cierra. Con ventas reales, predecir sería más difícil.

3. **Hay pocos días especiales para aprender.** En los casi cuatro años de datos solo hay 8 días de Carnaval y unas 4 veces cada fecha festiva (Día de la Madre, San Roque, Fin de Año, etc.). Con tan pocos ejemplos, el modelo se equivoca más en esos días que en los días normales.

llll
