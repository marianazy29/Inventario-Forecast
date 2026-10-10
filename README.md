# Pronóstico de ventas · Licorería UNAS FRÍAS? (Tarija)

Sistema que pronostica cuántas unidades se venderán de cada producto de una licorería durante los próximos 7 días. Incluye el proceso completo: generación y preparación de los datos, comparación de modelos, entrenamiento, una API y una página web para consultar los pronósticos y recomendaciones.

## Qué hace

- **Pronóstico semanal:** se elige un producto y una fecha, y el sistema muestra cuánto se venderá cada día de los 7 días siguientes y el total de la semana. Si las fechas ya pasaron, muestra también la venta real para comparar.
- **Recomendaciones:** para una fecha, sugiere los productos con mayor ganancia estimada según la demanda pronosticada.
- **Evaluación de modelos:** muestra los resultados de la prueba final frente a la línea base y los límites de la solución.

## Estructura del proyecto

```
Inventario-Forecast/
├── data/
│   └── Catalogo.xlsx         # catálogo de productos (único archivo de entrada, se creó a mano)
├── app/
│   ├── main.py               # API (FastAPI)
│   ├── schemas.py            # validación de los datos que recibe la API
│   ├── settings.py           # rutas, variables del modelo y configuración
│   ├── modeling.py           # funciones compartidas del modelo
│   └── tarija_calendar.py    # feriados, Carnaval y fechas festivas de Tarija
├── frontend/                 # página web (HTML, CSS y JavaScript)
├── tests/                    # pruebas automáticas
├── build_catalog.py          # 1. valida el catálogo
├── generate_data.py          # 2. genera las ventas sintéticas
├── preprocess.py             # 3. agrega las variables de calendario
├── split_data.py             # 4. separa los datos por fecha
├── select_model.py           # 5. compara modelos en validación (2025)
├── train.py                  # 6. prueba final (2026) y modelo para la API
├── analyze_errors.py         # análisis de errores (opcional, para poder documentar el infomre)
├── run.py                    # ejecuta todo el proceso y levanta la página
├── Dataset.md                # documentación del dataset
└── requirements.txt          # librerías con sus versiones exactas
```

Las carpetas `artifacts/` (modelo y métricas) y los archivos `.csv` de `data/` **no se incluyen**: se generan al ejecutar el proceso.

## Instalación

Requisitos: **Python 3.12 o superior** (probado con 3.14).

1. Abrir una terminal en la carpeta del proyecto.
2. Crear y activar un entorno virtual:

   **Windows**
   ```
   python -m venv env
   env\Scripts\activate
   ```

   **Linux o macOS**
   ```
   python3 -m venv env
   source env/bin/activate
   ```
3. Instalar las librerías:
   ```
   pip install -r requirements.txt
   ```

## Ejecución

```
python run.py
```

La primera vez, `run.py` ejecuta los 6 pasos del proceso (unos minutos) y después enciende el servidor. Luego, abrir en el navegador:

- **Página web:** http://127.0.0.1:8000/
- **Documentación de la API:** http://127.0.0.1:8000/docs

Las siguientes veces, si el modelo ya existe, `run.py` enciende directamente la página. 
Para rehacer todo (por ejemplo, después de cambiar el catálogo o las reglas del generador):

```
python run.py --reconstruir
```

Para detener el servidor: **Ctrl + C**.

### API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del sistema y última fecha con datos |
| GET | `/products?q=...` | Busca productos por nombre (sin importar tildes ni mayúsculas) |
| POST | `/forecast-week` | Pronóstico de 7 días desde una fecha, con el total semanal |
| POST | `/predict` | Pronóstico de un solo día, con su rango de confianza |
| POST | `/recommend` | Productos recomendados para una fecha |
| GET | `/metrics` | Resultados de la prueba final |

Ejemplo de consulta a `/forecast-week`:

```json
{ "date": "2026-10-01", "name": "PACEÑA LATA GRANDE" }
```

La API valida los datos recibidos y responde con mensajes claros: formato de fecha incorrecto, producto inexistente , nombre ambiguo o fecha fuera de rango.

## Datos

Los datos de ventas son **sintéticos**: los genera `generate_data.py` a partir del catálogo real de productos y de reglas definidas por mi persona (día de la semana, mes, feriados de Tarija, Carnaval, fechas festivas, tendencia de crecimiento y azar). Con la misma semilla (`SEED = 2026`) siempre se obtienen exactamente los mismos datos.

- 302 productos en 35 categorías.
- Ventas diarias del 1 de enero de 2023 al 30 de septiembre de 2026 (413.438 filas).
- Unidad de observación: un producto en un día.

El origen, las reglas del generador, las variables y las limitaciones se describen en detalle en [`Dataset.md`](Dataset.md).

## Modelo

**Problema:** pronosticar las unidades vendidas de cada producto en cada día, para planificar la semana.

**Variables del modelo:** producto, mes, día de la semana, fin de semana, feriado, Carnaval, fecha festiva, venta del mismo producto 7 días antes y promedio de ventas de las 4 semanas anteriores (que permite seguir la tendencia de crecimiento).

**Línea base:** vender lo mismo que el mismo día de la semana anterior.

**Validación temporal** (sin mezclar fechas, para no usar información del futuro):

| Conjunto | Período | Uso |
|---|---|---|
| Entrenamiento | 2023 a 2024 | Ajustar los modelos candidatos |
| Validación | 2025 | Comparar modelos y elegir el mejor |
| Prueba | enero a septiembre de 2026 | Evaluación final, una sola vez |

**Métricas:**

- **MASE (principal):** compara el error de cada producto con el de la línea base y promedia; todos los productos pesan igual. Menor que 1 significa mejor que la línea base.
- **MAE:** error promedio en unidades por producto y día.

### Selección del modelo (validación 2025)

| Modelo | MAE | MASE |
|---|---|---|
| **XGBoost con historial (400 árboles, profundidad 8)** | **6.34** | **0.7788** |
| XGBoost original (150 árboles, profundidad 6) | 6.36 | 0.7845 |
| XGBoost, solo venta de la semana anterior | 6.49 | 0.8066 |
| Ridge con historial | 6.69 | 0.8412 |
| Ridge sin historial | 6.74 | 0.8546 |
| Línea base | 8.88 | 1.0843 |

Se eligió XGBoost con historial por tener el menor MASE. XGBoost supera a Ridge porque cada grupo de productos tiene un patrón semanal distinto (por ejemplo, la cerveza sube mucho el sábado y los cigarros casi nada), algo que un modelo lineal no puede representar.

### Resultados en la prueba final (2026)

| Modelo | MAE | MAE relativo | MASE |
|---|---|---|---|
| Línea base | 8.91 | 1.0000 | 1.0584 |
| **XGBoost (elegido)** | **6.36** | **0.7138** | **0.7592** |
| Ridge con historial | 6.70 | 0.7514 | 0.8152 |

- El modelo elegido comete un **28.6 % menos de error** que la línea base.
- **Por semana** (total de 7 días por producto), el error promedio es del **15.9 %**, frente al 21.1 % de la línea base.

Para ver el análisis de errores por volumen, categoría, día de la semana y tipo de día:

```
python analyze_errors.py
```

## Pruebas

```
pytest
```

| Prueba | Archivo | Qué comprueba |
|---|---|---|
| Caso válido | `tests/test_api.py` | El pronóstico semanal devuelve 7 días con cantidades enteras no negativas y un total coherente |
| Caso difícil | `tests/test_api.py` | La semana de Carnaval, una fecha variable con pocos ejemplos, se pronostica con más ventas que una semana normal |
| Entrada inválida | `tests/test_api.py` | Una fecha mal escrita y un producto inexistente reciben un error claro |
| Flujo completo | `tests/test_full_flow.py` | En una copia limpia del proyecto, ejecuta los 6 pasos de este README y usa la API como la página |

Las pruebas de la API necesitan el modelo ya construido (`python run.py`) y no necesitan el servidor encendido. La prueba del flujo completo tarda unos minutos; para ejecutar solo las rápidas:

```
pytest -m "not slow"
```

## Limitaciones

- **Datos sintéticos:** el modelo aprende las reglas definidas en el generador. Su rendimiento con ventas reales de la licorería debe comprobarse cuando estén disponibles.
- **Mucha variación diaria:** las ventas de un producto en un día concreto varían mucho por azar. Por eso el pronóstico se presenta por semana, y los valores de cada día son orientativos.
- **Pocos días especiales:** solo hay 8 días de Carnaval y unos 4 por cada fecha festiva, por lo que el error es mayor en esas fechas.
- **Fechas lejanas:** para fechas posteriores al último dato real, el pronóstico se encadena día a día y pierde precisión a medida que la fecha se aleja.
- **Márgenes de ejemplo:** las recomendaciones usan márgenes de ganancia por categoría que son valores de ejemplo (`app/settings.py`); deberían reemplazarse por los márgenes reales.