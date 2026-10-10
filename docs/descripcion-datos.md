# Descripción de los datos

**Proyecto:** Clasificador de congestión de estaciones de TransMilenio (Bogotá)
**Archivo de datos:** `data/congestion.csv` (3000 filas, 8 columnas)
**Generador:** `src/dataset.py`

## 1. Origen de los datos

Los datos son **sintéticos**. No provienen de mediciones reales de TransMilenio: no existe un conjunto público de congestión por estación y hora para esta red que el equipo haya podido usar. Se generan con `src/dataset.py` a partir de:

| Insumo | Fuente | Naturaleza |
|---|---|---|
| Estaciones y líneas | `data/network.json` (misma red de la actividad 2) | Red reducida de 4 líneas y 20 estaciones; tiempos y coordenadas aproximados |
| Contexto de cada fila (hora, tipo de día, clima, evento) | Muestreo aleatorio con probabilidades fijadas por el equipo | Supuesto |
| Nivel de congestión | Puntaje latente con coeficientes fijados por el equipo | Supuesto |

Los coeficientes son estimaciones plausibles para uso académico, no calibradas con datos reales.

## 2. Diccionario de datos

| Columna | Tipo | Valores permitidos | Significado |
|---|---|---|---|
| `station` | texto (categórica) | 20 nombres de estación de `network.json` (por ejemplo `Calle 100`, `Ricaurte`, `Héroes`) | Estación donde se observa la congestión |
| `line` | texto (categórica) | `Caracas`, `Americas`, `NQS`, `Calle 26` | Línea de la fila; las estaciones de transbordo aparecen con cada una de sus líneas |
| `hour` | entero | 5 a 22 | Hora del día (formato 24 h) |
| `day_type` | texto (categórica) | `weekday`, `saturday`, `sunday_holiday` | Tipo de día |
| `is_transfer` | entero (0/1) | 0, 1 | 1 si la estación pertenece a dos o más líneas |
| `weather` | texto (categórica) | `dry`, `rain` | Clima en el momento |
| `near_event` | entero (0/1) | 0, 1 | 1 si hay un evento cercano a la estación |
| `congestion` | texto (**etiqueta**) | `low`, `medium`, `high` | Nivel de congestión a predecir |

Estaciones de transbordo en la red: `Calle 26` (Calle 26 y Caracas), `Centro Memoria` (Calle 26 y NQS) y `Ricaurte` (Americas y NQS). Hay 20 estaciones y 23 pares (estación, línea).

No hay valores nulos (verificado por `tests/test_dataset.py`).

## 3. Variable objetivo

`congestion` es una clase ordinal con tres niveles. Se define por dos umbrales sobre un puntaje latente que **no** se guarda en el CSV:

| Condición sobre el puntaje | Clase |
|---|---|
| puntaje < `LOW_MAX` = 0.5 | `low` |
| 0.5 <= puntaje < `MEDIUM_MAX` = 1.8 | `medium` |
| puntaje >= 1.8 | `high` |

## 4. Supuestos del puntaje latente

```text
puntaje = efecto_hora_día + TRANSFER·is_transfer + RAIN·[weather=rain]
          + EVENT·near_event + ruido gaussiano N(0, 0.5)
```

Constantes tomadas de `src/dataset.py` (unidades arbitrarias):

| Constante | Valor | Aplica cuando |
|---|---|---|
| `WEEKDAY_PEAK` | +2.0 | `weekday` y hora en 6, 7, 8, 17, 18, 19 |
| `WEEKDAY_DAYTIME` | +0.8 | `weekday` y hora entre 9 y 16 |
| (resto de horas hábiles) | 0.0 | `weekday` y hora 5, 20, 21, 22 |
| `SATURDAY` | +0.2 | `saturday` |
| `SUNDAY_HOLIDAY` | -0.8 | `sunday_holiday` |
| `LATE_NIGHT` | -1.0 | hora >= 21, se suma al efecto anterior |
| `TRANSFER` | +0.8 | `is_transfer` = 1 |
| `RAIN` | +0.6 | `weather` = `rain` |
| `EVENT` | +0.8 | `near_event` = 1 |
| `NOISE_STD` | 0.5 | desviación estándar del ruido |

Probabilidades de muestreo de las variables de contexto:

| Variable | Distribución |
|---|---|
| Par (estación, línea) | uniforme entre los 23 pares |
| `hour` | uniforme entre 5 y 22 |
| `day_type` | `weekday` 0.6, `saturday` 0.2, `sunday_holiday` 0.2 |
| `weather` | `rain` con probabilidad 0.25 (`RAIN_PROB`) |
| `near_event` | 1 con probabilidad 0.10 (`EVENT_PROB`) |

`is_transfer` no se muestrea: se deriva de la red.

## 5. Distribución del conjunto generado

Calculada sobre `data/congestion.csv` (semilla 42, 3000 filas).

| Clase | Filas | Proporción |
|---|---|---|
| `low` | 1228 | 0.409 |
| `medium` | 935 | 0.312 |
| `high` | 837 | 0.279 |

La clase mayoritaria (`low`) fija la línea base de exactitud: 0.410 sobre la partición de prueba.

| Variable | Distribución observada |
|---|---|
| `day_type` | `weekday` 1836, `saturday` 600, `sunday_holiday` 564 |
| `weather` | `dry` 2262, `rain` 738 |
| `near_event` | 0: 2705, 1: 295 |
| `is_transfer` | 0: 2230, 1: 770 |

Clase por tipo de día (proporción dentro de cada tipo):

| `day_type` | `low` | `medium` | `high` |
|---|---|---|---|
| `weekday` | 0.24 | 0.33 | 0.43 |
| `saturday` | 0.49 | 0.43 | 0.08 |
| `sunday_holiday` | 0.88 | 0.11 | 0.01 |

## 6. Procedimiento de generación

1. Se lee `data/network.json` y se construye el mapa estación -> líneas (`station_lines`).
2. Con un generador `numpy.random.default_rng(seed)` se muestrean `n` pares (estación, línea) y el contexto de cada fila.
3. `is_transfer` se calcula como 1 si la estación tiene dos o más líneas.
4. Se calcula el puntaje latente con las constantes de la sección 4 y se asigna la clase con los umbrales de la sección 3.
5. Se guardan las columnas `station, line, hour, day_type, is_transfer, weather, near_event, congestion`.

El proceso es determinista: la misma semilla produce el mismo archivo (verificado por `test_same_seed_is_deterministic`).

### Cómo regenerarlo

```bash
python -m src.dataset
```

Escribe `data/congestion.csv` con `n=3000` y `seed=42` y muestra las proporciones de clase. Para otros tamaños o semillas, desde Python:

```python
from src.dataset import generate_dataset
generate_dataset(n=5000, seed=7).to_csv("data/congestion.csv", index=False)
```

## 7. Limitaciones

| Limitación | Consecuencia |
|---|---|
| Los datos no son mediciones reales | El modelo solo aprende las reglas escritas en el generador más ruido; la exactitud no mide congestión real |
| Coeficientes y umbrales elegidos a mano | Otro conjunto de supuestos produciría otra distribución de clases |
| Variables no modeladas: incidentes, tamaño de flota, calendario escolar, puntos de la red fuera de las 20 estaciones | La congestión real depende de ellas |
| `station` y `line` no influyen en el puntaje (salvo a través de `is_transfer`) | El árbol puede usarlas en divisiones espurias |
| Red reducida (20 estaciones) con tiempos aproximados | No representa la red completa |
| Etiqueta derivada de un puntaje con ruido | Parte de los errores del modelo es ruido irreducible |

## 8. Cómo sustituirlos por datos reales

El esquema de columnas está pensado para que el reemplazo no cambie `train.py` ni `predict.py`.

| Paso | Detalle |
|---|---|
| Fuente | Datos abiertos del operador o de la autoridad de movilidad: validaciones (entradas) por estación y franja horaria, y ocupación o afluencia por estación |
| Agregación | Contar validaciones por estación, línea, fecha y hora |
| Etiqueta | Discretizar la afluencia en `low`, `medium`, `high` con cuantiles (por ejemplo terciles) por estación, o con umbrales de capacidad si el operador los publica |
| `day_type` | Derivar de la fecha con el calendario de festivos de Colombia |
| `weather` | Unir con datos meteorológicos horarios (por ejemplo precipitación mayor a un umbral) |
| `near_event` | Unir con un calendario de eventos masivos de la ciudad |
| `is_transfer` | Derivar de la red, como se hace hoy |
| Validación | Partición por fecha (entrenar en el pasado, probar en el futuro), no aleatoria, para evitar fuga de información |

Con datos reales la comparación contra la línea base y la validación cruzada seguirían siendo válidas, y la exactitud sí tendría lectura sobre congestión real.
