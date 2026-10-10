# Clasificador de congestión de estaciones de transporte masivo

## Descripción general

Este proyecto aplica aprendizaje supervisado a la red de TransMilenio (Bogotá) usada en la actividad anterior (4 líneas, 20 estaciones). Entrena un árbol de decisión que clasifica el nivel de congestión de una estación (`low`, `medium`, `high`) a partir de su contexto: hora, tipo de día, clima, eventos cercanos y si la estación es de transbordo.

El sistema combina:

- Un conjunto de datos sintético reproducible.
- Un árbol de decisión con reglas legibles.
- Selección de hiperparámetros por validación cruzada y comparación contra una línea base.
- Un predictor por línea de comandos.
- Pruebas automatizadas.

## Propósito académico

El proyecto demuestra la aplicación de conceptos de inteligencia artificial relacionados con:

1. Aprendizaje supervisado (clasificación).
2. Árboles de decisión y reglas de decisión.
3. Evaluación de modelos (partición entrenamiento/prueba, validación cruzada, matriz de confusión).
4. Toma de decisiones interpretable.

## Inicio rápido

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.dataset   # escribe data/congestion.csv
python -m src.train     # entrena, evalúa, guarda el modelo y la gráfica del árbol
python -m src.predict --station "Calle 100" --hour 7 --day weekday --weather rain --event 0
python -m pytest
```

Los mensajes del CLI están en español. `--day` acepta `weekday`, `saturday` o `sunday_holiday`; `--weather` acepta `dry` o `rain`; `--hour` va de 5 a 22. Los nombres de estación se comparan sin distinguir mayúsculas ni tildes y, si no hay coincidencia, se sugieren nombres parecidos (código de salida 2).

## Fuente de datos

No existe un conjunto de datos público de congestión por estación para esta red, por lo que los datos son **sintéticos**: `src/dataset.py` los genera a partir de supuestos explícitos, no de mediciones reales de TransMilenio. Las estaciones y líneas provienen de `data/network.json` (la misma red aproximada de la actividad anterior).

Cada fila es un par (estación, línea) con `hour`, `day_type`, `is_transfer`, `weather` y `near_event`. La etiqueta sale de un puntaje latente: las horas pico de día hábil (6-8 y 17-19) pesan más; las estaciones de transbordo, la lluvia y los eventos cercanos suman; los fines de semana y la noche restan; además se añade ruido gaussiano (semilla 42). Los coeficientes son constantes con nombre al inicio de `src/dataset.py`.

La consecuencia es importante: el modelo solo puede redescubrir las reglas escritas en el generador (más ruido). Su exactitud no dice nada sobre el comportamiento real de los pasajeros.

Detalle completo en [`docs/descripcion-datos.md`](docs/descripcion-datos.md).

## Modelo y evaluación

`DecisionTreeClassifier` dentro de un `Pipeline` de scikit-learn (codificación one-hot de las variables categóricas). Partición estratificada 80/20; `max_depth` y `min_samples_leaf` se eligen por validación cruzada de 5 particiones sobre el conjunto de entrenamiento. Resultados de `python -m src.train` con semilla 42 (3000 filas: 1228 `low`, 935 `medium`, 837 `high`):

| Métrica | Valor |
| --- | --- |
| Exactitud de la línea base (clase más frecuente) | 0.410 |
| Exactitud del árbol (20 % de prueba) | 0.787 |
| Exactitud con validación cruzada de 5 particiones (entrenamiento) | 0.759 (+/- 0.019) |
| Hiperparámetros elegidos | `max_depth=6`, `min_samples_leaf=10` |

Las variables más importantes son `hour`, `day_type`, `is_transfer`, `near_event` y `weather`, como cabe esperar del generador. La clase `medium` es la más difícil (F1 0.64) porque queda entre las otras dos. `src.train` imprime el conjunto completo de reglas (con `export_text`) y los primeros niveles se dibujan en `docs/img/arbol-decision.png`.

Informe completo con salidas reales: [`docs/informe-pruebas.md`](docs/informe-pruebas.md).

## Documentación

| Documento | Contenido |
| --- | --- |
| [`docs/descripcion-datos.md`](docs/descripcion-datos.md) | Origen, diccionario de datos, definición de la etiqueta, supuestos y limitaciones |
| [`docs/informe-pruebas.md`](docs/informe-pruebas.md) | Pruebas automatizadas, ejecución real y análisis de métricas |
| [`docs/INSTRUCCIONES.md`](docs/INSTRUCCIONES.md) | Instalación y ejecución paso a paso |

## Estructura del proyecto

```text
actividad-3/
├── README.md
├── requirements.txt
├── data/
│   ├── network.json          # estaciones y líneas (estimaciones aproximadas)
│   └── congestion.csv        # conjunto de datos sintético (generado)
├── docs/
│   ├── descripcion-datos.md
│   ├── informe-pruebas.md
│   ├── INSTRUCCIONES.md
│   └── img/arbol-decision.png  # gráfica del árbol (generada)
├── src/
│   ├── dataset.py            # generador de datos sintéticos
│   ├── train.py              # entrenamiento, evaluación, reglas, gráfica
│   └── predict.py            # CLI de predicción (es)
└── tests/
```

El modelo entrenado se guarda en `models/congestion_tree.joblib` (ignorado por Git); `src.predict` lo entrena al vuelo si no existe.

## Limitaciones

- El conjunto de datos es sintético; la congestión real depende de factores no modelados (incidentes, tamaño de la flota, calendarios escolares).
- Las filas se generan con coeficientes elegidos a mano y ruido, así que la exactitud reportada ilustra el método, no es una afirmación de desempeño real.
- Los nombres de estación y de línea se codifican con one-hot, por lo que el árbol puede usarlos en algunas divisiones espurias aunque el generador los ignora.
- La partición de prueba es una única muestra de 600 filas; con otras semillas se esperan diferencias de unos pocos puntos.
