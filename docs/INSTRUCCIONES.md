# Instrucciones de ejecución

Clasificador de congestión de estaciones de TransMilenio (Bogotá) con un árbol de decisión (scikit-learn).

## 1. Requisitos

- Python 3.10 o superior (desarrollado con Python 3.14).
- Git.
- Dependencias: `pandas`, `scikit-learn`, `matplotlib`, `pytest` (en `requirements.txt`).

## 2. Instalación

```bash
git clone <URL-del-repositorio>
cd actividad-3

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 3. Ejecución

Todos los comandos se ejecutan desde la raíz del repositorio, con el entorno virtual activo.

| Paso | Comando | Qué hace |
|---|---|---|
| 1. Generar datos | `python -m src.dataset` | Escribe `data/congestion.csv` (3000 filas, semilla 42) y muestra las proporciones de clase |
| 2. Entrenar y evaluar | `python -m src.train` | Elige hiperparámetros por validación cruzada, imprime métricas y reglas, guarda `models/congestion_tree.joblib` y `docs/img/arbol-decision.png` |
| 3. Predecir | `python -m src.predict --station ... --hour ...` | Predice el nivel de congestión de una estación |
| 4. Probar | `python -m pytest -v` | Ejecuta las 19 pruebas automatizadas |

Si falta `data/congestion.csv`, `src.train` lo genera; si falta el modelo, `src.predict` lo entrena y lo guarda.

### 3.1 Predictor

```bash
python -m src.predict --station "Calle 100" --hour 7 --day weekday --weather rain --event 0
```

| Argumento | Obligatorio | Valores | Por defecto |
|---|---|---|---|
| `--station` | sí | Nombre de una de las 20 estaciones; no distingue mayúsculas ni tildes | - |
| `--hour` | sí | Entero de 5 a 22 | - |
| `--day` | no | `weekday`, `saturday`, `sunday_holiday` | `weekday` |
| `--weather` | no | `dry`, `rain` | `dry` |
| `--event` | no | `0` (sin evento), `1` (evento cercano) | `0` |

Ejemplos:

```bash
python -m src.predict --station ricaurte --hour 7 --weather rain
python -m src.predict --station "Calle 100" --hour 21 --day sunday_holiday
python -m src.predict --station Héroes --hour 12 --day saturday
```

## 4. Cómo leer la salida

```text
Estación: Calle 100 | Hora: 7 | Día: weekday | Clima: rain | Evento: 0
Congestión prevista: alta
  alta: 0.79
  media: 0.21
  baja: 0.00
```

- Primera línea: parámetros usados (la estación aparece con su nombre oficial).
- `Congestión prevista`: la clase con mayor probabilidad (`alta`, `media` o `baja`).
- Las tres líneas siguientes son las probabilidades por clase, promediadas entre las líneas que sirven a la estación.

Códigos de salida: `0` éxito; `2` estación desconocida (con sugerencias) o argumentos inválidos (por ejemplo `--hour 3`).

```text
$ python -m src.predict --station "Calle 2" --hour 7
Estación desconocida: 'Calle 2'. ¿Quisiste decir: Calle 72, Calle 26, Calle 75?
```

## 5. Pruebas

```bash
python -m pytest -v
```

Resultado esperado: `19 passed`. Las pruebas del predictor usan un directorio temporal para el modelo y no modifican `models/`. El detalle está en `docs/informe-pruebas.md`.

## 6. Estructura del proyecto

```text
actividad-3/
├── README.md
├── requirements.txt
├── data/        # network.json (red) y congestion.csv (datos sintéticos)
├── docs/        # documentación e imagen del árbol
├── models/      # modelo entrenado (ignorado por Git)
├── src/         # dataset.py, train.py, predict.py
└── tests/       # pruebas con pytest
```

## 7. Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Se ejecutó fuera de la raíz o con `python src/train.py` | Ejecutar desde la raíz con `python -m src.train` |
| `ModuleNotFoundError: No module named 'pandas'` | Entorno virtual no activo o dependencias sin instalar | `source .venv/bin/activate` y `pip install -r requirements.txt` |
| La primera predicción tarda unos segundos | No hay modelo guardado y se entrena al vuelo | Ejecutar `python -m src.train` una vez |
