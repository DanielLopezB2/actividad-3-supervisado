"""Predict station congestion: python -m src.predict --station NAME --hour H [--day D] [--weather W] [--event 0|1]."""
import argparse
import difflib
import sys
import unicodedata

import joblib
import pandas as pd

from src import train
from src.dataset import DAY_TYPES, FEATURES, HOURS, station_lines

LEVELS = {"high": "alta", "medium": "media", "low": "baja"}


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.strip().casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def resolve_station(query: str, names: list) -> str:
    """Match a station name ignoring case and accents; raise ValueError with suggestions on a miss."""
    by_key = {_normalize(name): name for name in names}
    key = _normalize(query)
    if key in by_key:
        return by_key[key]
    close = difflib.get_close_matches(key, by_key, n=3, cutoff=0.5)
    message = f"Estación desconocida: {query!r}."
    if close:
        message += f" ¿Quisiste decir: {', '.join(by_key[c] for c in close)}?"
    raise ValueError(message)


def load_model():
    """Load the saved pipeline, training and saving it first if it does not exist."""
    path = train.MODEL_PATH
    if not path.exists():
        X, y = train.load_data()
        X_train, _, y_train, _ = train.split(X, y)
        model, _ = train.train(X_train, y_train)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return model
    return joblib.load(path)


def predict_proba(model, station: str, lines: list, hour: int, day: str, weather: str, event: int) -> dict:
    """Class probabilities for a station, averaged over the lines that serve it."""
    rows = pd.DataFrame(
        [
            {
                "station": station,
                "line": line,
                "hour": hour,
                "day_type": day,
                "is_transfer": int(len(lines) >= 2),
                "weather": weather,
                "near_event": event,
            }
            for line in lines
        ]
    )[FEATURES]
    mean = model.predict_proba(rows).mean(axis=0)
    return dict(zip(model.classes_, mean))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predicción del nivel de congestión de una estación.")
    parser.add_argument("--station", required=True, help="nombre de la estación")
    parser.add_argument("--hour", required=True, type=int, choices=HOURS, metavar="5-22", help="hora del día")
    parser.add_argument("--day", choices=DAY_TYPES, default="weekday", help="tipo de día (por defecto: weekday)")
    parser.add_argument("--weather", choices=["dry", "rain"], default="dry", help="clima (por defecto: dry)")
    parser.add_argument("--event", type=int, choices=[0, 1], default=0, help="evento cercano (por defecto: 0)")
    args = parser.parse_args(argv)

    lines_of = station_lines()
    try:
        station = resolve_station(args.station, list(lines_of))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    proba = predict_proba(load_model(), station, lines_of[station], args.hour, args.day, args.weather, args.event)
    best = max(proba, key=proba.get)
    print(f"Estación: {station} | Hora: {args.hour} | Día: {args.day} | Clima: {args.weather} | Evento: {args.event}")
    print(f"Congestión prevista: {LEVELS[best]}")
    for level in ("high", "medium", "low"):
        print(f"  {LEVELS[level]}: {proba.get(level, 0.0):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
