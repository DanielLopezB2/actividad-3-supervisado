import pytest

from src import predict, train


@pytest.fixture(scope="module", autouse=True)
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "tree.joblib"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train, "MODEL_PATH", path)
        yield path


def run(capsys, *argv):
    code = predict.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_missing_model_is_trained_on_the_fly(model_path):
    assert not model_path.exists()
    predict.load_model()
    assert model_path.exists()


def test_resolve_station_ignores_case_and_accents():
    names = ["Héroes", "Calle 26", "Normandía"]
    assert predict.resolve_station("heroes", names) == "Héroes"
    assert predict.resolve_station("  NORMANDIA ", names) == "Normandía"


def test_resolve_station_unknown_suggests_close_names():
    with pytest.raises(ValueError) as exc:
        predict.resolve_station("Calle 2", ["Calle 26", "Calle 72", "Héroes"])
    assert "Estación desconocida: 'Calle 2'." in str(exc.value)
    assert "¿Quisiste decir: " in str(exc.value) and "Calle 26" in str(exc.value)


def test_weekday_rush_hour_transfer_station_in_rain_is_high(capsys):
    code, out, _ = run(capsys, "--station", "ricaurte", "--hour", "7", "--day", "weekday", "--weather", "rain")
    assert code == 0
    assert "Congestión prevista: alta" in out


def test_sunday_night_non_transfer_station_is_low(capsys):
    code, out, _ = run(capsys, "--station", "Calle 100", "--hour", "21", "--day", "sunday_holiday")
    assert code == 0
    assert "Congestión prevista: baja" in out


def test_output_lists_probabilities_for_every_level(capsys):
    _, out, _ = run(capsys, "--station", "Héroes", "--hour", "12", "--day", "saturday")
    for label in ("alta", "media", "baja"):
        assert f"  {label}: " in out


def test_unknown_station_exits_2_with_suggestion(capsys):
    code, out, err = run(capsys, "--station", "Calle 2", "--hour", "7")
    assert code == 2
    assert out == ""
    assert "Estación desconocida: 'Calle 2'." in err and "Calle 26" in err


def test_hour_out_of_range_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        predict.main(["--station", "Calle 100", "--hour", "3"])
    assert exc.value.code == 2
