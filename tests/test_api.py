# Tests of the API with the real trained model.
# They need the model and the data: run "python run.py" before "pytest".

PRODUCT = "PACEÑA LATA GRANDE"


def test_valid_case_weekly_forecast(client):
    """Valid case: forecast of a normal week for a known product."""
    response = client.post("/forecast-week", json={"date": "2026-10-01", "name": PRODUCT})

    assert response.status_code == 200
    data = response.json()
    assert len(data["days"]) == 7
    assert data["days"][0]["date"] == "2026-10-01"
    assert data["days"][-1]["date"] == "2026-10-07"
    predictions = [day["predicted"] for day in data["days"]]
    assert all(isinstance(p, int) and p >= 0 for p in predictions)
    # The weekly total is the sum of the days (allowing for the rounding of each day)
    assert abs(data["week_total"] - sum(predictions)) <= 7


def test_difficult_case_carnival_week(client):
    """Difficult case: Carnival changes date every year and appears only 8 days in the data.
    The model must still recognize it: the Carnival week must sell more than a normal week."""
    carnival = client.post("/forecast-week", json={"date": "2026-02-12", "name": PRODUCT}).json()
    normal = client.post("/forecast-week", json={"date": "2026-03-12", "name": PRODUCT}).json()

    assert carnival["week_total"] > normal["week_total"]


def test_invalid_input(client):
    """Invalid input: the API must answer with a clear error instead of failing."""
    wrong_date = client.post("/forecast-week", json={"date": "01/10/2026", "name": PRODUCT})
    assert wrong_date.status_code == 422  # data with the wrong format

    unknown_product = client.post("/forecast-week", json={"date": "2026-10-01", "name": "producto que no existe"})
    assert unknown_product.status_code == 404  # product not found
    assert "No se encontró" in unknown_product.json()["detail"]
