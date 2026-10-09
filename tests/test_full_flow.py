import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from run import PIPELINE

PROJECT = Path(__file__).resolve().parent.parent

# Small program run inside the copy of the project: it uses the API like the web page does
API_FLOW = """
import json
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
found = client.get("/products", params={"q": "paceña lata"}).json()["results"][0]
week = client.post("/forecast-week", json={"date": "2026-10-01", "product_id": found["product_id"]})
metrics = client.get("/metrics").json()
print(json.dumps({"product": found, "status": week.status_code, "days": len(week.json()["days"]), "metrics": metrics}))
"""


@pytest.mark.slow
def test_full_flow_from_catalog_to_forecast(tmp_path):
    """End-to-end test: the same steps as the README, on a clean copy of the project.

    1. Copy the code and the catalog (without generated data, model or virtual environment).
    2. Run every pipeline step, in the same order as run.py.
    3. Check that the model and the metrics were created and that the model beats the baseline.
    4. Use the API like the web page: search a product and forecast its week.
    """
    copy = tmp_path / "project"
    shutil.copytree(PROJECT, copy, ignore=shutil.ignore_patterns(
        "env", "venv", ".venv", ".git", "__pycache__", "artifacts", "*.csv", "tests",
    ))

    for script, _ in PIPELINE:
        result = subprocess.run([sys.executable, script], cwd=copy, capture_output=True, text=True)
        assert result.returncode == 0, f"Falló {script}:\n{result.stderr}"

    for artifact in ["model.pkl", "product_encoder.pkl", "metrics.json", "prediction_interval.json"]:
        assert (copy / "artifacts" / artifact).exists(), f"No se creó {artifact}"

    result = subprocess.run([sys.executable, "-c", API_FLOW], cwd=copy, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    flow = json.loads(result.stdout.strip().splitlines()[-1])

    assert flow["product"]["product_id"] == "CVZ-001"
    assert flow["status"] == 200
    assert flow["days"] == 7
    # The chosen model must beat the simple rule of copying last week (MASE below the baseline)
    assert flow["metrics"]["xgboost"]["mase"] < flow["metrics"]["baseline"]["mase"]
