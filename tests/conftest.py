import os

import pytest

from app.settings import DATASET_PATH, ENCODER_PATH, MODEL_PATH


@pytest.fixture(scope="session")
def client():
    """HTTP client for the API, without starting a real server.

    The API needs the trained model and the processed data. If they do not exist,
    the API tests are skipped with a message explaining what to run first.
    """
    if not all(os.path.exists(p) for p in [MODEL_PATH, ENCODER_PATH, DATASET_PATH]):
        pytest.skip("Faltan el modelo o los datos: ejecuta primero 'python run.py'.")
    from fastapi.testclient import TestClient

    from app.main import app
    return TestClient(app)
