"""Basic smoke tests for the inference server."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.inference.server import app

client = TestClient(app)


def test_predict_endpoint_exists():
    """Verify the /predict endpoint responds (model not loaded → 500, not 404)."""
    payload = {
        "geom_mask": [[0.0] * 8 for _ in range(8)],
        "re": 100.0,
        "nx": 8,
        "ny": 8,
    }
    resp = client.post("/predict", json=payload)
    # 500 is expected (model not loaded), but endpoint must exist (not 404)
    assert resp.status_code != 404
