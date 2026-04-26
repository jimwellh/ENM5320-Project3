# tests/test_inference.py
"""Tests for the FastAPI inference server."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

import src.inference.server as server_module
from src.inference.server import app, _build_input, _denormalize

client = TestClient(app)

FAKE_STATS = {
    "u_mean": 0.5,  "u_std": 0.3,
    "v_mean": 0.0,  "v_std": 0.1,
    "p_mean": 0.2,  "p_std": 0.05,
}


# ── _build_input ──────────────────────────────────────────────────────────────

def _init_geometry():
    from src.data_gen.geometry import CylinderGeometry
    server_module._geometry = CylinderGeometry(nx=128, ny=128)


def test_build_input_shape():
    _init_geometry()
    x = _build_input(0.4, 0.5, 100.0)
    assert x.shape == (1, 3, 128, 128)


def test_build_input_re_channel_is_uniform():
    _init_geometry()
    x = _build_input(0.4, 0.5, 110.0)
    re_channel = x[0, 1]
    expected = (110.0 - 20.0) / 180.0
    assert torch.allclose(re_channel, torch.full_like(re_channel, expected), atol=1e-6)


def test_build_input_inlet_left_column_only():
    _init_geometry()
    x = _build_input(0.4, 0.5, 100.0)
    inlet = x[0, 2]
    assert (inlet[:, 0] == 1.0).all(), "Left column should be 1.0"
    assert (inlet[:, 1:] == 0.0).all(), "All other columns should be 0.0"


def test_build_input_geom_mask_contains_cylinder():
    _init_geometry()
    x = _build_input(0.5, 0.5, 100.0)
    mask = x[0, 0].numpy()
    assert mask[64, 64] == 1.0


# ── _denormalize ──────────────────────────────────────────────────────────────

def test_denormalize_applies_stats():
    server_module._stats = FAKE_STATS
    pred = np.zeros((3, 8, 8), dtype=np.float32)
    u, v, p = _denormalize(pred)
    assert np.allclose(u, FAKE_STATS["u_mean"])
    assert np.allclose(v, FAKE_STATS["v_mean"])
    assert np.allclose(p, FAKE_STATS["p_mean"])


def test_denormalize_output_shapes():
    server_module._stats = FAKE_STATS
    pred = np.random.randn(3, 128, 128).astype(np.float32)
    u, v, p = _denormalize(pred)
    assert u.shape == (128, 128)
    assert v.shape == (128, 128)
    assert p.shape == (128, 128)


# ── /predict endpoint ─────────────────────────────────────────────────────────

def _setup_mock_server(h=128, w=128):
    fake_pred = np.zeros((1, 3, h, w), dtype=np.float32)
    mock_model = MagicMock(return_value=torch.from_numpy(fake_pred))
    mock_model.eval = MagicMock(return_value=mock_model)
    server_module._model = mock_model
    server_module._stats = FAKE_STATS
    from src.data_gen.geometry import CylinderGeometry
    server_module._geometry = CylinderGeometry(nx=w, ny=h)


def test_predict_returns_503_when_model_not_loaded():
    server_module._model = None
    resp = client.post("/predict", json={"cx": 0.4, "cy": 0.5, "re": 100.0})
    assert resp.status_code == 503


def test_predict_returns_200_with_mock_model():
    _setup_mock_server()
    resp = client.post(
        "/predict",
        json={"cx": 0.4, "cy": 0.5, "re": 100.0, "export_usd": False},
    )
    assert resp.status_code == 200


def test_predict_response_has_correct_field_shape():
    _setup_mock_server()
    resp = client.post(
        "/predict",
        json={"cx": 0.4, "cy": 0.5, "re": 100.0, "export_usd": False},
    )
    data = resp.json()
    assert len(data["u"]) == 128
    assert len(data["u"][0]) == 128


def test_predict_response_includes_usd_path_when_requested():
    _setup_mock_server()
    from src.inference import usd_exporter
    original = usd_exporter.export_flow_to_usd
    usd_exporter.export_flow_to_usd = lambda u, v, p, **kw: "C:/fake/flow.usda"
    try:
        resp = client.post(
            "/predict",
            json={"cx": 0.4, "cy": 0.5, "re": 100.0, "export_usd": True},
        )
        data = resp.json()
        assert data["usd_path"] is not None
        assert data["usd_path"].endswith(".usda")
    finally:
        usd_exporter.export_flow_to_usd = original


def test_predict_usd_path_is_none_when_not_requested():
    _setup_mock_server()
    resp = client.post(
        "/predict",
        json={"cx": 0.4, "cy": 0.5, "re": 100.0, "export_usd": False},
    )
    data = resp.json()
    assert data["usd_path"] is None
