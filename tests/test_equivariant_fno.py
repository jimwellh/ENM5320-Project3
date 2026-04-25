"""Unit tests for EquivariantFNO architecture components."""
import pytest
import torch


def test_spectral_conv2d_output_shape():
    from src.models.equivariant_fno import SpectralConv2d
    sc = SpectralConv2d(in_channels=64, out_channels=64, modes=8)
    x = torch.randn(2, 64, 32, 32)
    y = sc(x)
    assert y.shape == x.shape


def test_spectral_conv2d_different_channels():
    from src.models.equivariant_fno import SpectralConv2d
    sc = SpectralConv2d(in_channels=8, out_channels=16, modes=4)
    x = torch.randn(2, 8, 32, 32)
    y = sc(x)
    assert y.shape == (2, 16, 32, 32)


def test_equivariant_fno_output_shape_small():
    from src.models.equivariant_fno import EquivariantFNO
    model = EquivariantFNO(in_channels=3, out_channels=3, modes=8, width=4, n_layers=2)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 3, 32, 32)


def test_equivariant_fno_output_shape_full():
    from src.models.equivariant_fno import EquivariantFNO
    model = EquivariantFNO(in_channels=3, out_channels=3, modes=16, width=16, n_layers=4)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    assert y.shape == (1, 3, 128, 128)


def test_equivariant_fno_rejects_wrong_out_channels():
    from src.models.equivariant_fno import EquivariantFNO
    with pytest.raises(AssertionError):
        EquivariantFNO(in_channels=3, out_channels=5, modes=8, width=4, n_layers=2)


def test_equivariant_fno_build_via_factory():
    from src.models.model_factory import build_model
    model = build_model(
        "equivariant_fno",
        in_channels=3, out_channels=3, modes=8, width=4, n_layers=2, group="p4"
    )
    assert model is not None
