"""Unified model construction interface."""

import torch.nn as nn
from .fno_baseline import FNOBaseline


def build_model(model_type: str, **kwargs) -> nn.Module:
    """Instantiate a model by name.

    Args:
        model_type: One of 'fno_baseline', 'equivariant_fno', 'pino'.
        **kwargs: Passed to the model constructor.

    Returns:
        Instantiated nn.Module.
    """
    if model_type == "fno_baseline":
        return FNOBaseline(**kwargs)
    elif model_type == "equivariant_fno":
        from .equivariant_fno import EquivariantFNO
        return EquivariantFNO(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
