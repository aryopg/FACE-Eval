"""Sampling-parameter resolution shared by the runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_SAMPLING_CONFIG = "config/sampling.yaml"


def resolve_sampling_params(model: str, cli_overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge per-model sampling config with explicit CLI overrides.

    Precedence: explicit CLI flag (non-None) > model pattern match > global defaults.
    """
    config = yaml.safe_load(Path(_SAMPLING_CONFIG).read_text())
    params: dict[str, Any] = dict(config.get("defaults", {}))

    model_lower = model.lower()
    for pattern, overrides in config.get("models", {}).items():
        if pattern.lower() in model_lower:
            params.update(overrides)
            break

    for key, val in cli_overrides.items():
        if val is not None:
            params[key] = val

    return params
