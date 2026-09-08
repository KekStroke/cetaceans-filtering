"""Reproducible, publication-facing validation helpers for animal2vec."""

from .protocol import (
    ProtocolError,
    SplitManifest,
    evaluate_selected_layer,
    load_split_manifest,
    select_layer_from_validation,
)
from .runtime import ConfigError, InputSpec, extract_features, input_spec_from_config

__all__ = [
    "ConfigError",
    "InputSpec",
    "ProtocolError",
    "SplitManifest",
    "evaluate_selected_layer",
    "extract_features",
    "input_spec_from_config",
    "load_split_manifest",
    "select_layer_from_validation",
]
