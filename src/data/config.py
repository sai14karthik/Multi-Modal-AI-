"""
Dataset configuration helpers.

Supports custom configs supplied via YAML.
If no config is provided, returns a minimal generic config.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Optional

import yaml


DEFAULT_CONFIG: Dict = {
    "dataset_name": "GenericDataset",
    "modalities": {
        "Mod1": {
            "folder": "Mod1",
            "classes": {
                "Class0": 0,
                "Class1": 1,
            },
        },
        "Mod2": {
            "folder": "Mod2",
            "classes": {
                "Class0": 0,
                "Class1": 1,
            },
        },
    },
    "metadata": {},
}


@lru_cache(maxsize=4)
def load_dataset_config(config_path: Optional[str]) -> Dict:
    """
    Load dataset configuration from YAML, falling back to DEFAULT_CONFIG.

    Args:
        config_path: Optional path to YAML file.

    Returns:
        Dictionary describing modalities, class mappings, and metadata pointers.
    """
    if not config_path:
        return DEFAULT_CONFIG

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "modalities" not in config:
        raise ValueError("Dataset config must define a 'modalities' section.")

    return config


def resolve_metadata_path(config: Dict, data_root: str) -> Optional[str]:
    """Return absolute path to metadata CSV if configured."""
    metadata_cfg = config.get("metadata", {})
    metadata_file = metadata_cfg.get("file")
    if not metadata_file:
        return None
    abs_path = metadata_file
    if not os.path.isabs(abs_path):
        abs_path = os.path.join(data_root, metadata_file)
    return abs_path

