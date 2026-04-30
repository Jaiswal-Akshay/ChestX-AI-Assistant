from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    torch.save(state, path)


def compute_pos_weights(train_csv_path: str, target_labels: list[str]) -> torch.Tensor:
    import pandas as pd

    df = pd.read_csv(train_csv_path)

    weights = []
    for label in target_labels:
        positives = float(df[label].sum())
        negatives = float(len(df) - positives)

        if positives == 0:
            weights.append(1.0)
        else:
            weights.append(negatives / positives)

    return torch.tensor(weights, dtype=torch.float32)