from pathlib import Path

DEFAULT_TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Pleural Effusion",
    "Pneumonia",
]

DEFAULT_CONFIG_PATH = "configs/baseline.yaml"

DEFAULT_OUTPUT_DIRS = [
    Path("data/processed"),
    Path("outputs/models"),
    Path("outputs/reports"),
    Path("outputs/figures"),
]