from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

from src.common.constants import DEFAULT_CONFIG_PATH, DEFAULT_TARGET_LABELS
from src.data.quality_checks import check_image_quality


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)


def clean_label(value: float, policy: str = "u_zeros") -> int:
    """
    Supported policies:
    - u_zeros: -1 -> 0
    - u_ones:  -1 -> 1
    """
    if pd.isna(value):
        return 0

    if value == 1:
        return 1

    if value == 0:
        return 0

    if value == -1:
        if policy == "u_ones":
            return 1
        return 0

    return 0


def extract_patient_id(path_str: str) -> str:
    """
    Example:
    CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
    -> patient00001
    """
    for part in str(path_str).split("/"):
        if part.startswith("patient"):
            return part
    return "unknown_patient"


def resolve_full_path(raw_root: str, relative_path: str) -> str:
    return os.path.join(raw_root, relative_path)


def compute_label_prevalence(df: pd.DataFrame, target_labels: List[str]) -> Dict[str, float]:
    if len(df) == 0:
        return {label: 0.0 for label in target_labels}
    return {label: float(df[label].mean()) for label in target_labels}


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Split ratios must sum to 1.0, but got {total:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )


def patient_level_split(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    patient_ids = df["patient_id"].unique()

    train_patients, temp_patients = train_test_split(
        patient_ids,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        shuffle=True,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=relative_test_ratio,
        random_state=seed,
        shuffle=True,
    )

    train_df = df[df["patient_id"].isin(train_patients)].copy()
    val_df = df[df["patient_id"].isin(val_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()

    return train_df, val_df, test_df


def run_quality_checks(
    df: pd.DataFrame,
    raw_root: str,
    min_size: int,
    std_threshold: float,
    contrast_threshold: float,
) -> Tuple[pd.DataFrame, int, int]:
    records = []
    removed_missing = 0
    removed_bad_open = 0

    for _, row in df.iterrows():
        full_path = resolve_full_path(raw_root, row["Path"])

        if not os.path.exists(full_path):
            removed_missing += 1
            continue

        try:
            image = Image.open(full_path).convert("L")
        except Exception:
            removed_bad_open += 1
            continue

        qc = check_image_quality(
            image,
            min_size=min_size,
            std_threshold=std_threshold,
            contrast_threshold=contrast_threshold,
        )

        record = row.to_dict()
        record["full_path"] = full_path
        record["too_small"] = qc["too_small"]
        record["blank"] = qc["blank"]
        record["low_contrast"] = qc["low_contrast"]
        record["remove_for_quality"] = (
            qc["too_small"] or qc["blank"] or qc["low_contrast"]
        )
        records.append(record)

    clean_df = pd.DataFrame(records)
    return clean_df, removed_missing, removed_bad_open


def save_summary_files(
    output_dir: Path,
    original_rows: int,
    cleaned_rows: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    removed_missing: int,
    removed_bad_open: int,
    flagged_df: pd.DataFrame,
    target_labels: List[str],
) -> None:
    summary_df = pd.DataFrame(
        {
            "metric": [
                "original_rows",
                "after_cleaning_rows",
                "train_rows",
                "val_rows",
                "test_rows",
                "removed_missing_files",
                "removed_unreadable_files",
                "flagged_too_small",
                "flagged_blank",
                "flagged_low_contrast",
                "train_patients",
                "val_patients",
                "test_patients",
            ],
            "value": [
                original_rows,
                cleaned_rows,
                len(train_df),
                len(val_df),
                len(test_df),
                removed_missing,
                removed_bad_open,
                int(flagged_df["too_small"].sum()),
                int(flagged_df["blank"].sum()),
                int(flagged_df["low_contrast"].sum()),
                int(train_df["patient_id"].nunique()),
                int(val_df["patient_id"].nunique()),
                int(test_df["patient_id"].nunique()),
            ],
        }
    )
    summary_df.to_csv(output_dir / "preprocessing_summary.csv", index=False)

    split_summary = {
        "train": {
            "rows": int(len(train_df)),
            "patients": int(train_df["patient_id"].nunique()),
            "label_prevalence": compute_label_prevalence(train_df, target_labels),
        },
        "val": {
            "rows": int(len(val_df)),
            "patients": int(val_df["patient_id"].nunique()),
            "label_prevalence": compute_label_prevalence(val_df, target_labels),
        },
        "test": {
            "rows": int(len(test_df)),
            "patients": int(test_df["patient_id"].nunique()),
            "label_prevalence": compute_label_prevalence(test_df, target_labels),
        },
    }

    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)


def build_processed_dataframe(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    labels_cfg = config["labels"]

    raw_root = data_cfg["raw_root"]
    input_csv = data_cfg["chexpert_csv"]
    frontal_only = data_cfg.get("frontal_only", True)
    target_labels = labels_cfg.get("target_labels", DEFAULT_TARGET_LABELS)
    uncertainty_policy = labels_cfg.get("uncertainty_policy", "u_zeros")

    df = pd.read_csv(input_csv)
    original_rows = len(df)

    print(f"Original rows: {original_rows}")

    if frontal_only and "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"].copy()
        print(f"After frontal-only filter: {len(df)}")

    keep_cols = ["Path"] + target_labels
    df = df[keep_cols].copy()

    df = df.dropna(subset=["Path"]).copy()
    df = df.drop_duplicates(subset=["Path"]).copy()

    for label in target_labels:
        df[label] = df[label].apply(lambda x: clean_label(x, policy=uncertainty_policy)).astype(int)

    df["patient_id"] = df["Path"].apply(extract_patient_id)

    print(f"After cleanup and label processing: {len(df)}")

    quality_cfg = config["quality"]
    flagged_df, removed_missing, removed_bad_open = run_quality_checks(
        df=df,
        raw_root=raw_root,
        min_size=quality_cfg["min_size"],
        std_threshold=quality_cfg["std_threshold"],
        contrast_threshold=quality_cfg["contrast_threshold"],
    )

    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    flagged_df.to_csv(processed_dir / "all_with_quality_flags.csv", index=False)

    if quality_cfg.get("remove_low_quality", True):
        df_final = flagged_df[flagged_df["remove_for_quality"] == False].copy()
    else:
        df_final = flagged_df.copy()

    print(f"Removed missing files: {removed_missing}")
    print(f"Removed unreadable files: {removed_bad_open}")
    print(f"Flagged too small: {int(flagged_df['too_small'].sum())}")
    print(f"Flagged blank: {int(flagged_df['blank'].sum())}")
    print(f"Flagged low contrast: {int(flagged_df['low_contrast'].sum())}")
    print(f"Remaining after quality filtering: {len(df_final)}")

    split_cfg = config["split"]
    train_df, val_df, test_df = patient_level_split(
        df=df_final,
        seed=config["seed"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
    )

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    save_summary_files(
        output_dir=processed_dir,
        original_rows=original_rows,
        cleaned_rows=len(df_final),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        removed_missing=removed_missing,
        removed_bad_open=removed_bad_open,
        flagged_df=flagged_df,
        target_labels=target_labels,
    )

    print("\nSaved files:")
    print(f"- {processed_dir / 'all_with_quality_flags.csv'}")
    print(f"- {processed_dir / 'train.csv'}")
    print(f"- {processed_dir / 'val.csv'}")
    print(f"- {processed_dir / 'test.csv'}")
    print(f"- {processed_dir / 'preprocessing_summary.csv'}")
    print(f"- {processed_dir / 'split_summary.json'}")

    return df_final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CheXpert data for training.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    build_processed_dataframe(config)


if __name__ == "__main__":
    main()