import os
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.data.quality_checks import check_image_quality

TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Pleural Effusion",
    "Pneumonia",
]

RAW_ROOT = "data/raw"
INPUT_CSV = "data/raw/CheXpert-v1.0-small/train.csv"
OUTPUT_DIR = "data/processed"


def clean_label(x):
    """
    Baseline label policy:
    1 -> 1
    0 -> 0
    -1 -> 0
    NaN -> 0
    """
    if pd.isna(x):
        return 0
    if x == 1:
        return 1
    return 0


def extract_patient_id(path_str: str) -> str:
    """
    Example path:
    CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
    -> patient00001
    """
    parts = str(path_str).split("/")
    for part in parts:
        if part.startswith("patient"):
            return part
    return str(path_str)


def resolve_full_path(path_str: str) -> str:
    return os.path.join(RAW_ROOT, path_str)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    print("Original rows:", len(df))

    # 1. Keep frontal only
    if "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"].copy()
        print("After frontal filter:", len(df))

    # 2. Keep only required columns
    keep_cols = ["Path"] + TARGET_LABELS
    df = df[keep_cols].copy()

    # 3. Remove missing paths
    df = df.dropna(subset=["Path"]).copy()

    # 4. Remove duplicate image paths
    df = df.drop_duplicates(subset=["Path"]).copy()
    print("After missing/duplicate cleanup:", len(df))

    # 5. Clean labels
    for label in TARGET_LABELS:
        df[label] = df[label].apply(clean_label).astype(int)

    # 6. Add patient id
    df["patient_id"] = df["Path"].apply(extract_patient_id)

    # 7. Quality checks
    records = []
    removed_missing = 0
    removed_bad_open = 0

    for _, row in df.iterrows():
        full_path = resolve_full_path(row["Path"])

        if not os.path.exists(full_path):
            removed_missing += 1
            continue

        try:
            img = Image.open(full_path).convert("L")
        except Exception:
            removed_bad_open += 1
            continue

        qc = check_image_quality(img)

        record = row.to_dict()
        record["too_small"] = qc["too_small"]
        record["blank"] = qc["blank"]
        record["low_contrast"] = qc["low_contrast"]
        record["remove_for_quality"] = (
            qc["too_small"] or qc["blank"] or qc["low_contrast"]
        )
        records.append(record)

    clean_df = pd.DataFrame(records)

    # Save quality report before filtering
    clean_df.to_csv(os.path.join(OUTPUT_DIR, "all_with_quality_flags.csv"), index=False)

    # 8. Remove low-quality rows
    filtered_df = clean_df[clean_df["remove_for_quality"] == False].copy()

    print("Removed missing files:", removed_missing)
    print("Removed unreadable files:", removed_bad_open)
    print("Flagged too small:", int(clean_df["too_small"].sum()))
    print("Flagged blank:", int(clean_df["blank"].sum()))
    print("Flagged low contrast:", int(clean_df["low_contrast"].sum()))
    print("Remaining after quality filter:", len(filtered_df))

    # 9. Patient-level split
    patient_ids = filtered_df["patient_id"].unique()

    train_patients, temp_patients = train_test_split(
        patient_ids, test_size=0.30, random_state=42
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.50, random_state=42
    )

    train_df = filtered_df[filtered_df["patient_id"].isin(train_patients)].copy()
    val_df = filtered_df[filtered_df["patient_id"].isin(val_patients)].copy()
    test_df = filtered_df[filtered_df["patient_id"].isin(test_patients)].copy()

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    # 10. Save summary table
    summary = pd.DataFrame(
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
            ],
            "value": [
                len(df),
                len(filtered_df),
                len(train_df),
                len(val_df),
                len(test_df),
                removed_missing,
                removed_bad_open,
                int(clean_df["too_small"].sum()),
                int(clean_df["blank"].sum()),
                int(clean_df["low_contrast"].sum()),
            ],
        }
    )
    summary.to_csv(os.path.join(OUTPUT_DIR, "preprocessing_summary.csv"), index=False)

    print("\nSaved:")
    print("- data/processed/all_with_quality_flags.csv")
    print("- data/processed/train.csv")
    print("- data/processed/val.csv")
    print("- data/processed/test.csv")
    print("- data/processed/preprocessing_summary.csv")


if __name__ == "__main__":
    main()