import os
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Pleural Effusion",
    "Pneumonia",
]

def clean_label(x):
    if pd.isna(x):
        return 0
    if x == 1:
        return 1
    if x in [0, -1]:
        return 0
    return 0

def main():
    csv_path = "data/raw/train.csv"   # update for your dataset
    df = pd.read_csv(csv_path)

    # Keep only frontal images if column exists
    if "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"].copy()

    # Keep required columns
    cols = ["Path"] + TARGET_LABELS
    df = df[cols].copy()

    # Clean labels
    for label in TARGET_LABELS:
        df[label] = df[label].apply(clean_label)

    # Create patient id from path if no explicit patient column
    df["patient_id"] = df["Path"].apply(lambda p: str(p).split("/")[2] if len(str(p).split("/")) > 2 else str(p))

    # Patient-level split
    patient_ids = df["patient_id"].unique()

    train_patients, temp_patients = train_test_split(
        patient_ids, test_size=0.30, random_state=42
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.50, random_state=42
    )

    train_df = df[df["patient_id"].isin(train_patients)].copy()
    val_df = df[df["patient_id"].isin(val_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("Saved splits:")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

if __name__ == "__main__":
    main()