import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

RAW_ROOT = "data/raw"
TRAIN_CSV = "data/processed/train.csv"

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Pleural Effusion",
    "Pneumonia",
]


def show_label_distribution(df):
    counts = df[LABELS].sum()
    plt.figure()
    counts.plot(kind="bar")
    plt.title("Label Distribution")
    plt.ylabel("Positive Cases")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_sample_images(df, n=4):
    sample_df = df.sample(n=min(n, len(df)), random_state=42)

    for _, row in sample_df.iterrows():
        img_path = os.path.join(RAW_ROOT, row["Path"])
        img = Image.open(img_path).convert("L")
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title(str({label: int(row[label]) for label in LABELS}))
        plt.axis("off")
        plt.show()


def show_pixel_histogram(df):
    img_path = os.path.join(RAW_ROOT, df.iloc[0]["Path"])
    img = Image.open(img_path).convert("L")
    arr = np.array(img)

    plt.figure()
    plt.hist(arr.flatten(), bins=50)
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv(TRAIN_CSV)
    print("Train rows:", len(df))
    print("\nMissing values:\n")
    print(df.isnull().sum())

    show_label_distribution(df)
    show_sample_images(df, n=4)
    show_pixel_histogram(df)


if __name__ == "__main__":
    main()