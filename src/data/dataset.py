import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Pleural Effusion",
    "Pneumonia",
]


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class CXRDataset(Dataset):
    def __init__(self, csv_file: str, image_root: str, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform or get_eval_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["Path"])

        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        labels = torch.tensor(
            [row[label] for label in TARGET_LABELS],
            dtype=torch.float32
        )

        return image, labels