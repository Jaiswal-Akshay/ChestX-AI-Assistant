import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Pleural Effusion",
    "Pneumonia",
]

class CXRDataset(Dataset):
    def __init__(self, csv_file, image_root="", transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["Path"])
        image = Image.open(img_path).convert("L")

        image = self.transform(image)

        labels = torch.tensor(
            [row[label] for label in TARGET_LABELS],
            dtype=torch.float32
        )

        return image, labels