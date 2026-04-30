from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.common.constants import DEFAULT_TARGET_LABELS


def get_train_transform(image_size: int = 224, grayscale: bool = True) -> transforms.Compose:
    if grayscale:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_eval_transform(image_size: int = 224, grayscale: bool = True) -> transforms.Compose:
    if grayscale:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class CXRDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        image_root: str,
        target_labels: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        grayscale: bool = True,
        return_metadata: bool = False,
    ) -> None:
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.target_labels = target_labels or DEFAULT_TARGET_LABELS
        self.grayscale = grayscale
        self.return_metadata = return_metadata

        if transform is None:
            self.transform = get_eval_transform(grayscale=grayscale)
        else:
            self.transform = transform

        required_columns = {"Path", *self.target_labels}
        missing = required_columns.difference(set(self.df.columns))
        if missing:
            raise ValueError(f"Missing required columns in {csv_file}: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, img_path: str) -> Image.Image:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path)
        image = image.convert("L" if self.grayscale else "RGB")
        return image

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        relative_path = row["Path"]
        full_path = os.path.join(self.image_root, relative_path)

        image = self._load_image(full_path)
        image = self.transform(image)

        labels = torch.tensor(
            [float(row[label]) for label in self.target_labels],
            dtype=torch.float32,
        )

        if self.return_metadata:
            metadata: Dict[str, str] = {
                "path": relative_path,
                "full_path": full_path,
                "patient_id": row.get("patient_id", "unknown_patient"),
            }
            return image, labels, metadata

        return image, labels