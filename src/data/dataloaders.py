from __future__ import annotations

from typing import List, Tuple

from torch.utils.data import DataLoader

from src.common.constants import DEFAULT_TARGET_LABELS
from src.data.dataset import CXRDataset, get_eval_transform, get_train_transform


def get_dataloaders(
    image_root: str,
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = 224,
    grayscale: bool = True,
    target_labels: List[str] | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    target_labels = target_labels or DEFAULT_TARGET_LABELS

    train_dataset = CXRDataset(
        csv_file="data/processed/train.csv",
        image_root=image_root,
        target_labels=target_labels,
        transform=get_train_transform(image_size=image_size, grayscale=grayscale),
        grayscale=grayscale,
    )

    val_dataset = CXRDataset(
        csv_file="data/processed/val.csv",
        image_root=image_root,
        target_labels=target_labels,
        transform=get_eval_transform(image_size=image_size, grayscale=grayscale),
        grayscale=grayscale,
    )

    test_dataset = CXRDataset(
        csv_file="data/processed/test.csv",
        image_root=image_root,
        target_labels=target_labels,
        transform=get_eval_transform(image_size=image_size, grayscale=grayscale),
        grayscale=grayscale,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader