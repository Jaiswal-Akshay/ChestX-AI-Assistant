from torch.utils.data import DataLoader

from src.data.dataset import CXRDataset, get_train_transform, get_eval_transform


def get_dataloaders(image_root="data/raw", batch_size=16, num_workers=0):
    train_ds = CXRDataset(
        csv_file="data/processed/train.csv",
        image_root=image_root,
        transform=get_train_transform(),
    )
    val_ds = CXRDataset(
        csv_file="data/processed/val.csv",
        image_root=image_root,
        transform=get_eval_transform(),
    )
    test_ds = CXRDataset(
        csv_file="data/processed/test.csv",
        image_root=image_root,
        transform=get_eval_transform(),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader