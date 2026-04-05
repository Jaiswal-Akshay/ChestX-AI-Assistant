from torch.utils.data import DataLoader
from src.data.dataset import CXRDataset

def get_dataloaders(image_root="", batch_size=16):
    train_ds = CXRDataset("data/processed/train.csv", image_root=image_root)
    val_ds = CXRDataset("data/processed/val.csv", image_root=image_root)
    test_ds = CXRDataset("data/processed/test.csv", image_root=image_root)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader