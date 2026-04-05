from src.data.dataloaders import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(image_root="data/raw/")

images, labels = next(iter(train_loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)