import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchvision import models

from src.data.dataloaders import get_dataloaders
from src.data.dataset import TARGET_LABELS

DEVICE = torch.device("cpu")
MODEL_PATH = "outputs/models/best_resnet18.pth"


def get_model():
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, len(TARGET_LABELS))
    return model


def main():
    print(f"Using device: {DEVICE}")

    _, _, test_loader = get_dataloaders(
        image_root="data/raw",
        batch_size=16,
        num_workers=0,
    )

    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)

    print("\nTest AUROC per class:")
    scores = []

    for i, label in enumerate(TARGET_LABELS):
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            print(f"{label}: {auc:.4f}")
            scores.append(auc)
        except ValueError:
            print(f"{label}: AUROC not defined")

    if scores:
        print(f"\nMean Test AUROC: {np.mean(scores):.4f}")


if __name__ == "__main__":
    main()