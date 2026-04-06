import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torchvision import models
from tqdm import tqdm

from src.data.dataloaders import get_dataloaders
from src.data.dataset import TARGET_LABELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(TARGET_LABELS)
MODEL_SAVE_PATH = "outputs/models/best_resnet18.pth"


def compute_class_weights(train_csv_path: str) -> torch.Tensor:
    """
    Compute positive class weights for BCEWithLogitsLoss.
    pos_weight = negatives / positives
    """
    df = pd.read_csv(train_csv_path)

    pos_weights = []
    for label in TARGET_LABELS:
        positives = df[label].sum()
        negatives = len(df) - positives

        if positives == 0:
            weight = 1.0
        else:
            weight = negatives / positives

        pos_weights.append(weight)

    return torch.tensor(pos_weights, dtype=torch.float32)


def get_model() -> nn.Module:
    """
    Load ResNet18, adapt for grayscale input and multi-label output.
    """
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Compute AUROC per class and mean AUROC.
    Handles cases where one class has only one label.
    """
    aurocs = {}

    for i, label in enumerate(TARGET_LABELS):
        try:
            score = roc_auc_score(y_true[:, i], y_prob[:, i])
            aurocs[label] = float(score)
        except ValueError:
            aurocs[label] = None

    valid_scores = [score for score in aurocs.values() if score is not None]
    mean_auroc = float(np.mean(valid_scores)) if valid_scores else None

    return aurocs, mean_auroc


def run_one_epoch(model, loader, criterion, optimizer=None):
    """
    Run one training or validation epoch.

    If optimizer is provided -> training mode
    If optimizer is None     -> evaluation mode
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_labels = []
    all_probs = []

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            probs = torch.sigmoid(outputs)

            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader)
    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)

    aurocs, mean_auroc = compute_auroc(y_true, y_prob)

    return epoch_loss, aurocs, mean_auroc


def save_metrics_text(epoch, train_loss, val_loss, train_aurocs, val_aurocs, train_mean, val_mean):
    """
    Append metrics to a training log file.
    """
    os.makedirs("outputs/reports", exist_ok=True)
    metrics_path = "outputs/reports/training_log.txt"

    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(f"\nEpoch {epoch}\n")
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Val Loss: {val_loss:.4f}\n")
        f.write(f"Train Mean AUROC: {train_mean}\n")
        f.write(f"Val Mean AUROC: {val_mean}\n")

        f.write("Train AUROC per class:\n")
        for label, score in train_aurocs.items():
            f.write(f"  {label}: {score}\n")

        f.write("Val AUROC per class:\n")
        for label, score in val_aurocs.items():
            f.write(f"  {label}: {score}\n")


def train(num_epochs=5, batch_size=16, lr=1e-4):
    """
    Main training function.
    """
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        image_root="data/raw",
        batch_size=batch_size,
        num_workers=0,
    )

    model = get_model().to(DEVICE)

    pos_weight = compute_class_weights("data/processed/train.csv").to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_mean_auroc = -1.0

    print(f"Using device: {DEVICE}")
    print("Target labels:", TARGET_LABELS)
    print("Class weights:", pos_weight.detach().cpu().numpy())

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")

        train_loss, train_aurocs, train_mean_auroc = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        val_loss, val_aurocs, val_mean_auroc = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Train Mean AUROC: {train_mean_auroc}")
        print(f"Val Mean AUROC:   {val_mean_auroc}")

        print("\nValidation AUROC per class:")
        for label, score in val_aurocs.items():
            print(f"  {label}: {score}")

        save_metrics_text(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_aurocs=train_aurocs,
            val_aurocs=val_aurocs,
            train_mean=train_mean_auroc,
            val_mean=val_mean_auroc,
        )

        if val_mean_auroc is not None and val_mean_auroc > best_val_mean_auroc:
            best_val_mean_auroc = val_mean_auroc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model to {MODEL_SAVE_PATH}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train(num_epochs=5, batch_size=16, lr=1e-4)