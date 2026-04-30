from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.training.metrics import multilabel_auroc
from src.training.utils import save_checkpoint


def run_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer=None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_targets = []
    all_probs = []

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        total_loss += loss.item() * images.size(0)

        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(probs)

    epoch_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    return epoch_loss, y_true, y_prob


def write_metrics_row(csv_path: Path, row: Dict[str, float], fieldnames: List[str]) -> None:
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    label_names: List[str],
    save_dir: str,
    early_stopping_patience: int = 3,
) -> Dict[str, float]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    metrics_csv = save_path / "metrics.csv"

    best_val_auroc = -1.0
    best_epoch = -1
    patience_counter = 0

    history = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_y, train_p = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        train_metrics = multilabel_auroc(train_y, train_p, label_names)

        val_loss, val_y, val_p = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        val_metrics = multilabel_auroc(val_y, val_p, label_names)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mean_auroc": train_metrics["mean_auroc"],
            "val_mean_auroc": val_metrics["mean_auroc"],
        }

        for label in label_names:
            row[f"train_auroc_{label}"] = train_metrics[label]
            row[f"val_auroc_{label}"] = val_metrics[label]

        fieldnames = list(row.keys())
        write_metrics_row(metrics_csv, row, fieldnames)
        history.append(row)

        print(
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"train_mean_auroc={train_metrics['mean_auroc']:.4f} | "
            f"val_mean_auroc={val_metrics['mean_auroc']:.4f}"
        )

        if val_metrics["mean_auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["mean_auroc"]
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auroc": best_val_auroc,
                    "label_names": label_names,
                },
                save_path / "best_model.pt",
            )
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auroc": best_val_auroc,
                "label_names": label_names,
            },
            save_path / "last_model.pt",
        )

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    return {
        "best_val_auroc": best_val_auroc,
        "best_epoch": best_epoch,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    criterion: nn.Module,
    device: torch.device,
    label_names: List[str],
) -> Dict[str, float]:
    test_loss, test_y, test_p = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )

    test_metrics = multilabel_auroc(test_y, test_p, label_names)
    test_metrics["test_loss"] = test_loss
    return test_metrics