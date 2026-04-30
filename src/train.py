from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.data.dataloaders import get_dataloaders
from src.models.model_factory import get_model
from src.training.trainer import evaluate_model, train_model
from src.training.utils import (
    compute_pos_weights,
    ensure_dir,
    load_yaml,
    resolve_device,
    save_json,
    save_yaml,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train chest X-ray classifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to config YAML file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)

    set_seed(config["seed"])

    target_labels = config["labels"]["target_labels"]
    image_root = config["data"]["raw_root"]
    batch_size = config["loader"]["batch_size"]
    num_workers = config["loader"]["num_workers"]
    image_size = config["image"]["size"]
    grayscale = config["image"]["grayscale"]

    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]

    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    save_dir = config["training"]["save_dir"]
    early_stopping_patience = config["training"]["early_stopping_patience"]
    device = resolve_device(config["training"]["device"])

    ensure_dir(save_dir)
    save_yaml(config, Path(save_dir) / "config_used.yaml")

    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"Labels: {target_labels}")

    train_loader, val_loader, test_loader = get_dataloaders(
        image_root=image_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        grayscale=grayscale,
        target_labels=target_labels,
    )

    model = get_model(
        model_name=model_name,
        num_classes=len(target_labels),
        pretrained=pretrained,
        grayscale=grayscale,
    ).to(device)

    pos_weights = compute_pos_weights("data/processed/train.csv", target_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        label_names=target_labels,
        save_dir=save_dir,
        early_stopping_patience=early_stopping_patience,
    )

    best_ckpt_path = Path(save_dir) / "best_model.pt"
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        label_names=target_labels,
    )

    print("\nFinal Test Metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    save_json(train_summary, Path(save_dir) / "train_summary.json")
    save_json(test_metrics, Path(save_dir) / "test_metrics.json")


if __name__ == "__main__":
    main()