from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, pretrained: bool = True, grayscale: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if grayscale:
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_densenet121(num_classes: int, pretrained: bool = True, grayscale: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    if grayscale:
        original_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def get_model(model_name: str, num_classes: int, pretrained: bool = True, grayscale: bool = True) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet18":
        return build_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
            grayscale=grayscale,
        )

    if model_name == "densenet121":
        return build_densenet121(
            num_classes=num_classes,
            pretrained=pretrained,
            grayscale=grayscale,
        )

    raise ValueError(f"Unsupported model name: {model_name}")