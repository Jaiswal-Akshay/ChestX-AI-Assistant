import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from src.data.dataset import TARGET_LABELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "outputs/models/best_resnet18.pth"


def get_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, len(TARGET_LABELS))
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def main():
    image_path = "data/raw/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"  # change this

    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    image = Image.open(image_path).convert("L")
    image = get_transform()(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).squeeze(0).cpu().numpy()

    print("\nPredicted probabilities:")
    for label, prob in zip(TARGET_LABELS, probs):
        print(f"{label}: {prob:.4f}")


if __name__ == "__main__":
    main()