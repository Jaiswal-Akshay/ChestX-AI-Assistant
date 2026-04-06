import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt

from src.data.dataset import TARGET_LABELS

DEVICE = torch.device("cpu")
MODEL_PATH = "outputs/models/best_resnet18.pth"
OUTPUT_DIR = "outputs/reports"


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))

        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def get_model():
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, len(TARGET_LABELS))

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def overlay_heatmap_on_image(gray_img, heatmap):
    gray_img = np.array(gray_img.resize((224, 224))).astype(np.float32)
    gray_img = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min() + 1e-8)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    gray_rgb = np.stack([gray_img, gray_img, gray_img], axis=-1)

    overlay = 0.6 * gray_rgb + 0.4 * heatmap_color
    overlay = np.clip(overlay, 0, 1)

    return gray_rgb, heatmap_color, overlay


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_path = "data/raw/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"  # change if needed

    model = get_model()
    transform = get_transform()

    pil_img = Image.open(image_path).convert("L")
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    top_idx = int(np.argmax(probs))
    top_label = TARGET_LABELS[top_idx]
    top_prob = float(probs[top_idx])

    gradcam = GradCAM(model, model.layer4[-1])
    cam = gradcam.generate(input_tensor, top_idx)
    gradcam.remove_hooks()

    gray_rgb, heatmap_color, overlay = overlay_heatmap_on_image(pil_img, cam)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(gray_rgb, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_color)
    plt.title(f"Grad-CAM\n{top_label}: {top_prob:.4f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "gradcam_result.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Top prediction: {top_label} ({top_prob:.4f})")
    print(f"Saved Grad-CAM figure to {save_path}")


if __name__ == "__main__":
    main()