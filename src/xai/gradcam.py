from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.models.base import BaseModel
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Produces a coarse localization map highlighting regions of the input
    image most influential for the model's prediction. Uses gradients of
    the target class score flowing into the final convolutional layer to
    weight the activation maps.

    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization", ICCV 2017.

    Args:
        model: Trained BaseModel subclass.
        target_layer: The convolutional layer to hook into. If None,
                      attempts to auto-detect the last conv layer.
        device: torch device. Auto-detected if None.
    """

    def __init__(
        self,
        model: BaseModel,
        target_layer: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(self.device)
        self.target_layer = target_layer or self._find_last_conv()

        self._activations = None
        self._gradients = None
        self._register_hooks()

    def _find_last_conv(self) -> torch.nn.Module:
        """Auto-detect the last Conv2d layer in the model."""
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model.")
        return last_conv

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate a Grad-CAM heatmap for a single image.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            Tuple of:
                - heatmap: np.ndarray of shape (H, W), values in [0, 1]
                - predicted_class: int
                - confidence: float (softmax probability of predicted class)
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        predicted_class = logits.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()

        if target_class is None:
            target_class = predicted_class

        # Backpropagate target class score
        score = logits[0, target_class]
        score.backward()

        # Global average pool the gradients over spatial dimensions
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # Resize to input image size
        cam = np.array(Image.fromarray(cam).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
        ))

        return cam, predicted_class, confidence


def load_and_preprocess(image_path: str | Path) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load an image and return both the preprocessed tensor and the
    original RGB numpy array for visualization.

    Args:
        image_path: Path to the chest X-ray image.

    Returns:
        Tuple of (input_tensor, original_rgb_array)
    """
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")
    original = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor = transform(img).unsqueeze(0)  # add batch dim
    return tensor, original


def visualize_gradcam(
    original: np.ndarray,
    heatmap: np.ndarray,
    predicted_class: int,
    confidence: float,
    true_label: Optional[int] = None,
    save_path: Optional[str | Path] = None,
    model_name: str = "",
):
    """
    Overlay Grad-CAM heatmap on the original image and display/save.

    Args:
        original: Original RGB image as numpy array (H, W, 3).
        heatmap: Grad-CAM heatmap as numpy array (H, W), values in [0, 1].
        predicted_class: Predicted class index.
        confidence: Model confidence for predicted class.
        true_label: Ground truth label (optional, shown in title if provided).
        save_path: If provided, save the figure to this path.
        model_name: Model name shown in figure title.
    """
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap)[:, :, :3]  # drop alpha
    overlay = 0.5 * original / 255.0 + 0.5 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    pred_name = CLASS_NAMES[predicted_class]
    title = f"{model_name} | Predicted: {pred_name} ({confidence:.2%})"
    if true_label is not None:
        true_name = CLASS_NAMES[true_label]
        correct = "✓" if true_label == predicted_class else "✗"
        title += f" | True: {true_name} {correct}"

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()