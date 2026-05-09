import torch
from pathlib import Path
from src.models.custom_cnn import LightCXR
from src.xai.gradcam import GradCAM, load_and_preprocess, visualize_gradcam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
model = LightCXR()
checkpoint = torch.load("experiments/results/lightcxr_best.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# Pick one pneumonia sample
image_path = list(Path("data/raw/test/PNEUMONIA").iterdir())[0]
tensor, original = load_and_preprocess(image_path)

# Generate Grad-CAM
gradcam = GradCAM(model)
heatmap, pred_class, confidence = gradcam.generate(tensor)

print(f"Predicted: {pred_class} | Confidence: {confidence:.4f}")
print(f"Heatmap shape: {heatmap.shape}, min: {heatmap.min():.3f}, max: {heatmap.max():.3f}")

# Visualize
visualize_gradcam(
    original, heatmap, pred_class, confidence,
    true_label=1,
    save_path="experiments/results/gradcam_lightcxr_sample.png",
    model_name="LightCXR"
)