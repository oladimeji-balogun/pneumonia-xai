"""
run_experiment.py

Trains, evaluates, and benchmarks a single CNN architecture for the
pneumonia detection benchmarking suite.

Usage:
    poetry run python run_experiment.py --model lightcxr
    poetry run python run_experiment.py --model resnet18
    poetry run python run_experiment.py --model mobilenetv2
    poetry run python run_experiment.py --model efficientnet_b0

Results are saved to experiments/results/<model_name>/:
    - <model_name>_best.pth        — best model checkpoint
    - <model_name>_metrics.json    — classification metrics on test set
    - <model_name>_benchmark.json  — computational benchmark results
    - <model_name>_history.json    — per-epoch training history
    - <model_name>_gradcam.png     — Grad-CAM visualization on a test sample
    - results_summary.csv          — appended row in the master results table
"""

import argparse
import json
import csv
import time
import torch
from pathlib import Path

from src.data.splitter import get_dataloaders, get_class_weights
from src.models.custom_cnn import LightCXR
from src.models.resnet import ResNet18
from src.models.mobilenet import MobileNetV2
from src.models.efficientnet import EfficientNetB0
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate, print_metrics
from src.evaluation.benchmark import benchmark_model, print_benchmark
from src.xai.gradcam import GradCAM, load_and_preprocess, visualize_gradcam

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR        = "data/raw"
RESULTS_DIR     = Path("experiments/results")
SUMMARY_CSV     = RESULTS_DIR / "results_summary.csv"
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-3
EPOCHS          = 30
PATIENCE        = 5
NUM_WORKERS     = 4
SEED            = 42

MODEL_REGISTRY = {
    "lightcxr":       LightCXR,
    "resnet18":       ResNet18,
    "mobilenetv2":    MobileNetV2,
    "efficientnet_b0": EfficientNetB0,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()


def save_json(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def append_to_summary(metrics: dict, benchmark: dict, model_dir: Path):
    """Append a row to the master results CSV."""
    row = {
        "model":             benchmark["model_name"],
        "accuracy":          round(metrics["accuracy"], 4),
        "f1":                round(metrics["f1"], 4),
        "auc_roc":           round(metrics["auc_roc"], 4),
        "precision":         round(metrics["precision"], 4),
        "recall":            round(metrics["recall"], 4),
        "specificity":       round(metrics["specificity"], 4),
        "tp":                metrics["tp"],
        "tn":                metrics["tn"],
        "fp":                metrics["fp"],
        "fn":                metrics["fn"],
        "trainable_params":  benchmark["trainable_params"],
        "flops":             benchmark["flops"],
        "gpu_mean_ms":       round(benchmark["gpu_mean_ms"], 3),
        "cpu_mean_ms":       round(benchmark["cpu_mean_ms"], 3),
    }

    write_header = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended to summary: {SUMMARY_CSV}")


def run_gradcam(model, model_dir: Path, device: torch.device):
    """Generate and save a Grad-CAM visualization on one pneumonia test sample."""
    sample_path = next(Path(DATA_DIR).joinpath("test/PNEUMONIA").iterdir())
    tensor, original = load_and_preprocess(sample_path)

    gradcam = GradCAM(model, device=device)
    heatmap, pred_class, confidence = gradcam.generate(tensor)

    save_path = model_dir / f"{model.model_name}_gradcam.png"
    visualize_gradcam(
        original, heatmap, pred_class, confidence,
        true_label=1,
        save_path=save_path,
        model_name=model.model_name,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Experiment: {model_name}")
    print(f"  Device:     {device}")
    print(f"{'='*60}")

    # Output directory for this model
    model_dir = RESULTS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )
    class_weights = get_class_weights(DATA_DIR)

    # ── Training ──────────────────────────────────────────────────────────────
    model = get_model(model_name)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        patience=PATIENCE,
        checkpoint_dir=model_dir,
        device=device,
    )

    train_start = time.time()
    history = trainer.train()
    train_duration = time.time() - train_start

    save_json(
        {**history, "train_duration_seconds": round(train_duration, 1)},
        model_dir / f"{model_name}_history.json",
    )

    # ── Load best checkpoint for evaluation ───────────────────────────────────
    checkpoint_path = model_dir / f"{model_name}_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nLoaded best checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f})")

    # ── Evaluation ────────────────────────────────────────────────────────────
    metrics = evaluate(model, test_loader, device)
    print_metrics(metrics, model_name)
    save_json(metrics, model_dir / f"{model_name}_metrics.json")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    benchmark = benchmark_model(model, device)
    benchmark["train_duration_seconds"] = round(train_duration, 1)
    print_benchmark(benchmark)
    save_json(benchmark, model_dir / f"{model_name}_benchmark.json")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    print("\nGenerating Grad-CAM visualization...")
    run_gradcam(model, model_dir, device)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    append_to_summary(metrics, benchmark, model_dir)

    print(f"\n{'='*60}")
    print(f"  Experiment complete: {model_name}")
    print(f"  Results saved to:   {model_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single model experiment.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to train and evaluate.",
    )
    args = parser.parse_args()
    main(args.model)