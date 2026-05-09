# Explainable Pneumonia Detection Under Computational Constraints

A benchmarking study of CNN architectures for pneumonia detection from chest X-rays, with a focus on balancing classification accuracy, computational efficiency, and clinical interpretability.

---

## Motivation

Pneumonia remains one of the leading causes of mortality worldwide, disproportionately affecting low-income regions where access to trained radiologists and high-end diagnostic infrastructure is severely limited. While deep learning models have demonstrated remarkable accuracy in detecting pneumonia from chest X-rays, their practical deployment in resource-constrained clinical environments is hindered by two critical challenges: **computational cost** and **lack of interpretability**.

This project addresses both. We benchmark four CNN architectures -- including **LightCXR**, a custom lightweight architecture designed explicitly for edge deployment — against pretrained baselines, and integrate Grad-CAM explainability to make model predictions clinically interpretable.

---

## Results Summary

| Model | AUC-ROC | Recall | Specificity | Params | CPU Inference |
|---|---|---|---|---|---|
| **LightCXR (ours)** | 0.9467 | 98.7% | 59.8% | **270K** | **2.96ms** |
| MobileNetV2 | 0.9622 | 99.5% | 67.1% | 2.2M | 24.24ms |
| EfficientNet-B0 | 0.9584 | 99.7% | 56.4% | 4.0M | 32.85ms |
| ResNet-18 | 0.9663 | 99.7% | 58.6% | 11.2M | 19.28ms |

**Key finding:** LightCXR achieves 0.9467 AUC-ROC at 270K parameters -- 6.5× faster on CPU than the next fastest model (ResNet-18), with only a 1.96% AUC-ROC gap. For deployment under hard compute constraints, this tradeoff is compelling.

---

## Project Structure

pneumonia-xai/
│
├── data/
│   ├── raw/                        # Kermany Chest X-Ray dataset (never modified)
│   │   ├── train/{NORMAL,PNEUMONIA}
│   │   ├── val/{NORMAL,PNEUMONIA}
│   │   └── test/{NORMAL,PNEUMONIA}
│   ├── processed/                  # Preprocessed data (if applicable)
│   └── metadata/                   # Dataset statistics
│
├── src/
│   ├── data/
│   │   ├── dataset.py              # PyTorch Dataset class
│   │   ├── transforms.py           # Augmentation pipelines
│   │   └── splitter.py             # DataLoader factory + 80/20 val split
│   │
│   ├── models/
│   │   ├── base.py                 # Abstract base model interface
│   │   ├── custom_cnn.py           # LightCXR — custom lightweight architecture
│   │   ├── resnet.py               # ResNet-18 wrapper
│   │   ├── mobilenet.py            # MobileNetV2 wrapper
│   │   └── efficientnet.py         # EfficientNet-B0 wrapper
│   │
│   ├── training/
│   │   └── trainer.py              # Training loop with early stopping
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # Accuracy, F1, AUC-ROC, precision, recall, specificity
│   │   └── benchmark.py            # FLOPs, parameter count, inference time
│   │
│   └── xai/
│       └── gradcam.py              # Grad-CAM implementation + visualization
│
├── experiments/
│   └── results/                    # Per-model checkpoints, metrics, plots
│       ├── lightcxr/
│       ├── mobilenetv2/
│       ├── efficientnet_b0/
│       ├── resnet18/
│       └── results_summary.csv     # Master results table
│
├── notebooks/
│   └── analysis.ipynb              # Results analysis and paper figures
│
├── paper/                          # Manuscript draft
├── tests/
├── run_experiment.py               # Main experiment runner
└── pyproject.toml
---

## Setup

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- CUDA-capable GPU recommended (CPU training supported but slow)

### Installation

```bash
git clone https://github.com/oladimeji-balogun/pneumonia-xai.git
cd pneumonia-xai
poetry install
```

### Dataset

Download the [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset and place it under `data/raw/` with the following structure:

data/raw/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
├── NORMAL/
└── PNEUMONIA/

> **Note:** The provided val split (16 images) is discarded. A proper validation set is carved from the training data using an 80/20 split at runtime.

---

## Running Experiments

Each architecture is trained, evaluated, and benchmarked independently:

```bash
poetry run python run_experiment.py --model lightcxr
poetry run python run_experiment.py --model mobilenetv2
poetry run python run_experiment.py --model efficientnet_b0
poetry run python run_experiment.py --model resnet18
```

Each run saves to `experiments/results/<model_name>/`:

| File | Contents |
|------|----------|
| `<model>_best.pth` | Best checkpoint (by val loss) |
| `<model>_metrics.json` | Test set classification metrics |
| `<model>_benchmark.json` | FLOPs, params, inference time |
| `<model>_history.json` | Per-epoch training history |
| `<model>_gradcam.png` | Grad-CAM visualization |
| `results_summary.csv` | Appended row in master results table |

---

## Analysis

After all experiments complete, open the analysis notebook to generate all paper figures:

```bash
poetry run jupyter notebook notebooks/analysis.ipynb
```

Figures produced:

- Accuracy–efficiency tradeoff plot (AUC-ROC vs parameter count)
- Classification metrics comparison (grouped bar chart)
- Inference latency comparison (GPU vs CPU)
- Confusion matrix grid
- Training curves
- Grad-CAM overlay grid

---

## LightCXR Architecture

LightCXR is a custom lightweight CNN designed around a strict **< 500K parameter budget** for deployment in resource-constrained environments. It uses depthwise separable convolutions as its primary building block -- factorizing standard convolutions into a depthwise pass (one filter per channel) followed by a pointwise 1×1 convolution -- reducing parameter count and FLOPs significantly while preserving representational capacity.

Stem conv       (3 -> 32,  stride=2)     -> 112×112
DS Block        (32 -> 64,  stride=2)    -> 56×56
DS Block        (64 -> 128, stride=2)    -> 28×28
DS Block        (128 -> 128)             -> 28×28
DS Block        (128 -> 256, stride=2)   -> 14×14
DS Block        (256 -> 256)             -> 14×14
DS Block        (256 -> 512, stride=2)   -> 7×7
Global Avg Pool                         -> 512
Classifier      (512 -> 2)
-----------------------------------------------------
Total parameters: 270,146
FLOPs:           130.5M
CPU inference:   2.96ms


---

## Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is applied to all trained models to produce visual explanations highlighting regions of the chest X-ray most influential for each prediction. This is critical for clinical trust -- a model that attends to anatomically plausible regions (lung fields, consolidation zones) provides stronger justification for its predictions than accuracy numbers alone.

---

## Training Details

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Initial LR | 1e-3 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | patience=5 |
| Max epochs | 30 |
| Batch size | 32 |
| Input size | 224×224 |
| Class imbalance | Weighted cross-entropy loss |
| Val split | 80/20 from training set (seed=42) |
| Augmentation | Horizontal flip, rotation (±10°), brightness/contrast jitter |

---

## Contributions

- **Survey and benchmark** of four CNN architectures on pneumonia detection, evaluated across both classification accuracy and computational cost
- **LightCXR**: a novel lightweight architecture achieving 0.9467 AUC-ROC at 270K parameters and 2.96ms CPU inference
- **Grad-CAM integration** for clinical interpretability across all benchmarked architectures
- **Reproducible pipeline** — every result traceable to exact hyperparameters and a fixed random seed

---

## Citation

> Balogun, O. Emmanuel, O. Faithfullness, A. (2025). *Explainable Pneumonia Detection Under Computational Constraints*. Manuscript in preparation.

---

## License

MIT