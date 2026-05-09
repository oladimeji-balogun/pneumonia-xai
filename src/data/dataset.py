import os
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


CLASS_MAP = {"NORMAL": 0, "PNEUMONIA": 1}


class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for the Kermany Chest X-Ray dataset.

    Expects the following directory structure:
        root_dir/
            NORMAL/
                *.jpeg
            PNEUMONIA/
                *.jpeg

    Args:
        root_dir (str | Path): Path to the split directory (e.g. data/raw/train).
        transform (callable, optional): Torchvision transform pipeline. Defaults to None.
    """

    def __init__(self, root_dir: str | Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        self._load_samples()
        self.class_weights = self._compute_class_weights()

    def _load_samples(self):
        for class_name, label in CLASS_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected class directory not found: {class_dir}"
                )
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    self.samples.append((img_path, label))

    def _compute_class_weights(self) -> torch.FloatTensor:
        total = len(self.samples)
        num_classes = len(CLASS_MAP)
        counts = [0] * num_classes

        for _, label in self.samples:
            counts[label] += 1

        weights = [total / (num_classes * count) for count in counts]
        return torch.FloatTensor(weights)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.FloatTensor:
        return self.class_weights