from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataset import ChestXRayDataset
from src.data.transforms import get_train_transforms, get_val_transforms


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders.

    The provided val split in the Kermany dataset contains only 16 images
    and is statistically unusable. This function discards it and carves a
    proper validation set from the training data using an 80/20 split.

    Args:
        data_dir (str | Path): Root data directory containing train/, val/, test/.
        batch_size (int): Batch size for all loaders. Default: 32.
        val_split (float): Fraction of training data to use for validation. Default: 0.2.
        num_workers (int): Number of subprocesses for data loading. Default: 4.
        seed (int): Random seed for reproducible splits. Default: 42.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    # Load full training set with augmentation transforms
    full_train = ChestXRayDataset(
        root_dir=data_dir / "train",
        transform=get_train_transforms(),
    )

    # Compute split sizes
    total = len(full_train)
    val_size = int(total * val_split)
    train_size = total - val_size

    # Reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    # Val subset should use val transforms, not augmentation.
    # We wrap it in a thin dataset that overrides the transform.
    val_dataset = _TransformOverrideDataset(val_subset, get_val_transforms())

    # Test set — deterministic, no augmentation
    test_dataset = ChestXRayDataset(
        root_dir=data_dir / "test",
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_class_weights(data_dir: str | Path) -> torch.FloatTensor:
    """
    Compute class weights from the training set for use in weighted loss.

    Args:
        data_dir (str | Path): Root data directory containing train/.

    Returns:
        torch.FloatTensor with per-class weights.
    """
    ds = ChestXRayDataset(root_dir=Path(data_dir) / "train")
    return ds.get_class_weights()


class _TransformOverrideDataset(torch.utils.data.Dataset):
    """
    Thin wrapper that applies a different transform to a Subset.

    random_split returns a Subset that delegates to the parent dataset's
    transform. Since the parent (full_train) uses augmentation transforms,
    we need this wrapper to apply clean val transforms to the val subset
    without modifying the parent dataset.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        img, label = self.subset[index]
        # img here is already a tensor from the parent's augmentation transform.
        # We need the raw PIL image — go through the parent dataset directly.
        original_index = self.subset.indices[index]
        img_path, label = self.subset.dataset.samples[original_index]

        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label