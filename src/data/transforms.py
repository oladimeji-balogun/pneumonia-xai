from torchvision import transforms

# ImageNet mean and std — used because all pretrained models (ResNet, MobileNet,
# EfficientNet) were trained on ImageNet. Grayscale X-rays converted to RGB
# still benefit from this normalization as it scales pixel values into the
# expected input distribution of pretrained weights.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Target size for all architectures. 224x224 is the standard input size for
# ResNet-18, MobileNetV2, and EfficientNet-B0.
IMAGE_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """
    Augmentation pipeline for training set.

    Augmentations are chosen to reflect realistic X-ray variability:
    - Horizontal flip: X-rays can be taken from either side
    - Small rotation: slight patient/equipment misalignment
    - Minor brightness/contrast shift: scanner calibration differences
    - No vertical flip: anatomically invalid for chest X-rays
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Deterministic pipeline for validation and test sets.
    No augmentation — only resize, tensor conversion, and normalization.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])