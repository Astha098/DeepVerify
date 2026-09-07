"""
dataset.py
----------
PyTorch Dataset for document-type classification, built on top of
data/processed/<class_name>/*.jpg  (produced by scripts/prepare_midv500.py).

Also provides a helper to build stratified train/val/test splits so results
are reproducible and don't leak the same physical document across splits.
"""

import random
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size=224, augment=False):
    """
    Two transform pipelines:
      - train (augment=True): random crop/flip/rotation/color-jitter, used
        in the augmentation-vs-no-augmentation experiment.
      - eval (augment=False): deterministic resize + normalize only.
    """
    if augment:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DocumentDataset(Dataset):
    """
    Expects a list of (filepath, label_idx) tuples. Use `make_splits` below
    to build train/val/test instances of this class from a processed
    directory of the form data/processed/<class_name>/*.jpg
    """

    def __init__(self, samples, class_to_idx, img_size=224, augment=False):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = build_transforms(img_size, augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(str(path))
        if image is None:
            # corrupt file safety net -- return a black image rather than crash a run
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, label


def make_splits(processed_dir="data/processed", val_frac=0.15, test_frac=0.15,
                 seed=42, img_size=224, augment_train=True):
    """
    Scans data/processed/<class_name>/*.jpg, builds a stratified split (same
    fraction of each class in train/val/test), and returns three
    DocumentDataset instances + the class_to_idx mapping.
    """
    processed_dir = Path(processed_dir)
    classes = sorted([d.name for d in processed_dir.iterdir() if d.is_dir()])
    if not classes:
        raise FileNotFoundError(
            f"No class folders found under {processed_dir}. "
            "Run scripts/prepare_midv500.py first (see README)."
        )
    class_to_idx = {c: i for i, c in enumerate(classes)}

    rng = random.Random(seed)
    train_samples, val_samples, test_samples = [], [], []

    for cls in classes:
        files = sorted((processed_dir / cls).glob("*.jpg"))
        rng.shuffle(files)
        n = len(files)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))
        val_files = files[:n_val]
        test_files = files[n_val:n_val + n_test]
        train_files = files[n_val + n_test:]

        label = class_to_idx[cls]
        train_samples += [(f, label) for f in train_files]
        val_samples += [(f, label) for f in val_files]
        test_samples += [(f, label) for f in test_files]

    train_ds = DocumentDataset(train_samples, class_to_idx, img_size, augment=augment_train)
    val_ds = DocumentDataset(val_samples, class_to_idx, img_size, augment=False)
    test_ds = DocumentDataset(test_samples, class_to_idx, img_size, augment=False)

    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    return train_ds, val_ds, test_ds, class_to_idx
