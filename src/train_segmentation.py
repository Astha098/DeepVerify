import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from segmentation_model import UNet

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_IMAGES = Path(
    "data/segmentation_grouped/train/images"
)

TRAIN_MASKS = Path(
    "data/segmentation_grouped/train/masks"
)

VAL_IMAGES = Path(
    "data/segmentation_grouped/validation/images"
)

VAL_MASKS = Path(
    "data/segmentation_grouped/validation/masks"
)

MODEL_PATH = Path(
    "results/best_segmentation_model_grouped.pth"
)

HISTORY_PATH = Path(
    "results/segmentation_training_history.json"
)

CURVES_PATH = Path(
    "results/segmentation_training_curves.png"
)

IMAGE_SIZE = 256

BATCH_SIZE = 4

LEARNING_RATE = 1e-3

EPOCHS = 20

EARLY_STOPPING_PATIENCE = 5

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# DATASET
# ============================================================

class DocumentSegmentationDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        augment=False
    ):

        self.image_dir = Path(image_dir)

        self.mask_dir = Path(mask_dir)

        self.augment = augment

        self.images = sorted(
            self.image_dir.glob("*.jpg")
        )

        if len(self.images) == 0:

            raise FileNotFoundError(
                f"No JPG images found in "
                f"{self.image_dir}"
            )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        mask_path = (
            self.mask_dir /
            f"{image_path.stem}.png"
        )

        if not mask_path.exists():

            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        # ----------------------------------------------------
        # Resize first
        # ----------------------------------------------------

        image = TF.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.NEAREST
        )

        # ----------------------------------------------------
        # TRAINING AUGMENTATION
        #
        # IMPORTANT:
        # geometric transforms are applied identically
        # to BOTH image and mask.
        # ----------------------------------------------------

        if self.augment:

            # Horizontal flip

            if random.random() < 0.5:

                image = TF.hflip(image)

                mask = TF.hflip(mask)

            # Small rotation

            if random.random() < 0.5:

                angle = random.uniform(
                    -10,
                    10
                )

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0
                )

                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0
                )

            # Mild affine transformation

            if random.random() < 0.4:

                angle = random.uniform(
                    -5,
                    5
                )

                translate = [
                    random.randint(-10, 10),
                    random.randint(-10, 10)
                ]

                scale = random.uniform(
                    0.95,
                    1.05
                )

                shear = random.uniform(
                    -3,
                    3
                )

                image = TF.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=[shear, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0
                )

                mask = TF.affine(
                    mask,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=[shear, 0.0],
                    interpolation=InterpolationMode.NEAREST,
                    fill=0
                )

            # ------------------------------------------------
            # Photometric augmentation
            #
            # Apply ONLY to image.
            # Never modify segmentation mask brightness/color.
            # ------------------------------------------------

            if random.random() < 0.5:

                brightness_factor = (
                    random.uniform(
                        0.8,
                        1.2
                    )
                )

                image = ImageEnhance.Brightness(
                    image
                ).enhance(
                    brightness_factor
                )

            if random.random() < 0.5:

                contrast_factor = (
                    random.uniform(
                        0.8,
                        1.2
                    )
                )

                image = ImageEnhance.Contrast(
                    image
                ).enhance(
                    contrast_factor
                )

        # ----------------------------------------------------
        # Convert image to tensor
        # ----------------------------------------------------

        image = TF.to_tensor(
            image
        )

        # ----------------------------------------------------
        # Convert mask to binary tensor
        # ----------------------------------------------------

        mask = np.array(
            mask,
            dtype=np.float32
        )

        mask = (
            mask > 127
        ).astype(
            np.float32
        )

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return image, mask


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(
        self,
        predictions,
        targets
    ):

        probabilities = torch.sigmoid(
            predictions
        )

        probabilities = probabilities.view(
            -1
        )

        targets = targets.view(
            -1
        )

        intersection = (
            probabilities *
            targets
        ).sum()

        dice = (
            2.0 * intersection + 1e-7
        ) / (
            probabilities.sum()
            + targets.sum()
            + 1e-7
        )

        return 1.0 - dice


# ============================================================
# COMBINED BCE + DICE LOSS
# ============================================================

class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = (
            nn.BCEWithLogitsLoss()
        )

        self.dice = DiceLoss()

    def forward(
        self,
        predictions,
        targets
    ):

        bce_loss = self.bce(
            predictions,
            targets
        )

        dice_loss = self.dice(
            predictions,
            targets
        )

        return (
            0.5 * bce_loss
            +
            0.5 * dice_loss
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    targets
):

    probabilities = torch.sigmoid(
        predictions
    )

    predictions = (
        probabilities > 0.5
    ).float()

    intersection = (
        predictions *
        targets
    ).sum()

    dice = (
        2.0 * intersection + 1e-7
    ) / (
        predictions.sum()
        + targets.sum()
        + 1e-7
    )

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum()

    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    return (
        dice.item(),
        iou.item()
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0.0

    for images, masks in loader:

        images = images.to(
            device
        )

        masks = masks.to(
            device
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            *
            images.size(0)
        )

    return (
        total_loss /
        len(loader.dataset)
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    total_dice = 0.0

    total_iou = 0.0

    total_samples = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                device
            )

            masks = masks.to(
                device
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                masks
            )

            batch_size = (
                images.size(0)
            )

            total_loss += (
                loss.item()
                *
                batch_size
            )

            dice, iou = (
                calculate_metrics(
                    outputs,
                    masks
                )
            )

            total_dice += (
                dice *
                batch_size
            )

            total_iou += (
                iou *
                batch_size
            )

            total_samples += (
                batch_size
            )

    return (
        total_loss /
        total_samples,

        total_dice /
        total_samples,

        total_iou /
        total_samples
    )


# ============================================================
# SAVE TRAINING CURVES
# ============================================================

def save_training_curves(
    history
):

    epochs = range(
        1,
        len(
            history["train_loss"]
        ) + 1
    )

    # Loss curve

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Segmentation Training Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        CURVES_PATH,
        dpi=150
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    device = get_device()

    print(
        f"Using device: {device}"
    )

    print(
        "\nDataset:"
    )

    print(
        "Similarity-grouped leakage-reduced split"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = (
        DocumentSegmentationDataset(
            TRAIN_IMAGES,
            TRAIN_MASKS,
            augment=True
        )
    )

    val_dataset = (
        DocumentSegmentationDataset(
            VAL_IMAGES,
            VAL_MASKS,
            augment=False
        )
    )

    print(
        f"\nTraining samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet().to(
        device
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = (
        BCEDiceLoss()
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    best_dice = 0.0

    best_epoch = 0

    epochs_without_improvement = 0

    history = {

        "train_loss": [],

        "val_loss": [],

        "dice": [],

        "iou": []

    }

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        (
            val_loss,
            dice,
            iou
        ) = validate(
            model,
            val_loader,
            criterion,
            device
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        history[
            "dice"
        ].append(
            dice
        )

        history[
            "iou"
        ].append(
            iou
        )

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        print(
            f"\nEpoch "
            f"[{epoch}/{EPOCHS}]"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss:   "
            f"{val_loss:.4f}"
        )

        print(
            f"Dice Score: "
            f"{dice:.4f}"
        )

        print(
            f"IoU Score:  "
            f"{iou:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.6f}"
        )

        # Scheduler monitors Dice

        scheduler.step(
            dice
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if dice > best_dice:

            best_dice = dice

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "Best model saved!"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >=
            EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    with open(
        HISTORY_PATH,
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    save_training_curves(
        history
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 55
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 55
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation Dice: "
        f"{best_dice:.4f}"
    )

    print(
        "\nBest model:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nTraining history:"
    )

    print(
        HISTORY_PATH
    )

    print(
        "\nTraining curves:"
    )

    print(
        CURVES_PATH
    )


if __name__ == "__main__":

    main()