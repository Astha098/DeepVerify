"""
train.py
--------
Trains DocumentClassifier on data/processed. Designed to run comfortably
on a MacBook Air M4 using the Apple MPS backend.

Example usage (from the project root, with venv active):

    python src/train.py --epochs 15 --model resnet18 --pretrained --augment
    python src/train.py --epochs 15 --model resnet18 --no-pretrained     # experiment: from scratch
    python src/train.py --epochs 15 --model resnet18 --pretrained --lr 0.01  # experiment: LR sweep
    python src/train.py --epochs 15 --model efficientnet_b0 --pretrained --augment

Each run writes:
    results/best_model.pt            - best checkpoint (by val F1)
    results/training_curves.png      - loss/accuracy curves
    results/run_<timestamp>.json     - full metric history + args, for comparing experiments
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import make_splits
from model import DocumentClassifier, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--model", default="resnet18",
                    choices=["resnet18", "resnet34", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--augment", action="store_true", default=False)
    p.add_argument("--patience", type=int, default=5, help="early stopping patience")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = sum(pred == label for pred, label in zip(all_preds, all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    device = get_device()
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds, class_to_idx = make_splits(
        args.data_dir, img_size=args.img_size, augment_train=args.augment
    )
    num_classes = len(class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = DocumentClassifier(
        num_classes=num_classes,
        backbone=args.model,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1, epochs_no_improve = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch:02d}/{args.epochs} ({time.time()-t0:.1f}s) "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "args": vars(args),
                "val_f1": val_f1,
            }, out_dir / "best_model.pt")
            print(f"  -> new best (val_f1={val_f1:.4f}), checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Save training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train_acc")
    axes[1].plot(history["val_acc"], label="val_acc")
    axes[1].plot(history["val_f1"], label="val_f1")
    axes[1].set_title("Accuracy / F1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    print(f"Saved training curves to {out_dir / 'training_curves.png'}")

    run_log = {"args": vars(args), "history": history, "best_val_f1": best_f1}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(out_dir / f"run_{stamp}.json", "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"Best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
