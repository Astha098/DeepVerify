"""
evaluate.py
-----------
Loads a trained checkpoint and reports precision/recall/F1 (per-class and
macro-averaged) plus a confusion matrix on the held-out TEST split (never
seen during training or model selection).

Usage:
    python src/evaluate.py --checkpoint results/best_model.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import make_splits
from model import DocumentClassifier, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/best_model.pt")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    train_args = ckpt["args"]

    _, _, test_ds, _ = make_splits(args.data_dir, img_size=train_args["img_size"], augment_train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = DocumentClassifier(
        num_classes=len(class_to_idx),
        backbone=train_args["model"],
        pretrained=False,  # weights come from the checkpoint
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(max(6, len(class_names) * 0.6), max(5, len(class_names) * 0.5)))
    sns.heatmap(cm, annot=len(class_names) <= 15, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (test set)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix to {out_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
