"""
train_autoencoder.py
---------------------
Trains the ConvAutoencoder (src/model.py) on GENUINE documents only, so it
learns to reconstruct authentic layouts well. At inference time, a document
that reconstructs poorly (high MSE) is flagged as visually anomalous --
this is the unsupervised half of the tamper-detection stage, complementing
the classical ELA check in preprocessing.py.

Usage:
    python scripts/train_autoencoder.py --epochs 20
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import ConvAutoencoder, get_device  # noqa: E402


class ImageOnlyDataset(Dataset):
    def __init__(self, root="data/processed", size=128):
        self.paths = sorted(Path(root).glob("*/*.jpg"))
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size)) / 255.0
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="results/autoencoder.pt")
    args = p.parse_args()

    device = get_device()
    ds = ImageOnlyDataset(args.data_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs in tqdm(loader, leave=False):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        print(f"Epoch {epoch}/{args.epochs} recon_loss={total_loss/len(ds):.5f}")

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved autoencoder to {args.out}")


if __name__ == "__main__":
    main()
