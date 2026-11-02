"""训练 LPRNet 车牌识别模型（CTC Loss）。

数据目录下是车牌小图，文件名格式：{车牌号}_{任意}.jpg
python train_rec.py --train datasets/plate_rec/train --val datasets/plate_rec/val
"""

import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from plate_ocr.chars import BLANK_IDX, CHAR2IDX, CHARS
from plate_ocr.lprnet import LPRNet
from plate_ocr.recognizer import ctc_greedy_decode, preprocess


class PlateDataset(Dataset):
    def __init__(self, root, augment=False):
        self.files = []
        for name in sorted(os.listdir(root)):
            label = name.rsplit(".", 1)[0].split("_")[0]
            if name.lower().endswith((".jpg", ".png")) and all(c in CHAR2IDX for c in label):
                self.files.append((os.path.join(root, name), label))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path, label = self.files[i]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.augment:
            img = cv2.convertScaleAbs(img, alpha=np.random.uniform(0.7, 1.3), beta=np.random.uniform(-30, 30))
        return torch.from_numpy(preprocess(img)), [CHAR2IDX[c] for c in label], label


def collate(batch):
    imgs, targets, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat = torch.tensor([c for t in targets for c in t], dtype=torch.long)
    return torch.stack(imgs), flat, lengths, list(labels)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    right = total = 0
    for imgs, _, _, labels in loader:
        logits = model(imgs.to(device)).cpu()
        for lg, gt in zip(logits, labels):
            right += ctc_greedy_decode(lg)[0] == gt
            total += 1
    model.train()
    return right / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="weights/lprnet.pth")
    args = ap.parse_args()

    device = torch.device(args.device)
    train_loader = DataLoader(PlateDataset(args.train, augment=True), batch_size=args.batch,
                              shuffle=True, num_workers=4, collate_fn=collate)
    val_loader = DataLoader(PlateDataset(args.val), batch_size=args.batch,
                            num_workers=4, collate_fn=collate) if args.val else None

    model = LPRNet(num_classes=len(CHARS)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    ctc = torch.nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    best = -1.0
    for epoch in range(args.epochs):
        total_loss = 0.0
        for imgs, targets, lengths, _ in train_loader:
            logits = model(imgs.to(device))                   # [B, C, T]
            log_probs = logits.permute(2, 0, 1).log_softmax(2)  # [T, B, C]
            input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long)
            loss = ctc(log_probs, targets, input_lengths, lengths)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        sched.step()

        acc = evaluate(model, val_loader, device) if val_loader else 0.0
        print(f"epoch {epoch + 1}/{args.epochs} loss {total_loss / len(train_loader):.4f} val_acc {acc:.4f}")
        if acc >= best:
            best = acc
            torch.save({"model": model.state_dict(), "chars": CHARS}, args.out)
    print(f"best val_acc {best:.4f}, saved to {args.out}")


if __name__ == "__main__":
    main()
