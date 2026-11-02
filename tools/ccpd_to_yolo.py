"""把 CCPD 转成 YOLO 检测数据集。

python tools/ccpd_to_yolo.py --src CCPD2019/ccpd_base --dst datasets/plate_det --val 0.1
"""

import argparse
import os
import random
import shutil
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from ccpd import parse_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="一个或多个 CCPD 图片目录")
    ap.add_argument("--dst", default="datasets/plate_det")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="每个目录最多取多少张，0 表示全部")
    args = ap.parse_args()

    files = []
    for d in args.src:
        names = [os.path.join(d, n) for n in os.listdir(d) if n.endswith(".jpg")]
        random.shuffle(names)
        files += names[: args.limit] if args.limit else names

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.dst, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.dst, "labels", split), exist_ok=True)

    for i, path in enumerate(files):
        split = "val" if random.random() < args.val else "train"
        try:
            (x1, y1, x2, y2), _ = parse_name(path)
        except ValueError:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = f"{i:07d}"
        shutil.copy(path, os.path.join(args.dst, "images", split, stem + ".jpg"))
        with open(os.path.join(args.dst, "labels", split, stem + ".txt"), "w") as f:
            f.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")

    with open(os.path.join(args.dst, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.dst)}\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n")
    print(f"done, {len(files)} images -> {args.dst}")


if __name__ == "__main__":
    main()
