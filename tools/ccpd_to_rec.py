"""从 CCPD 中裁剪车牌小图，生成 LPRNet 识别数据集（文件名即标签）。

python tools/ccpd_to_rec.py --src CCPD2019/ccpd_base CCPD2020/ccpd_green/train --dst datasets/plate_rec/train
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from ccpd import parse_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    n = 0
    for d in args.src:
        for name in os.listdir(d):
            if not name.endswith(".jpg"):
                continue
            try:
                (x1, y1, x2, y2), plate = parse_name(name)
            except ValueError:
                continue
            img = cv2.imread(os.path.join(d, name))
            if img is None:
                continue
            crop = cv2.resize(img[y1:y2, x1:x2], (94, 24))
            # 同一车牌可能出现多次，加序号避免覆盖
            cv2.imencode(".jpg", crop)[1].tofile(os.path.join(args.dst, f"{plate}_{n}.jpg"))
            n += 1
    print(f"done, {n} crops -> {args.dst}")


if __name__ == "__main__":
    main()
