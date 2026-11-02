"""命令行推理示例。

python demo.py --det weights/plate_det.pt --rec weights/lprnet.pth --img test.jpg --save out.jpg
"""

import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plate_ocr import PlatePipeline


def draw(img, results, font_path=None):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(font_path or "simhei.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    for r in results:
        x1, y1, x2, y2 = r["box"]
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        d.text((x1, max(y1 - 28, 0)), r["plateNumber"], fill=(255, 0, 0), font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="YOLOv8 检测权重")
    ap.add_argument("--rec", required=True, help="LPRNet 识别权重")
    ap.add_argument("--img", required=True)
    ap.add_argument("--save")
    ap.add_argument("--font", help="中文字体路径，用于在图上画车牌号")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    img = cv2.imdecode(np.fromfile(args.img, dtype=np.uint8), cv2.IMREAD_COLOR)
    results = PlatePipeline(args.det, args.rec, device=args.device)(img)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.save:
        cv2.imencode(".jpg", draw(img, results, args.font))[1].tofile(args.save)
