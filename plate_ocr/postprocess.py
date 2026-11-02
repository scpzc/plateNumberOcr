"""结果后处理：格式校验 + 简单的颜色判断。"""

import re

import cv2
import numpy as np

from .chars import PROVINCES

_PROV = "".join(PROVINCES)
# 普通蓝/黄牌：省份 + 字母 + 5 位
NORMAL_RE = re.compile(rf"^[{_PROV}][A-HJ-NP-Z][A-HJ-NP-Z0-9]{{4}}[A-HJ-NP-Z0-9学警港澳挂]$")
# 新能源：省份 + 字母 + 6 位，第 3 位或末位为 D/F
NEV_RE = re.compile(rf"^[{_PROV}][A-HJ-NP-Z]([DF][A-HJ-NP-Z0-9]{{5}}|[0-9]{{5}}[DF])$")

def normalize(text):
    text = text.replace("·", "").replace(" ", "").upper()
    return text


def is_valid(text):
    return bool(NORMAL_RE.match(text) or NEV_RE.match(text))


def guess_color(plate_bgr):
    """按 HSV 主色粗略判断底色：blue / yellow / green / white / black。"""
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = [c.flatten() for c in cv2.split(hsv)]
    colored = (s > 80) & (v > 60)
    if colored.mean() < 0.25:
        return "black" if np.median(v) < 90 else "white"
    hue = np.median(h[colored])
    if 100 <= hue <= 130:
        return "blue"
    if 15 <= hue <= 35:
        return "yellow"
    if 35 < hue < 100:
        return "green"
    return "unknown"
