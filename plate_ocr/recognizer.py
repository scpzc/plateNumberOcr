"""车牌字符识别：LPRNet + CTC 贪心解码。"""

import cv2
import numpy as np
import torch

from .chars import BLANK_IDX, CHARS
from .lprnet import LPRNet

INPUT_W, INPUT_H = 94, 24


def preprocess(plate_bgr):
    img = cv2.resize(plate_bgr, (INPUT_W, INPUT_H)).astype(np.float32)
    img = (img - 127.5) * 0.0078125
    return np.transpose(img, (2, 0, 1))


def ctc_greedy_decode(logits):
    """logits: [C, T]，返回 (文本, 平均置信度)。"""
    probs = torch.softmax(torch.as_tensor(logits), dim=0)
    conf, idx = probs.max(dim=0)
    text, scores, prev = [], [], BLANK_IDX
    for i, c in zip(idx.tolist(), conf.tolist()):
        if i != prev and i != BLANK_IDX:
            text.append(CHARS[i])
            scores.append(c)
        prev = i
    return "".join(text), (sum(scores) / len(scores) if scores else 0.0)


class PlateRecognizer:
    def __init__(self, weights=None, device="cpu"):
        self.device = torch.device(device)
        self.model = LPRNet(num_classes=len(CHARS), dropout=0).to(self.device)
        if weights:
            state = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(state.get("model", state))
        self.model.eval()

    @torch.no_grad()
    def __call__(self, plate_bgr):
        x = torch.from_numpy(preprocess(plate_bgr)).unsqueeze(0).to(self.device)
        logits = self.model(x)[0].cpu()
        return ctc_greedy_decode(logits)
