import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from ccpd import parse_name  # noqa: E402
from plate_ocr.chars import BLANK_IDX, CHAR2IDX, CHARS  # noqa: E402
from plate_ocr.lprnet import LPRNet  # noqa: E402
from plate_ocr.postprocess import guess_color, is_valid  # noqa: E402
from plate_ocr.recognizer import PlateRecognizer, ctc_greedy_decode  # noqa: E402


def test_lprnet_shape():
    out = LPRNet(len(CHARS)).eval()(torch.zeros(2, 3, 24, 94))
    assert out.shape == (2, len(CHARS), 18)


def test_ctc_decode():
    seq = ["京", "京", "-", "A", "-", "1", "1", "-", "1", "2", "3", "4"]
    logits = torch.full((len(CHARS), len(seq)), -10.0)
    for t, c in enumerate(seq):
        logits[BLANK_IDX if c == "-" else CHAR2IDX[c], t] = 10.0
    assert ctc_greedy_decode(logits)[0] == "京A11234"


def test_recognizer_runs():
    text, conf = PlateRecognizer()(np.zeros((40, 140, 3), dtype=np.uint8))
    assert isinstance(text, str) and 0 <= conf <= 1


def test_ccpd_parse():
    box, plate = parse_name("025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg")
    assert box == (154, 383, 386, 473)
    assert plate == "皖AY339S"


def test_valid():
    assert is_valid("川AQ3C28")
    assert is_valid("粤BD12345")
    assert not is_valid("川AQ3C2")
    assert not is_valid("AB12345")


def test_color():
    blue = np.zeros((24, 94, 3), dtype=np.uint8)
    blue[:] = (200, 80, 20)  # BGR
    yellow = np.zeros_like(blue)
    yellow[:] = (20, 200, 230)
    assert guess_color(blue) == "blue"
    assert guess_color(yellow) == "yellow"
