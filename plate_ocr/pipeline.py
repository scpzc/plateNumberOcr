"""检测 + 识别 + 后处理串起来。"""

from .detector import PlateDetector
from .postprocess import guess_color, is_valid, normalize
from .recognizer import PlateRecognizer


class PlatePipeline:
    def __init__(self, det_weights, rec_weights, device="cpu", conf=0.4):
        self.detector = PlateDetector(det_weights, conf=conf, device=device)
        self.recognizer = PlateRecognizer(rec_weights, device=device)

    def __call__(self, img_bgr):
        results = []
        h, w = img_bgr.shape[:2]
        for det in self.detector(img_bgr):
            x1, y1, x2, y2 = det["box"]
            crop = img_bgr[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
            if crop.size == 0:
                continue
            text, conf = self.recognizer(crop)
            text = normalize(text)
            results.append({
                "plateNumber": text,
                "plateColor": guess_color(crop),
                "valid": is_valid(text),
                "confidence": round(conf, 4),
                "box": det["box"],
                "detScore": det["score"],
            })
        return results
