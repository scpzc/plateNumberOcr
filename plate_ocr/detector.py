"""车牌检测：基于 ultralytics YOLOv8。"""


class PlateDetector:
    def __init__(self, weights, conf=0.4, device="cpu"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def __call__(self, img_bgr):
        res = self.model(img_bgr, conf=self.conf, device=self.device, verbose=False)[0]
        boxes = []
        for (x1, y1, x2, y2), score in zip(res.boxes.xyxy.int().tolist(), res.boxes.conf.tolist()):
            boxes.append({"box": [x1, y1, x2, y2], "score": round(score, 4)})
        return boxes
