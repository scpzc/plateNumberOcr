# 车牌号识别 OCR · 蓝兴 AI

> 官网：[https://www.88apis.com](https://www.88apis.com)  
> 产品页 / 在线体验：[https://www.88apis.com/plateNumberOcr.html](https://www.88apis.com/plateNumberOcr.html)  
> API 接口：[阿里云市场 · 车牌号识别](https://market.aliyun.com/detail/cmapi00074871)

本仓库是一个**基于开源方案的车牌识别入门项目**（YOLOv8 检测 + LPRNet 识别 + CCPD 数据集），方便学习车牌识别的完整流程：数据准备 → 检测训练 → 识别训练 → 推理。

如果你需要**直接可用于生产**的车牌识别（夜间、大角度、远距离、截图/PDF、双层黄牌、港澳牌等复杂场景），可以直接试用我们的在线服务，每天免费 10 次 👉 [88apis 车牌号识别](https://www.88apis.com/plateNumberOcr.html)

---

## 在线服务能力

多色多类型号牌识别，返回号牌与颜色类型。支持蓝牌、绿牌新能源、黄牌单层/双层、白牌、黑牌港澳及港澳常见号牌，输出规范化号牌号码。支持白天，夜晚，大角度拍摄识别，准确率高，可试用看效果。

```json
{
  "code": 200,
  "data": {
    "plateNumber": "川AQ3C28",
    "plateColor": "blue",
    "plateLayer": "single"
  }
}
```

- 覆盖主流号牌类型与新能源绿牌
- 返回号牌号码 + 颜色/类型，便于业务分流
- 支持图片和 PDF 输入
- 可与 VIN、行驶证识别互补做多源核验

**适用场景**：停车场 / 物流园进出场 · 车险报案录牌 · 充电站 / 洗车店 · 车队与路侧管理

---

## 开源示例项目

### 目录结构

```
plate_ocr/
  chars.py        字符表（省份 + 数字 + 字母，不含 I/O）
  lprnet.py       LPRNet 模型
  detector.py     YOLOv8 车牌检测
  recognizer.py   LPRNet 推理 + CTC 解码
  postprocess.py  车牌格式校验、HSV 颜色判断
  pipeline.py     检测 + 识别 串联
tools/
  ccpd.py         CCPD 文件名解析
  ccpd_to_yolo.py CCPD -> YOLO 检测数据集
  ccpd_to_rec.py  CCPD -> 车牌小图识别数据集
train_det.py      训练检测模型
train_rec.py      训练识别模型
demo.py           推理示例
```

### 安装

```bash
pip install -r requirements.txt
```

### 1. 准备数据

下载 [CCPD2019 / CCPD2020](https://github.com/detectRecog/CCPD)，然后转换：

```bash
# 检测数据集
python tools/ccpd_to_yolo.py --src CCPD2019/ccpd_base CCPD2020/ccpd_green/train --dst datasets/plate_det

# 识别数据集
python tools/ccpd_to_rec.py --src CCPD2019/ccpd_base CCPD2020/ccpd_green/train --dst datasets/plate_rec/train
python tools/ccpd_to_rec.py --src CCPD2020/ccpd_green/val --dst datasets/plate_rec/val
```

### 2. 训练检测模型

```bash
python train_det.py --data datasets/plate_det/data.yaml --epochs 100
```

### 3. 训练识别模型

```bash
python train_rec.py --train datasets/plate_rec/train --val datasets/plate_rec/val --epochs 100
```

### 4. 推理

```bash
python demo.py --det runs/detect/plate/weights/best.pt --rec weights/lprnet.pth --img test.jpg --save out.jpg --font simhei.ttf
```

输出示例：

```json
[
  {
    "plateNumber": "皖AY339S",
    "plateColor": "blue",
    "valid": true,
    "confidence": 0.9812,
    "box": [154, 383, 386, 473],
    "detScore": 0.9431
  }
]
```

### 测试

```bash
pip install pytest
pytest tests
```

---

## 开源方案的局限

CCPD 上很容易训到 99%，但直接用于真实业务通常会遇到：

| 问题 | 原因 |
| --- | --- |
| 非"皖"省份识别差 | CCPD 绝大部分是安徽车牌，省份分布极不均衡 |
| 黄牌、白牌、港澳牌识别差 | 训练数据里几乎没有这些样本 |
| 双层车牌识别乱码 | LPRNet 按单行序列识别 |
| 夜间、斜拍、远距离漏检 | CCPD 场景单一 |
| 截图、PDF、翻拍效果差 | 没见过此类数据 |

这些问题需要大量真实业务数据和针对性优化。如果不想自己踩坑，欢迎直接试用 👉 [https://www.88apis.com/plateNumberOcr.html](https://www.88apis.com/plateNumberOcr.html)

## 参考

- [CCPD: Chinese City Parking Dataset](https://github.com/detectRecog/CCPD)
- [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## License

MIT
