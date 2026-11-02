"""CCPD 文件名解析。

示例：025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
第 3 段是检测框（左上_右下），第 5 段是车牌字符索引。
"""

CCPD_PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
    "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O",
]
CCPD_ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]
CCPD_ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") + ["O"]


def parse_name(filename):
    parts = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0].split("-")
    if len(parts) < 7:
        raise ValueError(f"not a CCPD filename: {filename}")
    (x1, y1), (x2, y2) = [map(int, p.split("&")) for p in parts[2].split("_")]
    idx = [int(i) for i in parts[4].split("_")]
    plate = CCPD_PROVINCES[idx[0]] + CCPD_ALPHABETS[idx[1]] + "".join(CCPD_ADS[i] for i in idx[2:])
    return (x1, y1, x2, y2), plate.replace("O", "")
