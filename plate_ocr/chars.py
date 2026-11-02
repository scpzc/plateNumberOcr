"""LPRNet 字符表。最后一位为 CTC blank。"""

PROVINCES = [
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
]
SPECIAL = ["学", "警", "港", "澳", "挂", "使", "领"]
DIGITS = list("0123456789")
# 车牌中不使用字母 I 和 O
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")

BLANK = "-"
CHARS = PROVINCES + SPECIAL + DIGITS + LETTERS + [BLANK]
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
BLANK_IDX = len(CHARS) - 1
