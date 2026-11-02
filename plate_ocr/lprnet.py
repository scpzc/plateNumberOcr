"""LPRNet 的 PyTorch 实现，参考论文 LPRNet: License Plate Recognition via Deep Neural Networks。

输入 [B, 3, 24, 94]，输出 [B, num_classes, 18]，用 CTC 解码。
"""

import torch
import torch.nn as nn


class SmallBasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        mid = ch_out // 4
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, mid, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(mid, mid, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(mid, mid, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(mid, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=1),            # 0
            nn.BatchNorm2d(64),
            nn.ReLU(),                                            # 2 -> 全局上下文
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            SmallBasicBlock(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),                                            # 6 -> 全局上下文
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            SmallBasicBlock(64, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SmallBasicBlock(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),                                            # 13 -> 全局上下文
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(64, 256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(256, num_classes, kernel_size=(13, 1), stride=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),                                            # 22 -> 全局上下文
        ])
        self.context_ids = {2, 6, 13, 22}
        self.container = nn.Conv2d(64 + 128 + 256 + num_classes, num_classes, kernel_size=1)

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.context_ids:
                feats.append(x)

        ctx = []
        for i, f in enumerate(feats):
            if i in (0, 1):
                f = nn.functional.avg_pool2d(f, kernel_size=5, stride=5)
            elif i == 2:
                f = nn.functional.avg_pool2d(f, kernel_size=(4, 10), stride=(4, 2))
            f = f / torch.mean(torch.pow(f, 2))
            ctx.append(f)

        x = torch.cat(ctx, dim=1)
        x = self.container(x)
        return torch.mean(x, dim=2)  # [B, C, T]
