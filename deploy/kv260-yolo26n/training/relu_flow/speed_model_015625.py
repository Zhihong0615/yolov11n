"""Independent width-0.15625 ReLU-flow compiler and timing candidate.

This is an untrained structural probe until weight migration, six-class
training, real 512-image calibration and board quality qualification pass.
"""

import torch
from torch import nn
from torch.nn import functional as F

from model import C3k2NoSlice, Conv


class SmallSPPF(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = Conv(160, 80, act=False)
        self.m = nn.MaxPool2d(5, 1, 2)
        self.cv2 = Conv(320, 160)

    def forward(self, x):
        a = self.cv1(x)
        b = self.m(a)
        c = self.m(b)
        d = self.m(c)
        return x + self.cv2(torch.cat((a, b, c, d), 1))


class SmallRawDetect(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (40, 80, 160)
        c2, c3 = 16, 40
        self.one2one_cv2 = nn.ModuleList(
            nn.Sequential(Conv(c, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4, 1))
            for c in channels
        )
        self.one2one_cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(c, c, 3, groups=c), Conv(c, c3)),
                nn.Sequential(Conv(c3, c3, 3, groups=c3), Conv(c3, c3)),
                nn.Conv2d(c3, 6, 1),
            ) for c in channels
        )

    def forward(self, y16, y19, y22, clip8=False):
        maps = (y16, y19, y22)
        boxes = tuple(self.one2one_cv2[i](m) for i, m in enumerate(maps))
        classes = tuple(self.one2one_cv3[i](m) for i, m in enumerate(maps))
        if clip8:
            classes = tuple(F.relu(value) for value in classes)
        return boxes[0], classes[0], boxes[1], classes[1], boxes[2], classes[2]


class DpuFlowFast15625(nn.Module):
    def __init__(self, output_mode="six"):
        super().__init__()
        if output_mode not in ("six", "six_clip8"):
            raise ValueError(output_mode)
        self.output_mode = output_mode
        self._class_bias_shift_applied = False
        self.l0 = Conv(3, 16, 3, 2)
        self.l1 = Conv(16, 24, 3, 2)
        self.l2 = C3k2NoSlice(24, 40, 10)
        self.l3 = Conv(40, 40, 3, 2)
        self.l4 = C3k2NoSlice(40, 80, 20)
        self.l5 = Conv(80, 80, 3, 2)
        self.l6 = C3k2NoSlice(80, 80, 40, c3k=True)
        self.l7 = Conv(80, 160, 3, 2)
        self.l8 = C3k2NoSlice(160, 160, 80, c3k=True)
        self.l9 = SmallSPPF()
        self.l10 = C3k2NoSlice(160, 160, 80)
        self.l13 = C3k2NoSlice(240, 80, 40, c3k=True)
        self.l16 = C3k2NoSlice(160, 40, 20, c3k=True)
        self.l17 = Conv(40, 40, 3, 2)
        self.l19 = C3k2NoSlice(120, 80, 40, c3k=True)
        self.l20 = Conv(80, 80, 3, 2)
        self.l22 = C3k2NoSlice(240, 160, 80)
        self.detect = SmallRawDetect()

    def apply_clip8_class_bias_shift(self):
        if self.output_mode != "six_clip8" or self._class_bias_shift_applied:
            raise ValueError("Class shift requires a fresh six_clip8 speed model")
        with torch.no_grad():
            for tower in self.detect.one2one_cv3:
                tower[-1].bias.add_(8.0)
        self._class_bias_shift_applied = True

    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        y4 = self.l4(x)
        x = self.l5(y4)
        y6 = self.l6(x)
        x = self.l7(y6)
        x = self.l8(x)
        y9 = self.l9(x)
        y10 = self.l10(y9)
        y13 = self.l13(torch.cat((F.interpolate(y10, scale_factor=2, mode="nearest"), y6), 1))
        y16 = self.l16(torch.cat((F.interpolate(y13, scale_factor=2, mode="nearest"), y4), 1))
        y19 = self.l19(torch.cat((self.l17(y16), y13), 1))
        y22 = self.l22(torch.cat((self.l20(y19), y10), 1))
        if self.output_mode == "six_clip8" and not self._class_bias_shift_applied:
            raise RuntimeError("Apply class bias shift after loading weights")
        return self.detect(y16, y19, y22, self.output_mode == "six_clip8")
