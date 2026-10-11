"""Structural, untrained YOLO26n-derived DPU coverage probe.

This keeps a P3/P4/P5 CSP-PAN topology at 640x640, but replaces every SiLU
with ReLU, replaces the two PSA attention blocks with spatial convolutions,
and replaces C3k2 channel slices with parallel convolutions. The detection
head emits *raw* 10-channel maps (4 box channels + 6 class logits). No quality
claim or original YOLO26n semantic equivalence is implied by this module.

``packed`` output makes all three feature maps one NCHW DPU tensor using
nearest-neighbor expansion. Software can read one value per 2x2/4x4 tile for
P4/P5; this is a graph-output packing convention, not a resolution change.
"""

import torch
from torch import nn
from torch.nn import functional as F


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act=True, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU(inplace=False) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(channels * expansion)
        self.cv1 = Conv(channels, hidden, 3)
        self.cv2 = Conv(hidden, channels, 3)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class C3k(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.cv1 = Conv(channels, half)
        self.cv2 = Conv(channels, half)
        self.cv3 = Conv(channels, channels)
        self.m = nn.Sequential(Bottleneck(half, expansion=1.0),
                               Bottleneck(half, expansion=1.0))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2NoSlice(nn.Module):
    """C3k2 channel split implemented as two independent 1x1 convolutions.

    A trained C3k2 cv1 can be transplanted exactly before BN folding by
    dividing its output channels, BN parameters and running statistics in two.
    The probe has random weights, so that equivalence is not yet tested here.
    """

    def __init__(self, c1, c2, hidden, c3k=False):
        super().__init__()
        self.cv1a = Conv(c1, hidden)
        self.cv1b = Conv(c1, hidden)
        self.m = C3k(hidden) if c3k else Bottleneck(hidden)
        self.cv2 = Conv(3 * hidden, c2)

    def forward(self, x):
        a = self.cv1a(x)
        b = self.cv1b(x)
        return self.cv2(torch.cat((a, b, self.m(b)), 1))


class SPPF(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = Conv(256, 128, act=False)
        self.m = nn.MaxPool2d(5, 1, 2)
        self.cv2 = Conv(512, 256)

    def forward(self, x):
        a = self.cv1(x)
        b = self.m(a)
        c = self.m(b)
        d = self.m(c)
        return x + self.cv2(torch.cat((a, b, c, d), 1))


class RawDetect(nn.Module):
    """Original one-to-one box/class tower topology, before CPU TopK/decode."""

    def __init__(self):
        super().__init__()
        channels = (64, 128, 256)
        self.one2one_cv2 = nn.ModuleList(
            nn.Sequential(Conv(c, 16, 3), Conv(16, 16, 3), nn.Conv2d(16, 4, 1))
            for c in channels
        )
        self.one2one_cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(c, c, 3, groups=c), Conv(c, 64)),
                nn.Sequential(Conv(64, 64, 3, groups=64), Conv(64, 64)),
                nn.Conv2d(64, 6, 1),
            ) for c in channels
        )

    def forward(self, y16, y19, y22, split=False, class_relu=False):
        maps = (y16, y19, y22)
        boxes = tuple(self.one2one_cv2[i](m) for i, m in enumerate(maps))
        classes = tuple(self.one2one_cv3[i](m) for i, m in enumerate(maps))
        if class_relu:
            classes = tuple(F.relu(value) for value in classes)
        if split:
            return boxes[0], classes[0], boxes[1], classes[1], boxes[2], classes[2]
        return tuple(torch.cat((boxes[i], classes[i]), 1) for i in range(3))


class DpuFlowProbe(nn.Module):
    """Untrained, single-input, DPU-oriented YOLO26n-derived detector graph."""

    def __init__(self, output_mode="packed"):
        super().__init__()
        if output_mode not in ("packed", "separate", "six", "six_clip8"):
            raise ValueError(output_mode)
        self.output_mode = output_mode
        self._class_bias_shift_applied = False
        self.l0 = Conv(3, 16, 3, 2)
        self.l1 = Conv(16, 32, 3, 2)
        self.l2 = C3k2NoSlice(32, 64, 16)
        self.l3 = Conv(64, 64, 3, 2)
        self.l4 = C3k2NoSlice(64, 128, 32)
        self.l5 = Conv(128, 128, 3, 2)
        self.l6 = C3k2NoSlice(128, 128, 64, c3k=True)
        self.l7 = Conv(128, 256, 3, 2)
        self.l8 = C3k2NoSlice(256, 256, 128, c3k=True)
        self.l9 = SPPF()
        # Deliberate variant: PSA -> local CSP block, no Softmax.
        self.l10 = C3k2NoSlice(256, 256, 128)
        self.l13 = C3k2NoSlice(384, 128, 64, c3k=True)
        self.l16 = C3k2NoSlice(256, 64, 32, c3k=True)
        self.l17 = Conv(64, 64, 3, 2)
        self.l19 = C3k2NoSlice(192, 128, 64, c3k=True)
        self.l20 = Conv(128, 128, 3, 2)
        # Deliberate variant: attention-containing layer 22 -> Bottleneck CSP.
        self.l22 = C3k2NoSlice(384, 256, 128)
        self.detect = RawDetect()

    def apply_clip8_class_bias_shift(self):
        """Make `ReLU(logit+8)` the DPU class output for six_clip8 mode.

        ARM subtracts eight after final INT8 dequantization. Values below -8
        are floored to -8; their sigmoid is below the fixed AP floor 0.001.
        """
        if self.output_mode != "six_clip8" or self._class_bias_shift_applied:
            raise ValueError("Class shift requires a fresh six_clip8 model")
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
        y21 = torch.cat((self.l20(y19), y10), 1)
        y22 = self.l22(y21)
        if self.output_mode in ("six", "six_clip8"):
            if self.output_mode == "six_clip8" and not self._class_bias_shift_applied:
                raise RuntimeError("Apply the class bias shift after loading bridge weights")
            return self.detect(y16, y19, y22, split=True,
                               class_relu=self.output_mode == "six_clip8")
        p3, p4, p5 = self.detect(y16, y19, y22)
        if self.output_mode == "separate":
            return p3, p4, p5
        return torch.cat((p3, F.interpolate(p4, scale_factor=2, mode="nearest"),
                          F.interpolate(p5, scale_factor=4, mode="nearest")), 1)
