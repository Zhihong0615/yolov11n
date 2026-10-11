"""Independent width-0.25 ReLU-flow l2 Bottleneck removal hardware probe.

The 160x160 l2 CSP block retains its two cv1 branches and final cv2, but
removes the one internal Bottleneck. This changes the network function and
requires separate training/quality qualification before deployment.
"""

import torch
from torch import nn

from model import Conv, DpuFlowProbe


class C3k2NoBottleneck(nn.Module):
    def __init__(self, c1=32, c2=64, hidden=16):
        super().__init__()
        self.cv1a = Conv(c1, hidden)
        self.cv1b = Conv(c1, hidden)
        self.cv2 = Conv(2 * hidden, c2)

    def forward(self, x):
        return self.cv2(torch.cat((self.cv1a(x), self.cv1b(x)), 1))


class DpuFlowL2Drop(DpuFlowProbe):
    def __init__(self, output_mode="six"):
        super().__init__(output_mode)
        self.l2 = C3k2NoBottleneck()
