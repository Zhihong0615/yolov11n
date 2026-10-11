"""Width-0.25 ReLU-flow graph with only the first CSP hidden width halved.

The 160x160 layer2 Bottleneck remains. Its hidden channels are 8 instead of
16; all other stages and the six raw output tensors retain their widths.
This is a structural speed probe and needs separate training/quality checks.
"""

from model import C3k2NoSlice, DpuFlowProbe


class DpuFlowL2Half(DpuFlowProbe):
    def __init__(self, output_mode="six_clip8"):
        super().__init__(output_mode)
        self.l2 = C3k2NoSlice(32, 64, 8)
