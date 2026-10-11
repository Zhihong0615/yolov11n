#!/usr/bin/env python3
"""Move a trained ReLU-flow YOLO26n into the PyTorch 1.10 DPU graph.

Every C3k2 cv1 convolution/BN is partitioned by output channel into two
parallel convolutions. The two raw one-to-one Detect towers retain the exact
trained weights and output semantics before YOLO26 TopK/decode. This script
refuses to export if raw-map float parity fails on a real 640x640 road frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from model import C3k2NoSlice, DpuFlowProbe
from speed_model import DpuFlowFast1875
from l2half_model import DpuFlowL2Half


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split_cv1(source, first, second):
    state = source.state_dict()
    first_state, second_state = first.state_dict(), second.state_dict()
    if set(state) != set(first_state) or set(state) != set(second_state):
        raise ValueError("C3k2 cv1 submodule state names changed")
    hidden = first.conv.out_channels
    for key, value in state.items():
        if key == "bn.num_batches_tracked":
            first_state[key] = value.clone()
            second_state[key] = value.clone()
        else:
            if value.shape[0] != 2 * hidden:
                raise ValueError("C3k2 cv1 state shape changed: " + key)
            first_state[key] = value[:hidden].clone()
            second_state[key] = value[hidden:].clone()
    first.load_state_dict(first_state, strict=True)
    second.load_state_dict(second_state, strict=True)


def transplant_c3k2(source, target):
    if not isinstance(target, C3k2NoSlice) or len(source.m) != 1:
        raise ValueError("Unexpected C3k2 bridge shape")
    load_split_cv1(source.cv1, target.cv1a, target.cv1b)
    target.cv2.load_state_dict(source.cv2.state_dict(), strict=True)
    target.m.load_state_dict(source.m[0].state_dict(), strict=True)


def make_initialized_reference(yaml_path, checkpoint):
    original = YOLO(str(checkpoint)).model.cpu().eval()
    variant = YOLO(str(yaml_path))
    variant.load(str(checkpoint))
    source = variant.model.cpu().eval()
    source.model[22].m[0].load_state_dict(original.model[22].m[0][0].state_dict(), strict=True)
    return source


def raw_from_packed(packed):
    if tuple(packed.shape) != (1, 30, 80, 80):
        raise ValueError("Unexpected packed output shape")
    maps = (packed[:, 0:10], packed[:, 10:20, ::2, ::2], packed[:, 20:30, ::4, ::4])
    boxes = torch.cat([m[:, :4].reshape(1, 4, -1) for m in maps], -1)
    scores = torch.cat([m[:, 4:].reshape(1, 6, -1) for m in maps], -1)
    return boxes, scores


def raw_from_six(outputs):
    if len(outputs) != 6:
        raise ValueError("Expected six native box/class outputs")
    expected = ((4, 80), (6, 80), (4, 40), (6, 40), (4, 20), (6, 20))
    if any(tuple(value.shape) != (1, channels, width, width)
           for value, (channels, width) in zip(outputs, expected)):
        raise ValueError("Unexpected six raw output shapes")
    boxes = torch.cat([outputs[2 * i].reshape(1, 4, -1) for i in range(3)], -1)
    scores = torch.cat([outputs[2 * i + 1].reshape(1, 6, -1) for i in range(3)], -1)
    return boxes, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--initialize-from-reference", action="store_true")
    parser.add_argument("--architecture", choices=("relu_flow_025", "relu_flow_01875", "relu_flow_025_l2half"),
                        default="relu_flow_025")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.checkpoint, args.yaml, args.image):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.architecture != "relu_flow_025" and args.initialize_from_reference:
        raise ValueError("Alternate architecture requires its own sliced/trained checkpoint")
    source = (make_initialized_reference(args.yaml, args.checkpoint) if args.initialize_from_reference
              else YOLO(str(args.checkpoint)).model.cpu().eval())
    if (type(source.model[10]).__name__ != "C3k2" or
        type(source.model[22].m[0]).__name__ != "Bottleneck" or
        source.model[23].nc != 6 or source.model[23].reg_max != 1):
        raise ValueError("Checkpoint does not have the ReLU-flow derivative graph")
    if any(isinstance(m, torch.nn.SiLU) for m in source.modules()):
        raise ValueError("Derivative still contains SiLU")
    bridge = {"relu_flow_025": lambda: DpuFlowProbe("packed"),
              "relu_flow_01875": lambda: DpuFlowFast1875("six"),
              "relu_flow_025_l2half": lambda: DpuFlowL2Half("six")}[args.architecture]().eval()
    c3_indices = []
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 17, 19, 20, 22):
        original_layer = source.model[i]
        bridge_layer = getattr(bridge, "l%d" % i)
        if isinstance(bridge_layer, C3k2NoSlice):
            transplant_c3k2(original_layer, bridge_layer)
            c3_indices.append(i)
        else:
            bridge_layer.load_state_dict(original_layer.state_dict(), strict=True)
    head = source.model[23]
    bridge.detect.one2one_cv2.load_state_dict(head.one2one_cv2.state_dict(), strict=True)
    bridge.detect.one2one_cv3.load_state_dict(head.one2one_cv3.state_dict(), strict=True)
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode input image")
    prepared = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False,
                         scaleup=True, center=True)(image=frame)
    inp = np.ascontiguousarray(prepared[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
    x = torch.from_numpy(inp)
    with torch.no_grad():
        source_output = source(x)
        bridge_output = bridge(x)
    if not isinstance(source_output, tuple) or len(source_output) != 2:
        raise ValueError("Expected Ultralytics one-to-one evaluation pair")
    reference = source_output[1]["one2one"]
    boxes, scores = (raw_from_packed(bridge_output) if args.architecture == "relu_flow_025"
                     else raw_from_six(bridge_output))
    comparisons = {}
    for name, actual in (("boxes", boxes), ("scores", scores)):
        expected = reference[name]
        if actual.shape != expected.shape:
            raise ValueError("Raw %s shape differs: %r vs %r" % (name, actual.shape, expected.shape))
        delta = (actual - expected).abs()
        comparisons[name] = {"max_abs": float(delta.max()),
                             "mean_abs": float(delta.mean()),
                             "allclose_atol1e-4_rtol1e-3": bool(torch.allclose(actual, expected, atol=1e-4, rtol=1e-3))}
    if not all(item["allclose_atol1e-4_rtol1e-3"] for item in comparisons.values()):
        raise AssertionError("DPU bridge raw head differs from the trained graph: " + repr(comparisons))
    weights = args.out / "relu_flow_bridge_weights.npz"
    np.savez_compressed(weights, **{key: value.detach().numpy() for key, value in bridge.state_dict().items()})
    report = {
        "identity": "yolo26n-kv260-relu-flow-bridge",
        "architecture": args.architecture,
        "checkpoint_sha256": sha256(args.checkpoint),
        "variant_yaml_sha256": sha256(args.yaml),
        "source_model_sha256": sha256(Path(__file__).with_name({
            "relu_flow_025": "model.py", "relu_flow_01875": "speed_model.py",
            "relu_flow_025_l2half": "l2half_model.py"}[args.architecture])),
        "image_sha256": sha256(args.image),
        "bridge_weight_sha256": sha256(weights),
        "bridge_weight_file": str(weights),
        "c3k2_no_slice_transplanted_layers": c3_indices,
        "bridge_output": ("packed NCHW [1,30,80,80]; raw box+class maps before decode/TopK"
                          if args.architecture == "relu_flow_025" else
                          "six NCHW native box/class maps at 80,40,20; raw before decode/TopK"),
        "packed_extract": ("P3 channels0:10 every pixel; P4 channels10:20 at [::2,::2]; P5 channels20:30 at [::4,::4]"
                           if args.architecture == "relu_flow_025" else None),
        "raw_float_parity": comparisons,
        "source_script_sha256": sha256(Path(__file__)),
    }
    (args.out / "bridge_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
