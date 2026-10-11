#!/usr/bin/env python3
"""Quantize a full ReLU-flow DPU graph with either synthetic or real inputs.

Synthetic mode establishes compiler coverage only. Real mode requires a
transplanted bridge weight NPZ and the fixed train-only calibration list.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import pytorch_nndct
from pytorch_nndct.apis import torch_quantizer

from model import DpuFlowProbe
from speed_model import DpuFlowFast1875
from speed_model_015625 import DpuFlowFast15625
from l2drop_model import DpuFlowL2Drop
from l2half_model import DpuFlowL2Half


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_from_image(path):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode " + str(path))
    h, w = frame.shape[:2]
    gain = min(640 / h, 640 / w)
    new_w, new_h = round(w * gain), round(h * gain)
    dw, dh = (640 - new_w) / 2, (640 - new_h) / 2
    left, right = round(dw - 0.1), round(dw + 0.1)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if (new_w, new_h) != (w, h) else frame
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if padded.shape != (640, 640, 3):
        raise AssertionError("Unexpected letterbox shape")
    array = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
    return torch.from_numpy(array)


def load_weights(model, path):
    expected = model.state_dict()
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("Bridge NPZ does not exactly match DPU model state")
        state = {}
        for key, tensor in expected.items():
            value = torch.from_numpy(archive[key].copy())
            if value.shape != tensor.shape or value.dtype != tensor.dtype:
                raise ValueError("Bridge state shape/dtype mismatch: " + key)
            state[key] = value
    model.load_state_dict(state, strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calib", "test"), required=True)
    parser.add_argument("--output-mode", choices=("packed", "separate", "six", "six_clip8"), default="packed")
    parser.add_argument("--architecture", choices=("relu_flow_025", "relu_flow_01875", "relu_flow_015625", "relu_flow_025_l2drop", "relu_flow_025_l2half"),
                        default="relu_flow_025")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--calibration-list", type=Path)
    parser.add_argument("--quant-config-file", type=Path,
                        help="Explicit native VAI2.5 quantization configuration variant")
    parser.add_argument("--train-list", type=Path)
    parser.add_argument("--test-image", type=Path)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--fast-finetune-limit", type=int, default=0,
                        help="TRAIN calibration image count for native VAI 2.5 AdaQuant; 0 disables it")
    parser.add_argument("--fast-finetune-stems-file", type=Path,
                        help="Optional ordered TRAIN-calibration JPEG names for native AdaQuant")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(260)
    models = {"relu_flow_025": DpuFlowProbe,
              "relu_flow_01875": DpuFlowFast1875,
              "relu_flow_015625": DpuFlowFast15625,
              "relu_flow_025_l2drop": DpuFlowL2Drop,
              "relu_flow_025_l2half": DpuFlowL2Half}
    model = models[args.architecture](args.output_mode).eval()
    if args.weights is not None:
        if not args.weights.is_file():
            raise FileNotFoundError(args.weights)
        load_weights(model, args.weights)
    elif args.calibration_list is not None:
        raise ValueError("Real calibration requires a bridge weight NPZ")
    if args.output_mode == "six_clip8":
        model.apply_clip8_class_bias_shift()
    samples = []
    if args.calibration_list is not None:
        if args.train_list is None or not args.train_list.is_file():
            raise ValueError("Real calibration requires the frozen training list")
        allowed = {Path(line.strip()).resolve() for line in args.train_list.read_text().splitlines() if line.strip()}
        paths = [Path(line.strip()).resolve() for line in args.calibration_list.read_text().splitlines() if line.strip()]
        if args.limit < 1 or args.limit > len(paths) or len(set(paths)) != len(paths):
            raise ValueError("Invalid calibration list/limit")
        samples = paths[:args.limit]
        if any(path not in allowed or not path.is_file() for path in samples):
            raise ValueError("Calibration image is missing or outside the fixed training split")
        x = input_from_image(samples[0])
    else:
        # Structural coverage only: random weights and synthetic input.
        x = torch.rand(1, 3, 640, 640)
    if args.fast_finetune_stems_file and not args.fast_finetune_limit:
        raise ValueError("Fast-finetune image selection requires a positive limit")
    finetune_paths = []
    if args.fast_finetune_limit:
        if not samples or args.fast_finetune_limit > len(samples):
            raise ValueError("Fast finetune requires TRAIN-only calibration samples")
        if args.fast_finetune_stems_file:
            names = [line.strip() for line in args.fast_finetune_stems_file.read_text().splitlines()
                     if line.strip()]
            by_name = {path.name: path for path in samples}
            if (len(names) != args.fast_finetune_limit or len(set(names)) != len(names) or
                    any(name not in by_name for name in names)):
                raise ValueError("Fast-finetune stems must be unique members of TRAIN calibration")
            finetune_paths = [by_name[name] for name in names]
        else:
            finetune_paths = samples[:args.fast_finetune_limit]
    with torch.no_grad():
        y = model(x)
    shape = [list(t.shape) for t in y] if isinstance(y, tuple) else list(y.shape)
    print("float output", shape, "parameters", sum(p.numel() for p in model.parameters()), flush=True)
    quantizer = torch_quantizer(args.mode, model, (x,), output_dir=str(args.out),
                               device=torch.device("cpu"),
                               quant_config_file=(str(args.quant_config_file)
                                                  if args.quant_config_file else None))
    qm = quantizer.quant_model.eval()
    if args.fast_finetune_limit:
        if args.mode == "calib":
            # VAI 2.5's installed cache mode uses 20 layerwise iterations;
            # its default mix mode uses 100. Both are genuine AdaQuant, but
            # the CPU-only pinned container makes a bounded probe essential.
            from nndct_shared.utils import option_util
            option_util.set_option_value("nndct_ft_mode", 0)

            def forward_loop(module, image_paths):
                for image_path in image_paths:
                    module(input_from_image(image_path))

            print("native VAI2.5 fast_finetune TRAIN images", len(finetune_paths), flush=True)
            quantizer.fast_finetune(forward_loop, (qm, finetune_paths))
        else:
            # The calibration pass exports its refined parameter file and
            # marks quant_info.json fast_finetuned=true. Load both together.
            quantizer.load_ft_param()
    with torch.no_grad():
        if args.mode == "calib" and samples:
            for index, path in enumerate(samples):
                qm(input_from_image(path))
                print("calibrated %d/%d %s" % (index + 1, len(samples), path.name), flush=True)
        else:
            test_input = input_from_image(args.test_image) if args.test_image else x
            quant_output = qm(test_input)
    if args.mode == "calib":
        quantizer.export_quant_config()
    else:
        outputs = (quant_output,) if isinstance(quant_output, torch.Tensor) else quant_output
        stats = []
        for index, raw in enumerate(outputs):
            values = raw.detach().cpu().numpy()
            np.save(args.out / ("test_output_%d_fp32.npy" % index), values)
            stats.append({"index": index, "shape": list(values.shape), "min": float(values.min()),
                          "max": float(values.max()), "mean": float(values.mean()),
                          "std": float(values.std()),
                          "sha256": hashlib.sha256(values.tobytes()).hexdigest()})
        (args.out / "test_output_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        quantizer.export_xmodel(output_dir=str(args.out), deploy_check=False)
    report = {"status": "weighted_real_calibration_probe" if samples else "structural_coverage_only", "mode": args.mode,
              "output_mode": args.output_mode, "input_shape": list(x.shape),
              "architecture": args.architecture,
              "class_output_transform": "relu(raw_logit+8); ARM subtracts 8" if args.output_mode == "six_clip8" else None,
              "float_output_shape": shape,
              "parameters": sum(p.numel() for p in model.parameters()),
              "source_sha256": sha256(Path(__file__).with_name({
                  "relu_flow_025": "model.py", "relu_flow_01875": "speed_model.py",
                  "relu_flow_015625": "speed_model_015625.py",
                  "relu_flow_025_l2drop": "l2drop_model.py",
                  "relu_flow_025_l2half": "l2half_model.py"}[args.architecture])),
              "quantizer_entrypoint_sha256": sha256(Path(__file__)),
              "weights_sha256": sha256(args.weights) if args.weights else None,
              "quant_config_file_sha256": (sha256(args.quant_config_file)
                                           if args.quant_config_file else None),
              "calibration_list_sha256": sha256(args.calibration_list) if args.calibration_list else None,
              "train_list_sha256": sha256(args.train_list) if args.train_list else None,
              "test_image_sha256": sha256(args.test_image) if args.test_image else None,
              "fast_finetune_limit": args.fast_finetune_limit,
              "fast_finetune_mode": "native_vai25_cache_20_iters_per_layer" if args.fast_finetune_limit else None,
              "fast_finetune_stems_file_sha256": (sha256(args.fast_finetune_stems_file)
                                                 if args.fast_finetune_stems_file else None),
              "native_vai25_source_sha256": ({
                  relative: sha256(Path(pytorch_nndct.__file__).resolve().parent / relative)
                  for relative in ("apis.py", "qproc/base.py", "qproc/adaquant.py")
              } if args.fast_finetune_limit else None),
              "fast_finetune_samples": [{"path": str(path), "sha256": sha256(path)}
                                        for path in finetune_paths],
              "fast_finetune_parameter_sha256": (sha256(quantizer.processor.quantizer.param_file)
                                                    if args.fast_finetune_limit and
                                                    Path(quantizer.processor.quantizer.param_file).is_file() else None),
              "calibration_samples": [{"path": str(path), "sha256": sha256(path)} for path in samples]}
    (args.out / (args.mode + "_probe.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
