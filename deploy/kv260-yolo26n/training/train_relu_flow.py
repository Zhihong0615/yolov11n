#!/usr/bin/env python3
"""Fine-tune the distinct ReLU/CSP YOLO26n flow derivative on the fixed split.

Training keeps Ultralytics' dual Detect supervision and one-to-one validation.
Deployment later splits C3k2 cv1 into parallel convolutions and exports the
raw one-to-one maps before CPU boundary decode/TopK. It is a distinct variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO, settings


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--project-git-commit", default="unknown")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ReLU derivative training run")
    if not (0 < args.fraction <= 1):
        raise ValueError("fraction must be in (0, 1]")
    for path in (args.data, args.yaml, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    settings.update({"sync": False})
    os.environ["YOLO_OFFLINE"] = "true"
    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    run_dir = project / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    source = YOLO(str(args.checkpoint)).model.cpu().eval()
    variant = YOLO(str(args.yaml))
    variant.load(str(args.checkpoint))
    target = variant.model.cpu().eval()
    # The original layer 22 wraps its Bottleneck in Sequential(Bottleneck,
    # PSABlock). With attention disabled, the same Bottleneck moves to m[0].
    original_bottleneck = source.model[22].m[0][0].state_dict()
    replacement_bottleneck = target.model[22].m[0].state_dict()
    if (set(original_bottleneck) != set(replacement_bottleneck) or
        any(original_bottleneck[key].shape != replacement_bottleneck[key].shape
            for key in original_bottleneck)):
        raise ValueError("Layer 22 Bottleneck remap contract changed")
    target.model[22].m[0].load_state_dict(original_bottleneck, strict=True)
    remapped = {"model.22.m.0." + key: "model.22.m.0.0." + key
                for key in original_bottleneck}
    ss, ts = source.state_dict(), target.state_dict()
    transferable = {key for key in ts if key in ss and ts[key].shape == ss[key].shape}
    unequal = [key for key in transferable if not torch.equal(ts[key], ss[key])]
    if unequal:
        raise ValueError("Shape-compatible checkpoint state did not transfer: %r" % unequal[:12])
    if any(not torch.equal(ts[new], ss[old]) for new, old in remapped.items()):
        raise ValueError("Layer 22 Bottleneck state remap failed")
    by_layer = {}
    for key in sorted(transferable):
        layer = key.split(".")[1] if key.startswith("model.") else "other"
        info = by_layer.setdefault(layer, {"tensors": 0, "elements": 0})
        info["tensors"] += 1
        info["elements"] += ts[key].numel()
    target_types = [type(module).__name__ for module in target.modules()]
    if any(isinstance(module, torch.nn.SiLU) for module in target.modules()):
        raise ValueError("ReLU variant still contains SiLU")
    if any(name in ("PSABlock", "C2PSA", "Attention") for name in target_types):
        raise ValueError("ReLU variant still contains PSA attention")
    if (type(target.model[10]).__name__ != "C3k2" or
        type(target.model[22].m[0]).__name__ != "Bottleneck" or
        target.model[23].nc != 6 or target.model[23].reg_max != 1 or
        target.model[23].one2one_cv2 is None):
        raise ValueError("Unexpected ReLU flow graph or one-to-one head")
    if len(transferable) < 500:
        raise ValueError("Too little of the six-class source checkpoint transferred")
    structure_change = [
        "replace every SiLU with ReLU",
        "replace C2PSA layer 10 with C3k2 local convolution block",
        "disable PSA in layer 22 and retain its Bottleneck/Detect topology",
        "convert C3k2 cv1 channel slices into weight-equivalent parallel convolutions only at VAI export",
        "export one-to-one raw dense maps; perform final decode/TopK at the ARM output boundary",
    ]
    manifest = {
        "identity": "yolo26n-kv260-relu-flow-six-class-fixed-v1",
        "status": "dry_run" if args.dry_run else "training_started",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_sixclass_best_sha256": sha256(args.checkpoint),
        "variant_yaml_sha256": sha256(args.yaml),
        "data_yaml_sha256": sha256(args.data),
        "split_manifest_sha256": args.split_sha256,
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_state_tensors": len(ss), "target_state_tensors": len(ts),
        "transferred_state_tensors_exact": len(transferable),
        "transferred_state_elements_exact": sum(ts[key].numel() for key in transferable),
        "target_state_elements": sum(value.numel() for value in ts.values()),
        "remapped_layer22_bottleneck_tensors_exact": len(remapped),
        "remapped_layer22_bottleneck_elements_exact": sum(ts[key].numel() for key in remapped),
        "remapped_source_to_target_keys": remapped,
        "not_transferred_target_keys": sorted(set(ts) - transferable - set(remapped)),
        "transferred_by_layer": by_layer,
        "structure_change": structure_change,
        "validation_branch": "one-to-one (nms=False)",
        "training_script_sha256": sha256(Path(__file__)),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in
                      ("identity", "source_state_tensors", "target_state_tensors",
                       "transferred_state_tensors_exact", "transferred_state_elements_exact",
                       "target_state_elements", "remapped_layer22_bottleneck_tensors_exact",
                       "remapped_layer22_bottleneck_elements_exact",
                       "not_transferred_target_keys")}, indent=2), flush=True)
    if args.dry_run:
        return
    variant.train(
        data=str(args.data.resolve()), imgsz=640, batch=args.batch,
        epochs=args.epochs, workers=args.workers, fraction=args.fraction,
        patience=args.patience, seed=args.seed, deterministic=True,
        device=0, amp=True, cache=False, project=str(project), name=args.name,
        exist_ok=True, resume=False, nms=False, plots=False, verbose=True,
    )
    weights_dir = run_dir / "weights"
    manifest["status"] = "training_completed"
    manifest["weights"] = {name: sha256(weights_dir / name)
                           for name in ("best.pt", "last.pt") if (weights_dir / name).is_file()}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
