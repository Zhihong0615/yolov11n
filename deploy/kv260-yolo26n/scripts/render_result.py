#!/usr/bin/env python3
"""Render one frame from the KV260 runner's concatenated detection binary."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

CLASS_NAMES = ("Car", "Cyclist", "Pedestrian", "Tram", "Tricycle", "Truck")
MAX_DETECTIONS = 300
ROW_FLOATS = 6


def parse_result_file(path: Path) -> list[list[tuple[float, ...]]]:
    data = path.read_bytes()
    offset = 0
    frames: list[list[tuple[float, ...]]] = []
    while offset < len(data):
        if len(data) - offset < 4:
            raise ValueError(f"Truncated detection count at byte {offset}")
        count = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if count > MAX_DETECTIONS:
            raise ValueError(f"Detection count {count} exceeds {MAX_DETECTIONS}")
        byte_count = count * ROW_FLOATS * 4
        if len(data) - offset < byte_count:
            raise ValueError(f"Truncated detections for frame {len(frames)}")
        flat = struct.unpack_from("<" + "f" * (count * ROW_FLOATS), data, offset)
        offset += byte_count
        frames.append([tuple(flat[i:i + ROW_FLOATS])
                       for i in range(0, len(flat), ROW_FLOATS)])
    if not frames:
        raise ValueError("Result file contains no frames")
    return frames


def load_image_list(path: Path) -> list[Path]:
    images = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if not images:
        raise ValueError("Image list is empty")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_list", type=Path)
    parser.add_argument("result_bin", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = parse_result_file(args.result_bin)
    if args.frame_index < 0 or args.frame_index >= len(frames):
        raise ValueError(f"Frame index must be in [0, {len(frames) - 1}]")
    images = load_image_list(args.image_list)
    image_path = images[args.frame_index % len(images)]

    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    colors = ((55, 62, 225), (220, 125, 30), (180, 55, 170),
              (18, 150, 75), (25, 165, 230), (195, 165, 20))
    for x1, y1, x2, y2, score, class_id_float in frames[args.frame_index]:
        class_id = int(round(class_id_float))
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"Unknown class id: {class_id}")
        color = colors[class_id]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(image, p1, p2, color, 4, cv2.LINE_AA)
        label = f"{CLASS_NAMES[class_id]} {score:.3f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x = max(0, p1[0])
        y = max(text_h + baseline + 4, p1[1])
        cv2.rectangle(image, (x, y - text_h - baseline - 4),
                      (x + text_w + 8, y), color, -1)
        cv2.putText(image, label, (x + 4, y - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise OSError(f"Could not write preview: {args.output}")
    print(f"Rendered frame {args.frame_index}: {image_path} -> {args.output}")
    print(f"Detections: {len(frames[args.frame_index])}")


if __name__ == "__main__":
    main()
