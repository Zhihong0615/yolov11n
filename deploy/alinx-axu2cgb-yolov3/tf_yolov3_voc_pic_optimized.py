"""Faster headless runner for the original ALINX YOLOv3 DPU example.

The original example is kept unchanged.  This runner preserves its model,
anchors and decode logic, but removes display/file-I/O from the hot path,
uses wall-clock timing, caches fixed DPU tensor sizes, and limits NMS input
after confidence filtering.

Environment variables:
  YOLO_SHOW=1          show the annotated result (default: 0)
  YOLO_SAVE=1          save annotated results (default: 0)
  YOLO_MAX_IMAGES=N    process at most N images (default: all)
  YOLO_PRE_NMS_TOPK=N  keep at most N candidates per class (default: 200)
  YOLO_SCORE_THRESH=X  confidence threshold (default: original value)
"""

import os
import time

import cv2
import numpy as np
from dnndk import n2cube

from tf_yolov3_voc_pic import (
    ANCHORS_PATH,
    CLASSES_PATH,
    CONV_INPUT_NODE,
    CONV_OUTPUT_NODE1,
    CONV_OUTPUT_NODE2,
    CONV_OUTPUT_NODE3,
    INPUT_IMAGE_PATH,
    KERNEL_CONV,
    NMS_THRESH,
    OBJECT_TYPE,
    SCORE_THRESH as ORIGINAL_SCORE_THRESH,
    TEST_SET,
    boxes_and_scores,
    draw_bbox,
    get_anchors,
    get_class,
    pre_process,
)


SHOW = os.environ.get("YOLO_SHOW", "0") == "1"
SAVE = os.environ.get("YOLO_SAVE", "0") == "1"
MAX_IMAGES = int(os.environ.get("YOLO_MAX_IMAGES", "0"))
PRE_NMS_TOPK = max(1, int(os.environ.get("YOLO_PRE_NMS_TOPK", "200")))
SCORE_THRESH = float(os.environ.get("YOLO_SCORE_THRESH", str(ORIGINAL_SCORE_THRESH)))
OUTPUT_ROOT = os.environ.get("YOLO_OUTPUT_ROOT", "output_result_optimized")


def nms_boxes_fast(boxes, scores, iou_threshold, topk):
    """NMS compatible with the original example, with an early Top-K cap."""
    if boxes.size == 0 or scores.size == 0:
        return np.empty((0,), dtype=np.int32)

    order = np.argsort(scores)[::-1]
    if order.size > topk:
        order = order[:topk]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
    keep = []

    while order.size:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        width = np.maximum(0.0, xx2 - xx1 + 1.0)
        height = np.maximum(0.0, yy2 - yy1 + 1.0)
        intersection = width * height
        union = areas[current] + areas[rest] - intersection
        overlap = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )

        # The original code keeps overlap <= (1 - NMS_THRESH), so preserve
        # that behavior instead of silently changing detection semantics.
        order = rest[np.where(overlap <= (1.0 - iou_threshold))[0]]

    return np.asarray(keep, dtype=np.int32)


def eval_fast(yolo_outputs, image_shape, class_names, anchors):
    """Decode YOLO outputs and perform bounded per-class NMS."""
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    boxes = []
    box_scores = []
    input_shape = np.asarray(np.shape(yolo_outputs[0])[1:3], dtype=np.int32) * 32

    for i, output in enumerate(yolo_outputs):
        layer_boxes, layer_scores = boxes_and_scores(
            output,
            anchors[anchor_mask[i]],
            len(class_names),
            input_shape,
            image_shape,
        )
        boxes.append(layer_boxes)
        box_scores.append(layer_scores)

    boxes = np.concatenate(boxes, axis=0)
    box_scores = np.concatenate(box_scores, axis=0)

    # Discard anchors that cannot pass the threshold for any class before the
    # per-class loop.  This is usually the biggest software-side win.
    candidate_mask = np.max(box_scores, axis=1) >= SCORE_THRESH
    boxes = boxes[candidate_mask]
    box_scores = box_scores[candidate_mask]

    boxes_out = []
    scores_out = []
    classes_out = []
    for class_id in range(len(class_names)):
        class_mask = box_scores[:, class_id] >= SCORE_THRESH
        class_boxes = boxes[class_mask]
        class_scores = box_scores[class_mask, class_id]
        keep = nms_boxes_fast(class_boxes, class_scores, NMS_THRESH, PRE_NMS_TOPK)
        if keep.size:
            boxes_out.append(class_boxes[keep])
            scores_out.append(class_scores[keep])
            classes_out.append(np.full(keep.shape, class_id, dtype=np.int32))

    if not boxes_out:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    return (
        np.concatenate(boxes_out, axis=0),
        np.concatenate(scores_out, axis=0),
        np.concatenate(classes_out, axis=0),
    )


def image_paths():
    for test_set in TEST_SET:
        index = 1
        while True:
            image_name = test_set + "/img{:0>5d}.jpg".format(index)
            path = os.path.join(INPUT_IMAGE_PATH, image_name)
            if not os.path.exists(path):
                break
            yield image_name, path
            index += 1


def annotate(image, boxes, scores, classes, class_names):
    """Create the same style of annotated image only when requested."""
    image_h, image_w = image.shape[:2]
    draws = []
    for box, score, class_id in zip(boxes, scores, classes):
        top, left, bottom, right = box
        top = max(0, int(np.floor(top + 0.5)))
        left = max(0, int(np.floor(left + 0.5)))
        bottom = min(image_h, int(np.floor(bottom + 0.5)))
        right = min(image_w, int(np.floor(right + 0.5)))
        draws.append([left, top, right, bottom, score, class_id])
        cv2.putText(
            image,
            class_names[class_id] + ":" + str(round(float(score), 2)),
            (left, max(0, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (92, 51, 255),
            1,
        )
    return draw_bbox(image, draws, class_names)


def main():
    class_names = get_class(CLASSES_PATH)
    anchors = get_anchors(ANCHORS_PATH)
    object_dim = 3 * (5 + len(class_names))
    paths = image_paths()

    if SAVE:
        for test_set in TEST_SET:
            os.makedirs(os.path.join(OUTPUT_ROOT, test_set), exist_ok=True)

    n2cube.dpuOpen()
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
    task = n2cube.dpuCreateTask(kernel, 0)

    input_len = n2cube.dpuGetInputTensorSize(task, CONV_INPUT_NODE)
    output_sizes = [
        n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE1),
        n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE2),
        n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE3),
    ]

    timings = []
    processed = 0
    try:
        for image_name, image_path in paths:
            if MAX_IMAGES and processed >= MAX_IMAGES:
                break

            image = cv2.imread(image_path)
            if image is None:
                print("SKIP unreadable image: {}".format(image_path))
                continue

            image_shape = image.shape[:2]
            wall_start = time.perf_counter()

            pre_start = time.perf_counter()
            image_data = pre_process(image, (416, 416))
            pre_ms = (time.perf_counter() - pre_start) * 1000.0

            conv_start = time.perf_counter()
            n2cube.dpuSetInputTensorInHWCFP32(
                task, CONV_INPUT_NODE, image_data, input_len
            )
            n2cube.dpuRunTask(task)
            conv_out1 = n2cube.dpuGetOutputTensorInHWCFP32(
                task, CONV_OUTPUT_NODE1, output_sizes[0]
            )
            conv_out2 = n2cube.dpuGetOutputTensorInHWCFP32(
                task, CONV_OUTPUT_NODE2, output_sizes[1]
            )
            conv_out3 = n2cube.dpuGetOutputTensorInHWCFP32(
                task, CONV_OUTPUT_NODE3, output_sizes[2]
            )
            conv_ms = (time.perf_counter() - conv_start) * 1000.0

            yolo_outputs = [
                np.reshape(conv_out1, (1, 13, 13, object_dim)),
                np.reshape(conv_out2, (1, 26, 26, object_dim)),
                np.reshape(conv_out3, (1, 52, 52, object_dim)),
            ]

            post_start = time.perf_counter()
            boxes, scores, classes = eval_fast(
                yolo_outputs, image_shape, class_names, anchors
            )
            post_ms = (time.perf_counter() - post_start) * 1000.0
            total_ms = (time.perf_counter() - wall_start) * 1000.0
            timings.append((pre_ms, conv_ms, post_ms, total_ms))

            if SHOW or SAVE:
                result = annotate(image, boxes, scores, classes, class_names)
                if SAVE:
                    result_path = os.path.join(OUTPUT_ROOT, image_name)
                    cv2.imwrite(result_path, result)
                if SHOW:
                    cv2.imshow("YOLOv3 ALINX {} optimized".format(OBJECT_TYPE), result)
                    if cv2.waitKey(1) == 27:
                        break

            processed += 1
            print(
                "FRAME {}: detections={} pre={:.2f}ms dpu={:.2f}ms "
                "post={:.2f}ms total={:.2f}ms fps={:.2f}".format(
                    processed,
                    len(boxes),
                    pre_ms,
                    conv_ms,
                    post_ms,
                    total_ms,
                    1000.0 / total_ms if total_ms else 0.0,
                )
            )
    finally:
        if SHOW:
            cv2.destroyAllWindows()
        # These APIs are present in the Xilinx runtime used by this example;
        # guard them so the runner remains usable with older DNN runtimes.
        destroy_task = getattr(n2cube, "dpuDestroyTask", None)
        destroy_kernel = getattr(n2cube, "dpuDestroyKernel", None)
        if destroy_task:
            destroy_task(task)
        if destroy_kernel:
            destroy_kernel(kernel)
        n2cube.dpuClose()

    if timings:
        values = np.asarray(timings, dtype=np.float64)
        mean = values.mean(axis=0)
        print(
            "SUMMARY: frames={} pre={:.2f}ms dpu={:.2f}ms post={:.2f}ms "
            "total={:.2f}ms fps={:.2f}".format(
                len(timings),
                mean[0],
                mean[1],
                mean[2],
                mean[3],
                1000.0 / mean[3] if mean[3] else 0.0,
            )
        )


if __name__ == "__main__":
    main()
