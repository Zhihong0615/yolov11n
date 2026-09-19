"""Single-DPU pipelined runner.

The DPU remains serialized, but CPU preprocessing/postprocessing is overlapped
with the next DPU invocation.  This is a throughput mode; the original and
optimized single-frame runners remain available for latency measurements.
"""

import os
import queue
import threading
import time

import cv2
import numpy as np
from dnndk import n2cube

from tf_yolov3_voc_pic_optimized import *  # noqa: F401,F403


class _Result:
    def __init__(self):
        self._event = threading.Event()
        self._value = None
        self._error = None

    def set_result(self, value):
        self._value = value
        self._event.set()

    def set_error(self, error):
        self._error = error
        self._event.set()

    def result(self):
        self._event.wait()
        if self._error is not None:
            raise self._error
        return self._value


class _DpuExecutor:
    """Small Python-3.7-compatible single worker executor."""
    def __init__(self):
        self._jobs = queue.Queue()
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def _run(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            function, args, result = job
            try:
                result.set_result(function(*args))
            except Exception as error:
                result.set_error(error)

    def submit(self, function, *args):
        result = _Result()
        self._jobs.put((function, args, result))
        return result

    def shutdown(self):
        self._jobs.put(None)
        self._thread.join()


def _dpu_setup():
    class_names = get_class(CLASSES_PATH)
    object_dim = 3 * (5 + len(class_names))
    n2cube.dpuOpen()
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
    task = n2cube.dpuCreateTask(kernel, 0)
    input_len = n2cube.dpuGetInputTensorSize(task, CONV_INPUT_NODE)
    output_nodes = [CONV_OUTPUT_NODE1, CONV_OUTPUT_NODE2, CONV_OUTPUT_NODE3]
    output_sizes = [
        n2cube.dpuGetOutputTensorSize(task, node) for node in output_nodes
    ]
    shapes = [
        (1, 13, 13, object_dim),
        (1, 26, 26, object_dim),
        (1, 52, 52, object_dim),
    ]
    return class_names, task, input_len, output_nodes, output_sizes, shapes


def _dpu_infer(state, image_data):
    class_names, task, input_len, output_nodes, output_sizes, shapes = state
    start = time.perf_counter()
    n2cube.dpuSetInputTensorInHWCFP32(
        task, CONV_INPUT_NODE, image_data, input_len
    )
    n2cube.dpuRunTask(task)
    outputs = []
    for node, size, shape in zip(output_nodes, output_sizes, shapes):
        raw = n2cube.dpuGetOutputTensorInHWCFP32(task, node, size)
        outputs.append(np.asarray(raw, dtype=np.float32).reshape(shape).copy())
    return outputs, (time.perf_counter() - start) * 1000.0


def _dpu_close(state):
    n2cube.dpuClose()


def _next_item(paths):
    try:
        image_name, image_path = next(paths)
    except StopIteration:
        return None
    image = cv2.imread(image_path)
    if image is None:
        return _next_item(paths)
    start = time.perf_counter()
    data = pre_process(image, (416, 416))
    return image_name, image, data, (time.perf_counter() - start) * 1000.0


def main():
    max_images = int(os.environ.get("YOLO_MAX_IMAGES", "0"))
    paths = iter(image_paths())
    current = _next_item(paths)
    if current is None:
        print("No input images")
        return

    executor = _DpuExecutor()
    setup_future = executor.submit(_dpu_setup)
    state = setup_future.result()
    anchors = get_anchors(ANCHORS_PATH)
    future = executor.submit(_dpu_infer, state, current[2])
    total_start = time.perf_counter()
    processed = 0
    try:
        while current is not None:
            if max_images and processed >= max_images:
                break

            # Prepare the next frame while the DPU is busy with current.
            next_item = _next_item(paths)
            raw_future = future
            next_future = None
            if next_item is not None:
                next_future = executor.submit(_dpu_infer, state, next_item[2])

            outputs, dpu_ms = raw_future.result()
            post_start = time.perf_counter()
            boxes, scores, classes = eval_fast(
                outputs, current[1].shape[:2], state[0], anchors
            )
            post_ms = (time.perf_counter() - post_start) * 1000.0
            processed += 1
            print(
                "PIPE_FRAME {}: detections={} pre={:.2f}ms dpu={:.2f}ms "
                "post={:.2f}ms".format(
                    processed, len(boxes), current[3], dpu_ms, post_ms
                )
            )

            current = next_item
            future = next_future
    finally:
        executor.submit(_dpu_close, state).result()
        executor.shutdown()

    elapsed = time.perf_counter() - total_start
    print(
        "PIPE_SUMMARY: frames={} wall={:.2f}ms fps={:.2f}".format(
            processed, elapsed * 1000.0, processed / elapsed if elapsed else 0.0
        )
    )


if __name__ == "__main__":
    main()
