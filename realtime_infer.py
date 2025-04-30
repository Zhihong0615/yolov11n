import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time
import threading
from queue import Queue
from pycuda.compiler import SourceModule

# ----------------- CUDA核函数 -----------------
resize_norm_pad_kernel_code = """
__global__ void resize_pad_norm(
    unsigned char* input, float* output, 
    int iw, int ih, 
    int ow, int oh
){
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    int dz = blockIdx.z * blockDim.z + threadIdx.z; // 0:R,1:G,2:B

    if (dx >= ow || dy >= oh || dz >= 3) return;

    float scale = fminf((float)ow / iw, (float)oh / ih);
    int nw = scale * iw;
    int nh = scale * ih;
    int pad_x = (ow - nw) / 2;
    int pad_y = (oh - nh) / 2;

    int ix = (dx - pad_x) * 1.0f / scale;
    int iy = (dy - pad_y) * 1.0f / scale;

    float pixel = 114.0f / 255.0f; // 默认填充值

    if (ix >= 0 && ix < iw && iy >= 0 && iy < ih) {
        int x0 = floorf(ix);
        int y0 = floorf(iy);
        int x1 = min(x0 + 1, iw - 1);
        int y1 = min(y0 + 1, ih - 1);

        float dx1 = ix - x0;
        float dy1 = iy - y0;
        float dx0 = 1.0f - dx1;
        float dy0 = 1.0f - dy1;

        int idx00 = (y0 * iw + x0) * 3;
        int idx01 = (y0 * iw + x1) * 3;
        int idx10 = (y1 * iw + x0) * 3;
        int idx11 = (y1 * iw + x1) * 3;

        float v00 = input[idx00 + (2 - dz)];
        float v01 = input[idx01 + (2 - dz)];
        float v10 = input[idx10 + (2 - dz)];
        float v11 = input[idx11 + (2 - dz)];

        pixel = (dy0 * (dx0 * v00 + dx1 * v01) + dy1 * (dx0 * v10 + dx1 * v11)) / 255.0f;
    }

    int out_idx = (dz * oh + dy) * ow + dx;
    output[out_idx] = pixel;
}
"""
mod = SourceModule(resize_norm_pad_kernel_code)
resize_pad_norm_kernel = mod.get_function("resize_pad_norm")

# ----------------- 预处理 -----------------
def cuda_preprocess(img_np, input_gpu_ptr, ow=640, oh=384):
    ih, iw, _ = img_np.shape
    input_cpu = img_np.astype(np.uint8)
    d_image = cuda.mem_alloc(input_cpu.nbytes)
    cuda.memcpy_htod(d_image, input_cpu)

    block = (16, 16, 1)
    grid = ((ow + block[0] - 1)//block[0], (oh + block[1] - 1)//block[1], 3)

    resize_pad_norm_kernel(
        d_image, input_gpu_ptr,
        np.int32(iw), np.int32(ih),
        np.int32(ow), np.int32(oh),
        block=block, grid=grid
    )
    d_image.free()

# ----------------- 后处理 -----------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_iou(box, boxes):
    inter_left_top = np.maximum(box[:2], boxes[:, :2])
    inter_right_bottom = np.minimum(box[2:4], boxes[:, 2:4])
    inter_wh = np.clip(inter_right_bottom - inter_left_top, a_min=0, a_max=np.inf)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]
    box_area = (box[2]-box[0]) * (box[3]-box[1])
    boxes_area = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    union_area = box_area + boxes_area - inter_area
    return inter_area / (union_area + 1e-6)

def postprocess(pred, raw_w, raw_h, dst_width=640, dst_height=384, conf_thres=0.6, iou_thres=0.45):
    pred = pred[0]
    boxes = pred[:, :4]
    cls_logits = pred[:, 4:]
    cls_scores = sigmoid(cls_logits)

    conf = cls_scores.max(axis=1)
    labels = cls_scores.argmax(axis=1)

    keep = conf > conf_thres
    boxes = boxes[keep]
    conf = conf[keep]
    labels = labels[keep]

    if boxes.shape[0] == 0:
        return []

    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    scale = min(dst_width / raw_w, dst_height / raw_h)
    pad_x = (dst_width - raw_w * scale) / 2
    pad_y = (dst_height - raw_h * scale) / 2

    xyxy[:, [0,2]] = (xyxy[:, [0,2]] - pad_x) / scale
    xyxy[:, [1,3]] = (xyxy[:, [1,3]] - pad_y) / scale

    detections = np.concatenate([xyxy, conf[:, None], labels[:, None]], axis=1)
    detections = detections[detections[:, 4].argsort()[::-1]]

    boxes = detections.copy()
    keep = []
    while boxes.shape[0]:
        keep.append(boxes[0])
        if len(boxes) == 1:
            break
        ious = compute_iou(boxes[0], boxes[1:])
        boxes = boxes[1:][ious < iou_thres]

    return keep

# ----------------- 主流程 -----------------
def infer_stream(engine_path, video_source=0, save_path=None, target_fps=None):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    cap = cv2.VideoCapture(video_source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    save_writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    d_input = cuda.mem_alloc(1 * 3 * 384 * 640 * 4)
    for binding in engine:
        if engine.get_tensor_mode(binding) != trt.TensorIOMode.INPUT:
            output_shape = engine.get_tensor_shape(binding)
    output_data = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output_data.nbytes)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    names = ['Car', 'Cyclist', 'Pedestrian', 'Tram', 'Tricycle', 'Truck']
    color_map = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

    print("Start streaming...")
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        cuda_preprocess(frame, d_input)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_data, d_output, stream)
        stream.synchronize()
        end = time.time()

        detections = postprocess(output_data.transpose(0,2,1), raw_w=w, raw_h=h)

        for det in detections:
            left, top, right, bottom, score, label = det
            label = int(label)
            color = color_map[label % len(color_map)]
            caption = f"{names[label]} {score:.2f}"
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color, 2)
            cv2.putText(frame, caption, (int(left), int(top)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if save_writer:
            save_writer.write(frame)

        cv2.imshow("TRT RealTime Inference", frame)
        
        # 限帧处理
        if target_fps:
            elapsed = end - start
            delay = max(0, (1.0/target_fps) - elapsed)
            time.sleep(delay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    if save_writer:
        save_writer.release()
    cv2.destroyAllWindows()
    print(f"Total frames: {frame_id}")

# ----------------- 启动入口 -----------------
if __name__ == "__main__":
    engine_path = '/home/zhihong/yolov11n/outputs/train1/road_obstacle_yolov11/weights/best.engine'
    # 可以是文件路径，也可以是数字0（摄像头）
    video_source = '/home/zhihong/yolov11n/dataset/camera_videos/88.mp4'
    save_path = '/home/zhihong/yolov11n/outputs/train1/infer_videos/88_infered_realtime.mp4'
    infer_stream(engine_path, video_source=video_source, save_path=save_path, target_fps=100)
