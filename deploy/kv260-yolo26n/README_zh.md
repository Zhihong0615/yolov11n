# KV260 上的 YOLO26n 衍生模型推理源码

本目录整理了曾在 KV260 上运行的完整 C++ 推理路径：ARM 端 BGR 预处理、VART 调度、DPU 执行、INT8 检测头解码和结果可视化，并提供板端编译、单图启动和长测脚本。

硬件 overlay、DPU AXI/时钟配置、HLS 原型源码及其验证边界见[硬件复现指南](../../hardware/kv260-yolo26n/README_zh.md)。HLS 原型没有集成到本指南所述的 DPU-only 性能测试 bitstream。

## 兼容性、模型文件与当前验证状态

本目录虽放在 `yolov11n` 仓库中，但它运行的是 **YOLO26n 衍生 ReLU-flow 六类模型**，不是 `models/yolo11n.pt`，也不是 YOLOv11n TensorRT 模型。runner 要求输入 `[1,640,640,3]` INT8，以及 P3/P4/P5 六个输出张量（每尺度 4 个框通道、6 个分类通道）。YOLOv11n、YOLOv8n、YOLOv3 或其他 xmodel 均不能直接使用。

仓库不包含训练数据、模型权重、xmodel 或 Vitis AI 运行时二进制。精确复用历史模型时，xmodel 应与 [`model_contract.json`](model_contract.json) 中的 SHA-256 一致；重新训练/量化会生成不同 SHA，需重新检查输出张量、精度和数值一致性。若要改为 YOLOv11n，需要先将 YOLOv11n 训练/导出并编译为 DPU xmodel，再按其真实 XIR 输出重写和验证解码器；本项目没有声称这项移植已经完成。

本次开发环境没有连接 KV260，因此没有在本次分支上重新编译或运行板卡。这里记录的 C++ 源码与历史 325/650 MHz 长测 at-run source hash 一致，性能数据来自先前保存的板端实验记录。更换板卡、xmodel、overlay 或 runtime 后需重新验证。

## 目录内容

| 路径 | 用途 |
|---|---|
| `src/bench_cpp_dual_runner_pipeline.cpp` | ARM 预处理、VART/DPU 执行和 ARM 后处理三段软件流水 |
| `src/fast_letterbox.cpp` | 640×640 letterbox、BGR→RGB、INT8 写入；AArch64 使用 NEON |
| `src/fast_decode_six.cpp` | 六个 INT8 输出张量的 TopK、LTRB/stride 解码及坐标还原 |
| `scripts/build_on_board.sh` | KV260 原生编译 |
| `scripts/run_single_image.sh` | 单图推理并保存框图 |
| `scripts/run_long_test.sh` | 5×6,000 帧基本长测 |
| `scripts/render_result.py` | 结果二进制解析及 PNG 渲染 |
| `tests/test_result_binary.py` | 结果二进制格式和损坏输入测试 |
| `training/` | 派生网络训练入口、模型 YAML、VAI bridge/量化源码和 B4096 编译架构描述 |

三份 C++ 源码 SHA-256 与原板端长测 at-run 身份相同：runner `9501460e3c4fdca5f5e5d07dbfefc5eee0b4c8f1a270d4a68546d7efd9993df2`；letterbox `e3002d13a40bc4ccb5ad2a3356b98be29df32755a45eb0e4226d5129e48e235a`；decoder `755d14f6b2dda28eb109b31c31736e03881bc1532e582e4d298dba1f86b58542`。源码适用仓库根目录 MIT License；VART、XIR 和 OpenCV 等依赖须按各自许可安装。

## 已有板端结果

| 指标 | 记录值 |
|---|---:|
| 全路径平均吞吐率 | 91.5916 FPS |
| 全路径延迟 p50 / p95 / p99 | 24.4077 / 24.7195 / 25.9464 ms |
| 5 轮处理帧数 | 30,000，零丢帧、零错误 |
| U14 电源轨平均功耗 | 9.2707 W |
| U14 电源轨能量 | 0.10122 J/有效帧 |
| mAP50 / mAP50–95 | 54.4211% / 35.5115%，995 张验证图 |

计时从已解码 BGR 帧交给板端程序开始，到板端输出最终检测框结束；不含相机采集和 JPEG 解码。能量数据来自 INA260 对 U14 电源轨的测量，不是整板墙插能耗。原版 YOLO26n FP32 在同一 KV260 CPU 路径为 1.7377 FPS；52.71×是跨模型结构和数值精度的系统吞吐率比，不是同模型硬件加速比。最终 INT8 精度没有达到原工程设定的质量门槛。

## 从训练到板端运行

本节命令复现模型生成链路。训练数据、原始 YOLO26n 六类 checkpoint 和校准图像不在仓库中；不能用本仓库的 `models/yolo11n.pt` 代替 YOLO26n checkpoint。

### A. 训练 YOLO26n ReLU-flow 派生模型

历史训练使用 PyTorch 2.7.0+cu128、Ultralytics 8.4.174 和 CUDA GPU。准备 YOLO 格式数据 YAML（640 输入、六类且顺序为 `Car, Cyclist, Pedestrian, Tram, Tricycle, Truck`）、六类 YOLO26n checkpoint，以及固定 train/val split。先计算 split 清单的 SHA-256：

```bash
sha256sum /path/to/split_manifest.csv
```

在仓库根目录运行：

```bash
python3 deploy/kv260-yolo26n/training/train_relu_flow.py \
  --data /path/to/road_data.yaml \
  --yaml deploy/kv260-yolo26n/training/yolo26n-kv260-relu-flow.yaml \
  --checkpoint /path/to/yolo26n_six_class_best.pt \
  --project /path/to/training_runs \
  --name yolo26n_kv260_relu_flow \
  --epochs 100 --batch 16 --workers 8 --patience 20 --seed 42 \
  --split-sha256 YOUR_SPLIT_MANIFEST_SHA256
```

训练得到 `training_runs/yolo26n_kv260_relu_flow/weights/best.pt`。模型结构相对原版使用 ReLU、简化 PSA/C2PSA，并保留 one-to-one 三尺度头；因此这是 YOLO26n 衍生网络，不是只改变激活函数的原版模型。要复现本项目验证集指标，还必须使用原始数据和原始 split；其他数据或新 checkpoint 的质量数字不可沿用。

### B. 导出 DPU bridge 并检查浮点数值

用真实训练图片作为样本，导出量化所需的 DPU bridge 权重。脚本会比较 bridge 与训练图的原始检测头输出，不一致时会停止：

```bash
python3 deploy/kv260-yolo26n/training/relu_flow/export_bridge.py \
  --checkpoint /path/to/training_runs/yolo26n_kv260_relu_flow/weights/best.pt \
  --yaml deploy/kv260-yolo26n/training/yolo26n-kv260-relu-flow.yaml \
  --image /path/to/train_sample.jpg \
  --out /path/to/bridge
```

成功后生成 `relu_flow_bridge_weights.npz` 和 `bridge_manifest.json`。manifest 保存 checkpoint、配置、权重和 parity 的哈希。继续之前先确认 `raw_float_parity` 通过。

### C. 使用 512 张训练集图做 Vitis AI 2.5 PTQ

量化需要 Vitis AI 2.5 PyTorch/NNDCT 环境。`train.txt` 是训练 split 清单，`calibration_512.txt` 是其中唯一的 512 张校准图片；量化器会检查校准图片确实属于训练 split。路径使用可读的本机绝对路径：

```bash
python3 deploy/kv260-yolo26n/training/relu_flow/quantize.py \
  --mode calib --architecture relu_flow_025 --output-mode six_clip8 \
  --out /path/to/quantized \
  --weights /path/to/bridge/relu_flow_bridge_weights.npz \
  --calibration-list /path/to/calibration_512.txt \
  --train-list /path/to/train.txt --limit 512

python3 deploy/kv260-yolo26n/training/relu_flow/quantize.py \
  --mode test --architecture relu_flow_025 --output-mode six_clip8 \
  --out /path/to/quantized \
  --weights /path/to/bridge/relu_flow_bridge_weights.npz \
  --calibration-list /path/to/calibration_512.txt \
  --train-list /path/to/train.txt --limit 512 \
  --test-image /path/to/train_sample.jpg
```

第二条命令生成 `DpuFlowProbe_int.xmodel` 和量化探针记录。校准只用训练集图像；不要把 validation/test 图片放进校准清单。

### D. 编译成 KV260 B4096 xmodel

在装有 Vitis AI 2.5 compiler 的环境中运行（若在容器内，先把仓库和输出目录挂载进去）：

```bash
vai_c_xir.py \
  -x /path/to/quantized/DpuFlowProbe_int.xmodel \
  -a deploy/kv260-yolo26n/training/kv260-b4096-observed.arch.json \
  -o /path/to/compiled \
  -n yolo26n_relu_flow_int8
```

确认生成的 compiled xmodel，记录 SHA-256，并在支持的 KV260 DPU overlay 上检查 XIR 输入输出合同。当前 C++ runner 要求一个 `[1,640,640,3]` 输入及六个 P3/P4/P5 INT8 原始输出。若 Vitis AI 版本、架构描述或模型改变后不符合，停止并更新 decoder，不能把旧模型合同硬套上去。

### E. 检查板卡、复制模型、编译和推理

完成模型制作后，按下面步骤在 KV260 上运行。

## 板端运行步骤

### 1. 检查板卡软件

目标环境是 KV260/Kria K26，Ubuntu 22.04 aarch64、GCC 11.4.0、OpenCV 4.5.4、VART 2.5.0、XIR ABI 2，以及与 xmodel 相容的 DPUCZDX8G ISA1 B4096。历史板端 fingerprint 为 `0x0101000016010407`。

```bash
uname -m
g++ --version
pkg-config --modversion opencv4
vart_version
show_dpu
```

确认 `show_dpu` 能找到 B4096 DPU CU，且板端具备 VART 开发头文件和链接库。若系统只有 VART runtime 而没有 `/usr/include/vart/runner.hpp`，需按当前镜像的 AMD/Xilinx 文档安装匹配的开发包。不要用主机 x86 库替换板端 aarch64 库。

### 2. 准备 xmodel 与输入图片

在本机把匹配 xmodel 和道路图片复制到板端：

```bash
mkdir -p ~/road_demo/images
scp /本机路径/final_six_clip8_train512.xmodel ubuntu@板卡地址:~/road_demo/model.xmodel
scp /本机路径/道路图片.jpg ubuntu@板卡地址:~/road_demo/images/0001.jpg
```

在板端创建图片清单，每行一个绝对路径：

```bash
printf '%s\n' "$HOME/road_demo/images/0001.jpg" > ~/road_demo/images.txt
```

多图例子见 [`examples/images.txt.example`](examples/images.txt.example)。清单中的图像会在计时开始前全部解码进内存，并按顺序循环；JPEG 文件读取时间不在全路径计时边界内。

### 3. 复制源码并在板上编译

从本仓库根目录运行：

```bash
scp -r deploy/kv260-yolo26n ubuntu@板卡地址:~/kv260-yolo26n
ssh ubuntu@板卡地址
cd ~/kv260-yolo26n
bash scripts/build_on_board.sh
```

产物是 `build/kv260_yolo26n_runner`。编译脚本调用板端原生 `g++`，从 `pkg-config opencv4` 读取 OpenCV 参数，并链接 `vart-runner`、`vart-util` 和 `xir`；脚本不会修改 overlay。

常见编译问题：

- `opencv4` 找不到：安装该镜像对应 OpenCV 开发包，确认 `pkg-config --modversion opencv4` 成功。
- 找不到 VART 头文件或 `libvart-runner`：核对 Vitis AI runtime/development package 版本和 aarch64 库路径。
- 链接报 `undefined reference`：检查 `ldconfig -p | grep -E 'vart|xir'`，不要混用不同 ABI 的 VART/XIR。

### 4. 单图推理并查看效果

```bash
cd ~/kv260-yolo26n
bash scripts/run_single_image.sh \
  ~/road_demo/model.xmodel \
  ~/road_demo/images/0001.jpg \
  ~/road_demo/output
```

脚本预热 3 帧、计时 1 帧、使用 2 个 OpenCV 线程。输出目录包括输入清单、`result.bin` 检测结果、`result.bin.json` 摘要、`result.bin.trace.csv` 逐帧记录、`console.json` 和 `preview.png`。

在板端桌面打开图片：

```bash
xdg-open ~/road_demo/output/preview.png
```

如果 SSH 会话没有图形桌面，在本机另开一个终端：

```bash
scp ubuntu@板卡地址:~/road_demo/output/preview.png .
```

结果二进制按帧连接写入：每帧先写一个 little-endian `uint32 detection_count`，再写 `count × 6` 个 little-endian `float32`，字段顺序为 `x1,y1,x2,y2,score,class_id`。单帧 FPS 和 p95 不具有统计意义，不应作为性能结果。

### 5. 多帧持续测试

准备一组足够多且路径有效的输入图片后运行：

```bash
bash scripts/run_long_test.sh \
  ~/road_demo/model.xmodel \
  ~/road_demo/images.txt \
  ~/road_demo/long_run \
  5
```

每轮执行 200 帧预热和 6,000 帧测量，默认共 5 轮；每轮单独保存结果二进制、JSON 和逐帧 trace。它是便于复跑的基础测试脚本，不会自动完成参考输出逐框比较、INA260 能耗积分、温度监测或长期 trace 审计。要发表新的性能数据，需另外保存模型/源码/overlay 哈希及传感器记录，并核对帧数、顺序、丢帧和错误。

短测可直接调用 runner：

```bash
build/kv260_yolo26n_runner \
  ~/road_demo/model.xmodel \
  ~/road_demo/images.txt \
  ~/road_demo/short.bin \
  20 256 2
```

参数顺序是 `MODEL.xmodel IMAGE_LIST OUTPUT.bin WARMUP_FRAMES MEASURED_FRAMES OPENCV_THREADS`。不得通过跳帧、缩小输入或主机 GPU 推理来抬高 KV260 FPS。

## 数据流与张量合同

1. OpenCV 读取已解码 BGR 图像，按居中 letterbox、线性插值和 `(114,114,114)` 填充成 640×640。
2. ARM 端将 BGR 转为 RGB，归一化并按 fix-point=6 量化成 NHWC INT8 `[1,640,640,3]`。
3. VART 将输入提交给单个 DPU CU。两个 VART Runner 是软件提交上下文，共用同一物理 DPU，不代表两颗 DPU。
4. DPU 输出 P3/P4/P5 六个 INT8 原始张量；ARM 端执行分数变换、TopK、LTRB/stride 解码、阈值过滤、反 letterbox 和坐标裁剪。
5. one-to-one 输出不执行 NMS，最多保留 300 个检测框。神经网络在 DPU 上计算，预处理、调度和最终解码运行在 PS/ARM。

这是跨帧软件流水，不代表 PL 内有逐层自制卷积硬件流水。缓冲区是板端应用可访问的内存，不能据此称为 BRAM。

程序启动时会校验一输入、六输出、输入形状和输出通道。如果报 `Expected one 640x640 INT8 input and six outputs`、`Unexpected DPU output shape` 或 `Missing class/box output scale`，说明给错了 xmodel，不要靠改阈值绕过。

## 与原工程的关系

更完整的训练精度、量化损失、性能测试口径和硬件结构说明，保存在原工程的 `reports/KV260_YOLO26n_FINAL_HANDOVER.md`、`reports/full_dpu_final_ptq512/README.md` 和 `experiments/relu_final_ptq512_325_formal/`。这些文件不随本目录自动同步；引用 FPS 时要同时说明 mAP 和当前质量门槛未通过的事实。
