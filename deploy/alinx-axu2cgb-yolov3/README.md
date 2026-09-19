# ALINX AXU2CGB / DPU YOLOv3

这是根目录 YOLOv11n 软件项目的板端延续工程。根目录保留原来的
PyTorch/ONNX/TensorRT 软件流程；本目录记录在 ALINX AXU2CGB 上已经复现并验证的
DNNDK/DPU YOLOv3 office 检测流程。

> 注意：当前板端模型是 YOLOv3 office 模型，并不是把根目录的 YOLOv11n 权重直接
> 放到了 DPU 上。要在该板上部署 YOLOv11n，需要另外完成算子兼容性检查、量化和
> Vitis AI 编译。

## 目标平台

- Board: ALINX AXU2CGB
- SoC: `xczu2cg-sfvc784-1-i`，AArch64，2 个 Cortex-A53
- Board Linux: `5.4.0-xilinx-v2020.1`
- DPU: `DPUCZDX8G_1`，1 个计算单元
- 当前 xclbin: kernel clock 300 MHz，data clock 150 MHz
- 生成/运行环境: Vitis/Vivado 2020.1 兼容平台，DNNDK/n2cube 运行时

## 目录内容

- `tf_yolov3_voc_pic.py`：原始单线程脚本
- `tf_yolov3_voc_pic_optimized.py`：低延迟优化版，默认关闭显示和逐帧写盘
- `tf_yolov3_voc_pipelined.py`：CPU 预处理/后处理与下一次 DPU 调用重叠的吞吐版
- `libdpumodeltf_yolov3_office.so`：与当前 DPU 模型匹配的 AArch64 模型库
- `modellib/`、`model_data/`：后处理工具、类别和 anchor 配置
- `test_sample/t1/`：用于复现测试的样例图像
- `dpu_current.xclbin`：当前板端工作的 DPU xclbin 备份
- `PERFORMANCE.md`：实测结果和分析

## 板端运行

将本目录复制到开发板后，在板端项目目录执行：

```sh
cd /home/root/Alinx_DNN/tf_yolov3_office_deploy
YOLO_MAX_IMAGES=30 python3 -u tf_yolov3_voc_pic_optimized.py
```

需要吞吐优先时运行流水线版：

```sh
YOLO_MAX_IMAGES=30 python3 -u tf_yolov3_voc_pipelined.py
```

如需窗口显示或保存检测结果：

```sh
YOLO_SHOW=1 YOLO_SAVE=1 YOLO_MAX_IMAGES=10 python3 -u tf_yolov3_voc_pic_optimized.py
```

运行环境需要板端已有匹配的 `n2cube`、OpenCV、NumPy、Python 3 和 DPU 驱动；
模型库必须与 xclbin/DPU 架构匹配。`dpu_current.xclbin` 可作为当前工作的备份，
通常应放到板端 `/media/sd-mmcblk1p1/dpu.xclbin`。

## 当前实测性能

真实墙钟计时、关闭窗口和逐帧写盘时：

| 版本 | 帧数 | 吞吐 |
| --- | ---: | ---: |
| 原始脚本 | 10 | 2.55 FPS |
| 单线程优化版 | 10 | 3.03 FPS |
| DPU/CPU 流水线版 | 10 | 3.87 FPS |
| DPU/CPU 流水线版 | 30 | 4.15 FPS |

前 5 帧的原始后处理与优化后处理逐框结果一致。测试样例中存在少量 JPEG 损坏
警告，不影响已经完成的性能复现，但正式数据集建议先重新编码。

当前 `dpuRunTask` 的墙钟时间约为 220--224 ms，说明现有一个 DPU 计算单元是主要
瓶颈。增加 DPU 单元、修改 DPU 架构或提高时钟，需要用匹配的 2020.1 平台重新生成
`.xclbin`，并重新编译模型库；不能只替换其中一个文件。

## 公开内容与安全边界

本目录只包含可复现工程、模型运行库、xclbin 备份和测试样例；本机 SSH 私钥、串口
脚本、临时网络配置、SD 卡救援镜像和缓存文件均未提交。

## License

本仓库中由项目作者编写的源代码采用 MIT License。AXU2CGB 平台文件、DPU 运行时、
生成的模型库、xclbin 以及测试数据可能包含厂商或其他上游内容，使用时仍需遵循其
各自的许可和分发条件。
