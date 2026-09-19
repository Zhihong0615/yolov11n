# YOLOv3 性能优化记录

测试平台：ALINX AXU2CGB，Linux 5.4.0-xilinx-v2020.1，AArch64，
`DPUCZDX8G_1`，当前 xclbin 的内核时钟为 300 MHz、数据时钟为 150 MHz。

## 实测结果

测试使用板端 `test_sample/t1`，真实墙钟计时，不使用原脚本的
`time.process_time()`。

| 版本 | 帧数 | 平均吞吐 |
| --- | ---: | ---: |
| 原始脚本（关闭显示和写盘） | 10 | 2.55 FPS |
| 单线程优化版 | 10 | 3.03 FPS |
| DPU/CPU 流水线版 | 10 | 3.87 FPS |
| DPU/CPU 流水线版 | 30 | 4.15 FPS |

前 5 帧的原始后处理与优化后处理逐框比较结果完全一致。样例图像中仍有
JPEG 损坏警告，建议正式测试前重新编码这些样本。

## 运行

单线程、低延迟模式：

```sh
cd /home/root/Alinx_DNN/tf_yolov3_office_deploy
YOLO_MAX_IMAGES=30 python3 -u tf_yolov3_voc_pic_optimized.py
```

吞吐模式：

```sh
cd /home/root/Alinx_DNN/tf_yolov3_office_deploy
YOLO_MAX_IMAGES=30 python3 -u tf_yolov3_voc_pipelined.py
```

默认关闭窗口和逐帧 JPEG 写盘；需要输出图片时可设置 `YOLO_SAVE=1`，需要
窗口显示时可设置 `YOLO_SHOW=1`。

## DPU 重配置状态

当前板端 xclbin 已匹配 AXU2CGB 平台，并使用一个 DPU 计算单元。增加计算单元、
改变 DPU 架构或提高时钟都需要使用匹配的 Vivado/Vitis 2020.1 平台重新生成
`.xclbin`，并重新编译 DPU 模型；不能只替换硬件文件而继续使用旧模型。

本机暂未安装 Vivado/Vitis，因此本轮没有刷写新的 bitstream，当前可工作的 DPU
配置保持不变。
