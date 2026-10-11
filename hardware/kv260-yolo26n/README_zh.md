# KV260 YOLO26n 硬件文件与复现指南

本目录补上了仓库上一版遗漏的硬件材料：325/650 MHz DPU overlay 文件、AXI/时钟配置，以及两组可独立综合或仿真的 HLS 原型。

## 先区分板端实测设计与 HLS 原型

| 部分 | 在历史板端性能测试中实际使用 | 本目录提供的内容 |
| --- | --- | --- |
| KV260 可编程逻辑 | 单个 AMD/Xilinx DPUCZDX8G B4096 CU；DPU 时钟配置为 `aclk=325 MHz`、`ap_clk_2=650 MHz` | 可加载的 325/650 MHz `.bit.bin`、DTBO、XRT 元数据 xclbin，以及生成该配置所用的 Vitis 时钟 hook 和链接配置 |
| ARM/PS | 图像预处理、VART 调度、检测头解码和结果输出 | 见 [`../../deploy/kv260-yolo26n/`](../../deploy/kv260-yolo26n/) 的 C++ 源码和板端操作手册 |
| HLS 单卷积原型 | **没有**接入性能测试的 DPU overlay | 一个 YOLO26n 卷积层的 INT8、分块、数据流 HLS 设计及 C/RTL 验证证据 |
| HLS 注意力原型 | **没有**接入性能测试的 DPU overlay | PSA 注意力的 FP32 HLS 变体、C 仿真/综合脚本和资源报告；属于独立研究原型 |

历史性能数字只属于已记录的 DPU 部署和对应 ReLU-flow INT8 xmodel，不是 HLS 原型的性能。HLS 源码没有被包装进该 xclbin，也没有产生可在板上运行的完整 YOLO 检测器 bitstream。

## 实际运行的 PL 与 AXI 结构

部署架构是 PS + DDR + 一个 DPU CU。DPUCZDX8G 内部 PE、卷积阵列和指令调度属于 AMD/Xilinx 加密 RTL IP；本仓库不含其可编辑 RTL 源码。板上数据路径为：

```text
ARM/OpenCV
   │ 输入张量写入 DDR，VART 提交任务
   ▼
单个 DPUCZDX8G B4096 ── AXI 主端口 ── DDR 中的指令、权重和特征张量
   │
   ▼
ARM/VART 取回输出张量并解码检测结果
```

该平台的连接配置如下。它描述 DPU 的 AXI 连接，**不表示**另有一个 PL AXI DMA IP：

| DPU 端口 | 方向/用途 | 平台连接 | 实际数据 |
| --- | --- | --- | --- |
| `S_AXI_CONTROL` | AXI4-Lite 从端控制 | accelerator control interconnect；基址 `0xA0010000` | DPU 控制寄存器 |
| `M_AXI_GP0` | DPU AXI 主端口 | PS `HPC1` | 指令和 profiling 地址等访问 |
| `M_AXI_HP0` | 128-bit DPU AXI 主端口 | PS `HP1` | DPU 张量/特征数据访问 |
| `M_AXI_HP2` | 128-bit DPU AXI 主端口 | PS `HP3` | DPU 张量/特征数据访问 |

`HPC1` 为带一致性支持的 PS-PL 端口；`HP1/HP3` 为高性能 DDR 访问端口。实际运行时张量由 VART `TensorBuffer` 管理，DPU 通过这些 AXI 主端口读写 DDR。应用侧有双 Runner 的软件流水，但物理上仍是一个 DPU CU。

325/650 MHz 是配置及实现中的时钟域：`DPUCZDX8G_1.aclk` 为 325 MHz，`DPUCZDX8G_1.ap_clk_2` 为 650 MHz，平台其他时钟仍独立运行。时钟改接由 [`source/dpu_clock_325_650_post_overlay.tcl`](platform/dpu325-b4096/source/dpu_clock_325_650_post_overlay.tcl) 完成，链接配置见 [`source/dpu_clock_325_650_dpu_only.cfg`](platform/dpu325-b4096/source/dpu_clock_325_650_dpu_only.cfg)。历史实现报告记录了 setup/hold 路由结果；最小 slack 约为 0.000 ns，因此它说明该特定构建满足工具报告的门槛，不代表有充足的时序裕量。复核摘要见 [325 MHz DPU 实现说明](platform/dpu325-b4096/IMPLEMENTATION.md)。

## 文件清单

```text
hardware/kv260-yolo26n/
├── README_zh.md
├── platform/dpu325-b4096/       # 历史 325/650 MHz DPU-only overlay 包及构建配置
└── hls/
    ├── pure_pl_stream/          # 单个 INT8 3×3 卷积层；有 C/RTL 仿真证据
    └── attention/               # FP32 PSA 注意力 HLS 原型；无完整板端实现
```

### DPU overlay 包

[`platform/dpu325-b4096/`](platform/dpu325-b4096/) 含板端加载所需的 `.bit.bin`、DTBO、XRT 元数据 xclbin、`shell.json`、Device Tree 源文件、时钟 hook 及校验和。可用下面的步骤先校验：

```bash
cd hardware/kv260-yolo26n/platform/dpu325-b4096
sha256sum -c SHA256SUMS
```

其中 `.bit`、`.bit.bin`、`.xclbin` 是受 [`LICENSE-BINARIES`](platform/dpu325-b4096/LICENSE-BINARIES) 约束的 AMD/Xilinx 二进制文件，不属于仓库根目录 MIT 源码许可。DPU RTL 本身没有随这些二进制开放。

#### 将已构建的 overlay 安装到 KV260

只有在确认板卡型号、XRT/VART 版本与 DPU 指纹兼容后才执行。当前会话没有连接板卡，所以此处是可复现的历史部署流程，并非本次新测结果。

1. 在本机将 package 目录复制到板端暂存区：

   ```bash
   scp -r hardware/kv260-yolo26n/platform/dpu325-b4096 \
     ubuntu@<KV260_IP>:~/dpu325-b4096
   ```

2. 登录板端，重新校验文件并安装运行时文件：

   ```bash
   cd ~/dpu325-b4096
   sha256sum -c SHA256SUMS
   sudo install -d /lib/firmware/xilinx/dpu325-b4096
   sudo install -m 0644 dpu325-b4096.bit.bin dpu325-b4096.dtbo \
     dpu325-b4096.xclbin shell.json /lib/firmware/xilinx/dpu325-b4096/
   sudo xmutil listapps
   ```

3. 保留当前 VART 配置以备回退，再切换到新 app：

   ```bash
   cp /etc/vart.conf ~/vart.conf.before-dpu325
   sudo xmutil unloadapp
   sudo xmutil loadapp dpu325-b4096
   printf 'firmware: %s\n' \
     /lib/firmware/xilinx/dpu325-b4096/dpu325-b4096.xclbin \
     | sudo tee /etc/vart.conf
   sudo xbutil --batch program -d 0000:00:00.0 \
     -u /lib/firmware/xilinx/dpu325-b4096/dpu325-b4096.xclbin
   xbutil --batch examine -d 0000:00:00.0 -r dynamic-regions
   show_dpu
   ```

   历史包的 XRT UUID 为 `E4A88BD4-2D7D-14C8-2F40-567A0F32A66D`，DPU CU 基址为 `0xA0010000`，`show_dpu` 指纹为 `0x0101000016010407`。实测 `show_dpu` 应以运行推理的 `ubuntu` 用户执行，避免本机 VART lock 权限问题。

4. 按 [`deploy/kv260-yolo26n/README_zh.md`](../../deploy/kv260-yolo26n/README_zh.md) 编译 C++ runner，使用与 `model_contract.json` SHA 一致的 xmodel 运行单图测试。只有检查输入输出合同、正确性和温度后再做长测。

5. 回退到板上原有 AMD `benchmark-b4096` app：

   ```bash
   sudo xmutil unloadapp
   sudo xmutil loadapp benchmark-b4096
   sudo xbutil --batch program -d 0000:00:00.0 \
     -u /lib/firmware/xilinx/benchmark-b4096/kv260-benchmark-b4096.xclbin
   sudo install -m 0644 ~/vart.conf.before-dpu325 /etc/vart.conf
   xbutil --batch examine -d 0000:00:00.0 -r dynamic-regions
   show_dpu
   ```

   确认 UUID 回到 `58E67848-75EA-02F9-BBB1-A7904FFFE459`。若镜像中没有原有 `benchmark-b4096` app/xclbin，不要执行回退命令；应先从对应 AMD KV260 镜像恢复原平台。

### 自行重建 325/650 MHz DPU overlay

本仓库包含钟频链接配置、post-overlay BD hook、实现策略 Tcl、Device Tree 和最终板端包。重新生成 bitstream 还需要 Vivado/Vitis 工具、匹配的 KV260 平台源、DPU encrypted RTL、原始硬件 handoff 及完整实现流程。DPU RTL 需从 AMD/Xilinx 官方 [kria-vitis-platforms](https://github.com/Xilinx/kria-vitis-platforms) 的 `xlnx_rel_v2022.1` / commit `5780bec2519950702df773f3c742ecbe5ee40863` 获取。历史时钟实现使用 Vitis/Vivado 2024.2，不能仅靠本仓库这些 Tcl 从空目录一键重建；完整 Vivado 工程/DCP 没有提交。

历史包可直接加载运行；若要复现或修改 PL，请先准备官方平台和工具，按 AMD 平台源重建 `.xsa`、DPU RTL kernel `.xo`，再以本目录 config/hook 链接、综合、布线、检查完整时序/DRC/CDC，最后重新打包并生成新的 SHA-256/UUID。**不能把重新生成的二进制冒充成本目录的已测版本，也不能跳过板端兼容性检查。**

## HLS 1：独立 INT8 卷积数据流原型

源码在 [`hls/pure_pl_stream/`](hls/pure_pl_stream/)。它对应冻结 YOLO26n 图中的一个 3×3 卷积形状：输入 `[1,16,160,160]`、输出 `[1,8,160,160]`，每层 29,491,200 MAC。输入按 NHWC INT8 打包为 128-bit 像素字，输出打包为 64-bit 字；16×16 输出 tile 带一圈 halo，形成 18×18 局部 BRAM tile。三段 HLS `DATAFLOW` 通过有界 FIFO 连接：tile loader、权重/偏置/重定标参数缓存与 MAC 计算、requant/LUT 与 tile store。

该原型的 412 个 64-bit 参数字保存权重、偏置、定点 multiplier/shift 和激活 LUT。测试向量使用自洽的参数，不含训练得到的卷积权重或真实校准表，因此只能验证接口/数值/控制，不构成已部署模型精度结果。

在安装 Vitis HLS 2024.2 和对应 XCK26 device support 的 Linux 主机上：

```bash
cd hardware/kv260-yolo26n/hls/pure_pl_stream
export VITIS_HLS=/path/to/Vitis_HLS/2024.2
"$VITIS_HLS/bin/vitis_hls" -f csim.tcl
"$VITIS_HLS/bin/vitis_hls" -f synth.tcl
"$VITIS_HLS/bin/vitis_hls" -f cosim.tcl
```

历史深度 324 的完整输出 RTL co-simulation 对 204,800 个输出值零误差，并无死锁。其 HLS C-synthesis 估计为 40 BRAM18、96 DSP、约 23.9k LUT；在 10 ns 目标下 HLS 报告估算 7.3 ns。深度 64/324 的报告与日志在 `evidence/` 中。这个数字不含 placement/routing，也不代表板端时序或整网 FPS。

## HLS 2：PSA 注意力研究原型

源码在 [`hls/attention/`](hls/attention/)。这些 FP32 HLS kernel 对 YOLO26n 导出图中的两头、400-token PSA 注意力切片做 QK 点积、Softmax 和 V 加权归约，包含 online、K/V cache、树形点积和多 lane 变体。源码用 full-row Softmax 数值参考做 C 仿真；不是 INT8 kernel，也不是本项目最终 DPU 图的一部分。

| 变体 | BRAM18 | URAM288 | DSP | HLS 最坏延迟估计（每块，200 MHz） |
| --- | ---: | ---: | ---: | ---: |
| online | 24 | 0 | 18 | 467 ms |
| cached | 13 | 12 | 22 | 498 ms |
| tree | 13 | 12 | 80 | 269 ms |
| tree8 | 20 | 16 | 98 | 219 ms |
| tree8 + value4 | 26 | 16 | 98 | 132 ms |

上表是源级 HLS 综合估计，不是板上延迟。针对 KV260 extensible XSA 的 tree8 `.xo` 综合报告还估算 128 BRAM18、16 URAM、98 DSP，最坏 365 ms/块（100 MHz 目标），说明该 FP32 原型不适合作为当前实时部署方案。更多原始 HLS 报告见 `hls/attention/reports/`。

使用 Vitis HLS 2024.2 可单独重跑源码综合与仿真：

```bash
cd hardware/kv260-yolo26n/hls/attention
vitis_hls -f run_online_synth.tcl
vitis_hls -f run_tree8_synth.tcl
vitis_hls -f run_csim.tcl
```

每个 Tcl 的 `open_project -reset` 只清理该变体的本地 `build*` 输出。板级 `.xo` 报告仅作资源/时钟估算；全系统 link 曾因主机资源限制中止，**没有** combined DPU+HLS xclbin、放置布线结果或板测。

## 性能与证据边界

- 325/650 MHz DPU-only overlay 有历史实现报告、板端加载和模型运行记录；更换板卡镜像、runtime、xclbin 或模型后要重测。
- 91.5916 FPS 属于历史 ReLU-flow INT8 全路径长测；不是 DPU-only 子图 FPS，也不是 HLS 卷积/注意力 prototype 的结果。
- HLS 单层 C/RTL co-sim 不等价于完整网络实现；HLS 资源估计不能加到 DPU 资源表后宣称已集成。
- 本次提交没有连接板卡，因此没有重新加载 overlay、编译板端程序或验证图像输出。
