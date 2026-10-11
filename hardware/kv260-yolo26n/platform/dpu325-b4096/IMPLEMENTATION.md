# 325/650 MHz overlay 实现摘要

本目录保存历史板端使用的 DPU-only 325/650 MHz package。全量时序、资源、DRC、CDC、功耗估算与 30,000 帧同模型性能记录留在原工程的归档中；下表列出该实现的关键事实，避免把时钟设定误说成频率测量。

## 实现和运行身份

- 硬件：KV260 `xck26-sfvc784-2LV-c`、单个 `DPUCZDX8G_1` B4096 CU。
- DPU clocks：`aclk=325 MHz`、`ap_clk_2=650 MHz`；平台 VCU 等其他时钟域保持其原频率。
- XRT metadata UUID：`E4A88BD4-2D7D-14C8-2F40-567A0F32A66D`。
- DPU CU base：`0xA0010000`；硬件 fingerprint：`0x0101000016010407`。
- 生成 bitstream 的 SHA-256：`7bf937d103019b261b984ff63d4f5782e049c04586a924c441bb93889bb6afb5`。
- runtime metadata xclbin 的 SHA-256：`37f4ad970a577e24bc3631d8febb6a3f0f2977863776c02270d8ac7fcae21ef8`。

## 时序结果

修复后 routed design 的 setup/hold TNS 为 0、失败端点为 0；最小 slack 报告约为 0.000 ns，650 MHz DPU 内部路径 WNS 约 +0.001 ns。该结果贴近边界，适用于描述这一次实现通过了工具门槛，不能描述为有充足裕量。板端部署后进行了模型输出和长稳测试。

历史 routed utilization：LUT 70,402/117,120（60.11%）、FF 135,992/234,240（58.06%）、BRAM 94.5/144（65.63%）、URAM 56/64（87.5%）、DSP 710/1,248（56.89%）。这是包含平台的路由结果。Vivado 默认 activity 的片上功耗估值为 7.537 W，报告置信度为 Low；它不是板端功耗。板端工作负载下测得的 U14 rail power 应引用性能报告，不要将两个功耗口径混用。

## 原始记录的边界

最终 repaired routed DCP、完整 Vivado 工作树和 post-route audit 原始输出没有随仓库发布。仓库保留了本 overlay 包、可见的 Vitis clock hook/config 和本摘要，因此可加载已有包并审阅关键修改；若要从源码重建，需按主目录说明取得 AMD/Xilinx 官方 DPU/平台源并重跑完整实现，重新计算校验和，不能用新构建替换已测 package 后继续沿用旧性能数据。
