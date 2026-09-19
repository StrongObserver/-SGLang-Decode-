# 简历词条与实现对应

本表对应提供的 PDF 中 P1 的三个负责模块；P2 与实习不在本仓库范围。源 PDF 和原资料仅阅读，未修改，也未复制进公开仓库。

| 简历表述 | 具体实现 | 验证入口（本次未运行） |
|---|---|---|
| GPU 紧凑任务表、直接索引 | `kernels/plan.py::count_scan/emit_records`；`kernels/attention.py::_attention_task` | `tests/gpu/test_decode.py` 对比 GPU 记录与 CPU 规格 |
| 按分片计算量分桶 | `make_order_and_packs` 整数桶号、稳定 cumsum rank | `planning/reference.py::bucket_order` |
| 动态领取长任务 | `persistent_decode` 重桶优先包队列、scalar atomic | D1 模式；任务覆盖检查 |
| 限量打包短任务 | `make_order_and_packs` 的 width、packed_tasks、singleton tail | A/B/B64 的包数与预算用例 |
| 均匀批次静态轮转 | 设备模式判断、R1 步长循环 | uniform 与 Graph 模式切换用例 |
| 初代紧凑轮转对照 | R0 只建前缀、消费者二分定位 | `benchmarks/bench_decode.py --workload B` |
| 按请求页表定位 KV | `attention.py` 的 paged / CSR 两个地址入口，真实 strides | 非连续 KV、乱序物理页、尾页用例 |
| 原分片边界与输出槽位 | 上游 requested_splits + 32 对齐；按原 r/h/s 写回 | CPU 边界规格与 GPU 输出参考 |
| 有效片数与归并一致 | `merge.py::merge_splits` 使用本轮 counts，保留 LSE 数学 | partial/LSE 预填 NaN 后输出对照 |
| 连续缩批避免读旧数据 | counts 写满容量，merge 只读有效 split | `test_graph.py` 的 32→7→1→32 |
| 只读任务索引跨头组共享 | R、P 长度为 E，而非 E×G；task 恢复 head group | `runtime/workspace.py` 与 GPU 记录检查 |
| 同一步兼容层计划复用 | `DecodeGraph._step`；`ScheduledTritonBackend._prepared` | 两层 Graph、`DECODE_REUSE_PLAN=0/1` 消融 |
| 每层独立 reset／固定网格 | LayerBuffers、counter.zero_、固定 workers | counter 毒化、指针稳定用例 |
| SGLang 整模型接入 | `integrations/sglang_backend.py` + 单 import patch | `benchmarks/bench_model.py` 真实 Engine.generate |

## 数字如何对应

| 用户提供资料中的历史记录 | 本仓库状态 |
|---|---|
| B 单层 237→214 μs，降低 9.7% | 相应 R0/D4、B 输入和完整调用计时入口已写；未重新测量，不能将数字视为此版本结果 |
| A 整模型 Decode 单步 22.0→20.6 ms，降低 6.4% | SGLang 接入与计划复用代码已写；未恢复原始模型快照、运行日志和精确单步测量环境 |
| 尾页、重排、缩批的历史正确性／访存检查 | 用例与 Compute Sanitizer 入口已写；本次没有执行，不能写作通过 |

`bench_graph` 只覆盖 attention 堆叠；`bench_model` 报真实生成全程墙钟（含 Prefill），两者都不能冒充历史的固定快照整模型 GPU Decode 单步数字。历史入口还需要模型输入更新到设备采样结束的准确事件边界及原实验依赖记录。项目的功能还原不依赖重现上述数字。
