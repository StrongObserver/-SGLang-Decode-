# SGLang Decode Task Scheduling

基于 SGLang 的分页注意力 Decode 任务划分优化。将有效请求分片组织为 GPU 紧凑任务表，在同一分页 GQA 计算核上比较轮转、静态均衡、动态领取和限量打包，并在 CUDA Graph 下复用同一步的兼容层计划。

**这是根据保留资料重建的源码工程。核心代码、SGLang 接入、对照与验证入口已写入；本次仅做粗略静态检查，未编译、未运行测试或 GPU 实验。历史成绩不是这个版本重新测得的结果。**

## 从这里阅读

| 内容 | 入口 |
|---|---|
| 简历三个负责模块对应到哪些代码 | [简历实现对照](docs/resume_mapping.md) |
| 数据流、任务映射、Graph 生命周期 | [架构说明](docs/architecture.md) |
| 固定版本 SGLang 接入／回退 | [接入说明](docs/integration.md) |
| 计时边界、验证脚本和未验证事项 | [验证说明](docs/validation.md) |
| 借鉴、上游源码与许可证 | [来源声明](THIRD_PARTY_NOTICES.md) |

## 核心实现

- **紧凑计划**：GPU count/scan → SoA records → 稳定工作量分桶及包描述。记录跨头组共享，任务直接定位原请求／分片。
- **多种调度**：R0 前缀搜索轮转、R1 直接索引轮转、贪心静态列表、D1 单任务领取、D4 最多四项打包；每桶保留末轮单任务。
- **分页计算**：Triton grouped-query attention、真实 stride／页表寻址、FP32 在线 softmax 状态、按原分片槽位归并。提供 SGLang CSR token 索引入口。
- **运行时与 Graph**：固定缓冲及网格、每层独立 counter/scratch、逐次回放建表、步内兼容层共享、容量与模型类型回退。
- **接入与对照**：固定 SGLang 版本的 backend 子类和可撤销单 import 补丁；CPU 规格、输出 oracle、缩批回放用例、单层及模型基准。

## 目录

```text
src/sglang_decode/
├── config.py               # 容量、头布局、执行模式
├── kernels/                # GPU 建表、持久 attention、分片归并
├── planning/               # 规划调度、CPU 规格、离线选择表
├── runtime/                # workspace、执行器、CUDA Graph 生命周期
├── integrations/           # SGLang backend 接入与回退
└── reference/              # FP32 数值参考
benchmarks/                 # workload、上游/FlashInfer 对照、计时与结果记录
tests/unit/                 # 任务覆盖、包预算、容量、策略规格
tests/gpu/                  # 数值、寻址、Graph 连续缩批
configs/                    # A100 示例容量与未校准策略
scripts/                    # 接入补丁、粗略静态检查、可选诊断与结果汇总
third_party/                # 原样上游 baseline、来源哈希与许可证
docs/                       # 架构、简历映射、接入、验证边界
```

## 使用范围

重建接口目标为 [SGLang v0.4.6.post5](https://github.com/sgl-project/sglang/tree/7e257cd666c0d639626487987ea8e590da1e9395)，支持 BF16/FP16、完整上下文、单 token GQA Decode。独立入口支持多 token 物理页；该固定版本的 SGLang 接入使用 page_size=1 CSR KV 索引。Prefill、模型与 KV 分配继续使用上游。

默认 AUTO 采用直接轮转 R1；没有将资料里的历史收益硬编码为动态启用条件。`STATIC` 的 GPU 规划器是参考实现，性能比较另外包含更高效的主机 heap 规划及其上传成本。动态是否优于强静态保持待测。

不触发项目导入、编译或 GPU 执行的检查入口：

```bash
python3 scripts/static_check.py
```

安装、GPU 测试、基准和服务启动命令见[接入说明](docs/integration.md)与[验证说明](docs/validation.md)，本次均未执行。仓库没有模型权重、个人简历原文件或预制性能日志。

## 许可证

本地代码使用 Apache-2.0。上游源码保留原版权和许可；Triton 参考部分的 MIT 声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
