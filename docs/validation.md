# 验证入口与本次交付状态

本次仅执行 Python AST 解析、JSON 解析、Git diff 空白检查与人工阅读。没有导入 torch/Triton/SGLang，没有 Python 字节码编译、Triton JIT、构建、单元测试、GPU 测试或性能实验。测试文件存在不代表测试通过。

## 可用的后续入口

以下命令只是说明，运行它们会触发相应测试／GPU 编译，**本次未执行**：

```bash
# 纯 CPU 规格测试
PYTHONPATH=src:. python3 -m pytest tests/unit

# GPU 输出、元数据、缩批和 Graph 检查
DECODE_RUN_GPU_TESTS=1 PYTHONPATH=src:. python3 -m pytest -o addopts='' -m gpu tests/gpu
bash scripts/sanitize.sh --run-gpu

# 单层：准备 + reset + attention + merge，全部调度变体
PYTHONPATH=src:. python3 -m benchmarks.bench_decode --run-gpu --workload B --output results/B

# 可选上游/FlashInfer，对齐 dtype、长度和实际 KV；需要对应依赖
PYTHONPATH=src:. python3 -m benchmarks.bench_decode --run-gpu --workload A \
  --external upstream flashinfer --output results/A_external

# Graph attention 堆叠，不含模型其他层
PYTHONPATH=src:. python3 -m benchmarks.bench_graph --run-gpu --output results/stack

# 真实模型完整生成，需先按 integration.md 接入
PYTHONPATH=src:. python3 -m benchmarks.bench_model --run-gpu --model /path/to/model-snapshot \
  --token-ids /path/to/prompts.json --mode R1 --reuse-plan 1 --output results/model.json

python3 scripts/summarize_results.py results/B/samples.csv
```

## 覆盖与边界

- CPU 规格包含 A 的 152 条记录／1216 任务、B 的 96 条记录／768 任务／705 包、B64 的 1329 包，以及零长度、尾片、预算和无重漏。
- GPU 检查覆盖所有调度模式、乱序页、实际 stride、BF16 与 FP32 参考输出、设备元数据对照。
- Graph 检查覆盖 32→7→1→32、连续更新、两层 scratch/counter 隔离、模式来回切换、指针稳定和 NaN／计数器毒化。
- Compute Sanitizer 是访存与同步补充，不能代替输出对照；用例未覆盖的并发、多设备和模型变体仍需单独验收。

## 计时与策略

单层默认 200 次预热、每轮 1000 次、5 轮交替顺序。每轮保存 CUDA Event 均摊和完整墙钟；包括主机静态规划与上传的比较必须使用完整墙钟。Event 可能包含 CPU 提交空档，不能标作纯 kernel 忙碌时间。轮均值的中位数与范围不等于逐调用 P99。

外部上游基线将 fixture 的 CSR 与分片数预先备好；SGLang baseline 调用包含其原 stage1/stage2。FlashInfer baseline 每次执行 plan+run。两者的准备边界应在正式报告中保留，不能声称已完成上游参数搜索或在此版本证明最快基线。统一头组织的调度归因使用 R0/R1/STATIC/D1/D4；外部库比较是不同 kernel 的完整入口比较。

`calibrate_policy.py` 从完整墙钟原始行选择路径：动态必须比最快静态至少快 3% 且跨轮范围分离；独立 holdout 的退化超过 2% 拒绝输出。范围分离只是保守门槛，不是统计显著性证明。输入范围由调用者根据真实校准簇冻结；两个不同文件也不自动证明数据独立，需在 manifest 中保存独立输入哈希与采样来源。

若 CPU 静态赢，脚本拒绝把它冒充成 GPU LPT 写入 AUTO；应先完成主机静态在线接入／成本验证，再改策略。目前 AUTO 没有校准数据，故保留 R1。`configs/policy.unvalidated.json` 故意不能用于启用动态模式。

## 结果管理

`results/` 默认忽略，不预置合成实验成绩。基准运行生成 manifest 与逐轮 CSV；模型入口保存完整生成记录。工作区中没有原始工程的测量文件。GPU 编译是否成功、正确性、资源使用量和真实速度均保持未验证。
