# SGLang 接入方式

固定参考为 `v0.4.6.post5`，提交 `7e257cd666c0d639626487987ea8e590da1e9395`。这是本次还原选择的接口版本，不声称就是丢失工程的版本。较新的 SGLang 已移动 kernel 路径，不能直接替换为 main。

以下是未来使用指令，本次没有安装依赖、应用外部补丁或执行任何 GPU 命令：

```bash
git clone https://github.com/sgl-project/sglang.git /path/to/sglang
git -C /path/to/sglang checkout 7e257cd666c0d639626487987ea8e590da1e9395
python3 -m pip install -e /path/to/sglang/python
python3 -m pip install -e .
python3 scripts/patch_sglang.py /path/to/sglang
DECODE_MODE=R1 DECODE_REUSE_PLAN=1 python3 -m sglang.launch_server \
  --model-path /path/to/model-snapshot --attention-backend triton \
  --dtype bfloat16 --triton-attention-num-kv-splits 16
```

补丁脚本核对 HEAD 后仅替换 `model_runner.py` 的一个 import。可用 `--revert` 撤销这个替换；不会覆盖其他本地改动。没有自动下载权重或修改已安装环境。

## 调用链

上游 `init_forward_metadata` 生成 KV 索引和 num_kv_splits；子类记录本步容量及计划待准备状态。第一兼容层复制长度／分片配置到固定缓冲并建计划；随后各层使用自己的 Q、KV、scratch 和 counter。没有把所有层共用实际 KV 指针。

CUDA Graph capture 时，第一层的 GPU 建表调用被录入模型 Graph。Replay metadata hook 更新 upstream 索引及 active；重放时由已捕获的 kernel 重建本步计划。Python `_prepared` 只决定捕获时在图里放几份建表节点，不能被误认为每次 replay 的动态判断。禁用共享后，每层都录入建表节点，用于共享开关消融。

`DECODE_DISABLE=1` 保留完整上游 Decode；`DECODE_MODE=R0/R1/STATIC/D1/D4` 强制消融路径；`AUTO` 使用已验证 policy，否则 R1。`DECODE_BATCH_CAPACITY` 默认 64。改变这些捕获配置后应重建进程及 Graph。

当前适配只面向单进程内串行执行的完整 GQA Decode，其他模式回到上游；并发执行实例需要独立 backend。模型配置不支持时不创建扩展 engine。CPU 预检查使用上游已知 shape，设备 n/error 仍保留第二层有界保护。

## 未完成运行验收的接口风险

源码已按固定版本核对 metadata、KV pool、forward_decode、capture/replay hook 和 ModelRunner 选择入口，但没有实际加载 SGLang 或模型。依赖组合、具体模型配置字段、Graph 捕获及 kernel 资源限制须在日后获准运行后确认。没有使用“已部署”“兼容最新版本”或“已复现性能”的描述。
