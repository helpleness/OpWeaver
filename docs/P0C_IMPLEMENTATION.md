# P0-C Optimization Lab 实施记录

## 真实执行边界

消费 P0-B 的 `ModelArtifact`。不执行 PyTorch 导出，不重建模型库，不改源码和原始图/权重。首个稳定 backend ID 为 `onnxruntime.cpu`，固定 CPUExecutionProvider；能力探测读取当前 ORT 版本/可用 Provider/GraphOptimizationLevel，只有已实现且可用的后端能启动。Olive 和 NVIDIA Model Optimizer 明确为未实现。

链路：选模型 → Hash/外部权重校验 → 创建 reference Session → 校验并保存输入 → baseline → ORT 官方 graph optimization 序列化 → 独立新图及外部权重 → 重新加载 → 注册到 P0-B 模型库 → 多样本验证 → 从已注册 Artifact 重载并 benchmark → 比较 → 保存 JSON/Markdown。Baseline-only 也建立独立持久化实验。完整实验总是重新测试两侧，不沿用不匹配的旧 baseline。

## 模块与复用

| 文件 | 职责 |
|---|---|
| worker/opweaver/lab_types.py | 1.0 版本 Benchmark/Optimization/Validation 配置与结果、Experiment、BackendCapability、OptimizationJob |
| benchmark.py | 高精度计时、ORT Session 工厂、环境/Hash、同条件比较；采样器接收 runner，与优化后端分离 |
| optimization.py | OptimizationBackend 的 Probe/Capabilities、Validate、Optimize 和 Progress 回调；ORT Adapter/注册表 |
| lab_inputs.py | Seeded 数值 Tensor、NPZ allow_pickle=False、输入名/类型/秩/静态维度/共享符号/Batch/体积校验、输入快照 |
| lab_validation.py | 全输出、多样本、Shape/DType、浮点容差、整数/bool 精确比较、空 Tensor 和非有限值处理 |
| lab_runner.py | 独立子进程实验流水线，复用 ModelTasks.load/import_onnx/ModelArtifact/RunStore |
| lab_tasks.py / code_tasks.py | 复用 CaptureTasks 的进程所有权、锁、取消/超时、事件和原子 job.json；增加可选 progress 文件轮询钩子 |
| process_owner.py | 监控监督 Worker 的进程存活，防止 Worker 崩溃后孤立的 ORT 任务继续发布结果 |
| lab_reports.py | 原子实验 JSON、完整 Markdown、源/优化模型与权重/输入的显式导出 |
| frontend/src/stores/lab.ts | 独立事件路由、项目/选择请求防串、终结去重、历史/配置恢复 |
| OptimizationLab.vue / LatencyChart.vue | 实际配置、进度、原始采样曲线、指标/错误/图比较、跨实验比较和导出 |

Experiment Schema 来自 Pydantic，生成共享 JSON Schema 与 TypeScript。Artifact 新增可选 parent_model_id、experiment_id、target_environment；旧 P0-B 记录读取时补默认值，路径仍在 `.opweaver/imports/<id>`，Model Imports 可以直接重开优化模型。图比较按钮复用原 Graph Canvas 和 Inspector，不维护第二份图。

## Benchmark 语义

`perf_counter_ns` 测量一次同步 `session.run` 的墙钟时间，包含 Runtime 输出分配，排除输入生成、Session 创建和进度持久化。初始化和 warmup 推理耗时合计单独记录，warmup=0 时合计为 0。进度回调最多每 200ms 更新一次（阶段结束强制更新），短模型采样循环中不逐次写报告。保存未截断的全部计时样本，NumPy mean/percentile 计算 Mean/P50/P95/Min/Max。requests/s = 1000/mean_ms，concurrency 固定 1；只有用户明确声明并通过 batch_input axis 0 检查时才计算 samples/s。

两侧 Session 默认 ORT_DISABLE_ALL；用户可选择 manifest 支持的相同 Session 等级。线程数、顺序执行、内存模式/reuse/CPU arena、Provider 都显式设置并记录。模型生成等级与计时 Session 等级分离。环境记录 Python/NumPy/ORT、CPU/架构/逻辑核数/OS，Benchmark 记录模型 ID/Hash、seed/输入配置、实际 Shape/DType/数据 Hash、batch、warmup/runs、Session Options 与原始延迟。

Comparison 只有在环境/输入 Hash/Benchmark 配置指纹一致、有限正分母时输出 speedup。模型 Hash 不属于条件指纹，因为比较对象本来是不同图。Latency change 和 Throughput change 保留符号；变慢为 Regression。无测量、失败、零/缺失值或不同条件不产生假加速。跨历史实验使用其 optimized benchmark（没有则 baseline），同样检查条件；不保证不同时刻的机器负载或热状态相同。

优化使用 ORT SessionOptions.optimized_model_filepath。额外显式设置 `session.optimized_model_external_initializers_file_name=optimized.onnx.data` 和 size threshold=0，确保外部权重重新序列化到新目录，避免保留原文件的相对引用。原模型、其项目副本及外部文件执行前后校验 Hash。产物记录运行环境与序列化参数，All 可能具有平台相关布局；支持某等级不等于知道执行了哪些 pass，applied_passes 诚实保留 unknown/null。

## 输入与数值验证

随机模式每个样本使用 seed+index，float 标准正态，整数 [0,8)，bool [0,2)。每个动态维度要求具体配置，不自动填 Batch=1。支持 float16/32/64、有/无符号 int8/16/32/64、bool dense Tensor；ORT 对具体模型的 kernel 支持仍以实际 Session/Run 为准。字符串、complex、稀疏、sequence/map、BF16/float8 未接入。

用户 NPZ 每文件一组输入，名称必须完整匹配、dtype 不能静默转换、Shape/共享符号必须一致；验证至少两个不同文件。检查 ZIP 解压大小/重复 entry，禁止 pickle。总输入快照上限 256 MiB；所有实际输入保存 NPZ 和 Hash，导出后可复现。

浮点规则为 `abs(candidate-reference) <= atol + rtol*abs(reference)`，参考量在右侧。NaN 即使两侧匹配也失败；同号 Inf 匹配通过，其他 Inf 失败；误差统计仅对有限值配对。零参考值不参与相对误差，另计非零差异数；无有效分母为 null。整数/bool 不转浮点而精确比较（覆盖 uint64 大整数）；Shape/DType 不一致、输出名称/数量/顺序不同、非 Tensor/不支持 dtype 均失败。空 Tensor 仅在 Shape/DType 一致时通过。每个样本/输出独立保存指标和判断。

数值一致性不测 Accuracy/F1/Perplexity，不证明所有输入等价。task_quality 保留 null 作为将来任务评估的结果入口，无可点击的假精度评估按钮。随机 token/索引值不能保证合法，应使用真实输入 NPZ。

## 任务与持久化

公共任务信封维持 CREATED/RUNNING/STOPPING/COMPLETED/FAILED/CANCELLED，Lab 另有 Pending/Validating/Running/Succeeded/Failed/Cancelled 用户状态、当前 phase/progress 与五个 phase 状态。子进程原子写 progress.json；监督线程每 200ms 检查，推送 lab.task，界面每秒轮询补偿。短阶段可能在两次轮询间结束，持久化阶段结果/日志仍保留。

Cancel/timeout 终止 ORT 所在子进程，不在主 Worker 内等待 native kernel 的合作取消。Windows 虚拟环境可能包含 launcher/真实解释器两层，因此终止进程树，同时由取消标记/owner watcher 保证实际 ORT 进程退出。Worker 意外结束时 owner watcher 退出子进程；重开历史将未结束任务标为 Failed，保留已完成阶段和错误。Windows 报告原子替换遇到短暂文件共享锁时采用有界重试，持续权限错误仍明确失败。实验与训练事件完全分开，运行 Benchmark 时拒绝启动同 Worker 的其他训练/Forward/捕获/导入任务。外部程序造成的系统负载仍无法消除。

`.opweaver/optimizations/<id>/` 保存 request/job/progress/experiment JSON、result.json、report.md、stdout/stderr.log、inputs NPZ 与临时 ORT 输出；SQLite metadata 的 optimization: 前缀存摘要，不混入训练 runs。优化模型进入同一 model-import: 索引。无自动 GC，长时间多实验需用户管理磁盘。

导出在显式父目录创建 `opweaver-<id>`，拒绝覆盖同名目录；JSON/Markdown 包含配置、环境、原始耗时、比较、验证、Graph 统计、日志/失败和模型链接。复制源/优化 ONNX 与全部 external weights、输入样本；每个引用先检查路径归属与 Hash。没有本地模型上传行为。

## 验收与边界

运行 `scripts/test.ps1` 和 `cd frontend; npm run test:e2e`；离线示例包含 CNN/external-CNN、多输入三输出、动态维度、自定义不兼容节点。操作步骤见 README 的 Optimization Lab 节。

当前只做 ORT CPU graph optimization。未实现量化、剪枝、Olive 工作流、NVIDIA、GPU/并发 Benchmark、RSS、任务级数据集评估、复杂图 Diff、远程或插件市场。原图和优化图可以分别打开；仅按实际算子分布报告变化，不凭节点减少推断融合或加速。模型分析仍受 P0-B 主 protobuf 256 MiB/20k 分析节点/24 MiB 报告限制。实验超时 3–1800s、计时最多 100k、验证 2–16 样本；这是轻量本地闭环，未声称覆盖大型 LLM。

下一阶段接入真实 Olive Adapter：依赖/版本/模型兼容性探测、workflow 配置、进程执行/产物解析，复用本轮 Benchmark/Validation/Experiment。NVIDIA 应按格式、具体功能和硬件/软件分别声明能力，不默认所有平台可用。

依据：[ORT Graph Optimization](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)、[官方 Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)、[ORT 官方 external initializer 序列化用法](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/optimizer.py)。
