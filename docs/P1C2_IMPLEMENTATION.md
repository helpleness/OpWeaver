# P1-C2：逐层量化误差诊断

2026-10-23。独立 QuantizationDiagnosticEngine 接收已有 ModelArtifact 和 diagnostic 角色数据快照，复用导入器、DatasetFeeds、进程生命周期与现有 GraphCanvas。不会触发 Olive Workflow，也不重新量化。真实测量及原生验收记录见 [VALIDATION.md](VALIDATION.md)。

## 使用

1. 在已保存项目的 Optimization → **Quantization diagnostics** 选择 FP32 Artifact、对应 QDQ Artifact 和 Diagnostic Dataset。数据可在原有 Benchmark & datasets 面板创建，支持 NPZ、已实现的 PNG/JPEG+CSV 适配器及其预处理/输入映射。
2. 设置 Max saved batches、Max capture tensors、原图节点多选/算子过滤、工作 Tensor 内存/临时磁盘预算及 Timeout。一个 sample 明确表示一个保存批次/一次推理 feed，example 数单独保存，默认取 manifest 顺序前3批。多输入/动态 batch 使用每批实际 Shape。
3. Run layer diagnostic 后查看 validate/instrument/capture/weights/complete 日志与逐批进度；小模型可能在一次前端轮询之间完成，完整进度仍在报告日志。Cancel 会终止诊断子进程。
4. 查看匹配覆盖率、Activation/Weight 两种排名；支持 MAE、Max、RMSE、Cosine、算子/匹配状态/名称过滤和 Top N。无有效值显示不可用，不能当作零误差。
5. 点击 FP32 / INT8 图定位按钮，打开现有 Model Imports 画布，展开分组、居中并选中对应节点。Inspector 显示 Tensor、匹配理由、指标、有效批次、原图推断 Shape 与实际展平采集 Shape。屏幕外虚拟节点先移入视口，测量后精确居中。
6. 节点颜色使用本报告 activation MAE 的 p50/p90，也可覆盖为有限、非负、low≤high 的阈值；灰色表示不可比较。这是相对误差分布，不是通用质量门限或根因判断。权重误差独立显示。
7. 选择历史记录可恢复排名和图定位。Export diagnostic report 在新目录保存 JSON/Markdown/运行日志；导出保存的测量，不重新执行模型。源模型/数据缺失时显示当前 availability，历史统计保留，报告仍可导出。

## 真实 ORT 接入

开发机实际 Worker 环境：ONNX Runtime **1.31.0**、ONNX **1.22.0**、NumPy **2.5.0**，CPUExecutionProvider。Olive **0.13.0** 仍在独立环境，仅用于已有量化及回归夹具。执行时将四个 helper 的实际签名与环境写入报告，并校验 instrumentation 后缀，避免静默套用不兼容版本。

`diagnostic_capture.py` 使用 `qdq_loss_debug.modify_model_output_intermediate_tensors` 创建临时模型，再裁剪为获准的采集输出及 helper Reshape。使用真实 `collect_activations`，graph optimization disable、sequential、intra/inter线程1、关闭 arena/memory pattern。两侧消费同一个 DatasetFeeds feed 对象，顺序和预处理一致；每批记录内容指纹。

ORT 将输出展平；保留实际采集 `[elements]` 和原图 symbolic/concrete Shape，绑定实际输入符号并验证元素数量后才 reshape 计算。未知 rank/数据依赖 Shape 无法做预算预检时标 Unsupported。诊断耗时含模型准备和 Session 创建，不能当作 Benchmark 延迟。

所有 helper 只操作 `.opweaver/diagnostics/<id>/temporary/` 中的模型副本，包括外部权重；源 Artifact Hash 在运行前后复验。ORT 可能写 sibling inferred 文件，因此推断也只针对副本。

## 匹配证据与覆盖率

`diagnostic_matching.py` 独立于数值引擎。先用真实 `create_activation_matching` 的名称规则规划 QDQ 对，再在真实采集结果上调用该 helper。额外规则必须通过实际 Q/DQ 边、常量参数、SSA Tensor 名/输出端口、算子/domain/opset/attributes/input arity 与递归上游生产者契约验证。

Node Name 不参与配对：节点改名仍匹配。不会因为同类型/相邻位置就配对；Tensor 改名、融合、替换生产者、Flatten→Reshape 或不一致上游可能保守地成为 Partially Matched / Unmatched，且不输出该行误差。没有实现完整 ONNX 等价性证明。

状态为 Matched / Partially Matched / Unmatched / Unsupported。每行保存原始 `onnx_<index>`、候选计算节点 ID、实际 QDQ Tensor 生产者 ID、原始 Node Name/唯一性、两侧 Tensor/Shape、method/reason。ID 与已有 ONNX 导入图一致；保留 C3 所需标识，不声称已实现节点排除闭环。

覆盖率 = 请求原始节点范围中至少有一个有效捕获批次的 activation 输出数 / 该范围全部 activation 输出数。权重不计入分母；超限省略的输出仍计入范围并记录 omitted。部分 feed 无效时仅聚合有效 feed，状态与计数明确显示。来源链无法确认时警告，仍只描述用户显式选择的两份模型。

## 数值和权重

`diagnostic_metrics.py` 在真实浮点值上以 float64 累积 MAE/MSE/RMSE/max absolute/relative/cosine。MAE/MSE 按有效元素数加权；RMSE=sqrt(global MSE)；max 横跨有效元素；cosine 为有定义的逐 feed cosine 的均值，另存 defined_cosine_samples。另存等权 mean_sample_mae，避免混淆两种平均。

相对误差只覆盖参考值非零元素，并记录 zero/defined 数；全零参考或零向量的对应指标为 null。Shape/dtype不一致、整数、空 Tensor、NaN/Inf、统计溢出不强制置零。每个 Tensor 每个 feed 只保存一次统计，某行失败不重复计入其他行。

`diagnostic_weights.py` 以实际 initializer identity/DQ 参数建立受限权重模型，调用真实 `create_weight_matching`。权重独立于 activation/数据批次，记录整数 dtype、原始 Shape、axis、scale/zero-point Shape/数量及前16个值；per-channel 按真实 axis 广播，INT32 bias 与 INT8 weight 分别保留。

安装版本 helper 存在整数减 zero-point 溢出的可能，因此额外用拓宽后的 affine DQ 检查 helper 返回值；二者不一致时显示 Unsupported 并扣留指标。没有把原始 INT8 编码直接与 FP32 数值比较。

## 资源、隔离与持久化

| 项目 | 默认 / 上限与语义 |
|---|---|
| 保存批次 | 3 / 16；按 manifest 顺序，不随机重采样 |
| 采集 Tensor | 32 / 256；两侧总和，包含 QDQ pre/post 输出 |
| 原图范围 | 未选时前64个节点；最多64个显式节点，报告最多256个 activation 行 |
| 权重 | 可关闭，最多64条；单独估计匹配/统计工作空间 |
| 工作 Tensor 内存 | 128 / 512 MiB；输入+暴露输出+最大 Tensor 的保守统计空间，不是总进程 RSS 限制 |
| 临时磁盘 | 512 / 1024 MiB；创建前保守估计副本/权重模型，阶段记录观测峰值 |
| 模型 | 每侧含外部权重最多128 MiB |
| Timeout | 180 / 900秒，最小3秒；父进程超时可中止原生 ORT 推理 |
| 临时策略 | 成功/失败/取消/中断恢复均清理本次 temporary；不会清理用户原模型 |

预算不包含模型解析、Session/kernel 内部 RAM，报告不将工作空间估计冒充总进程峰值。超限应缩小节点范围/关闭权重/提高预算；非本轮支持的大模型不默认暴露所有输出。

任务复用 CaptureTasks 的独立子进程、owner watcher、进度轮询、进程树终止；诊断与训练/导入/Benchmark 等计算任务互斥。异常保留失败记录，Worker 退出不导致桌面进程崩溃。任务 terminal 状态始终由生命周期 state 推导，含 supervisor 硬超时和异常退出路径。

JSON Schema：`schemas/quantization-diagnostic.schema.json`；TypeScript 由生成脚本派生。SQLite 只保存任务摘要；`.opweaver/diagnostics/<id>/diagnostic.json`/`report.md` 保存统计、匹配、Hash、数据指纹、源量化实验/配置、环境、日志和限制。每批原始激活归约后释放，不进入数据库/前端/导出文件。报告 viewer 24 MiB 上限；恢复时核对引用，并将遗留运行状态标 Failed、清理临时文件。

## 当前边界与后续

已实现 QDQ 顶层 float32/float16 activation 诊断和基础常量权重分析。真实验证覆盖实际安装 ORT1.31.0 的 float32 Conv/Gemm/MatMul、多输入、动态 batch、NPZ/图片、per-channel、部分匹配/非有限结果/失败与生命周期、UI图联动和原生恢复；不声称所有 float16/任意算子组合均已实测。

当前不支持 QOperator、custom/function/control-flow 子图、数据依赖中间 Shape、任意 Tensor 重命名/融合后的可靠映射、GPU profiling、INT4/LLM 专项。QOperator 明确失败，不隐式转换成 QDQ。诊断数据必须为 diagnostic 角色；与最终评估有可检测重叠时拒绝，其他角色重叠提示，沿用 C1 指纹覆盖边界。

C3 建议基于保存的原始 ID/Node Name 唯一性、匹配证据和真实误差选择待验证节点，创建新实验并验证排除实际生效，再以独立验证数据比较原始/全量INT8/排除节点三组的质量与 Benchmark。高误差只表示值得调查，不直接推断根因。本轮没有实现诊断选点重新量化、自动混合精度搜索或敏感性因果判定。
