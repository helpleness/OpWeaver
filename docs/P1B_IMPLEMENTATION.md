# P1-B：ONNX 静态量化、共享数据集与质量评估

实现日期：2026-10-23。增量扩展已有 P0/P1-A，沿用 ModelArtifact、Olive Adapter、实验任务、Validation 和同一个 BenchmarkEngine。

## 已实现的真实链路

NPZ / InputConfig → 校验、规范化及只读数据集快照 → Olive DataConfig → OnnxStaticQuantization → 注册新模型 → 独立数据数值评估 / 可选 Top1 → 同条件 CPU Benchmark → 保存、重开及完整导出。

Olive 运行于独立 `.tools/olive-env`，实际版本 0.13.0，ORT 1.31.0。Worker 没有安装 Olive；没有失败后改用自行调用 ORT 的回退。静态量化由 Olive 内部调用 ORT，`olive_data.py` 是版本化产品代码，经 DataConfig.user_script 显式注册固定 DataContainer；高级 JSON 不接受任意 data_config/user_script/shell。

| 参数 | 实际支持 / 边界 |
|---|---|
| Pass | OnnxStaticQuantization、独立 OnnxQuantizationPreprocess；原 Peephole/OpVersionConversion 保留 |
| 校准方法 | MinMax、Entropy、Percentile，均有真实执行测试 |
| 表示 | QDQ、QOperator；均有真实 ONNX 产物及 CPU 执行测试 |
| 权重 / 激活 | int8 / uint8；已验证 QDQ S8S8、U8U8，QOperator U8S8；不开放 S8U8 和 CPU QOperator signed activation |
| 粒度 | per-tensor、int8 权重 per-channel；uint8 per-channel 明确拒绝 |
| 对称性 | activation_symmetric / weight_symmetric 分开传入 |
| 过滤 | op_types_to_quantize、op_types_to_exclude、nodes_to_exclude；算子名与实际 ORT 注册表比对 |
| 预处理 | quant_preprocess=False；需要时显式添加 OnnxQuantizationPreprocess，保留其独立 Artifact、footprint、产物和导出 |
| QDQ 配置 | prepare_qdq_config=False，避免自动扩大用户选定的量化算子范围 |

保留 Pass 顺序、开关、模板、高级通用 JSON、实际原生配置、候选、原始 stdout/stderr。记录的是执行结果与图中实际观测，不能从文件名、节点数变化或期望参数推断已量化、已融合或已加速。

## 数据格式与操作

1. 导入 ONNX，在 Optimization Lab 选择 Microsoft Olive 并配置独立环境。
2. 添加 OnnxStaticQuantization，按需在前面加入显式预处理。
3. 一个 `.npz` 表示**一个已包含 batch 维度的输入批次**；每个输入一个命名数组，不再次叠加 batch。多输入模型必须提供全部输入，允许 `{"image":"pixels"}` 这样的模型名 → NPZ key 映射。
4. “Calibration & evaluation datasets” 中分别创建两个数据集，NPZ 路径逐行填写或用原生多选框选择。也可 Copy current InputConfig 生成有明确 seed/shape 的合成测试数据；评估默认 seed 与校准不同。动态输入需明确具体 shape。
5. 可选标签 key 必须是非负整数 `[batch]`；Top1 还要求明确输出名及 `[batch, classes]` scores。没有标签不会生成 accuracy。
6. 运行后查看执行状态、误差、质量阈值、图中实际 Q/DQ/整数权重以及性能；从历史重开或导出。

数据在 `.opweaver/datasets/<id>/batch-*.npz`；manifest 保存 schema、来源、role、映射、seed、模型 hash、每批 shape/dtype/min/max/mean、文件和内容指纹、批次数和样本数。SQLite 只存元数据，不存 Tensor。一个批次一个批次读取，最多 512 批，单 NPZ 压缩/展开及 Tensor 头预算 256 MiB，整个数据集 Tensor 预算 1 GiB，准备过程 45s 上限。批次数小于 32 会提示代表性风险；合成数据另有明确提示。样本数按第一个非标量输入的 axis 0 统计，scalar-only 为每批一例，manifest 标明口径。

严格校验缺失/损坏/空文件、键名映射、dtype/rank/静态维度/共享符号维度、非有限数据、object/pickle、解压预算、标签 shape。快照复用时校验 manifest 与每批 SHA256；重开实验会检查引用是否仍可用，缺失或改变不会抹掉历史测量。

校准集与评估集有独立 ID、角色和内容指纹。即使不同 ID、不同文件名，只要输入内容相同也会检出交集。默认拒绝重叠；只有用户显式允许 reuse 才执行，并标记“不是独立评估”。

## 三层质量语义

- 执行有效性：运行、输出名称/数量/shape/dtype、有限值。异常属于执行失败。
- 数值差异：逐批逐输出 MAE、max absolute、max relative、atol/rtol 及超容差元素比例。INT8 allclose 失败仍保留产物并继续 Benchmark；原 FP32 工作流仍保持严格一致性规则。
- Quality Gate：用户可设置最大 MAE、最大 absolute error、最大超容差比例、最大 Top1 accuracy drop。`passed / failed / not_configured` 与实验执行状态分离；阈值失败仍保留性能数据。未启用任务评价明确 `not_evaluated`。

`TaskEvaluator` 是扩展边界，ClassificationTop1 已真实实现，使用用户提供的标签，分别测原模型和候选 Top1 / accuracy drop。测试是明确构造的监督 toy task，不能代表 ImageNet 或真实业务精度。检测、分割、生成、LLM evaluator 尚未实现。

Benchmark 的两个 session 使用同一评估集首批输入、同一 ORT CPU/provider、线程、图优化、warmup 和次数。模型加载与数据准备不计入 session.run 耗时；保存每次采样、mean/P50/P95、请求吞吐以及有明确 batch 时的样本吞吐。conditions_hash 和 input_digest 相同才比较；可能报告 Regression。该单批计时不代表整个数据集的延迟分布。

## 实验与图证据

Experiment 1.0 兼容增加 calibration_dataset_id、evaluation、datasets、quantization、data_availability、preprocessed_artifacts；缺字段的旧记录继续打开。预处理与量化模型均保留来源和实验关联。原始文件不回写。Model Inspector 展示实际 Q/DQ、QOperator、INT8/UINT8 权重、scale/zero point 的有限元数据及剩余 float initializers；控制流子图仍保持原有语义，不扁平化。

Olive 的 `*-calibration-usage.json` 记录真实 reader 的 dataset ID、指纹、read_calls 和 unique_batches；未消费全部批次则不能报告量化成功。导出包含 JSON/Markdown、原/预处理/量化模型和外部数据、两类数据集 manifest/NPZ、实际工作流/footprint/日志/读取证据。导出重定位模型及数据文件；配置保留原执行路径作为溯源，换电脑需重新选择环境/注册数据后再执行。

## 可复现实验与测试

生成离线样例：

```powershell
. scripts/environment.ps1
& $env:OPWEAVER_PYTHON scripts/generate_quantization_examples.py
```

`examples/quantization/quantization-cnn.onnx` 是有固定随机权重的合成 CNN，输入 image `[B,3,8,8]`，输出 scores `[B,32]`。calibration 32 批、evaluation 8 批，每批 2 例，NPZ key 为 pixels；seed 2026 / 12026。未训练，也不主张任务精度。fixture.json 使用相对文件路径，可在其他工作目录重新生成。

浏览器真实实验 `7d5b5a1eaf6245db8dd7496b2778ce19`：MinMax / QDQ / int8 weights & activation / per-channel，独立数据 32 / 8 批；20,914 B → 8,977 B（约 -57.08%）。MAE 0.01422653597、max absolute 0.2914562225。mean 0.013758 → 0.023499 ms，实际**变慢**；0 MAE gate failed，执行和 Benchmark 成功。微小模型 CPU 计时受噪声影响，不能推广为量化性能结论。更多独立运行与原生验收见 VALIDATION.md。原生完整验收使用 Peephole→Opset18→INT8，20,914 B→9,161 B（-56.20%），mean 0.012540→0.027068 ms，报告 Regression；32 批校准及 8 批独立评估、质量 gate/图/导出/关闭重启均已实际完成。

最终完整脚本：140 Python（含 19 量化专项，实际 Olive 用例无跳过）、24 Vitest、前端 TypeScript/Vite、Go 实际 Worker/Olive/静态量化 RPC 集成、go vet 全通过。Playwright 11 条全通过，覆盖 NPZ 映射/量化/gate/报告/导出/重开/Inspector，以及原有训练/模型导入/源码协作/鼠标连续拖动。

## 当前限制与下一阶段

本地标准 ONNX / CPU / Olive 0.13.x 合同。其他版本需要重新探测验证；没有 GPU/NVIDIA、动态量化、INT4、量化训练或 LLM 专项量化。512 批/1 GiB 的数据集管理有界，不是大型数据湖；没有自动 GC 或项目整体重定位。数据准备是有时间上限的同步 RPC，量化/评估/Benchmark 复用可取消、超时、进程崩溃恢复的任务系统。当前 reader 逐批加载 NPZ，不流式切分单个巨大 NPZ。

下一阶段建议：真实代表性数据与标签/预处理 pipeline、可扩展任务 evaluator、量化误差逐层定位、更多模型和 CPU 组合矩阵、数据集迁移/GC，然后再单独验证 GPU/INT4/LLM，避免把接口预留当作已经执行的能力。
