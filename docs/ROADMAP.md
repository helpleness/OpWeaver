# 分阶段路线

按 2026-10-23 修订任务书重新编号。旧版 P0/P1 是已有可运行基线，不等于本任务书的全部 P0 已完成。本轮按最终指令先完成 P0-A，保留原架构和训练功能。

## 已有基线

Wails/Go/Vue/Pinia/Vue Flow，Design Graph 1.0，Eager 模型、Autograd、合成训练、CPU/CUDA 检测、真实指标、SQLite 历史、停止/Checkpoint 精确恢复、OperatorRegistry、基础 PassManager/DCE、测试和 Windows 构建。

## P0-A — 已完成

- Monaco 项目源码编辑、多文件、保存、搜索、错误、源码跳转。
- 图/代码/双视图、可调分隔条、Inspector 显隐、连续拖动与稳定光标。
- AST 静态发现、显式子进程 FX 捕获、Shape/Dtype、原模型 Forward 对照、超时/取消/日志。
- 15 个内置算子；Linear/Conv2d/Norm/Sequential/嵌套模块/Add/Concat/残差等可追踪结构。
- 可读可执行 PyTorch 生成和受支持结构往返；节点 ID、Source Mapping、Hash/修订/同步状态。
- Visual 代码差异审阅，Code-First 只读图和直接构造参数局部改写；失败保留有效图，磁盘冲突拒绝覆盖。
- example.square 注册示例：构造、Shape、Autograd、代码生成、捕获的真实测试。
- 任务书 A–D、鼠标回归、原训练链路验证，结果见 VALIDATION.md。

当前限制：FX 为配置示例输入上的 CPU/eval 分析；动态控制流不支持，未知节点保留为分析；没有静态等价证明。任意手写模型不保证转成 Design Graph。Code-First 不开放训练或任意反向改线，生成模型不迁移原权重。第三方自动插件加载、OS 沙箱、完整 Python 项目依赖打包未实现。

## P0-B — 已完成

1. ONNX 文件作为独立 ModelArtifact 导入，原始模型只读保留。
2. ONNX Graph origin 和来源/版本，复用现有 Analysis Graph 渲染与 Inspector。
3. 实读节点、属性、权重、输入输出、动态维度、Shape/Dtype 和未知算子诊断。
4. 模型 Inspector 的真实参数量、文件大小、算子分布；不可靠 FLOPs 显示未测量。
5. 统一 ImporterRegistry，PyTorch 复用 FX 并可回到 Monaco；ModelArtifact/Metadata/Diagnostic/GraphImportResult。
6. 属性子图及外层捕获、多输出、搜索、高亮、分组/展开、600 节点回归、异步导入和持久化。
7. 离线模型与任务书 A–G 验证，详情见 VALIDATION.md。

## P0-C — ONNX CPU 优化闭环，尚未实现

1. ONNX Runtime CPU Execution Provider，真实配置输入及推理。
2. Optimization Adapter：Probe/Capabilities/Validate/Optimize/Progress/Result；Artifact、任务、配置、报告均有版本与 lineage。
3. Baseline → ORT graph optimization → 固定输入数值校验 → Optimized Benchmark → 比较/导出。
4. 同条件 warmup/次数/Batch/Shape，记录逐次耗时、均值/P50/P95/吞吐、模型大小和环境。
5. 输出 Shape/Dtype、atol/rtol、最大/平均绝对误差；失败或变慢如实报告，原模型不覆盖。
6. 离线测试 F，无 GPU 也可完成；此阶段之前不创建假 Optimization Lab。

## P1 — Microsoft Olive 和训练泛化

Olive 可选 Adapter，检测依赖/能力并调用真实工具，不重实现算法。Dataset/DataLoader、输入/target 映射、validation split、Scheduler、可恢复 sampler、训练插件。增量事件 sequence、分页指标、日志轮转、幂等启动、Artifact 重定位、历史图恢复。

## P2 — 可选 NVIDIA/Kernel/复杂模型能力

NVIDIA Model Optimizer Adapter 按 GPU/CUDA/格式探测；Triton/CUDA Kernel、参考实现、Forward/Backward 数值验收、真实 Benchmark/Profiling/autotuning/回退。Transformer/Attention/GQA/MQA/KV Cache/MoE/LoRA、大模型量化和分布式/远程 Worker 分项落地。算子逻辑、Kernel 实现和后端能力分离，不能把路线表当作当前能力。

跨阶段产品事项：可信插件来源与版本锁、Python runtime 管理/可选 CPU bundle、签名安装包、中文本地化、DPI/辅助技术验证、子图 ModuleSpec 执行。CPU 基础功能不应依赖 NVIDIA 环境。
