# P1-A — Microsoft Olive Adapter

本阶段增量接入真实 **Olive 0.13.x local CPU workflow API**，实测 0.13.0、Python 3.13.3、CPU Torch 2.11.0、ONNX 1.23.2、ORT 1.31.0、onnxscript 0.7.2、onnx-ir 1.1.0、onnxoptimizer 0.4.2。Worker 环境仍为原来的 ONNX/PyTorch/CUDA 环境，没有安装 Olive。P0-C ORT Adapter、BenchmarkEngine、输入和 Validation Engine 保留原实现。

## 安装与使用

1. 在项目根目录显式执行 `scripts/setup_olive.ps1`，创建 `.tools/olive-env`。支持 `-EnvironmentPath` 和 `-BasePython`。CPU Torch 从官方 CPU index 安装，其余验证版本在 requirements-olive.txt。完整安装快照写到虚拟环境的 installed-requirements.txt。应用启动和探测均不会安装包。
2. 在已保存项目导入 `examples/onnx/redundant-identity.onnx`，进入 Optimization Lab，选择 Microsoft Olive。
3. 配置独立 interpreter 或虚拟环境目录，点击 Save & detect environment。Windows 检查 Scripts/python.exe，POSIX 路径支持 bin/python；后者仅路径/进程契约支持，未进行原生 GUI 验收。带空格路径有真实测试。
4. 添加 OnnxPeepholeOptimizer，可追加 OnnxOpVersionConversion（示例 target_opset=18）。可调整顺序、禁用、删除和修改参数。输出文件名、外部权重阈值可配置。
5. Validate & preview 使用安装版本的 RunConfig 验证。Save workflow 保存通用配置（含 source ID、环境、输入、Benchmark/Validation），同 Workflow ID 显式保存会更新该模板；历史 Experiment 不变。Saved workflows 可恢复再执行。
6. Advanced generic workflow JSON 可编辑、验证和应用。原生 Olive JSON 作为只读生成预览；实际执行配置在实验 Configuration & logs 中独立保留。
7. Run optimization experiment 执行 baseline → 实际 Olive workflow → 新 ModelArtifact → 多样本 Validation → 相同条件 ORT CPU Benchmark。Graph changes 可查看前后模型；Export 保存模型、输入、JSON、Markdown、工作流和完整原始日志。

## 环境与能力

OptimizationEnvironment 是独立且版本化的后端环境引用，按项目保存到 SQLite metadata。快照包含 Python/库版本、venv 前缀、安装目录、CPU EP、探测错误和遥测状态。状态包括 Not Installed、Available、Missing Dependency、Version Incompatible、Environment Error。当前允许经验证的 0.13.x API；其他 minor 版本明确不可用。探测缓存 30s，显式检测会刷新。未配置或不可用只影响 Olive，ORT/建模/训练仍正常。

能力由独立解释器实际导入 Pass 并读取 `_default_config` 参数契约，结合固定的本地功能白名单产生。支持 Peephole 的 onnxscript_optimize、onnxoptimizer_optimize、fuse_reshape_operations，以及 Opset Conversion 的 target_opset。未安装 onnxoptimizer 时其可选开关不可启用。自定义 operator domain 模型的这些 Pass 被保守标记为未验证/不可用；不是所有 ONNX 模型都承诺能优化。Opset 转换后的 CPU 兼容性仍由实际加载检查，不能仅凭安装状态推断。

Olive 0.13.0 的当前依赖组合可能未自动安装 requests；安装清单显式补齐。环境探测会实际导入依赖和 workflow API，不能仅凭包元数据判定成功。

## 调用和产物

`optimization_environment.py` 负责环境解析、探测和短时配置校验；`olive_adapter.py` 实现统一 Adapter 和工作流转换；`olive_runner.py` 由配置的 Python 以 `-I -u` 启动，调用 `olive.workflows.run.run.run`。显式参数数组，无 shell。RunConfig 解析和 run 调用使用副本，因为 Olive 会规范化并修改传入字典；保存的通用/生成配置与实际展开配置分开。

工作流固定 LocalSystem/CPUExecutionProvider，无 evaluator、search、共享 cache、远程资源或用户脚本。非启用 Pass 不放入有序原生 passes，启用 Pass 按顺序执行。每次实验全新目录与 cache，保留实际 footprint 的 Pass 类型、参数、时间、父子模型关系；验证实际已产生产物的执行序列与请求一致，不能把部分成功当作整体成功。

读取 WorkflowOutput/ModelOutput 的实际 ONNXModelHandler 路径，处理 Olive 返回的小写类型和目录模型资源。候选模型必须在该实验目录中。保留所有返回候选的配置/路径/来源与图签名，当前本地串行无搜索流程选第一个返回候选，不宣称它是性能最优。所选候选注册到共享 ModelArtifact，保存 parent_model_id / experiment_id。原文件和外部权重不覆盖，前后 hash 检查沿用 P0-C。

请求 Pass ≠ 实际执行 Pass ≠ 已证明的 rewrite。对加载外部权重后的 ONNX graph 计算 digest/node 观察值，明确表示观察到的变化或未观察到可应用的图转换；不从节点减少推断融合规则或加速。`applied_passes` 保持未知。若候选附带尚不支持的 inference_settings，返回明确兼容性失败，避免破坏两侧相同 Benchmark 配置。

所有数值验证和 Benchmark 均在原 Worker 的现有 ORT CPU 引擎执行。相同输入文件/seed、动态 Shape、NPZ、多输入、多输出、atol/rtol 和精确整数/bool 规则沿用原实现。两侧相同 CPU EP、线程、session graph level、warmup/runs/Batch。报告区分优化环境与 Benchmark 环境；允许未变化、变慢、体积增加及数值失败。

## 生命周期、日志和恢复

沿用 CaptureTasks 的后台 Lab 子进程、进程树终止、超时和取消。嵌套 Olive 子进程也监视 owner PID 和 cancel marker，Lab/Worker 意外退出后不能继续优化。Olive stdout/stderr 写入实验内独立原始文件，实时追加到 phase 日志；没有百分比时 progress=null，UI 只显示运行状态和原始日志。

实验 `.opweaver/optimizations/<id>/olive/` 保留 request.json、native-workflow.json、executed-workflow.json、pass-results.json、execution.json、result.json（可用时）、stdout.log、stderr.log、output/cache/tmp。中断/失败保存已取得的配置、错误和日志，未完成模型不标记成功。恢复时正在运行的历史标为失败；取消为 Cancelled。取消不影响已有实验、原模型和训练历史。

清理策略是保留所有实验目录便于诊断，不自动删除原始/失败产物；成功模型另外复制注册，后续清理实验 cache 不依赖源模型。当前没有自动 GC 或 UI 清理操作，长期使用需在所有相关任务结束后手动管理磁盘。导出所选验证模型、输入和工作流原始文件；未选候选的元数据保留，但大型 cache/临时文件不自动复制到导出包。

默认在任何 Olive import 前设置 `OLIVE_DISABLE_TELEMETRY=1`、`OTEL_SDK_DISABLED=true`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_HUB_DISABLE_TELEMETRY=1`。此处没有在线模型下载、模型上传、云系统或远程代码入口。这是本地调用策略，不是操作系统网络沙箱。

## 验证与边界

test_olive.py 包含独立空 venv、错误路径、版本/依赖分类、真实原生配置校验、多 Pass 顺序、参数/脚本拒绝、真实冗余 Identity/外部权重 CNN/动态多输入输出、NPZ 回放、无图变化、共享数值和 Benchmark、导出/重开、真实进程取消/超时，以及补充的进程故障注入（后者不代替真实执行测试）。Go 和 Playwright 也通过应用 RPC/UI 调用真实 Olive。最终计数、桌面结果及可复现产物见 VALIDATION.md。

只完成上述两个本地 ONNX Pass。Transformer 专用优化、量化/剪枝、校准、自动搜索、GPU/云端工作流、NVIDIA ModelOpt、任意原生 JSON 和用户脚本、远程模型下载均未实现。测试的小模型不代表大型 LLM 或所有 ONNX 算子兼容；大模型 Pass 会占用独立进程内存，当前未做内存性能测量。下一阶段宜按模型/校准数据和 EP 契约增量接入一个量化 Pass，继续复用当前验证及性能测量。

官方依据：[Olive 配置选项](https://microsoft.github.io/Olive/reference/options.html)、[ONNX transformations](https://microsoft.github.io/Olive/features/onnx-transformations.html)、[隐私和遥测](https://microsoft.github.io/Olive/privacy.html)。实际调用和参数以安装的 0.13.0 wheel 源码与真实执行测试为准，未混用其他版本的 pass_flows API。
