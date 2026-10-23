# 验收记录：P1-B、P1-A、P0-C、P0-B、P0-A 与已有基线

## P1-B 验收（2026-10-23，UTC+8）

| 检查 | 最终实际结果 |
|---|---|
| scripts/test.ps1 / Pytest | **140 passed，0 skipped**；含 19 项共享数据/真实量化/质量专项 |
| Vitest | **24 passed**；包括实验恢复期间阻止默认选择覆盖的竞争回归 |
| Go / go vet | **9 tests / 3 packages passed**；真实 Worker、Olive 以及静态量化/数据集 RPC/导出集成，vet 通过 |
| Ruff / vue-tsc / Vite | **通过**；52 个 Python 文件；仅既有 Monaco 大 chunk 提示 |
| Playwright | **11 passed**；真实 NPZ→Olive INT8→独立评估→gate→Benchmark→导出→重载→Inspector，以及 P0/P1-A 和鼠标回归 |
| Wails Windows amd64 | **构建通过，原生 WebView2 完整操作及关闭重启通过** |

新增真实测试覆盖：输入映射、多输入、动态 Batch、缺失/损坏/空/非有限/dtype/shape/object 数据、指纹改变/重叠、明确允许复用、独立评估、MinMax/Percentile/Entropy、QDQ S8S8 / U8U8、QOperator U8S8、int8 per-channel、Gemm 排除、显式预处理及阶段 Artifact lineage/导出、标签 Top1、gate 通过/失败、数据引用缺失重开。量化测试没有用 Mock 替代 Olive。原有进程取消/超时/崩溃恢复、训练、代码协作、导入和 Benchmark 回归保持通过。

原生项目仍为 `projects/model-mvkdmssb`。通过原生窗口导入 `examples/quantization/quantization-cnn.onnx`，Artifact **91276a09ba3641e4ae44ee937cd78666**；保留原 Peephole → Opset18，再添加 Static Quantization，MinMax/QDQ/int8 activation & weights/per-channel、weight symmetric=True、activation symmetric=False。校准 **384535f249a74a368a4ac78b9715feb6**：32 批/64 例；评估 **52047c8149c74443914b8805295fc22d**：8 批/16 例。均是明确的合成数据，映射 image→pixels，内容独立，未配置任务 evaluator，不虚构 accuracy。初次原生多选产生的 3 批小快照保留为历史，最终实验实际使用的是后续完整的 32 批快照。

原生实验 **bcf56d0c8b064ef3a5680d696b1420e8**，量化 Artifact **23e5e664b7054e369b652891eba8275d**：

| 指标 | 原始 | 量化 |
|---|---:|---:|
| 模型总字节，含 external data | 20,914 | 9,161 |
| 平均延迟 ms | 0.012540 | 0.027068 |
| P50 ms | 0.012400 | 0.026100 |
| P95 ms | 0.013000 | 0.030920 |
| requests/s，concurrency=1 | 79,744.82 | 36,943.99 |

两侧相同 ORT 1.31.0 CPU、1/1 线程、session optimization disable、warmup10/runs100、评估首批输入；conditions_hash/input_digest 一致。用户未声明 batch，所以没有 samples/s 指标。体积 **-56.20%**，延迟 **+115.85% / Regression**，不主张性能提升。微小合成 CNN 的 CPU 数据不能代表业务模型。

真实输出观测：4 QuantizeLinear、8 DequantizeLinear、2 INT8 权重（Conv `[16,3,3,3]`、Gemm `[144,32]`）。Conv/Relu/Gemm 有 QDQ 输入边界；不从这些边界声称所有内部计算都是 INT8。图从 5 到 17 个算子，增加 Q/DQ 并不等于复杂度回退或优化失败。实际 reader 日志 read_calls=32 / unique_batches=32，确认经过真实 Olive DataConfig 消费数据。

独立 8 批共 512 个输出元素：MAE **0.014226535973648424**、max absolute **0.2914562225341797**、max relative（排除零参考值）**5.197156715666615**；atol1e-5/rtol1e-4 的超容差比例 **99.609375%**。有损模型 allclose 未通过，执行有效；用户示例阈值 MAE≤0 的 Quality Gate failed，Experiment Succeeded，Benchmark/产物保留。没有把这个示例零误差阈值作为默认业务阈值。

原生 Export model & reports → Windows 目录选择 → `test-results/opweaver-bcf56d0c8b064ef3a5680d696b1420e8/`。导出原模型/量化模型/外部权重、32+8 批 NPZ/manifest、输入回放、实际 Olive 配置/footprint/日志/消费证据、report.json/report.md。用导出的两侧模型重新创建 ORT session，重算 8 批数据，MAE/max 与保存报告一致。

原生 Graph changes → View optimized graph，Model Inspector 实际显示 QDQ/Q4/DQ8/2 整数权重以及量化图。关闭应用并重新启动后，自动恢复 bcf56d0c 实验、原始 CNN 选择、32/8 批数据引用、误差、gate 和性能回退。原有 Synced 源码/设计图与 `139f2860`（30 epochs、480 steps、loss0.0564）的训练记录保持。

交付 `build/bin/OpWeaverStudio.exe`：**16,651,776 B**，2026-10-23 17:16:53，SHA256 **f44d128feb443a7db5ab9c491fc43f2bc6ec59afb88c1547ab3d985096dffcac**。Python/Worker/Olive 仍是外部运行时；随包 worker 与源代码逐文件校验。未验证干净电脑安装、签名包、所有 DPI 或 GPU。更多能力和边界见 [P1-B 实施记录](P1B_IMPLEMENTATION.md)。

## P1-A 验收（2026-10-23，UTC+8）

| 检查 | 实际结果 |
|---|---|
| Pytest / scripts/test.ps1 | **121 passed**：P0-C 原 100 项 + 21 项 Olive 参数化用例；另定向复查错误环境配置、真实嵌套解释器取消退出 |
| Vitest | **23 passed**，新增工作流预检期间取消正确停止新任务，不误操作上次实验 |
| Go / go vet | **9 tests / 3 packages passed**；既有应用集成测试增加真实 Olive 环境配置、原生工作流校验、保存、执行、验证、Benchmark 和导出 |
| Ruff / vue-tsc / production build | **通过**；46 个 Python 文件；仅保留既有 Monaco chunk 体积提示 |
| 安装脚本 / pip check | **通过**；显式独立 venv、CPU Torch、完整 freeze，No broken requirements |
| Playwright | **10 passed**，包括真实 Olive 全界面流程和 P0-A/B/C 回归；高级 JSON 的合法/非法配置亦纳入最终用例 |
| Wails desktop | **Windows amd64 构建通过**；真实界面配置、执行多 Pass、导出、关闭后重启恢复 |

实际 Olive **0.13.0** / Python **3.13.3** / Torch **2.11.0+cpu** / ONNX **1.23.2** / ORT **1.31.0** / onnxscript **0.7.2** / onnx-ir **1.1.0** / onnxoptimizer **0.4.2**。独立位置 `.tools/olive-env`；Worker 仍为原 `.venv` 的 CUDA Torch/ONNX 1.22.0/ORT CPU 环境，未安装 Olive。探测和执行均关闭 Olive 遥测，工作流无远程下载/上传或用户脚本。

本阶段 A–H：

- A：真实空 venv（带空格路径）Not Installed；坏路径 Environment Error；缺依赖/版本不兼容分类；真实安装 Available，CPU EP/版本/安装路径实际导入验证。
- B：真实 RunConfig 校验，有序两 Pass、禁用 Pass 不执行、参数/输出/源模型/序列化、保存与重开；非法 Pass、重复 ID、非法参数/类型、空工作流、shell/user_script hook 在任务创建前拒绝。
- C/D/E：真实 Olive workflow（未用 Mock 替代），Identity 冗余、外部权重 CNN、动态多输入/三输出（包含 bool）、NPZ 回放；共享三样本验证与 Benchmark，输入/环境/session 条件指纹一致。额外关闭优化开关，确认无图变化时不虚构转换；测试不要求一定加速。
- F：真实嵌套 Olive 解释器进程取消并确认 Windows 进程退出、实际任务超时；故障注入补充 Olive 报错、无结果文件、候选文件缺失，保留失败证据。原始模型/其他实验不变。已有 owner-watch/Worker kill-restart 测试保持通过。
- G：SQLite/磁盘恢复和导出重新加载；原生应用关闭重启后，实验配置、环境、数值与性能记录恢复，优化模型仍在共享模型库。
- H：原 Graph ↔ Code、拖动稳定性、训练/Checkpoint、ONNX/Inspector、ORT 四等级、Benchmark、Validation 与历史流程保持通过。

原生验收的项目仍为 `projects/model-mvkdmssb`；原先 Synced 源码/图和 `139f2860`（30 epochs / 480 steps）的训练记录保留。通过原生 Import model 导入 `examples/onnx/redundant-identity.onnx`，模型 Artifact `8482c9d0…`；选择 Microsoft Olive，检测独立环境，添加 Peephole → Opset Conversion(target_opset=18)，保存 `native_olive_acceptance`，再点击 Run optimization experiment。

首条原生实验 **ed77e877483441c68316f8fbb6c59784**，优化 Artifact **6cdede319bcd4749895c28c39f2ec428**：

- Olive footprint 真实记录两个已产生产物的 Pass；图从 3 个节点变为 1 个（两个 Identity 消失，保留 Relu）。
- 模型总文件体积 205 → 158 bytes；三组 `[2,4]` float32 输入，所有最大/平均误差为 **0**。
- 同一 Worker ORT CPU、session optimization=disable、单线程、warmup=10、timed runs=100，mean **0.008032 → 0.005721 ms**，该次观测 **1.404×**。这是小模型的单次序列测量，不承诺其他模型或重跑相同加速。
- 原生目录选择器导出到 `test-results/opweaver-ed77e877483441c68316f8fbb6c59784/`，含 report.json/report.md、原/优化 ONNX、3 个 NPZ、通用/生成/实际配置、Pass footprint 和完整原始日志。使用导出 bundle 单独重新加载两侧 ORT，三样本逐元素完全一致。
- 源模型 SHA256 `238e593dba0bcb9bd9eae44aa601b42d7ef95601c1131ece3060e0aceb966211`。历史和模型均未覆盖。

最终 exe：`build/bin/OpWeaverStudio.exe`，**16,636,416 bytes**，2026-10-23 **16:07:55**；SHA256 **f041e9568b9a41e353e3cd5f2b480aa7b4ef9bb39db9b098c046126c88d88224**。同目录带 Worker 源码，Python 和独立 Olive 环境仍是外部依赖。

该交付包再通过全部 **10 项内嵌前端 E2E（2.6 分钟）**，包括高级 JSON 合法/非法配置和保存模板的 NPZ 草稿同步/恢复；修复 Vue 响应式对象直接 structuredClone 的问题。随后在原生界面恢复 `native_olive_acceptance` 再执行，实验 **4c2f1573a317462dba9982a456b63dc1**，优化 Artifact **9510e31ddd2c44958ceb4d6ea153adfd**：两 Pass、3→1 节点、205→158 bytes、三样本误差仍全部为 0；mean **0.006965→0.005722 ms**，该次观测 **1.217×**。原生导出在 `test-results/opweaver-4c2f1573a317462dba9982a456b63dc1/`。两次运行的性能差异如实保留，不取代原实验。

再次关闭并重启交付包后，`4c2f1573` 最新记录、环境与 `native_olive_acceptance` 配置、两个实际执行 Pass 和原训练记录正常恢复；导出的模型/NPZ 另行重新加载，三组输出逐元素一致。测试用 HTTP/Vite 辅助进程均已停止，原生应用保留打开供查看。

边界：两个本地标准 ONNX Pass 和上述 CPU 模型有实测；Transformer 专用 Pass、量化/剪枝/校准、GPU/云端/自动搜索、NVIDIA ModelOpt 均未实现或验证。Linux/macOS 原生 GUI、大模型内存性能及所有 ONNX 算子覆盖不作为本轮完成项。详情见 [P1-A 实施记录](P1A_IMPLEMENTATION.md)。以下保留历史验收记录。

## P0-C 最终验收（2026-10-23，UTC+8）

| 检查 | 实际结果 |
|---|---|
| `scripts/test.ps1` / Pytest | **100 passed**，原 66 项 + 34 项 Lab 参数化测试 |
| Vitest | **22 passed**，包括任务隔离、终结事件防回退、重开配置、指标缺失/负数 |
| Go 真实 Worker 集成 / go vet | **9 tests / 3 packages passed**；App→导入→Baseline→ORT 优化→验证→Benchmark→模型重载→导出→实验比较 |
| Ruff check / format | **通过**，41 个 Python 文件 |
| vue-tsc / Vite production | **通过**；既有 Monaco chunk 大小警告保留 |
| 最终桌面 exe 内嵌资源的 Playwright | **9 passed，1.3 分钟**；新增两条 Lab 流程含 NPZ 重放 |
| `scripts/build.ps1` | **通过**，Wails 2.17.0 / Windows amd64，2026-10-23 14:54:56，exe 16,624,128 bytes |
| npm audit | **0 vulnerabilities** |
| 原生 Wails/WebView2 | **通过** Baseline、完整优化、多样本验证、原生目录对话框导出、关闭/重启后恢复 |

ORT **1.31.0 / CPUExecutionProvider**；保留 Python 3.13.3 / PyTorch 2.11.0 / ONNX 1.22.0。没有安装或依赖新的 CUDA 环境，没有下载模型或真实数据集。

P0-C A–H 覆盖：

- A：真实 CPU 推理逐次高精度采样；统计重新从原始样本计算，初始化/warmup/稳态分开，Batch 未声明时 samples/s 为 unavailable。
- B：Disable/Basic/Extended/All 均实际序列化并重载；普通 CNN 和 external-CNN 四个等级均验证。原文件及外部权重 Hash 不变，新图/权重拥有独立 Artifact 和 lineage/目标环境。
- C：3 组同输入、多输出数值比较；容差/零参考/NaN/有符号 Inf/空 Tensor/类型或 Shape 错误，uint64 大整数和 bool 精确比较，不用浮点近似替代。
- D：两个输入、三个输出（包含 bool），seeded random 和真实 NPZ 文件；UI 将已保存的 NPZ 输入重放到新实验。输入名称/Shape/DType/共享符号、动态维度未填、pickle 和恶意 NPY 长度头被拒绝。
- E：快/相同/慢、体积增大、零或缺失指标、条件不同，回退为 Regression；历史实验比较同样检查条件指纹，不强制正收益。
- F：SQLite 摘要、完整配置/环境/原始计时/错误/报告持久化，浏览器重开和原生关闭/重启均恢复；优化模型在同一 P0-B 模型库重开。
- G：不兼容自定义算子、错误输入、损坏优化文件、子进程异常退出、中断历史、实际推理中取消、实际 3s 超时、监督进程死亡、Windows 短暂文件共享锁。验证失败不产生 Passed 声明。
- H：既有 Graph ↔ Code、源码补丁/失败保留、连续拖动/光标、ONNX Inspector/搜索/分组/子图/外部权重/600 节点、空图建模→训练→重开→Resume 和 960×680 检查全部继续通过。

原生实测使用现有项目 `projects/model-mvkdmssb` 和已有 Artifact `a05d8cdc439f4c91b12b47a1af7f323e`，image Shape 显式 `[2,3,8,8]`，seed=42，warmup=10，runs=100，线程 1/1，双方 Session graph optimization=disable，artifact level=extended，验证 3 组、atol=1e-5/rtol=1e-4。

保存实验 **d4a4eef273e04b9f82af9380e0a7847f**，优化 Artifact **28e123dcaafa49b0bdfb932c9a83eb32**：

| 实测指标 | Baseline | Optimized |
|---|---:|---:|
| 文件合计 | 1201 B | 1596 B |
| ONNX 算子数 | 5 | 4 |
| Mean | 0.010687 ms | 0.010089 ms |
| P50 | 0.0105 ms | 0.0097 ms |
| P95 | 0.011905 ms | 0.01272 ms |
| 计时样本 | 100 | 100 |
| 输出一致性 | 参考 | 3 组通过；max/mean abs error=0 |

该次 Mean speedup 约 1.059×，体积 **增加 32.89%**，P95 比 baseline 更高；只是这次短模型/顺序 CPU 测量，不是普遍优化收益。未声明 Batch，界面没有推导 samples/s。真实图统计出现 com.microsoft::FusedConv；未根据节点数宣称必然加速或列出未观察到的 pass。

原生导出目录 `test-results/opweaver-d4a4eef273e04b9f82af9380e0a7847f` 包含完整 report.json/report.md、原始图+weights.bin、优化图+optimized.onnx.data、3 个实际输入 NPZ。独立从导出目录重新创建两侧 Session 并运行保存输入，输出验证再次通过。重启桌面后同一实验/配置/指标恢复；原 Synced Design Graph 和 30 epochs / 480 steps 训练记录仍存在。

曾修复的实际阻塞：ORT 默认序列化保留旧 external data 相对引用；Windows venv 仅杀 launcher 导致解释器继续运行；高频报告替换遭遇短暂共享锁；warmup 计时排除进度持久化并节流采样期间日志。均有针对性检查，最终全套通过。

当前边界：ORT CPU graph optimization；数值一致性只覆盖记录样本。量化/剪枝、Olive/NVIDIA Adapter、GPU/并发/内存 Benchmark、任务数据集精度、复杂图 Diff 和 Linux/macOS 原生 GUI 均未作为本轮完成项。部署到其他电脑仍需外部 Python 环境。详细模块/约束见 P0C_IMPLEMENTATION.md。以下是历史阶段记录。

## P0-B 最终自动验收（2026-10-23，UTC+8）

| 检查 | 结果 |
|---|---|
| Pytest | **66 passed**，含新增 17 项模型导入验证和原 P0-A/训练测试 |
| Vitest | **17 passed**，含分组/方向遍历、失败保留、过期项目事件隔离 |
| Go `OPWEAVER_INTEGRATION=1 go test -count=1 -v ./...` | **9 passed / 3 packages**，含 App→真实 Worker→ONNX 子进程导入/加载/路径拒绝 |
| Ruff check / format、go vet | **通过** |
| vue-tsc + Vite production | **通过**，Monaco 体积警告仍存在，无构建错误 |
| Playwright，最终桌面 exe 的内嵌生产资源 | **7 passed，约 1.1 分钟** |
| Wails 2.17.0 Windows amd64 / scripts/build.ps1 | **通过**，复制最新 Worker 和全部离线模型 |
| npm audit | **0 vulnerabilities** |

新增 ONNX 1.22.0。CPU 完成全部 P0-B 测试，无 ONNX Runtime/CUDA 依赖。外部权重只读元数据/流式复制，不把 Tensor 内容写到前端或 SQLite。

任务书 A–G 的实际覆盖：

- A：离线 CNN Conv/Relu/MaxPool/Flatten/Gemm，5 个算子、11 个可视节点、10 条边；输出 `[B,3]`，Pool `[B,2,3,3]`。4 个 Initializer、113 元素、452 逻辑字节，源文件 Hash/复制件字节一致。
- B：Identity 分支和 Add 汇合，验证 Add 两个真实生产者，重复操作数和多输出 Split 的 out_1 端口不误连，省略输入不产生假边。
- C：`[B,None,4]` 保留符号与未知维度；UI `[B,?,4]`，不填入假定 batch/维度。
- D：com.example::Fancy 和未知标准算子保留属性/Domain/边，显示 UNKNOWN_OPERATOR。Checker 完整性和执行支持分开，不伪称已做推理。
- E：损坏 protobuf、缺 external data、路径穿越、越界字节范围、未定义输入结构均失败。UI 显示错误并保留上一有效图，主 Worker 可继续导入。
- F：P0-A Visual 往返、ReLU→GELU、精确源码局部修改、失败保留、鼠标回归继续通过；原空白图手工连线→Forward→训练→重开→Resume 和 960×680 检查继续通过。统一 PyTorch Importer 复用 FX，Open Python 回到 Monaco/映射。
- G：导入后重载，Artifact/报告/源 Hash/历史均保留；原模型未修改。SQLite 仅保留任务/Artifact 摘要，报告和权重为独立文件。
- If 两个独立子图及外层 x 捕获、真实 external-cnn+weights.bin、600 节点模型自动折为 3 组并可搜索最后节点；重新加载大模型仍默认分组。所有 7 条 E2E 检查 pageerror 或原有错误断言。

曾修复：子图名称被输出 Tensor 局部变量覆盖；分析图视口渲染误影响 Design Graph 的节点 DOM；PyTorch 算子数误包含输入/输出；重开大图时分组监听未立即执行。增加/保留对应验收，未通过放宽测试掩盖。

截图：`test-results/p0b-model-inspector.png`。限制：256 MiB protobuf / 20,000 分析节点 / 24 MiB 报告 / 120s 超时；确定性网格布局、会话布局缓存，属性有明确摘要上限。ONNX 子图只读，Python 独立源码未自动打包依赖/权重；没有 ONNX 推理、优化、Benchmark 或无损 Python 重建。

以下 P0-A 结果为上阶段记录，当前 P0-B 结果以上表为准。

日期：**2026-10-23（UTC+8）**。以下结果来自本轮真实执行，不是预设示例指标。

## 本轮 P0-A 最终结果

| 命令 | 结果 |
|---|---|
| `scripts/test.ps1`：Pytest | **49 passed**，其中 Code/FX/AST/生成/捕获测试 16 项 |
| Ruff check / format | **通过** |
| `npm test` | **12 passed** |
| `npm run build` | **通过**，vue-tsc + Vite production；Monaco 体积警告，无构建错误 |
| `OPWEAVER_INTEGRATION=1 go test -count=1 -v ./...` | **8 passed / 3 packages**，真实 Worker 无缓存验证 |
| `go vet ./...` | **通过** |
| `npm run test:e2e`，最终 Go 内嵌 production 资源 | **4 passed，50.1s** |
| `scripts/build.ps1` | **通过**，Wails 2.17.0 Windows amd64，复制最新 Worker/示例 |
| `npm audit --audit-level=low` | **0 vulnerabilities** |

本轮环境仍为下文的 Windows/Python 3.13.3/PyTorch 2.11.0+cu130/Go 1.27.3/Node 20.19.2。新增 Monaco 0.57.0，没有引入 ONNX/ORT 等 P0-B/C 依赖。全部 P0-A 自动验收使用 CPU。

### 任务书 A–D 和鼠标回归

- A：实际残差模型包含 Linear、LayerNorm、ReLU、Add，FX 节点和精确构造映射；另测 Sequential Conv2d/BatchNorm/GELU/Concat、构造参数传递及原模型/FX 输出比较。
- B：实际在 Monaco 搜索并将 `nn.ReLU()` 改为 `nn.GELU()`，保存、解析；同一节点更新为 GELU，重载后 Design Graph 和 Synced 状态一致。
- C：生成 MLP 和残差代码、真实加载 Forward、重新捕获；图端末层参数 3→5，经 diff 应用、解析一致。Code-First 精确局部补丁改 head.out_features，保留无关 helper，Tensor Inspector 实际输出 `[4,5]`，重载保留分析图。
- D：动态 Tensor 条件触发 FX 明确错误；源码不被覆盖，前版有效图保留，显示 Parse failed · graph stale。未知 sin 节点保留可见、分析只读。打开文件的副作用 marker 测试确认 AST 不执行模型。
- 捕获超时和 Stop 杀死子进程，随后任务可正常执行。Windows CRLF/BOM 保留原始内容 Hash；构造赋值歧义、复杂表达式、磁盘 Hash 冲突、非法算子参数拒绝改写。
- 自定义 Square 注册验证真实 Autograd、代码生成和 FX 捕获，不使用假节点。
- 鼠标测试每次移动 3px、连续 35 次取样：位置不出现超过 1px 的反向回跳，单步跳动小于 12px，松开偏差小于 2px，总位移与 105px 目标差小于 5px。检查拖动期间子元素边界保持 grabbing 光标；取消实时 20px 吸附、拖动中旧坐标回写及无变化布局历史。
- 原有空白图手工连线→Forward→CPU 训练→重载→Resume 流程继续通过，并检查 960×680 布局和 pageerror。两个代码 E2E 均断言无浏览器运行时错误。
- 最终视觉检查修复 Code-First 误显示空白 Design Graph 问题计数；Problems 改用当前源码诊断并支持定位。Tensor Inspector 去掉固定 batch 4 文案。新增 Code-First 无假问题的断言，最终四条 E2E 全部再通过。

截图：`test-results/p0a-split.png`、`p0a-stale-graph.png`、`studio-trained.png`、`studio-960.png`（均 gitignored）。测试通过后没有增加伪造 Benchmark 或优化页。

### P0-A 原生 Wails/WebView2 冒烟

关闭旧的已保存桌面窗口后运行最新 exe，确认原项目、布局和 30 epochs 的训练记录仍可恢复。实际点击 Split、生成代码、Monaco diff 预览及 Apply、Parse graph；桌面显示 **Synced**，生成 src/model.py 已保存。独立捕获报告为 COMPLETED、195 参数、真实输出 `[4,3]`（float32），原模型与 FX 在该输入上的 atol=1e-6/rtol=1e-5 对照通过。Monaco 编辑和 ReLU→GELU 的键盘输入链路另由 production 浏览器 E2E 完整验证；没有把它冒充成全部 WebView2 输入法/DPI 组合的验证。

### 当前限制

该 P0-A 历史阶段只完成 P0-A；P0-B/P0-C 的当前结果见本文前面的验收记录。Olive/NVIDIA/Kernel 仍未实现。FX 仅 CPU/eval、具体示例输入与静态构造分支，不证明任意输入下等价；动态 Python、任意广播、复杂输出或未知操作不强行投影为可编辑图。Code-First 不直接训练、生成代码不迁移已有权重。

源码编辑为项目 src 下 UTF-8 Python 文件；多包/相对导入项目和第三方插件加载 UI 未完整实现。独立进程不是安全沙箱，超时不是对用户代码后代进程的 OS 作业级资源约束。Python/PyTorch 仍为外部运行时。系统级鼠标驱动、远程桌面光标、全部 DPI/多屏组合没有在本轮验证；鼠标回归范围是应用画布拖动/光标及编辑器定位。

## 先前基线验收（保留原始记录）

下文 33/7/6/1 测试数是 P0-A 开发前的基线，当前结果以上表为准。下文 P0/P1/P2 等为旧任务书阶段编号，当前路线见 ROADMAP.md。

## 环境

- Windows，Python 3.13.3，PyTorch 2.11.0+cu130，Pydantic 2.14.1。
- Go 1.27.3，Wails 2.17.0，Node.js 20.19.2。
- 默认和全部核心训练回归使用 CPU。
- 检测到 NVIDIA GeForce RTX 4060 Laptop GPU；另有独立 CUDA 冒烟验证。

## 自动测试与构建

| 命令 / 测试组 | 结果 | 覆盖 |
|---|---|---|
| `python -m pytest -q` | **33 passed** | IR 往返、Schema 无漂移、错误节点/端口/环/Shape/Dtype/参数/算子、运行时端口契约、拓扑、共享参数、多输入输出、自定义多输出插件、Autograd、Optimizer、DCE/回退、训练/停止/恢复/错误 |
| `npm test` | **7 passed** | Pinia 增删/参数/边/Undo/Redo/Save/Load，验证错误，Inspector 真正同步 IR，曲线不伪造数据 |
| `OPWEAVER_INTEGRATION=1 go test ./...` | **6 tests passed / 3 packages** | 原子项目修订、恢复连接/布局、目录穿越/坏布局拒绝，真实 Worker 协议错误/kill/restart，打包资源目录的 Python 定位，本地服务 Host/Origin 检查 |
| `go vet ./...` | **通过** | Go 静态检查 |
| `ruff check` / `ruff format --check` | **通过** | Python lint/格式 |
| `npm run build` | **通过** | vue-tsc 类型检查 + Vite production |
| `npm run test:e2e` | **1 passed** | 从空白画布开始的完整真实后端流程，约 17s |
| `scripts/build.ps1` / `wails build` | **通过** | Windows amd64 原生桌面 exe 和资源复制 |
| `npm audit` | **0 vulnerabilities** | 本轮已安装依赖的 npm advisory 检查 |

GitHub Actions CPU CI 已提交配置，但没有推送仓库或声称远程 CI 已运行。

## E2E 的实际步骤

1. 启动构建后的 Go 本地服务和独立 Python Worker。
2. UI 创建 Empty graph 项目，拖入 Input，添加 Linear/ReLU/Linear/Output。
3. Inspector 编辑第二个 Linear 的 in_features=16、out_features=3。
4. 真实鼠标拖动四条端口连接，验证 Graph is valid、5 nodes / 4 edges。
5. Shift 多选两个节点，验证选中态。
6. Save，验证 All changes saved。
7. Forward，验证 `[4, 3]` 真实输出。
8. Train，等待 COMPLETED，检查真实曲线和 Console 的 Epoch 20/20 日志。
9. 重载页面，从持久化项目恢复四条连接、实验记录。
10. 从 Checkpoint 恢复至总计 22 epochs，检查恢复日志从 epoch 21 开始。
11. 检查 960×680 窗口无文档水平溢出、画布仍有可用面积、主操作可见；浏览器没有 pageerror。

本地截图（gitignored）：`test-results/studio-trained.png`、`test-results/studio-960.png`。失败时 Playwright 保存 trace，不在代码中填入截图曲线。

## 原生桌面实测

使用实际 Wails/WebView2 窗口完成：

- 创建项目并保存图/布局。
- Forward：**195 参数，输出 [4, 3]**，Tensor Inspector 显示实算 min/max/mean。
- CPU 训练 20 epochs / 320 steps：最后 batch **Loss 0.0092，训练 batch Accuracy 100%**；观察实际 Loss 下降曲线和 Checkpoint。
- 关闭应用，再次启动；自动恢复已保存模型和 SQLite 中 COMPLETED 记录。
- 点击 Resume，读取保存的 Checkpoint，继续至 **30 epochs / 480 steps**；最后 batch Loss 约 0.0564。单 batch Loss 有波动，不把它称作验证集精度。
- 保存界面中的布局调整，再关闭以更新构建。

## 数值和恢复验证

- 离线 `scripts/demo.py --residual`：99 参数，Forward [4, 3]，12 epochs / 192 steps。首 epoch 平均 Loss **0.857805**，末 epoch 平均 Loss **0.099363**。
- CPU 中断与连续训练比较：相同 seed、图、Optimizer、数据条件下，最终全部参数 `torch.equal`。
- 另用带 Dropout 的图，在 epoch 中途第 2 step 取消，再恢复；最终参数与不中断训练逐项完全一致。
- MSE/CrossEntropy × SGD/AdamW 四种组合完成训练。
- 无 CUDA 分支通过模拟 `is_available=False` 验证：CPU 仍可训练，CUDA 请求返回明确错误；这不等于在另一台物理无 GPU 主机上实测。
- 本机 CUDA 冒烟：MLP、2 epochs、64 samples、4 steps，COMPLETED，checkpoint 保存成功。没有声称测过多 GPU、分布式或 CUDA 设备迁移恢复。
- 模拟 SQLite 写入失败仍能向客户端报告 FAILED；不把任务留在永久 RUNNING。

## 曾发现并已修复的问题

- Wails 2.10.2 与 Go 1.27 的绑定生成失败；固定到 Wails 2.17.0 后构建和原生运行通过。
- Schema 漂移测试最初的 fixture 相对路径多向上一级；修正后测试通过。
- 旧 Vitest 测试依赖存在 advisory；更新到 4.1.12，并锁定兼容 Node 20 的 Vue test-utils 后 audit 为零。
- Inspector 单节点选中状态可能覆盖画布多选；拆分 Inspector 当前节点与画布 selection，并增加 Shift 多选 E2E 验证。
- Worker 管道需要在 Wait 前完成 stdout 读取；修正终结响应读取和协议超大消息恢复路径。
- 从带资源的 build/bin 目录直接启动时，原实现未查找仓库虚拟环境。增加向祖先目录查找 .venv，并用独立 Go 回归测试固定该行为。

## 尚未验证 / 当前限制

- Linux/macOS 原生 GUI 构建、Windows 安装包/签名、干净电脑部署、远程 CI 未运行。
- 此历史基线验收时尚无优化工作区；当前 ORT CPU 优化/导出/Benchmark 见本文 P0-C 记录。Quantization/其他编译后端仍未实现。
- 原生窗口大小通过 CSS/浏览器 960×680 验证；未覆盖所有 Windows DPI、多显示器、辅助技术组合。
- 仅单训练任务、float32、合成数据二维单输入/输出；不提供 train/validation split，Accuracy 是当前训练 batch 指标。
- 默认事件/SQLite 保存完整有界快照，长实验存储效率有限；不承诺高吞吐指标平台。
- 图/布局旧修订未自动回收；SQLite Checkpoint 路径是绝对路径，项目移动后的重定位未实现。
- Checkpoint 恢复要求完整 IR 一致；没有 UI 一键还原历史图。跨设备/版本不保证 bitwise 一致。
- PyTorch/Python 未打包进 exe；第三方插件无沙箱和加载 UI。
- 没有把原生关闭过程、磁盘断电或任意恶意插件当作已完全验证的场景。

### P0-B 原生桌面实测

实际在新 Wails/WebView2 窗口通过 Project Explorer 的 Import model 选择 external-cnn.onnx，使用注册表探测格式并导入。Inspector 显示 Tiny CNN、5 算子、4 Initializer/113 元素、452 逻辑字节、weights.bin；主文件 749 B、合计 1201 B。Checker passed，ONNX 未执行推理，源 Hash 与原文件一致。

关闭并重新启动桌面，原 30 epochs 训练记录、Synced 设计图和 Tiny CNN Artifact 均保留；从 Explorer 再次打开已存分析。此前模型/代码均为 All changes saved。浏览器自动测试另验证节点属性、搜索/分组/子图/大图重载和完整训练恢复；没有宣称覆盖所有 DPI/输入法/大规模 LLM。
