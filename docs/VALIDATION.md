# 验收记录：P0-B、P0-A 与已有基线

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

本轮最终指令是先完成 P0-A，因此 ONNX 导入（P0-B）、ORT 优化/数值误差/Benchmark 闭环（P0-C）、Olive/NVIDIA/Kernel 尚未实现。FX 仅 CPU/eval、具体示例输入与静态构造分支，不证明任意输入下等价；动态 Python、任意广播、复杂输出或未知操作不强行投影为可编辑图。Code-First 不直接训练、生成代码不迁移已有权重。

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
- P2–P5 产品功能未实现；优化工作区未执行 quantization/compile/export/benchmark。
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
