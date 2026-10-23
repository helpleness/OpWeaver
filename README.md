# OpWeaver Studio · 算织

**Visual AI Model Engineering IDE** — *Weave Models. Forge Kernels.*

面向模型工程的本地桌面 IDE。当前完成 **P0-A Graph ↔ Code 双向协作** 和 **P0-B 真实模型导入/分析**，保留已有图设计、训练和 Checkpoint 功能。

客户端：**Wails 2 + Go + Vue 3 + TypeScript + Vue Flow + Pinia + Monaco**。计算：**独立 Python Worker + PyTorch Eager/FX**。CPU 即可完成基础流程，不需要下载数据集。

## 已经实现

- 独立、版本化 Graph IR，语义图与 UI 布局分别保存；JSON Schema 自动生成 TypeScript 类型。
- 15 个内置算子：Input、Output、Linear、Conv2d、ReLU、GELU、BatchNorm1d/2d/3d、LayerNorm、Add、Multiply、Concat、Flatten、Dropout。节点参数编辑、Shape/Dtype/Device 检查、拓扑排序、循环及缺失依赖诊断。
- 画布拖入节点、多端口连线、分支与残差、平移缩放、框选、多选、删除、Undo/Redo、Minimap。连续拖动、稳定系统光标；网格保留显示，不再实时吸附造成跳动。
- Graph / Split / Code 三种视图，可调分隔条、可隐藏 Inspector；Monaco 真正编辑项目 `src/*.py`，支持多文件、保存、搜索、行号、语法高亮、错误和源码跳转。
- AST 静态入口发现；显式 FX 捕获 + ShapeProp + 原模型 Forward 对照，支持 Sequential、嵌套模块、分支/残差。保留未知节点，动态控制流失败时保留上版有效图。
- Design Graph → 可执行可读 PyTorch 类 → FX 重新解析；稳定节点 ID、Source Mapping 双向定位、修订/内容 Hash/同步状态。
- Visual Authoring 生成代码先审阅 diff；手写代码不允许整文件再生成覆盖。Code-First 图只读，对可精确定位的直接构造参数提供局部 AST 补丁，保留注释和无关代码。
- IR → `torch.nn.Module`，注册子模块/参数、Autograd、多输入/多输出、参数共享。
- 真实 Forward 和 Tensor 统计，CPU 默认训练，可选检测到的 CUDA；SGD/AdamW、MSE/CrossEntropy、Epoch/Batch/LR/Seed。
- 实际 Loss/Accuracy/LR/Step/Epoch/时间、Worker 日志、SQLite 实验记录。
- 每个 epoch 和停止时保存 Checkpoint；恢复模型、Optimizer、随机数、epoch/batch/step 和指标。
- 项目保存/加载、最近项目恢复、Worker 错误隔离和意外退出后的重新连接。
- OperatorPlugin 基础接口、PassManager、DCE 与 Eager 数值验收/回退接口。
- 统一 Model Importer 注册入口：真实 ONNX ModelProto、PyTorch 源码（复用原 FX 捕获），独立 ModelArtifact/Metadata/Diagnostics/GraphImportResult。
- ONNX IR/Opset、节点/属性/多输出/可选输入、Shape inference、符号/未知维度、Initializer、External Data 原样复制与 Hash；未知算子保留。If/Loop/Scan 属性子图可独立打开，控制流不扁平化。
- Model Imports 工作区和 Model Inspector：节点搜索定位、上下游高亮、模块/算子分组、展开、视口按需渲染、布局缓存、可调 Inspector、统计/权重/诊断、导入历史和项目重开。

**范围边界**：Code-First 当前执行 CPU/eval 捕获与前向，不能直接连接训练系统；Visual Design Graph 保留已有训练。合成训练只接受单个二维 float32 输入/输出；模型构建器支持多输入/多输出。Concat Design 节点有两个输入，不自动支持任意广播或 Python 控制流。捕获结果不等于导出或权重迁移，数值对照只覆盖配置的示例输入。ONNX 图只读，不执行推理，也不还原 PyTorch 源码。P0-C 优化/Benchmark、第三方插件加载 UI、Kernel 和 LLM 尚未实现。

打开/导入 Python 文件只做 AST 检查。点击 **Parse graph** 才在可取消、超时的独立子进程中执行用户代码；它拥有当前用户权限，**不是安全沙箱**。全部模型默认本地处理。

## Windows 11 快速启动

需要 Python 3.10+（推荐 3.13）、Node.js 20.19+、Go 1.25+、WebView2 Runtime。项目固定 Wails **2.17.0**；本轮实测 Python 3.13.3、PyTorch 2.11.0、Go 1.27.3、Node 20.19.2。

在仓库根目录运行：

```powershell
# 安装依赖；Go 缺失时下载到仓库 .tools，验证官方 SHA256
.\scripts\bootstrap.ps1 -DownloadGo

# 已有可信 PyTorch 安装时可创建复用系统包的 venv，避免重复下载
# .\scripts\bootstrap.ps1 -DownloadGo -UseSystemTorch

# 原生桌面开发（前端热更新）
.\scripts\dev.ps1

# 生成 Windows 桌面程序与 Python/示例源码资源
.\scripts\build.ps1
.\build\bin\OpWeaverStudio.exe
```

当前工作目录已经准备好 `.venv`、`.tools` 和前端依赖，可直接运行构建产物。**exe 没有内嵌 Python/PyTorch**；发布到另一台电脑时需安装运行环境并设置 `OPWEAVER_PYTHON`，详见 [开发文档](docs/DEVELOPMENT.md)。

## 跑通第一个模型

1. 启动应用；默认展示 Synthetic classifier 示例。点击 **Save**，选择项目名称和新的空目录，Starting point 选择 **Current graph**，创建项目。
2. 点击 **Forward**，Tensor Inspector 应显示 `[4, 3]` 输出和真实统计。
3. 打开 **Training** 配置 Epoch/Batch/LR/Optimizer/Loss/Device；点击 **Train model**。底部显示真实曲线，Console 显示 epoch 日志。
4. Checkpoint 自动写入项目的 `checkpoints/`。运行中 **Stop training** 会在 batch 边界停止并保存状态。
5. 关闭并重新打开应用；最近项目和 SQLite 历史会恢复。点击 **Resume**，将 **Total target epochs** 增加到已完成 epoch 之后继续。

从空白图开始：New → Empty graph，依次拖入 Input、Linear、ReLU、Linear、Output；首个 Linear 配置 `8→16`，第二个 `16→3`，从右侧输出端口连接到下一节点的左侧输入端口。等到 **Graph is valid** 后保存、Forward、Train。也可点击库中算子直接添加。Models 库提供带 Add 的残差示例。

注意：训练使用开始时的图快照。训练中编辑图不会改变正在运行的模型；恢复 Checkpoint 要求当前 IR 与保存的图完全匹配。保存的 `.graph.json` 快照位于 Checkpoint 旁边。

## Graph ↔ Code 使用

1. 保存项目后点击 **Code** 进入双视图。Visual 模式点击 **Graph → Code · review diff**，选择新文件名，审阅并应用差异。
2. 检查模型入口、Constructor JSON 和 Example inputs，点击 **Parse graph**。默认 MLP 输入为 `[4,8]`；生成后解析应显示 **Synced**。
3. 图中修改末层 `out_features`，再次生成、审阅、应用并解析。手工将 `nn.ReLU()` 改成 `nn.GELU()`，保存后解析，原节点 ID 保留；成功反向同步会保存设计图。
4. 导入 [residual_model.py](examples/pytorch/residual_model.py)，选择 `ResidualModel`，保持默认输入并解析。Code-First 展示真实只读分析图，点击节点定位源码。Inspector 可对精确映射的 `head.out_features` 预览并应用局部补丁；重新解析确认 Shape。
5. [sequential_cnn.py](examples/pytorch/sequential_cnn.py) 使用输入 `[2,3,8,8]`。[dynamic_control_flow.py](examples/pytorch/dynamic_control_flow.py) 用于验证捕获失败、诊断和保留上版有效图。

外部修改会触发 Hash 冲突，不覆盖磁盘文件；使用编辑器的 **Reload file from disk** 再处理冲突。捕获记录和日志保存在 `.opweaver/captures/<id>/`，同步记录保存在 `.opweaver/code-state.json`。更多限制和实现依据见 [P0-A 实施记录](docs/P0A_IMPLEMENTATION.md)。

## 导入真实模型（P0-B）

1. 创建/打开项目，在 Project Explorer 点击 **Import model…**，或切换 **Model Imports**。
2. 选择本地 `.onnx` / `.py`，点击 **Detect format / inspect entry points**。格式识别和解析由后端注册的 importer 决定。
3. ONNX 点击 **Import and analyze**；Python 选择类/工厂、Constructor 和 Example inputs 后点击 **Import and capture Python**。Python 此步会显式执行源码；`.pt/.pth` 默认拒绝自动反序列化。
4. 在画布查看真实结构，点击节点查看属性、各输入/输出 Tensor 和相关权重。使用 Search、Group、Trace、Subgraph 选择框；点击 Overview 查看统计、算子分布和 Diagnostics。
5. **Re-analyze saved model** 创建新的分析记录；项目重开后模型下拉框和 Explorer 仍可加载已有记录。Python 分析可通过 **Open Python** 回到原有 Monaco/Source Mapping 工作区。

离线示例在 [examples/onnx](examples/onnx)：CNN、残差、动态维度、自定义 Domain、多输出、If、外部权重和 600 层图。可用 `python scripts/generate_onnx_examples.py`（先设置 PYTHONPATH=worker）重建，不下载权重。`corrupt.onnx` 是故意损坏的负例。

ONNX **Initializer elements 包括常量，不等于可训练参数**；逻辑 Tensor 字节数与实测文件大小分别展示。FLOPs 显示未测量；schema 可识别不代表推理后端支持。外部数据保留相对路径，缺失、越界、目录穿越或逃逸符号链接会明确失败，保留上版有效分析。

当前限制：protobuf 主文件最多 256 MiB（可配合 external weights），全模型最多 20,000 个分析节点，报告最多 24 MiB，任务最长 120s；超过限制明确失败，不发布部分成功图。大型属性数组/字符串只显示带截断标记的摘要，原文件保持完整。PyTorch 导入目前针对独立源码文件，未自动收集多包依赖或序列化权重。完整边界见 [P0-B 实施记录](docs/P0B_IMPLEMENTATION.md)。

## 自动化验证

```powershell
# Python + Vue/Pinia + TypeScript/Vite build + Go/真实 Worker 集成 + go vet
.\scripts\test.ps1

# 不启动 GUI 的真实 CPU 示例
$env:PYTHONPATH = 'worker'
.\.venv\Scripts\python.exe scripts\demo.py --residual

# 端到端 UI：终端 1 启动本地开发适配器
.\scripts\dev.ps1 -Mode server
# 终端 2（需要已安装 Chrome）
cd frontend
npm run test:e2e
```

E2E 覆盖空白建模→训练/恢复、P0-A 双向同步/局部改写/失败保留/鼠标、P0-B 导入/Inspector/搜索/分组/自定义算子/动态维度/子图/外部权重/600 节点/持久化/Python 复用；另检查 960×680 窗口和浏览器运行时错误。实测结果见 [验收记录](docs/VALIDATION.md)。

## 目录

```text
app.go / main.go          Wails 宿主与统一 API
internal/worker/          子进程、RPC、取消、重启
internal/project/         原子 manifest、图/布局修订、源码 Hash 冲突检查
frontend/src/             Vue Flow、Monaco、Pinia 同步状态、真实指标 UI
worker/opweaver/          IR、注册表、FX/AST/代码生成、捕获任务、训练、Pass
schemas/                 共享 JSON Schema（类型源）
examples/                MLP / Residual 图与 PyTorch 离线源码示例
scripts/                 安装、开发、构建、测试、示例
docs/                    架构、协议、扩展、路线和验收说明
```

进一步阅读：[架构](docs/ARCHITECTURE.md) · [Graph IR](docs/GRAPH_IR.md) · [Plugin API](docs/PLUGIN_API.md) · [协议](docs/PROTOCOL.md) · [路线](docs/ROADMAP.md) · [Windows/WSL2 开发](docs/DEVELOPMENT.md)。

许可证：[Apache-2.0](LICENSE)。
