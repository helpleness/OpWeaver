# 架构与实施清单（基线 + P0-A）

## 执行边界

```text
Vue + Pinia + Vue Flow + Monaco
        │  Request(method, JSON) / worker:event
Wails / Go App
        ├── ProjectStore → filesystem (manifest, graph revisions, UI layout)
        ├── Workspace → Python source + content hash / code-state
        └── Worker Client → isolated Python child process
                              ├── Pydantic Graph IR / OperatorRegistry
                              ├── Validation → GraphModule (nn.Module)
                              ├── AST / Codegen / Source Mapping
                              ├── CaptureTasks → disposable FX/Forward child
                              ├── TrainingEngine → PyTorch Eager (CPU/CUDA)
                              ├── RunStore → SQLite metadata/experiments
                              └── artifacts → checkpoint + graph snapshot
```

Go 不实现 Tensor、Autograd 或训练计算；GPU 调用全部位于独立 Worker。桌面 WebView 仅持有图 JSON、参数、统计数据和路径。大型 Tensor/模型权重留在 Python 与文件系统，不经 Go RPC 反复复制。

## 实施清单与当前状态

| 阶段 | 交付 | 状态 |
|---|---|---|
| P0 | Go/Wails + Vue/Vite 基础、独立 Worker、版本化通信 | 完成 |
| P0 | Graph IR、JSON Schema、TypeScript 生成类型 | 完成 |
| 基线 / P0-A | 15 个算子注册、DAG 验证、Shape/参数/端口诊断、代码生成入口 | 完成 |
| P0 | 基础画布、Inspector、Undo/Redo、项目保存加载 | 完成 |
| P1 | 真实 Eager 模型、多输入输出、Add 残差、共享参数 | 完成 |
| P1 | 合成训练、状态、日志、指标、停止 | 完成 |
| P1 | SQLite 历史、Checkpoint、应用重开后恢复 | 完成 |
| P1 | 核心自动化测试、真实 Worker 集成、桌面/UI 验收 | 见 VALIDATION.md |
| P0-A | Monaco、源码文件、Graph ↔ Code、Source Mapping、同步/冲突、FX 子进程与 Forward 验证 | 完成 |
| 新 P0-B / P0-C | ONNX 导入、ORT 优化/Benchmark | 尚未实现，下一阶段 |
| 新 P1 / P2 | Olive、NVIDIA、外部插件、子图、Kernel、大模型 | 路线已记录，未实现 |

## 权威数据与同步

Python `ir.py` 的 Pydantic 模型是 Design IR 的唯一类型源；`source.py` 定义 AnalysisGraph 和 SourceMapping。`scripts/generate_schema.py` 生成这三者和 TrainingConfig 共四个 JSON Schema；`npm run schema` 生成对应前端类型。Go 对 Graph 使用 `json.RawMessage`，避免维护第三份手工 DTO；保存和加载前调用 Worker 解析。语义错误允许保存未完成模型，Schema 格式错误拒绝保存。布局独立版本化。

Pinia 保存语义图与独立 layout。Vue Flow 节点是由 IR/布局派生的视图，画布临时属性不进入 IR。参数变更触发 250ms 去抖验证；请求带本地修订计数，过期结果不能覆盖新图。连线先本地拒绝自环、重复输入与环路，再由 Worker 验证 Shape/Dtype。训练前强制验证并保存快照。

## Worker 与调度

### P0-A 图和代码协作

Design Graph 表达受支持、可编辑和可执行的结构；FX Analysis Graph 独立记录原始 FX 节点/边/属性/Shape/Dtype/来源。两者复用 GraphCanvas/OperatorNode，不把 Vue Flow 的临时状态写回 IR。Analysis origin 为 `pytorch.fx`；`onnx` 只预留类型值，没有 ONNX 导入器。

`source.py` 的 AST 检查不执行文件。用户选择顶层模型类/工厂、Constructor JSON 和具体 Example inputs，点击 Parse 才启动 `code_tasks.py` 管理的独立 `code_runner` 进程。它用实际安装的 PyTorch FX symbolic_trace/ShapeProp；自定义 Tracer 记录调用位置，并通过 registry 识别受支持 leaf module。原模型 eval Forward 和 FX Forward 使用相同示例输入比较输出；不一致时禁止投影为可编辑 Design Graph。此校验不证明所有输入下等价。

FX 比 torch.export 更适合这一阶段可读的 module path、leaf 类型和源码关联，因此本轮选择 FX。未实现 torch.export/ATen 导入。未知 FX 操作仍留在分析图，只有全图可表达、语义验证通过时才产生 Design Graph。动态数据控制流追踪失败返回诊断，保留源文件和上一张有效图，不发布不完整成功图。

`codegen.py` 从已验证的 Design Graph 和 registry 回调生成正常 nn.Module、构造函数、forward 和分支。惰性元数据保存节点 ID/输入/表达式/输出身份，反向捕获保留 ID 与布局。生成的是模型结构，不携带或转换旧模型权重。可执行 IR 和生成器当前是 float32 能力；分析允许更多 dtype，但不伪称可编辑。

Visual Authoring 以 Design IR 为主；每次 Graph → Code 都展示 Monaco diff，只有仍匹配上次生成 Hash 的文件可再次整文件生成。手工改动后必须使用新文件名或局部补丁。Code-First 以源码为主，图结构只读；`patch_parameter` 只改当前类 __init__ 中唯一、直接、字面量 nn 构造参数，拒绝重赋值、复杂表达式、别名不明确和展开参数。保留其余源码字节；应用前 Worker 校验算子参数，Go 校验磁盘 Hash，应用后需重新捕获验证实际行为。

SourceMapping 保存 file/start/end/module_path/node_id 和 exact/inferred/unmapped；只有确切定义可局部改写。一个位置可映射多个算子。源码光标选择与图点击导航有方向标记，避免跳行反馈环；拖动期间使用 Vue Flow 实时坐标，仅结束时提交布局，取消 20px 实时吸附，避免选中态触发回跳。

`useCode` 管理多文件未保存缓冲、修订、Hash 和 synced/pending/parsing/parse_failed/stale/conflict/read_only。磁盘变更使用 expected_hash 拒绝覆盖；diff 生成后图或源码再变更时拒绝应用旧预览。Visual 反向同步保存 Design Graph；重开同时核对代码 Hash 和图签名，防止旧设计图显示 Synced。图和 code-state 不是一个文件系统事务，因此检查仍有必要。

每次捕获生成独立 ID，记录 request/result/job/stdout/stderr 到 `.opweaver/captures/<id>/`，SQLite metadata 的 `capture:<id>` 存任务，不混入训练 runs。默认 30s，RPC 可配置 3–120s；Stop/timeout 杀死捕获子进程，主 Worker 仍可响应。不是限制任意用户代码行为的资源沙箱；用户代码自行创建的后代进程没有 OS 作业级约束。分析日志/报告保留在文件系统，尚无 GC。

### 既有训练任务

首版用 stdio **JSON Lines RPC v1**：小型本地程序无需部署 gRPC 服务、端口管理或生成三套桩代码，容易发现 stdout 污染与进程退出。请求 ID、流事件、能力查询、Artifact 路径已经隔离在 Client 接口；远程执行未来采用 gRPC（见 PROTOCOL.md），不把 stdio 假装成远程方案。

Go 按需启动一个 Worker，维护请求关联、60s 调用超时、stderr 日志与退出检测。训练本身是异步任务，不受启动 RPC 超时限制；长任务通过 `threading.Event` 在 batch 边界取消。正常退出向 Worker 发送 shutdown，等待保存，超过关闭窗口后终止进程。Worker 崩溃使等待请求失败，下次请求启动新进程；历史中未终结任务标记 FAILED。

一个 Worker 同时只跑一个训练任务。任务生命周期：CREATED → VALIDATING → QUEUED → RUNNING → COMPLETED；取消经过 STOPPING → CANCELLED，任一阶段错误进入 FAILED。QUEUED 当前只是生命周期边界，不是多任务等待队列。取消验证阶段任务可能尚无可保存模型。

事件携带有界完整快照，UI 另外每 2s 轮询活跃状态以弥补丢失事件。当前保留最多 10,000 个 step 指标和 500 条任务日志，不适合大规模长期实验。P2 将改为带序号的增量事件与分页指标存储。

## 持久化与一致性

- `project.opweaver.json` 指向同一保存修订的 `graph.<timestamp>.json` 和 `layout.<timestamp>.json`。
- 先写入两个独立不可变文件，flush 后最后原子替换 manifest；失败不会让原 manifest 指向半写文件。Windows 下 Go rename 行为已通过保存/加载测试。
- 旧图/布局修订保留，当前无 GC；手动清理前要保留 manifest 引用。
- SQLite `metadata` 保存项目/model 版本；`runs` 保存配置、环境、任务状态、真实指标、日志和 Artifact 路径。
- Checkpoint 先写临时文件再 `os.replace` 发布；旁边保存启动时的完整 IR。
- Checkpoint 包含模型/Optimizer、CPU/CUDA RNG、数据 seed、epoch/next_batch/step。数据 shuffle 使用独立按 epoch 派生的 generator。
- 恢复要求 IR SHA256 一致；只允许更改总 epoch 和执行设备，防止把不同训练条件无声混合。

该设计保证应用正常关闭/重新打开和进程级错误恢复，不宣称抵抗断电或所有文件系统故障。历史存在 SQLite 中，图与权重存在文件系统；没有跨两者的全局事务。

## 扩展边界

`OperatorPlugin` 是当前实现的动态注册边界。`ModuleSpec` / `PluginReference` 已进入 IR，但当前遇到子图执行或外部插件声明会明确拒绝。其余 Model/Training/Optimization/Kernel/Backend 插件仅有契约设计，避免大量不执行的占位实现。

PassManager 在原始图的深拷贝上变换，验收失败返回原始图并给出原因。DCE 的 Eager 验收重用同一组权重，在若干 batch 大小上比较输出；这只是有界测试，不是数学等价证明。剪枝、量化与编译需要专用 Artifact、Calibration 和验证数据，不能直接套用 DCE 的接口实现。

## 信任边界

Python 子进程隔离故障，不是权限沙箱。P0-A UI 显式提示 Parse 会执行本地 Python；打开/导入不执行，没有自动发现插件功能。捕获时用户代码拥有当前账户权限，未来可信隔离需容器/OS 权限、文件/网络/资源配额。Checkpoint 用 `torch.load(weights_only=True)`，且只接受当前项目 checkpoints 路径；这不是对所有恶意文件的完整隔离。

本地 HTTP/SSE 适配器仅为开发/E2E，绑定 `127.0.0.1:34115`，检查 Host/Origin 与 RPC 自定义头；没有远程用户认证，不用于公网或团队服务。原生产品默认直接使用 Wails bindings。
