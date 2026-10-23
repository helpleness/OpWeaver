# Graph IR 1.0

权威定义：`worker/opweaver/ir.py`；生成 Schema：`schemas/graph-ir.schema.json`；前端生成类型：`frontend/src/generated/graph-ir.ts`。所有对象 `extra=forbid`，未知版本/字段拒绝解析。

| 类型 | 责任 |
|---|---|
| Graph | `ir_version`, `id`, `metadata`, `nodes`, `edges`, `modules`, `plugins` |
| Node | 唯一稳定 ID、OperatorSpec、参数、输入输出 Port、可选 share_with/module_ref |
| Edge | 唯一 ID、source/source_port、target/target_port |
| Port | 稳定端口名，可选显式 TensorSpec 契约 |
| TensorSpec | 正整数或符号维度、dtype、device |
| OperatorSpec | 版本化算子 ID，如 core.linear@1 |
| ModuleSpec | 子图引用和公开端口；当前只有扩展格式，不执行 |
| GraphMetadata | 模型名称、model_version、描述 |
| PluginReference | 插件 ID、版本、种类；外部加载当前拒绝 |

节点 ID 不包含 `.`，可直接作为 PyTorch ModuleDict key，正则 `^[A-Za-z_][A-Za-z0-9_-]*$`。端口名必须与注册表声明的顺序和名称完全匹配。数组中 Input/Output 节点出现顺序定义模型位置参数/返回值顺序，拓扑排序不会改变公开接口。

一个输入端口只允许一个数据源；输出端口可广播到多个消费者。Add 声明 `a`、`b` 输入和 `out` 输出。首版 Add 要求相同 Shape/Dtype，不自动广播。输出节点以 `in` 接收 Tensor；一个模型可有多个 Output，返回 tuple。

Linear 变换最后一维且必须等于 in_features。Flatten 支持整数连续维度折叠，不能折叠未解析符号；应保留 `B`。符号在 Forward 时检查跨输入一致性，不能用符号掩盖已知静态尺寸冲突。前向预览/合成数据适配器只解析 B；更一般的符号可由 Python 调用者提供真实 Tensor。

TensorSpec Schema 预留 float32/64/16/bfloat16/int64/bool，但当前内置算子能力仅声明 float32，使用其他 dtype 时明确报错。device `any` 表示由运行配置选择；`cpu`/`cuda` 是约束，不代表自动跨设备复制。

`share_with` 引用同图的未共享同类型/同参数节点，Builder 使用同一 nn.Module 对象；不能引用自身、形成共享链或共享 Input/Output。UI 当前没有共享参数编辑控件，可通过 IR/Python API 使用。

## 校验流程

1. Pydantic Schema 与节点 ID、字段版本检查。
2. 解析 OperatorRegistry 和参数 Schema，检查端口声明、模块和插件能力。
3. 校验边引用、端口方向、重复生产者、未连接的必需输入。
4. Kahn 拓扑排序；未处理完所有节点则报告环路。
5. 按依赖推导 Shape/Dtype/Device，校验显式端口契约，传播上游失败位置。

返回 `{ valid, issues, order, tensors }`。issue 带 severity/message/node/edge；tensors 以 `node_id:port_name` 索引。用户可保存尚未连接完成的语义图，但不能执行无效图。

## 布局

布局另存为：

```json
{"version":1,"positions":{"input":{"x":60,"y":150}}}
```

不把 Vue Flow 的 selected、computedPosition、dragging、Handle 等内部状态写入 IR。当前保留节点位置；viewport/选中态不持久化，重新打开后自动 fit。

## 演进规则

修改语义或必需字段时升级 ir_version 并提供显式迁移函数与回归 fixture。首版只接受 1.0，不进行猜测式迁移。循环/条件控制流未来用显式节点或 Python Module 表达；当前 DAG 限制不意味着后续只能支持 Sequential。

重新生成：

```powershell
$env:PYTHONPATH = 'worker'
.\.venv\Scripts\python.exe scripts\generate_schema.py
cd frontend
npm run schema
```

生成脚本同时刷新两个内置示例，测试会比较提交 Schema 与当前 Pydantic 生成结果。

## P0-A 分析 IR 与源码映射

`source.py` 独立定义 `AnalysisGraph(schema_version=1.0)`，生成 `analysis-graph.schema.json` 和 TypeScript 类型。它包含 origin、ID/name、来源 Hash、read_only、节点、共用 Edge 和诊断；不冒充 Design Graph。节点保存 kind/target 标识、module_path、Tensor metadata、已识别算子及参数；来源路径/入口、Forward 统计和校验报告保存在外层 CaptureResult。未知算子不删除，完整捕获失败不发布新有效图。

`SourceMapping` 另有 `source-mapping.schema.json`：file_path、start_line/end_line、module_path、graph_node_id、status（exact/inferred/unmapped）、reason。行号为一开始，缺乏可靠位置时为 null。直接构造定义是 exact，捕获调用栈位置是 inferred；不把 inferred 当成可精确改写的定义。一个源码位置允许多个节点引用。

PyTorch Analysis Graph 可在全部操作/端口/Shape/数值对照受支持时投影为 Design Graph，保留原始分析和来源。P0-B 已实现 ONNX origin：source_model_id 关联版本化产物，节点 details 保存 domain/op_type/opset、各端口 Tensor、schema 状态和子图链接；parameters 保存真实属性。ONNX 保持只读，不承诺恢复 PyTorch 源码。

GraphImportResult（schemas/onnx-report.schema.json）统一封装模型产物、ModelMetadata、主/子分析图、输入输出、权重摘要、统计和结构化 ModelDiagnostic。子图保留外层自由 Tensor 捕获；UI 分组仅是视图投影，不更改原图/控制流。
