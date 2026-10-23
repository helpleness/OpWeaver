# 插件与 Pass 契约

## 当前可用：OperatorPlugin

`worker/opweaver/registry.py` 提供 `OperatorRegistry.register(plugin)` 和 `get(id, version)`。重复 ID/version 拒绝注册；缺失版本报出可读错误。Worker 内置注册 15 个插件，前端通过 `registry` RPC 读取实际 manifest/参数 JSON Schema，不维护第二份算子列表。

OperatorPlugin 字段：

- `id`, `version`, `label`, `category`, `description`。
- 有序 `inputs` / `outputs` 端口名。
- Pydantic `parameters` 类：默认值、类型、约束、额外字段策略。
- `factory(params) -> nn.Module`：真正注册 Parameter、Buffer、子模块，提供 forward。
- `infer(input_specs, params) -> output_specs`：不能依赖真实 GPU Tensor；失败须给出有意义的异常。
- `dtypes`, `devices`, `autograd`, `torch_export`, `onnx`, `custom_kernel`。
- 可选 `python_module(params) -> constructor` 或 `python_expression(inputs, params) -> expression`，以及 `python_imports`：可执行代码生成入口。
- 可选 `module_type` 和 `module_parameters(module) -> dict`：FX leaf 类型和参数序列化入口，按精确类型匹配，不将未知子类误作内置模块。
- `manifest()` 返回上述声明和后端能力，当前后端为 pytorch.eager。

自定义模块可在受信任的 Python 进程里手工注册，例如：

```python
from torch import nn
from opweaver.registry import OperatorPlugin, EmptyParams, builtin_registry, identity_shape
from opweaver.builder import GraphModule

registry = builtin_registry()
registry.register(OperatorPlugin(
    id="example.silu", version="1", label="SiLU", category="Activation",
    inputs=("in",), outputs=("out",), parameters=EmptyParams,
    factory=lambda params: nn.SiLU(), infer=identity_shape,
    description="Trusted application-provided SiLU",
))
# graph 中的 Node 应引用 example.silu@1，声明 in/out
# model = GraphModule(graph, registry)
```

这不是 UI 自动加载机制。已有 Monaco 编辑器和显式模型捕获，但 Worker 不从任意路径自动安装/加载算子插件。传入 Graph.plugins 或 ModuleSpec 执行声明仍明确失败。

完整示例在 `worker/opweaver/custom_example.py`：`register_example(registry)` 注册 `example.square@1`，包括真实 nn.Module、Shape、Autograd、构造代码、FX 类型/参数和导入语句。`test_custom_operator_codegen_example` 验证注册→构图→代码生成→捕获→Backward。示例使用显式传入的受信 registry；并未悄悄启用全局第三方加载器。没有 codegen 回调的插件可继续 Eager 执行，但生成代码会清楚失败。

算子若提供 `torch.autograd.Function`，由其 forward/backward 实现控制自定义梯度；不能只设置 `autograd=True` 就声称支持训练。声明 torch.export/ONNX 能力前必须有独立导出/数值回归测试。内置 manifest 暂将两者设为 false，含义是本产品未验证或开放该路径。

## 三级扩展与未来契约

| 插件类别 | 计划的入口 | 产物/能力 | 实现状态 |
|---|---|---|---|
| OperatorPlugin | params schema / infer / factory | Eager Module、Backward、兼容性 | 基础实现 |
| ModelPlugin | public ports / nested Graph / state mapping | 复用子图、导入模型 | P2 契约 |
| TrainingPlugin | dataset / collate / loss / checkpoint hooks | 输入/target 映射、可恢复数据游标 | P2 契约 |
| OptimizationPlugin | PassSpec / transform / verify | 版本化 Artifact、验证报告 | 基础 PassManager，无插件加载 |
| KernelPlugin | signatures / dispatch / backward / build | Triton/CUDA 二进制、缓存键、设备约束 | P4 契约 |
| BackendPlugin | probe / lower / compile / run / benchmark | 目标硬件能力、运行时 artifact | P3–P4 契约 |

Level 1 是当前 PyTorch 内置算子；Level 2 已提供 Python 编辑/显式捕获和受信 API 注册，插件加载 UI 和来源管理仍未实现；Level 3 才接入 Triton/CUDA。编译、量化、导出是能力矩阵的交集，不做“任意用户代码自动可导出”的保证。

## PassManager

`PassSpec` 声明：ID/version、input_type、stage、backends、operators、precisions、calibration、changes_parameters、changes_structure、requires_retraining、artifact_type、validation。

stage 为 `pre_training` / `training` / `post_training` / `export_ir` / `backend`。Manager 对候选图深拷贝变换，调用外部 verifier；不通过或异常时保留原图并返回 reason。`dead_code_elimination` 从 Output 反向追踪依赖和参数共享 owner；`verify_eager` 复制权重、eval、固定输入、比较不同 batch 输出并恢复 RNG。

```python
from opweaver.passes import PassManager, PassSpec, dead_code_elimination, verify_eager

manager = PassManager()
manager.register(PassSpec(
    id="core.dce", version="1", input_type="GraphIR", stage="pre_training",
    backends=("pytorch.eager",), operators=(), precisions=("float32",),
    calibration=False, changes_parameters=False, changes_structure=True,
    requires_retraining=False, artifact_type="GraphIR",
    validation="CPU forward equivalence on fixed samples",
), dead_code_elimination)
# result = manager.run("core.dce", graph, verify_eager)
```

该接口没有执行 Backend/精度能力自动协商，调用者须在选择 Pass 前匹配声明。未来 Manager 将加入显式 context（模型权重、校准数据、训练状态、后端能力）及 Artifact lineage。PTQ/QAT、剪枝恢复训练不是无状态图重写，不可用简单 Transform 替代。

## 信任与部署

进程隔离只防止 GPU/训练异常直接使 UI 进程崩溃，不隔离用户权限。第三方插件可访问 Worker 账户可访问的文件与网络；P2 必须增加显式来源与版本锁定、信任审批、撤销、能力探测。服务端/不受信任插件需要容器或其他 OS 权限隔离、资源配额和超时。禁止在 Vue 中 eval Python，也不通过自动安装插件消除 Missing Operator 错误。
