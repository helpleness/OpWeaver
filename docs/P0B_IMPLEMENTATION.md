# P0-B 增量实施

基于已验证的 P0-A、15 算子、Monaco、训练和恢复基线，不重新初始化项目。

1. 扩展共用 AnalysisNode/details/source_model_id，新增 ModelImporter、ModelArtifact、ModelMetadata、ModelDiagnostic、GraphImportResult；权重保存独立文件。
2. 复用已有异步子进程任务生命周期，实现 ONNX 原样复制、外部权重路径校验、真实 checker/Shape inference 和图解析。
3. 同一 GraphCanvas 接受外部只读分析图，保持 Design/PyTorch/ONNX 来源区别。
4. 实现导入历史、模型/节点/输入输出/属性/权重 Inspector、节点搜索、错误与取消、重开加载。
5. 提供离线生成的轻量模型，验证多输出/动态维度/外部权重/未知算子/非法文件，再回归 P0-A 和训练，构建 Windows 桌面。

P0-B 只分析模型，不执行推理或声称能从 ONNX 还原 PyTorch。ORT 推理、优化、Benchmark 为 P0-C。FLOPs 无可靠测量时显示 Not measured。ONNX schema 已知仅表示可识别，不等于某推理后端支持。

已落地模块：importers.py 注册 ONNX/PyTorch；PyTorch 直接复用 P0-A capture 并保存参数、输入和映射，可回到 Monaco。model_tasks.py 复用异步生命周期，import_runner.py 执行；ModelBrowser/ModelInspector/useModels/analysis-view 复用同一 GraphCanvas，增加搜索、上/下游高亮、模块/算子折叠和子图切换。

已验证 600 个真实 Identity 节点的图，默认折叠为 3 组，并可搜索最后一个节点。视口渲染只用于外部分析图，原 Design Graph 保留原行为；布局为确定性网格和会话缓存。Inspector 可调宽度，各属性段可折叠；权重检索最多显示 100 条匹配记录，报告仍保留全部摘要。

限制：主 protobuf 256 MiB、全模型 20,000 个分析节点、报告 24 MiB、导入 120s；超限明确失败。外部权重可流式复制，但不承诺任意大小模型的 C++ Shape inference 成功。大量数组/字符串为带截断标记的预览；原模型原样保存。ONNX 子图只读，不递归编辑。未提供跨后端支持矩阵、任意 ONNX→PyTorch 重建或可信沙箱。PyTorch 独立源码不自动打包多文件依赖和模型权重。

官方 API 依据：[ONNX Python API](https://onnx.ai/onnx/repo-docs/PythonAPIOverview.html)、[Shape inference](https://onnx.ai/onnx/api/shape_inference.html)、[External data](https://onnx.ai/onnx/repo-docs/ExternalData)。本机实测使用已安装的 ONNX 1.22.0，运行与格式合法性检查分别记录。

## P0-B 完成状态与限制

本轮完成真实 ONNX/PyTorch 导入和分析。复用 P0-A 的注册算子、FX、Monaco、Go RPC 与原训练；未重新初始化。新增元数据与 diagnostics 均实际序列化并由 UI 消费，非空接口。

源模型、External Data 均原样保留；Shape inference 不写回。ONNX 模型合法性与运行时支持分开记录。未知算子、自定义域保留属性和连线；If 子图和隐式外层依赖可查看。权重只发送摘要，Initializer 元素不冒充可训练参数，FLOPs 未测量。

限制：protobuf 256 MiB、20,000 分析节点、24 MiB 报告、120s 导入超时；超限失败保留旧图。布局目前为确定性网格和会话缓存。大型数组/字符串显示明确截断的预览，原文件不变。未执行 ONNX 推理、Benchmark、优化、ONNX→Python 恢复或第三方 runtime 支持认证。Python 导入针对独立源码，不自动打包依赖或模型权重。

下一步 P0-C 直接消费已注册 ModelArtifact 的托管路径、Hash、I/O、Shape/DType、Opset、外部权重和诊断，接 ONNX Runtime CPU，完成基线→优化→数值验证→相同条件 Benchmark→导出，不再建立平行模型库。
