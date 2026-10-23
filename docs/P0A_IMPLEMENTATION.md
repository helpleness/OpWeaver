# P0-A 增量实施清单

本轮按 2026-10-23 新任务书执行，先完成 P0-A。保留已有 Wails/Vue Flow、Graph IR、训练与 Checkpoint 功能。P0-B ONNX 和 P0-C 优化闭环随后实施，不以不可运行页面代替。

## 已有与缺失

- 已有：Design Graph 1.0、算子注册表、真实 PyTorch Eager 构建/训练、项目读写、错误隔离、测试。
- 缺失：项目 Python 文件编辑、Graph→Code、Code→Graph、Source Mapping、同步状态/冲突处理。
- 已发现交互问题：选中态变化会重新传入旧的节点坐标，拖动时可能回跳；20px 实时吸附让慢速拖动呈跳跃状。修复时保留网格显示，拖动使用连续坐标，停止后只提交实际改变的位置。

## 顺序

1. 修复拖动状态与光标稳定性，执行已有回归测试。
2. 扩展同一个 OperatorRegistry 的构造/Shape/代码生成入口；实现独立 Analysis Graph 和 Source Mapping。
3. 实现 AST 静态发现（打开文件不执行）、显式 FX 捕获、子进程超时/取消、真实 Forward、完整诊断。
4. 实现可读可执行代码生成和保留无关源码的 AST 局部补丁，所有写入提供 diff 及乐观 Hash 冲突检查。
5. 接入 Monaco、图/代码/双视图、可调分隔条、文件切换、映射导航与同步状态。
6. 验证任务书 A–D、鼠标拖动回归、已有训练链路；更新架构、路线、开发和验收记录。

## 边界

实施已落地：新增 source/codegen/capture/code_runner/code_tasks，扩展原 registry，Go workspace 文件服务，以及同一 GraphCanvas 的分析图渲染、Monaco 和 Pinia 同步状态。README/ARCHITECTURE/PROTOCOL/PLUGIN_API/ROADMAP 已同步，实测结果见 VALIDATION.md。

- Design Graph 继续表示可执行且可编辑的受支持模型；FX Analysis Graph 保存原始语义、来源、未知算子和映射置信度。未来 ONNX 使用单独 origin。
- Code-First 不支持自由改线；仅对能准确定位的直接模块构造参数提供可验证局部改写。其他结构只读。
- 未成功捕获时保留源码和上一张有效图，明确标记图已过期。捕获是用户显式动作，Worker 子进程不是沙箱。
- 不在主 Worker 的 stdin 读取循环执行用户模型；专用子进程承担导入、捕获和 Forward，支持超时/取消并保存任务报告。
