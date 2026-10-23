# Worker Protocol v1

传输：UTF-8、每行一个 JSON 对象；Python `-u` 启动、stdout 仅协议、stderr 日志。环境探测、校验、Forward 是请求/响应；Train.Start 迅速返回 Job，之后通过事件流报告进度。Go Client 是可替换传输边界。

```json
{"protocol":1,"id":"42","method":"validate","params":{"graph":{}}}
{"protocol":1,"id":"42","result":{"valid":false,"issues":[],"order":[],"tensors":{}}}
{"protocol":1,"id":"42","error":{"code":"ValueError","message":"Missing operator ..."}}
{"protocol":1,"event":"metric","job_id":"...","data":{"id":"...","state":"RUNNING","metrics":[]}}
```

上面是 envelope 结构示意，空 graph 不满足 Graph Schema。正式 Schema 为 `schemas/worker-protocol.schema.json`。

| 方法 | params | result |
|---|---|---|
| environment | {} | Python/PyTorch/CUDA/device/notice |
| registry | {} | OperatorPlugin manifests |
| validate | graph | valid/issues/order/tensors |
| forward | graph | parameters/device/outputs(shape/dtype/min/max/mean) |
| train.start | graph/config/project/resume? | Job |
| train.stop | id | Job（已请求取消） |
| train.status | id | 该 Worker 的活跃/终结 Job 快照 |
| train.history | project | SQLite 中最近 100 个实验；旧进程未完成任务改为 FAILED |
| code.inspect | source/file_path | AST entries/diagnostics，不执行源码 |
| code.generate | graph/file_path/class_name | Python text/hash/source_map，不写文件 |
| code.patch | source/class_name/module_path/key/value/operator_id/parameters/expected_hash | 已验证参数的局部 text/diff/hash，不写文件 |
| code.capture.start | project/file_path/entry/constructor/inputs/expected_hash/execute_confirmed/timeout | 异步 CaptureJob |
| code.capture.status / stop | id | CaptureJob 快照 / 请求终止子进程 |
| shutdown | {} | 完成取消/保存后终止读取循环 |

Go 的统一 `App.Request(method, payloadJSON) -> resultJSON` 另外实现 `app.info`、`project.create/save/load`、`example`、`dialog.open/directory/python`、`workspace.list/read/write/import`。workspace 限制项目内 `src/*.py` 和版本化 code-state 文件，源码上限 2 MiB；写入 expected_hash 不匹配拒绝覆盖。import 只复制源文件，不执行。Go 将 capture 的项目相对 path 转成已校验的绝对 file_path，再交给 Worker；执行前再次核对磁盘 SHA256。

它拒绝未知方法；没有任意 Shell 命令 RPC。P0-A 显式 Python 捕获当然会执行用户代码，不能把“无 Shell RPC”等同于权限沙箱。Wails 和开发 HTTP 使用同一接口，计算行为无 Mock 分支。

RPC 参数保留为 JSON，不在 Go 复制 Python IR 类型。Worker 解析 Graph 与 TrainingConfig，返回可读错误；UI 中显示 Console/Problems/错误条。Go 请求超时会释放等待项，训练任务必须单独 stop，不能把 RPC 等待结束当成 GPU 操作已经取消。

## 事件与一致性

`state`、`metric`、`log` 携带同一 Job 快照：id/state/config/graph_id/graph_digest/metrics/logs/checkpoint/error/environment/created_at。事件快照保留最近 10,000 个 metrics、500 个 logs。Go 额外生成 `worker.stderr` / `worker.exited` 宿主事件，它们不是 Python RPC schema 的消息。

UI 的 2s 状态轮询用作流中断补偿。SSE 缓冲可丢弃事件，不保证重放；SQLite 和 checkpoint 是重开后恢复来源。此版本不宣称 exactly-once，也没有远程鉴权、任务幂等键或分布式调度。客户端请求超时后应查询当前状态，不能盲目重复 Train.Start。

`code.task` 单独携带 CaptureJob：`{"protocol":1,"event":"code.task","data":{"id":"...","state":"RUNNING","project":"...","source_hash":"...", ...}}`。不带训练 job_id，前端不将它混入训练指标。捕获生命周期 CREATED → RUNNING → COMPLETED/FAILED，取消 STOPPING → CANCELLED；UI 700ms 轮询补偿，终结事件按 ID 去重。任务含 result、error、logs、artifact_dir，结果保存 AnalysisGraph、可选 Design Graph、Source Mapping、实际 Forward 统计和输入范围内的数值对照报告。

## Artifact

Checkpoint 路径是本地文件引用，不传 Tensor 字节。当前路径必须属于项目 checkpoints 目录，加载使用 weights_only。远程化后 ArtifactReference 将包含 content digest、size、媒体类型、storage URI、生产者版本，采用独立对象存储/数据通道；不要把权重塞进 gRPC 单个消息。

## 后续 Protobuf/gRPC 迁移

计划服务：Environment/GetCapabilities、Graph/Validate、Job/Start/Get/Cancel/Watch、Artifact/Resolve。新增客户端 session、幂等 request key、事件 sequence/resume token、结构化错误 code/locations、能力协商、TLS/认证与远程取消确认。保留 GraphIR JSON payload + schema version，或生成同源 Protobuf schema，避免同时手工维护三套 Graph DTO。

JSON Lines 继续作为本地开发 transport；gRPC Worker 可由相同 Go Client 接口适配。首版没有提供不可运行的 proto stub。

## P0-B API

| 方法 | 参数 | 结果 |
|---|---|---|
| model.importers | {} | 已注册格式、版本、扩展名、可用性、是否执行 Python |
| model.probe | source | 格式识别；Python 使用原 AST 静态入口发现 |
| model.import.start | project/source/timeout；Python 另需 entry/constructor/inputs/execute_confirmed | 异步 ModelJob |
| model.import.status / stop | id | 任务快照 / 请求终止 |
| model.list | project | SQLite 中的任务及 Artifact 摘要 |
| model.load | project/id | 版本化 GraphImportResult，从报告文件读取 |
| dialog.model | {} | Go 原生模型文件选择器 |

model.task 与 code.task 使用相同的 CREATED/RUNNING/STOPPING/COMPLETED/FAILED/CANCELLED 快照信封，但前端独立路由，不赋值给训练 Job。最多 120s 导入子进程超时，Go RPC 仅等待启动响应。结果/图和权重放在文件系统，SQLite 存摘要；记录可重开。导入 API 不执行 ONNX 推理；P0-C 使用下述独立 API。

## P0-C API

P1-A 兼容扩展：`lab.backends` 接受可选 project/source_model_id 并返回 environment/passes。`lab.environment.configure`（environment.python_path）与 `lab.environment.probe` 检测独立解释器；`lab.workflow.preview` 验证 config 并返回 generic/native/validated，`lab.workflow.save` 保存模板，`lab.workflow.list` 恢复项目模板。以上均使用现有 Request JSON RPC；环境/工作流写入前验证项目。

OptimizationConfig 新增可选 environment 和 workflow，旧记录缺失字段按默认值恢复。Experiment 新增 workflow_execution，OptimizationResult 新增 workflow。schema_version 仍为 1.0（兼容可选字段扩展）；执行前冻结环境，实际原生配置、Pass footprint、候选和日志随实验保存。没有百分比的 Olive 阶段 progress=null。

| 方法 | 参数 | 结果 |
|---|---|---|
| lab.backends | {} | BackendCapability 列表，未实现/依赖缺失不假装可运行 |
| lab.start | project/config (ExperimentConfig 1.0) | 异步任务；config.mode=baseline 或 optimize |
| lab.status / lab.stop | id | 实时任务快照 / 杀死实验子进程 |
| lab.history | project | SQLite 持久化摘要，恢复时将中断任务标 Failed |
| lab.load | project/id | 完整 Experiment 1.0；含两侧配置、环境、原始计时和验证 |
| lab.compare | project/reference_id/candidate_id | 两个保存实验的有效 Benchmark、同条件比较或不可比原因 |
| lab.export | project/id/directory | JSON/Markdown/模型/权重/输入 bundle 的实际路径 |
| dialog.inputs | {} | 多选 NPZ 文件路径；桌面专用 |

`lab.task` 复用公共子进程快照信封，另含 status、phase、progress、phases、logs 和最终 Experiment result。前端单独路由，避免训练 Job 被替换；轮询 1s 补偿，终结事件去重，迟到项目事件丢弃。Go 校验已保存 project，Python 再校验输入/模型/ID/写入目录和内容 Hash。长任务在独立子进程运行，启动/状态 RPC 不等待整个实验。超时和取消终止子进程，不将 HTTP 断连误当作取消。

## P1-B API / 兼容字段

lab.dataset.create(project, config: DatasetConfig) → 已验证的数据集 manifest/ID；lab.dataset.list(project) → 元数据及引用可用性；lab.dataset.load(project,id) → 校验 manifest 和批次内容后返回元数据。格式 npz-batches-v1，Tensor 始终留在文件系统。数据准备 512 批/1 GiB/45 秒上限，不属于长任务的取消接口；量化运行仍使用 lab.start/status/stop。

OptimizationConfig 可选 calibration_dataset_id；ExperimentConfig.evaluation 包含 dataset_id、allow_calibration_reuse、task/output_name、gates。ValidationResult 扩展 execution_valid/numerical_summary/quality_gate/task_quality；Experiment 扩展 datasets/quantization/preprocessed_artifacts/data_availability，旧 1.0 缺字段恢复默认值。Quality Gate failed 不等于执行 Failed。带标签 Top1 仅支持显式 [batch,classes] 输出；未配置任务不输出 accuracy。

静态量化原生 DataConfig 仅由 Adapter 生成，固定使用受控产品模块；高级通用 JSON 不接受任意 user_script/data_config。lab.load 重新验证引用可用性，缺数据保留历史测量；lab.export 要求引用模型和数据仍完整。

## P1-C1 数据与评价扩展

沿用 lab.dataset.create/list/load。DatasetConfig.mode 新增 images，image 包含 manifest_path/root/input_name/class_mapping/batch_size/drop_last/provenance/preprocessing。preprocessing 明确 resize/center_crop/color/layout/scale/mean/std/dtype/interpolation/exif_transpose；完整格式见 P1C1_IMPLEMENTATION.md。role 新增 validation/diagnostic/final_evaluation，兼容 evaluation。manifest 的 dataset_version=2.0，保存 source_adapter/source_files/class_mapping/label_distribution/preprocessing/overlaps，旧快照仍可读取。

ExperimentConfig.evaluation 新增可选 final_dataset_id。Experiment 新增 baseline_evaluation（baseline-only 原模型评价）与 final_evaluation（独立最终评价）；validation.task_quality 保留原字段并新增 Top5、两侧详细统计、accuracy_drop_percentage_points 和 accuracy_change_percent。baseline=0 时相对百分比为 null。calibration/evaluation 引用 final 角色或 final 与它们有可检测重叠均拒绝。diagnostic 目前只是数据用途，尚无诊断任务 RPC。
