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
