# P1-C1：本地分类数据与真实任务评估

2026-10-23。本轮按 P1-C 任务允许的子阶段顺序，完成 C1 的数据→FP32→Olive INT8→同集评估→Benchmark→导出与恢复。C2/C3 尚未实现，不能据本轮输出声称已有逐层诊断、敏感节点归因或图中选点重新量化闭环。实测与构建记录见 [VALIDATION.md](VALIDATION.md)。

## 能力与边界

| 状态 | 内容 |
|---|---|
| 已实现 | 本地 PNG/JPEG + UTF-8 CSV；显式输入名称、标签→分数列映射、预处理、批处理与快照 |
| 已实现 | 保留 NPZ / seeded InputConfig；calibration / validation / diagnostic / final_evaluation 角色，兼容原 evaluation |
| 已实现 | 原模型独立 Top1/Top5；原模型和候选在同一带标签验证集上的配对评价；正确/错误数、每批标签/预测、百分点与相对百分比 |
| 已实现 | 跨数据集检测相同原图、批次及有界逐样本 Tensor；校准/验证默认拒绝重用，最终评估始终拒绝与校准/验证重叠 |
| 已实现 | 最终评估字段默认不选，结果单独保存，不参与当前 Quality Gate 或量化配置生成 |
| 仅保存用途 | diagnostic 角色可导入、校验、持久化和预览；没有中间激活采集入口 |
| 当前不支持 | C2 qdq_loss_debug 接入、Tensor 匹配、逐层激活/权重误差及图联动；C3 诊断选点排除、新实验构造及三组候选筛选 |
| 已有 P1-B 能力保留 | Olive Pass 的手动 nodes_to_exclude 参数；这不等于完成 C3 的诊断选点、排除证明和反馈闭环 |
| 当前不支持 | 检测/分割/文本适配器、多输入图片模型、F1/mAP/IoU/Perplexity、用户脚本、GPU/INT4/LLM 专项量化 |

## 操作与离线样例

1. 在已保存项目导入 `examples/classification/color-classifier.onnx`。这是解析构造的六颜色原型分类器，**不是业务训练模型**。48 张本地 PNG/JPEG 和四个 CSV 由 `scripts/generate_classification_examples.py` 生成，无需下载数据。
2. 在 Optimization Lab 选择模型，Dataset source 选 **Local image classification CSV**。先选 Calibration 角色，CSV 填 `examples/classification/calibration.csv` 的绝对路径；Image root 留空表示 CSV 的父目录。
3. 输入名选 `image`，标签映射填 `{"red":0,"green":1,"blue":2,"yellow":3,"cyan":4,"magenta":5}`。数值是 `scores` 的列索引，不按文件名或字母顺序推断。
4. Batch=2、float32、Resize=`[10,10]`、Center crop=`[8,8]`、RGB、NCHW、scale=`0.00392156862745098`、mean=`[0,0,0]`、std=`[1,1,1]`、bilinear、EXIF 开启、provenance=Synthetic test fixture。保存后预览 6 批 / 12 例 / 6 类，输入 `[2,3,8,8]`。
5. 以 Validation / Diagnostic / Final evaluation 角色分别保存另外三个 CSV。它们使用独立生成的图片。生产数据应由用户确认代表性和覆盖范围；工具不会从正确率推断代表性。
6. 展开 Task evaluator，选择 Classification Top1 / Top5，输出名填 `scores`。可先使用 ORT 后端点击 **Run baseline only**，真实执行原模型评价。图片批次提供动态 Batch 的实际值，无需同时手填随机输入 Shape。
7. 切换 Microsoft Olive，指定既有独立 `.tools/olive-env`，Save & detect environment，添加 OnnxStaticQuantization。本轮验证用 MinMax / QDQ / int8 / per-channel，沿用 P1-B 的真实 Olive DataConfig、量化引擎和 CPU BenchmarkEngine。
8. 查看 Classification quality、数值误差、质量门限及真实延迟/体积。无标签或未选 evaluator 不输出 accuracy；执行有效、数值误差和任务精度分别报告。
9. Export model & reports 导出 JSON/Markdown、模型/外部权重、规范 NPZ、manifest、原图与 CSV；关闭重启后从历史恢复引用和测量。

CSV 必须是且仅是 `path,label` 两列；路径相对 Image root，标签必须在映射中：

```csv
path,label
validation/sample-00.png,red
validation/sample-01.jpg,green
```

预处理为 `(pixel × scale − mean) / std`，三通道 mean/std 按 RGB/BGR 转换后的顺序应用。Resize/Crop 采用 `[height,width]`，支持 null，裁剪不隐式补边。依次处理 EXIF、转 RGB（灰度展开、alpha 丢弃）、resize/crop、色序、归一化、布局和显式 dtype。uint8 要求 scale=1、mean=0、std=1；不假设 ImageNet 均值、标准差或类别顺序。

最后不足一批时默认保留，静态 Batch 模型可能不匹配；用户可明确 drop_last，丢弃数量进入警告和标签统计。空数据集或丢弃后为空会拒绝。解码成功还必须通过 ORT 输入名、dtype、rank、固定/共享符号 Shape 检查。

## 数据契约、可复现性与资源

`dataset_sources.py` 的 DatasetSource / DatasetPreprocessor 是实际适配器抽象；NPZDatasetSource 和 ImageClassificationSource 发布同一 feed/labels/provenance 批次契约。DatasetManifest 是实际序列化契约；DatasetIterator 复用 DatasetFeeds，为验证和 Olive 提供磁盘批次迭代。DatasetEvaluator / ClassificationEvaluator 与 Olive 独立；TaskEvaluator 保留配对评价扩展点。当前只实现分类，不把未来指标名称当作已注册功能。

数据位于 `.opweaver/datasets/<id>/`：规范化 `batch-*.npz`、`manifest.json`、`source/manifest.csv`、原图副本、可回放 `source/replay.csv`。记录 schema1.0 / dataset_version2.0 / adapter1.0、Pillow/NumPy 版本、完整预处理、标签映射/分布、seed、样本/批次数、模型 Hash、各源文件 Hash、标签 Hash、Tensor 指纹及数据指纹。不同标签映射也影响指纹。SQLite 保存元数据，Tensor 留在磁盘。

原图重复和预处理后完全相同的图片 Tensor 均拒绝。还校验损坏/空/缺失文件、路径越界、未知标签、非法列索引、dtype/shape、非有限值、std≤0。失败导入清理本次未注册的部分快照。重用时校验 manifest、原图/CSV 副本和 NPZ SHA256；原始外部目录之后可移动/删除，快照保持独立。快照缺失/改变会显示不可用，保留历史测量，导出拒绝不完整引用。

上限：4,096 图片、512 批、CSV 2 MiB、单图 16 MiB/约 16M 像素、源文件总量 256 MiB、处理后单图约 4M 像素、叠批输入 128 MiB、全部 Tensor 1 GiB、manifest 4 MiB，准备 45s。NPZ 沿用原有解压/展开限制。逐例重叠 Hash 只覆盖前 4,096 例且要求输入 Batch 轴兼容；所有批次 Hash 保留，范围不足会提示。原图 Hash 不受重新分批或预处理改变影响；不同编码、近似相似图片的语义重复不保证检出。

创建时记录与既有所有角色的可检测重叠和警告；运行时重新检查所选 calibration / validation / final。显式允许 calibration reuse 只放行验证集，不能放行最终评估重叠。最终评估默认不选，不用于当前调参、门限或候选筛选。人反复查看最终结果后手动调参仍可能引入泄漏，工具不推断人的研究决策。

导出均为本地文件。新项目回放时导入导出的模型，以 `datasets/<role>/source/replay.csv` 新建数据集，root 留空，使用保存的映射和预处理。原项目 Dataset ID、source_model_id 和绝对路径不会自动迁移。回放 CSV 只列实际纳入的样本，drop_last 丢弃的图片不会进入回放。

## 评价定义

标签是非负整数 `[batch]`，输出须为有限浮点 `[batch,classes]`；类别数跨批一致，映射索引须落在列范围内。Top1=argmax（并列取首列），Top5 为稳定降序前五列；不隐式 softmax 或重排类别。少于五类明确 not_applicable，不能以 TopK=min(5,C) 冒充 Top5。

- Accuracy=`correct / samples`，保存 incorrect、每批 labels/predictions 和统计口径。
- 原 accuracy_drop=`baseline − candidate`，为比例差。
- 百分点下降=`(baseline − candidate) × 100`。
- 相对百分比变化=`(candidate / baseline − 1) × 100`；baseline=0 时未定义，保存 null。

错误分类测试使用真实 ONNX 模型交换两类的输出权重：12 例从 12 对变成 8 对，下降 33.333 个百分点；反向 8 对→12 对则增长 33.333 个百分点、相对增长 50%。Top5 不变。另把六类 logits 全部取负，使真实类别排最后，验证 Top1/Top5 均为0；以该模型作参考再比较时，相对变化为 null。准确率只覆盖保存的标签、数据和预处理，不证明生产精度。

## 主要文件与后续

- `worker/opweaver/dataset_sources.py`：NPZ / 图片源、预处理与源文件快照。
- `datasets.py` / `lab_types.py`：统一快照/迭代、manifest、角色与重叠隔离，兼容旧记录。
- `task_evaluation.py` / `lab_runner.py`：独立、配对分类及最终评估，复用数值 Validation/Benchmark。
- `lab_reports.py`：JSON/Markdown 与原图/CSV 导出。
- `frontend/src/components/{ImageDatasetEditor,DatasetPanel,TaskQualityCard,OptimizationLab}.vue`：配置、预览和质量展示。
- `classification_examples.py`、生成脚本、`examples/classification/`：离线合成模型与四个独立分割。
- `worker/tests/test_image_datasets.py`、组件测试、`frontend/e2e/optimization-lab.spec.ts`：真实图片/ORT/Olive/UI 测试。

本轮没有逐层诊断指标或节点排除改进结果可报告。下一步 C2 应在独立临时模型上接入当前 ORT 的 qdq_loss_debug，验证匹配/未匹配状态、分批采集/预算、取消/超时/清理，再做误差列表与 Graph Inspector 联动。C3 随后验证原始节点唯一性、实际 Olive 排除行为及同集重新量化，比较原模型、原 INT8 和调整后 INT8；不能以本轮整体输出 MAE 代替逐层误差。
