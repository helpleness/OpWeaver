# 按事件聚合的连续记忆与公开语料扩展

本轮取消“判断更正类别后覆盖”的方案。`revision_reader_01` 至 `04` 是未接受的开发分支，不作为正式修复。新入口为 `episode_demo.py`。

## 实际实现与边界

- `episode_memory.py` 把片段编码成按顺序排列的 256 维浮点向量，SQLite 保存 float16 向量、地址、说话来源和写入顺序，不保存聊天原文或 token ID。读取会重建基座输入嵌入，不重建文字后再塞进提示。
- 同一事件的所有匹配片段按写入顺序一起读取，不平均，不删除旧说法，不用更正分类器、六字段输出分类器、固定拒答或答案模板门控裁定回答。
- 冻结 Qwen2.5-1.5B-Instruct；普通聊天显式绕过记忆通道。新的编码器是宽度投影、可学习局部上下文卷积和嵌入解码器，共 787,200 参数。
- 这版保留序列长度，主要压缩嵌入宽度。它仍可能近似恢复原措辞，不是固定大小的抽象事件摘要，也不是已经证明的无限上下文架构。单片段上限 192 token，事件读取预算 4,096 token，超出时显式返回状态。
- 地址归组仍沿用旧原型的 16 个项目 × 16 个讨论次数，匹配门槛 0.99。`train_event_identity.py` 只补训事件地址，既不判定更新意图，也不裁定事件内容。开放域指代消解尚未实现。
- 旧模型的 0.8 字段支持概率没有同定义的新对应量。新架构取消了该字段分类路径，不能将“没有调低数字门槛”解释为同等拒答可靠性。错误回答须独立审查。

## 数据与训练

初始控制版本 `episode_memory_02`：512 组合成经历、896 个片段，覆盖六类内容与单次叙述、更正、质疑、补充的交叉组合。训练词汇仅 253 个 tokenizer token，重建接近无损，因此仅是信息保真基线。400 步重建、120 步已知事实 QA；没有固定拒答监督。

公开语料版本 `episode_memory_open_01`：

| 来源 | 固定版本 | 数据许可 |
|---|---|---|
| [HuggingFaceFW/FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) 中文 `cmn_Hani` | `af9c13333eb981300149d5ca60a8e9d659b276b9` | ODC-By-1.0 |
| [OpenAssistant/OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) ready conversation trees | `179dd21fc55192153d94adb0e0ce8f69e222bf75` | Apache-2.0 |

来源数据卡和 SHA-256 校验记录保存在 `corpora/open_01/<source>/`。Infinity-Instruct 需要额外访问许可，本轮没有绕过其访问条件，也没有采用许可不明的镜像。

下载压缩文件合计 4,895,994,818 字节。过滤后正文共 3,221,541,544 字节，853,248 条记录；其中训练 819,055 条 / 3,090,652,199 字节，验证 17,243 条 / 65,704,931 字节，测试 16,950 条 / 65,184,414 字节。按完整文档 URL 或整棵对话树哈希划分 96%/2%/2%，只做标准化文本精确去重，没有声称消除了所有近重复。

**候选语料量不等于实际训练量。** 从训练集确定性抽取 24,000 篇中文网页和 8,000 条对话，每条取最多 192 token。样本总计 5,892,570 token，抽样优化实际覆盖 26,768 个不同片段 / 4,930,827 token。2,400 步重建累计公开语料 token 暴露 10,597,138（重复抽样计入）、合成材料 1,005,347；80 步冻结基座分布蒸馏另读取 15,056 token、监督 960 个预测位置。没有对话答案 SFT 或固定拒答目标。

公开验证片段的相对嵌入 MSE 从 0.6511 降至 0.4857。这是通道保真指标，不能当成问答准确率。基座参数未更新，训练峰值显存约 3.79 GB。

## 对照与审查

开发实验 `episode_eval_02` 为五组同题 120 条输出。其逐条 Luna 初审和主助手独立评审均已完成，引用校验通过；13 条分歧、1 个内部一致性标志已复核，随机抽查 12 条一致项。它暴露了编码损失和未找齐后续片段的问题，随后用于诊断，不能再称作最终未见测试。

正式新测试 `episode_final_01` 使用另一个种子、全新事件与问题／修订表述，在 8/32/64/128 个用户轮次检查。七组同题：原始基座完整历史、原始基座最近窗口、旧记忆版、小数据事件版、公开语料事件版、公开语料加归组改进版、同一检索事件原文基座诊断。各组用相同基座和生成上限；同一事件原文组是评估控制，不属于记忆运行路径。

补充真实 OASST2 测试对话检验用户需求概括和说话来源，整棵对话树来自保留测试集。此项已知对话归组，是编码／落盘／读取诊断，不能用于宣称自动开放域归组成功。

正确性不由模板、关键词、固定拒答语或模型自身放行决定。流程：GPT-5.6-Luna 逐条盲审 → 独立主助手证据判断 → 校验引用／覆盖／内部一致性 → 复核分歧 → 随机抽查一致项。模型判断和复核都不是人工金标准；保留原始判断与复核理由。最终结果见独立报告。

## 复现入口

```powershell
C:\Python313\python.exe -X utf8 -m experiments.episodic_memory.open_corpus --root experiments/episodic_memory/corpora/open_01
C:\Python313\python.exe -X utf8 -m experiments.episodic_memory.open_corpus --root experiments/episodic_memory/corpora/open_01 --prepare
C:\Python313\python.exe -X utf8 -m experiments.episodic_memory.train_open_memory --corpus experiments/episodic_memory/corpora/open_01/prepared --out <新运行目录>
C:\Python313\python.exe -X utf8 -m experiments.episodic_memory.episode_demo --run <运行目录> --db <新数据库> --owner me write --id event-1 --text "项目及讨论次数明确的经历叙述"
C:\Python313\python.exe -X utf8 -m experiments.episodic_memory.episode_demo --run <相同运行目录> --db <相同数据库> --owner me ask --question "关于该事件的问题"
```

创建输出目录的命令拒绝覆盖既有目录。`corpora/` 已加入 Git 忽略，原始语料不进入仓库提交。
