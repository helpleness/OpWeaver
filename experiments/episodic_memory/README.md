# 磁盘事件记忆最小原型

最新长上下文对照已完成：两段同源历史在约 2K/8K/16K/24K/32K token 检查，原始冻结 Qwen2.5-1.5B-Instruct 接收完整未截断历史，记忆版从同一消息流编码落盘后自动读取。各 80 条回答经 Luna 初审、证据校验与复核，整句合格为 33/80 对 66/80，有问题为 39/80 对 10/80；同题 35 次改善、2 次退步。仅两段受控事件加真实阅读材料对话，不能当作自然聊天泛化结论。记忆版另有 3 次漏读分歧却偶然答对。正式输出在 `runs/long_context_02`，评审在 `runs/long_context_luna_01`；[14页完整报告](../../exports/episodic_memory_report_20261026/长上下文记忆模型实验报告.pdf)与[逐题对照附录](../../exports/episodic_memory_report_20261026/逐题对照与证据.html)已导出，全部标签为 AI 评审而非人工金标准。

最新：[事件片段记忆与 3.22 GB 公开语料实验](EPISODE_MEMORY_RESULTS.md)已完成。基座冻结，实际抽样优化覆盖 493 万不同 token；新事件测试中公开语料版整句合格 20/24，小数据版 19/24，但真实对话 8 例仍无完整合格记忆摘要。旧 5 次门控误拒与 3 次明确更正回归通过，未核实冲突仍会选边。全部 216 条最终／开放对话／回归回答经过 Luna 初审、独立核查与一致项抽查，保留失败案例。数据与复现步骤见[协议](EPISODE_MEMORY_PROTOCOL.md)。

上一版：[连续聊天检查点实验](STREAM_RESULTS.md)。128个用户轮次，8/32/64/128轮检查，三组共72条输出经GPT-5.6-Luna初审、分歧复核和随机抽查。记忆可保留早期事实，但仍有5次表达误拒与3次更正处理失败；不能宣称长期记忆已解决。

已增加 [GPT-5.6-Luna 独立逐条初审](LUNA_REVIEW_RESULTS.md)：48/48 条完成，与主助手原评审有 9 条分歧；评审器也存在内部不一致，原始判断、证据和分歧复核全部保留。

最新进展：已实际重训 `fact_reader_01`，2,400 个语言目标中固定拒答目标为 0，正式演示入口对无法确认的回答返回结构化状态。48 条小规模对照输出已逐条对证据评审；发现正确读取仍会被旧模板门控拦截，尚不能宣称整体性能提升。见 [取消固定拒答监督的实验报告](FACT_READER_RESULTS.md)。

> **评分方法纠正（2026-10-26）：** 近期自由文本评估把固定拒答句、原型范围和事实正确性混在一起，旧分数不能作为通用语义准确率或可靠拒答能力。已发现 32 道有证据可答的题被要求拒答，以及 4 条正确生成被答案模板拦截。见 [纠正说明与证据审计](EVALUATION_CORRECTION.md)。本轮没有重训；13 条定向复核不代表总体成绩。

这是 OpWeaver 仓库中的独立研究实验，不修改现有应用或 worker。前五轮使用小型编码器和分类读取器；第六轮接入冻结的Qwen2.5-0.5B-Instruct，具备原基座的普通聊天路径。所有版本用向量保存经历，读取器不访问历史原文；目前都不是通用长期记忆数字人。

首轮旧模型实验：正常模板的已存事件答对率96.9%，内容清零后0%，错配后15.2%；未见表达模板0%。见 [历史结果与限制](RESULTS.md) 和 [完整实验报告](runs/pilot_03/REPORT.md)。

第二轮改进在新留出句式上：只换提问87.7%，只换事件叙述82.9%，两者都换76.3%；原句96.7%。显著改善但尚未完全解决，仍有有记忆却回答UNKNOWN的情况。见 [配对对比结果](PARAPHRASE_RESULTS.md) 和 [预先固定的实验方案](PARAPHRASE_PROTOCOL.md)。旧模型和数据库原样保留。

第三轮将“两者都换”的答对率提高到87.9%，UNKNOWN从104/480降至45/480；证据和内容置信门槛未降低。屏蔽正确事件的480个问题及新历史的1536个未知事件问题均未错误作答。但已存事件错答从10例变成13例，仍有读取错误，不能称为全面解决。见 [证据与拒答对比](EVIDENCE_RESULTS.md)。最终checkpoint为`runs/evidence_round_03/verified/model.pt`，可直接读取上一轮writer保存的数据库。

第四轮专门分析45次拒答与13次错答，增加字段化读取、独立问题意图/地址编码、地址一致度对证据分数的上限约束。旧集改善为21次拒答、0次错答，但最终新句式A错答125→155、B错答158→5，结果不稳定，**本轮候选未通过新句式泛化验证**。保留第一阶段失败及最终新事件/新句式复测，见 [残余拒答与错误回答报告](RESIDUAL_RESULTS.md) 和 [方案及公开修正](RESIDUAL_PROTOCOL.md)。实验checkpoint为`runs/typed_round_02/model.pt`；仍可读原数据库，双门槛保持0.99与0.8，不能将门槛不变理解成不会增加错误。

第五轮把问题理解与记忆读取单独评估，使用独立中文预训练前端，并做同架构“随机/预训练 × 窄分布/组合数据”四组对照。独立新结构分类准确率从38.0%提高到88.7%；两套全新事件的同时改写测试中，拒答140→51、294→226，错答158→21、84→55。仍未彻底解决，第二套端到端正确率仅41.5%，主要剩余拒答来自冻结的地址判断。见 [问题理解分离实验](LANGUAGE_RESULTS.md) 和 [预定方案](LANGUAGE_PROTOCOL.md)。模型不具备自由聊天能力。

### 第八轮：语义读取与扩展拒答测试

候选版本位于`runs/semantic_memory_03`，结果见[语义读取与拒答报告](SEMANTIC_RESULTS.md)，训练前方案及开发过程见[SEMANTIC_PROTOCOL.md](SEMANTIC_PROTOCOL.md)。普通聊天与记忆查询仍是显式分开的入口。此版仍限定六字段和有限词表，不支持任意人物/事实、多字段或是非问题。

已补做[原始Qwen同题对照](BARE_QWEN_RESULTS.md)：直接加载未接适配器的同一官方1.5B模型，比较无经历、全部可用事件文字和指定正确事件文字。全部文字组严格目标事实匹配55/64，同义核查后57/64；向量记忆系统55/64。先前33→55比较的是两套记忆系统，不能据此声称提高了基座通用理解能力。完整提示和原始输出在`runs/bare_qwen_comparison_02`；评分核查不将不同措辞的拒答算作事实错误。

问题先由冻结Qwen在字段定义上下文中编码，再由训练头选择字段；读取器使用统一字段指令和对应连续记忆槽。新增缺失事实识别、潜变量/生成一致性验证与同地址矛盾检查。核验语法限制输出范围，不能代替开放域语义核验。训练有720组事件、1440叙述视图、6480问题；最终独立校准120题、测试240题，覆盖15种条件。

```powershell
# 写入真实文本时不需要提供字段标签。新建独立的示例库。
C:\Python313\python.exe -m experiments.episodic_memory.semantic_demo --run experiments/episodic_memory/runs/semantic_memory_03 --db experiments/episodic_memory/runs/semantic_memory_03/my_demo.sqlite --owner me write --id demo-1 --text "花园项目第1次讨论。时间第3天，和小林交流。发生提出建议。感受高兴，因为获得支持。关注后续计划。"

# 在独立进程中只凭问题和磁盘向量回答。
C:\Python313\python.exe -m experiments.episodic_memory.semantic_demo --run experiments/episodic_memory/runs/semantic_memory_03 --db experiments/episodic_memory/runs/semantic_memory_03/my_demo.sqlite --owner me ask --question "花园项目第1次讨论，和我交谈的人是谁？"
C:\Python313\python.exe -m experiments.episodic_memory.semantic_demo --run experiments/episodic_memory/runs/semantic_memory_03 chat --text "请解释为什么会下雨。"

# 重评最终模型：out必须是新目录；测试会验证校准对应的权重哈希。
C:\Python313\python.exe -m experiments.episodic_memory.evaluate_semantic_memory --run experiments/episodic_memory/runs/semantic_memory_03 --suite experiments/episodic_memory/runs/semantic_suite_03 --split test --out experiments/episodic_memory/runs/my_semantic_test

# 核心组件与现有持久化等检查。
C:\Python313\python.exe -m unittest experiments.episodic_memory.test_memory experiments.episodic_memory.test_qwen_memory experiments.episodic_memory.test_semantic_memory
```

完整训练链保留在三个运行目录的metadata和training.jsonl中，可按以下顺序使用新目录复现：

1. `semantic_data --out <suite1>`，`train_semantic_memory --prepare --suite <suite1>`，然后`train_semantic_memory --suite <suite1> --out <run1>`（3000步监督、600步读取）。
2. `prepare_semantic_revision --source <suite1> --out <suite2>`；`train_semantic_memory --prepare --source <run1> --suite <suite2> --canonical-reader`；再去掉`--prepare`，增加`--out <run2> --warmup 0 --query-steps 1500 --steps 400`。
3. `prepare_semantic_revision --source <suite2> --out <suite3> --final-calibration`；`train_semantic_memory --prepare --source <run2> --suite <suite3> --canonical-reader --contextual-query --warmup 0`；再去掉`--prepare`，增加`--out <run3> --query-steps 1000 --steps 0`。
4. 先用`evaluate_semantic_memory --run <run3> --suite <suite3> --split calibration --out <run3>/calibration`校准，再使用`--split test --out <run3>/test`。所有模块命令前加`C:\Python313\python.exe -m experiments.episodic_memory.`。

第一、二版开发失败仍在`semantic_memory_01/calibration`与`semantic_memory_02/calibration`，不能把这些已用于开发的结果当作最终独立校准。第三版仅使用新的seed9207校准；三个suite中的最终test.json逐字相同。

### 第七轮：Qwen2.5-1.5B同系列对照

后续[逐 token 拒答诊断](QWEN_CONFIDENCE_DIAGNOSIS.md)：1.5B新问法24条目标事实匹配却拒答的输出中，19条只有事实片段之外的token未过门槛；另1条跨字段混读通过局部记忆接口干预定位。两个模型各120条生成复现一致，权重与部署门槛未改。

官方1.5B权重在`runs/qwen_1_5b_backbone`，最终适配器在`runs/qwen_1_5b_memory_01/adapter.pt`；原0.5B版本保持不变。新模型不复用旧隐藏向量，单独重算特征并重建SQLite。见[规模对照结果](QWEN_SCALE_RESULTS.md)及[预先固定方案](QWEN_SCALE_PROTOCOL.md)。

测试包括旧题回归、全新事件/问法，以及“正确事件文字 / 正确字段列表 / 正常向量检索 / 正确事件向量oracle / 空记忆”的双模型对照。前两个是评估能力的文字诊断，不进入部署ask接口。

```powershell
# 1.5B普通聊天；这里显式指定run，默认入口仍保留原0.5B。
C:\Python313\python.exe -m experiments.episodic_memory.qwen_demo --run experiments/episodic_memory/runs/qwen_1_5b_memory_01 chat --text "用两句话解释为什么会下雨。"

# 从1.5B的持久化实验数据库读取，不向查询程序提供经历原文。
C:\Python313\python.exe -m experiments.episodic_memory.qwen_demo --run experiments/episodic_memory/runs/qwen_1_5b_memory_01 --db experiments/episodic_memory/runs/qwen_1_5b_memory_01/evaluation/memories.sqlite --owner test-0002 ask --question "运动项目第4次讨论，参与的人是谁？"

# 重训：所有out都必须是新目录；使用已下载权重和对应1.5B缓存。
C:\Python313\python.exe -m experiments.episodic_memory.train_qwen_structured --backbone experiments/episodic_memory/runs/qwen_1_5b_backbone --reuse experiments/episodic_memory/runs/qwen_1_5b_features_01 --out experiments/episodic_memory/runs/my_qwen_1_5b --freeze-writer --reader-lr 0.0001 --batch 16
C:\Python313\python.exe -m experiments.episodic_memory.evaluate_qwen_memory --run experiments/episodic_memory/runs/my_qwen_1_5b

# 新建固定比较集，分别运行两个模型；--name决定suite内的输出子目录。
C:\Python313\python.exe -m experiments.episodic_memory.compare_qwen_scale --suite experiments/episodic_memory/runs/my_scale_comparison --prepare
C:\Python313\python.exe -m experiments.episodic_memory.compare_qwen_scale --suite experiments/episodic_memory/runs/my_scale_comparison --run experiments/episodic_memory/runs/qwen_memory_03 --name qwen_0_5b
C:\Python313\python.exe -m experiments.episodic_memory.compare_qwen_scale --suite experiments/episodic_memory/runs/my_scale_comparison --run experiments/episodic_memory/runs/my_qwen_1_5b --name qwen_1_5b
C:\Python313\python.exe -m experiments.episodic_memory.report_qwen_scale --new experiments/episodic_memory/runs/my_qwen_1_5b --suite experiments/episodic_memory/runs/my_scale_comparison --output experiments/episodic_memory/runs/my_scale_comparison/REPORT.md
```

如需重新生成1.5B缓存：`python -m experiments.episodic_memory.prepare_qwen_features --backbone experiments/episodic_memory/runs/qwen_1_5b_backbone --out 新缓存目录`。训练器校验模型revision和特征维度，拒绝混用缓存。1.5B默认与0.5B保持相同门槛，但概率校准不相同，解释结果时必须同时看原始正确率和门控后表现。

### 第六轮：Qwen生成式读取与磁盘连续记忆

使用官方`Qwen/Qwen2.5-0.5B-Instruct`，基座在`runs/qwen_backbone`；最终适配器在`runs/qwen_memory_03/adapter.pt`。此前两个候选保留在`qwen_memory_01`和`qwen_memory_02`。见[训练与对照结果](QWEN_MEMORY_RESULTS.md)、[实验方案及验证修正](QWEN_MEMORY_PROTOCOL.md)。

当前电脑使用`C:\Python313\python.exe`，CUDA PyTorch与Transformers已安装。以下命令从`D:\MyGoPro\OpWeaver`执行；每个数据库/实验输出使用新路径，避免覆盖已有记忆：

```powershell
# 普通聊天：保持原基座路径，不自动检索记忆。
C:\Python313\python.exe -m experiments.episodic_memory.qwen_demo chat --text "用两句话解释为什么会下雨。"

# 新建自己的数据库；随后写入一条当前经历（只有写入时接触这段文本）。
C:\Python313\python.exe -m experiments.episodic_memory.qwen_demo --db experiments/episodic_memory/runs/my_qwen_memory.sqlite init
C:\Python313\python.exe -m experiments.episodic_memory.qwen_demo --db experiments/episodic_memory/runs/my_qwen_memory.sqlite --owner me write --text "画册项目第1次讨论。时间第3天，和小林交流。发生叫错名字。感受尴尬，因为担心评价。关注他人反应。"

# 新进程从盘读取；无需再次输入事件，也不打开训练数据。
C:\Python313\python.exe -m experiments.episodic_memory.qwen_demo --db experiments/episodic_memory/runs/my_qwen_memory.sqlite --owner me ask --question "画册项目第1次讨论，当时和谁交流？"
```

CLI输出`raw`（模型原始生成）、`answer`（门控后最终答案）、`evidence`（地址证据）、`confidence`（所有生成token的最小条件概率），以及检索轨迹。`chat`与`ask`是显式不同入口，还没有训练自动回忆模式选择器。输入事件/问题特征预算192 token，超长报错。

重训与评估（复用本机已存在的冻结特征缓存，随机种子和训练数据固定）：

```powershell
C:\Python313\python.exe -m experiments.episodic_memory.train_qwen_structured --out experiments/episodic_memory/runs/my_qwen_run --freeze-writer --reader-lr 0.0001 --batch 16
C:\Python313\python.exe -m experiments.episodic_memory.evaluate_qwen_memory --run experiments/episodic_memory/runs/my_qwen_run
C:\Python313\python.exe -m experiments.episodic_memory.report_qwen_memory --run experiments/episodic_memory/runs/my_qwen_run --output experiments/episodic_memory/runs/my_qwen_run/REPORT.md
C:\Python313\python.exe -m unittest experiments.episodic_memory.test_memory experiments.episodic_memory.test_qwen_memory
```

若需要重建数据与特征，用`train_qwen_memory --smoke --out 新目录`生成seed7101的数据与冻结特征，然后把该目录作为`train_qwen_structured --reuse 新目录`参数。冒烟权重不会复用。依赖沿用`requirements-language.txt`；无需安装PEFT或训练整个基座。

训练先用字段监督与嵌入对齐，让8个槽保留人物、日期、事件、感受、原因、关注点、项目和次数；再固定写入端，只训练把槽位投影为32个连续输入嵌入的读取模块。正常推理不把槽位argmax解码成答案文字，Qwen自行生成答案。载荷FP16共4KiB/事件，32维地址128字节；SQLite没有原文、token ID和答案列。

重要实验范围：仍使用`generate_data.py`声明的有限事实值和16×16场景地址。可测试这些值的组合与改写，不能保证新人物、新项目、任意日期或相同地址的多条冲突经历。基座能普通聊天不意味着记忆模块已具有开放领域泛化能力。分堆索引只管理地址，全部地址仍缓存到CPU内存，尚未实现超大规模磁盘索引。

### 第五轮运行方式

本轮只替换字段理解，原记忆checkpoint保持`runs/typed_round_02/model.pt`；新前端位于`runs/language_round_02/pretrained_compositional/`。两者分别加载，旧SQLite不需重写。前端约877万参数，来源为[UER中文RoBERTa Mini](https://huggingface.co/uer/chinese_roberta_L-4_H-256)，是分类编码器，不是生成语言模型；同结构随机初始化对照也完整保留。

安装额外依赖见`requirements-language.txt`。下载的原预训练权重在`runs/language_backbone`，来源revision和哈希在`runs/language_comparison_01/provenance.json`。首次获取可用`huggingface_hub.snapshot_download`，只需要config、vocab、tokenizer配置和模型权重，不执行远程代码。

```powershell
python -m experiments.episodic_memory.diagnose_understanding
python -m experiments.episodic_memory.train_language --out experiments/episodic_memory/runs/my_language_run
python -m experiments.episodic_memory.evaluate_language --trained experiments/episodic_memory/runs/my_language_run --out experiments/episodic_memory/runs/my_language_comparison
python -m experiments.episodic_memory.audit_language_regression --trained experiments/episodic_memory/runs/my_language_run --out experiments/episodic_memory/runs/my_language_audit
python -m experiments.episodic_memory.report_language --trained experiments/episodic_memory/runs/my_language_run --comparison experiments/episodic_memory/runs/my_language_comparison --audit experiments/episodic_memory/runs/my_language_audit

python -m experiments.episodic_memory.demo --checkpoint experiments/episodic_memory/runs/typed_round_02/model.pt --intent-checkpoint experiments/episodic_memory/runs/language_round_02/pretrained_compositional --db experiments/episodic_memory/runs/typed_round_02/restart_demo_template3.sqlite --owner case_a query --question "画册项目第1次讨论，当时和我交流的是谁？"
```

输出目录必须是新目录。`language_round_01`是审计发现正视图跨验证重复的作废试运行，未用于最终测试；有效对照是修复后完整重训的`language_round_02`。测试家族6–9和事件90611/90612未参与微调，旧已暴露失败集只作开发回归。新前端接受更广中文符号，但原地址编码器仍是有限词表；CLI会返回`routing_unknown_tokens`，不能把字段理解能力等同于完整寻址能力。

### 第四轮实现与复现

`diagnose_residuals.py`对旧结果进行真实事件隔离、真实任务槽探针和意图诊断。这些使用标签的干预只存在于诊断脚本。
`train_typed.py`训练独立问题编码器、槽字段解码器和地址/证据判断。写入器及原查询检索向量编码器冻结。`models.py`中启用`typed_reader`时，答案概率显式边缘化事件与字段的概率；启用`grounded_context`时，证据不能高于项目分布一致度乘以次数分布一致度。推理不接收真实任务类别或支持事件ID。
`residual_language.py`分离训练、第一阶段测试和最终测试完整句式。`evaluate_typed.py`为每个模型分别计算问题状态；不要用共享旧问题状态的`evaluate_evidence.py`评估本轮模型。

以下命令依赖前几轮保存的checkpoint与训练库，输出目录须不存在。`demo`的使用方法不变，只需更换checkpoint路径。

```powershell
python -m experiments.episodic_memory.diagnose_residuals
python -m experiments.episodic_memory.train_typed --out experiments/episodic_memory/runs/typed_round_01
python -m experiments.episodic_memory.evaluate_typed --candidate experiments/episodic_memory/runs/typed_round_01 --out experiments/episodic_memory/runs/typed_comparison_01
python -m experiments.episodic_memory.train_typed --base experiments/episodic_memory/runs/typed_round_01 --grounded-context --out experiments/episodic_memory/runs/typed_round_02
python -m experiments.episodic_memory.evaluate_typed --candidate experiments/episodic_memory/runs/typed_round_02 --out experiments/episodic_memory/runs/typed_comparison_02 --seeds 90391 90392 --view-offset 2 --development-comparison experiments/episodic_memory/runs/typed_comparison_01
python -m experiments.episodic_memory.report_typed --comparison experiments/episodic_memory/runs/typed_comparison_02 --run experiments/episodic_memory/runs/typed_round_02
```

`runs/typed_comparison_01`是首阶段新测试，后来用于分析失败，最终报告明确标成开发集；`runs/typed_comparison_02`中的90391/90392与句式2/3才是最终独立复测。原始结果与失败记录均保留。

## 环境与入口

在仓库根目录运行。依赖 Python 3.10+、PyTorch、NumPy、标准库 SQLite。
当前机器已验证 `C:\Python313\python.exe`、PyTorch 2.11.0+cu130、RTX 4060 Laptop 8GB。

```powershell
python -m unittest experiments.episodic_memory.test_memory -v

python -m experiments.episodic_memory.train --out experiments/episodic_memory/runs/my_run --small --steps 5000 --stage-b-steps 500 --batch-size 8 --train-histories 192 --eval-histories 8 --test-events 96

python -m experiments.episodic_memory.evaluate --run experiments/episodic_memory/runs/my_run --split test --count 600
python -m experiments.episodic_memory.evaluate --run experiments/episodic_memory/runs/my_run --split test_template --count 600
python -m experiments.episodic_memory.verify_restart --run experiments/episodic_memory/runs/my_run
python -m experiments.episodic_memory.report --run experiments/episodic_memory/runs/my_run
```

输出目录必须不存在；程序拒绝覆盖已有实验。`runs/` 被本目录 `.gitignore` 忽略，模型、数据库和数据仍实际保存在本机。

`--small` 使用维度128、事件编码器2层、问题编码器1层、读取器1层，约105万参数；不加该参数使用维度192、4/2/2层。每个事件4个记忆槽，64维寻址向量。默认使用 BF16 混合精度（GPU支持时），可用 `--no-amp` 关闭；持久化内容始终为FP16，地址为FP32。

## 事件与问题

输入是受控中文，例如：

```text
画册项目第1次讨论。时间第3天，和小林交流。发生叫错名字。感受尴尬，因为担心评价。关注他人反应。
```

问题例如：

```text
画册项目第1次讨论，当时感受是什么？
画册项目第1次讨论，当时评价的原因是什么？
```

六类答案：人物、日期、事件、感受、原因、关注点，输出57类中的一类（含UNKNOWN）。当前问题带有“项目+第几次讨论”的显式场景线索。不同历史中相同场景的答案重新随机分配，避免只记住场景到答案的固定映射。感受与原因是输入中明确给出的记录，不是推断出真实主观体验。

词表为预先声明的字符和短语片段，包含全部模板的符号，不通过测试答案学习。`demo` 遇到词表外字符会拒绝输入；本模型尚不适用于任意自由对话。

数据集保存监督真值、原文和事件ID；它们只供生成器、训练器和评估器使用。真正的数据库没有答案列或原文列。`demo query` 只加载checkpoint与数据库，不打开数据集。

## 模型

1. 事件Transformer编码当前事件。
2. 四个可训练pooling queries对事件做交叉注意力，得到 `[4, dim]` 内容记忆。辅助任务引导各槽分别关注事件、人物时间、感受原因、关注点。
3. 寻址向量由事件编码器的CLS表示单独投影到64维并归一化；内容槽不再承担保存所有地址线索的责任。可用 `--address-source slots` 重现初始方案。
4. 问题编码器独立产生问题表示和归一化地址。
5. 读取器只接收问题表示、被选择的内容向量和地址相似度。地址的log-softmax作为交叉注意力偏置；答案分类器输出57类。输入中没有正确事件ID、答案或历史文本。

寻址不是查字典：推理时不解析项目/讨论次数，也不查询文字字段，全部候选排序来自学习得到的向量相似度。但默认训练增加了显式场景线索的辅助分类监督，不能宣称无监督学会了自由语言寻址。

## 动态分组

`cluster.py` 在每个owner内部执行在线余弦聚类：

- 相似度达到0.65则加入，否则创建新组。
- 第二相似组同样达阈值、且与第一名差值不超过0.05时，最多允许第二归属。
- 成员超限则做最多10轮二分聚类；退化/失衡时使用确定性的平衡拆分。
- flush时合并小于16成员、中心相似度至少0.90、合并后不超上限的组。
- 只改成员关系和中心，绝不平均或删除事件内容。

索引中心和地址向量可以缓存于CPU，事件内容按需从磁盘读取。此实现扫描分组中心，未声称百万规模的常数时间检索。原型运行默认每组32成员，每次选2组；`MemoryStore`通用默认上限64。最多取8条事件，32个latent tokens进入显存。小模型每次最多读取8KiB内容载荷，不包含地址索引、SQLite页和模型参数。

## 存储和版本

`store.py` 使用SQLite事务同时写入事件、分组、成员关系。事件内容用小端FP16 BLOB，地址用小端FP32 BLOB。表只有：

- `events(event_id, owner, timestamp, memory, address)`
- `groups(owner, cid, centroid)`
- `memberships(owner, cid, event_id)`
- `metadata(key, value)`：schema、writer SHA256、形状、聚类配置。

owner间严格隔离。重复事件ID拒绝覆盖；更正应写新事件。损坏形状、非有限数和编码器版本不匹配会报错。训练中冻结事件编码器后才建库，第二阶段确认writer指纹完全不变。升级writer需要重建库；没有解决已丢弃原文时的表示迁移问题。

默认保留全部事件；分组是访问优化，不是遗忘策略。有限磁盘上的价值筛选与记忆巩固不在本次最小验证中。

## 数据和损失

`generate_data.py`生成完全独立的训练、验证、测试历史。正常模板为0/1；`test_template`只使用未参与训练的模板2。每个owner内场景唯一，每条事件的六类内容独立随机化。每段历史额外生成8条未观察事件作为未知问题，绝不写入数据库。测试中20%问题来自这些未观察事件。重复场景跨owner的映射不同。

默认首轮规模：192个训练历史×32条存入事件=6144条训练事件；每个测试历史96条事件，超过训练历史长度。模型训练不把一个历史拼成超长上下文；每次只处理8个候选短事件。

第一阶段：

- `L_answer`：57类答案交叉熵；20%训练问题遮掉正确事件，目标为UNKNOWN。
- `L_route`：问题地址与候选事件地址的InfoNCE/交叉熵，温度0.07。正确事件的位置每次打乱，负例包含相同项目。
- `L_aux`：各槽内容的六项分类损失的平均。
- `L_context`：事件/问题地址分别预测显式项目和讨论次数，使用相同分类头，平均两端损失。仅在训练期提供监督标签。

最终默认 `L = L_answer + 1.0 L_route + 0.1 L_aux + 1.0 L_context`。
可用 `--context-weight 0 --route-weight 0.2 --address-source slots` 恢复原始损失/寻址基线。
增加context监督是初次实验路由坍塌后的修改，结果报告保留失败对照，不把有监督线索学习当作架构创新。

第一阶段AdamW 3e-4，第二阶段1e-4，梯度裁剪1.0，无梯度累积。所有事件随机初始化学习；没有调用大模型API。

第二阶段：冻结writer并建立磁盘库。真实路由选候选，读取器只拿到这些候选；没有找到支持事件时训练它输出UNKNOWN。路由训练仍可使用监督目标的地址向量改善查找，但不会把目标内容偷偷放进回答候选。离散分组和top-k不反向传播。最终读写路由不是完全可微系统。

第一阶段用验证集总损失选checkpoint，测试集不参与训练和checkpoint选择。第二阶段固定步数，保存最终checkpoint；不要根据测试结果反复挑模型再把它称作独立测试。

## 评估口径

`evaluate.py`报告：

- `oracle`：候选中保证有支持事件，候选位置仍随机。检查编码/读取上限。
- `clustered`：真实分组路由和磁盘读取。
- `flat`：扫描当前owner全部地址，比较分组漏检及速度。
- `zero`：保持候选和地址，内容向量全部清零。
- `shuffled`：保持地址，把候选内容循环错配到其他事件。
- `no_appraisal`：把感受原因槽、关注点槽清零。它只证明对这些槽的依赖，不等同于证明不使用其他槽存储相关信息。

分别报告有记忆问题准确率、Recall@8、找对后的准确率、未知问题错误作答率，以及六类任务的准确率。20%未知问题使“总答UNKNOWN”的总体准确率也是20%；必须看`present_accuracy`，不能只看总体准确率。

时间只包括CPU路由、SQLite内容读取和GPU回答，不包括问题编码；候选地址通常已缓存，因此不是冷盘I/O基准。并行跑其他GPU任务会影响时间。`reopen_payload_exact`验证关闭重开连接后的索引与向量精确一致；独立进程测试使用下面CLI。

## 跨进程记忆演示

将路径替换成自己的实验路径；每条命令是独立进程：

```powershell
python -m experiments.episodic_memory.demo --checkpoint experiments/episodic_memory/runs/my_run/model.pt --db experiments/episodic_memory/runs/my_run/demo.sqlite write --event-id example-001 --text '画册项目第1次讨论。时间第3天，和小林交流。发生叫错名字。感受尴尬，因为担心评价。关注他人反应。'

python -m experiments.episodic_memory.demo --checkpoint experiments/episodic_memory/runs/my_run/model.pt --db experiments/episodic_memory/runs/my_run/demo.sqlite query --question '画册项目第1次讨论，当时感受是什么？'
```

输出的softmax概率没有做校准，不是可靠性保证。UNKNOWN表示没有找回足够可靠证据，不表示事件从未发生。

## 改写泛化与配对测试

新训练选项：`--paraphrases`生成同义子句、随机句序、随机场景位置和提问包装；显式排除诊断模板2和留出模板3的完整问句。声明词表仍为184，不增加新词符号；改变的是这些词在训练中的覆盖和组合。

`--intent-bottleneck`让模型从问题文本预测六类意图，再以预测概率加权六个可学习嵌入作为读取器输入。标签只用于训练损失，不传入推理读取器。问题地址仍由原问题表示产生。这是针对有限六类问题的结构约束，不能直接等同通用意图理解。

`--consistency-weight 0.2`对同一事件的两个训练表述约束内容槽和事件地址一致，同时约束同义问题地址一致。槽内容辅助监督防止仅靠一致性产生无信息表示；是否真正依赖内容仍由清零消融检查。每批额外编码正事件视图，不对所有负例重复编码。

```powershell
python -m experiments.episodic_memory.train --out experiments/episodic_memory/runs/my_paraphrase --small --steps 5000 --stage-b-steps 500 --batch-size 8 --train-histories 192 --eval-histories 8 --test-events 96 --paraphrases --intent-bottleneck --consistency-weight 0.2

python -m experiments.episodic_memory.factorial --run experiments/episodic_memory/runs/my_paraphrase --out experiments/episodic_memory/runs/my_diagnostic --template 2 --count 600
python -m experiments.episodic_memory.factorial --run experiments/episodic_memory/runs/my_paraphrase --out experiments/episodic_memory/runs/my_fresh --template 3 --count 600
python -m experiments.episodic_memory.verify_restart --run experiments/episodic_memory/runs/my_paraphrase --template 3
```

两组factorial各自比较原句、只换问题、只换事件、两者都换。它们从同一test事实生成四种条件，保持owner、目标答案和问题采样相同，变化事件时另建数据库，不覆盖正常库。每次使用新的输出目录。`reference_summary`和`reference_address`是使用原问法内部表示的诊断干预，不是部署模式，也不能算正式推理准确率。

checkpoint保存新增配置，旧checkpoint仍可加载。新writer需要自己的数据库；不能拿新模型直接解释旧模型的记忆向量。原型通过保留实验语料重编码建库，尚未解决原文已销毁时的记忆迁移。

## 文件与实验边界

第三轮关于证据不足与错误拒答的诊断、失败尝试和对比见 [证据实验结果](EVIDENCE_RESULTS.md) 与 [实验方案](EVIDENCE_PROTOCOL.md)。该轮冻结全部编码器，旧数据库可直接复用。

```powershell
# 训练一个新的语义证据读出模型（输出目录必须不存在）
python -m experiments.episodic_memory.train_evidence --base experiments/episodic_memory/runs/paraphrase_structured --out experiments/episodic_memory/runs/my_evidence --steps 3000 --batch-size 16 --semantic --balanced-evidence

# 用上一轮的数据库和新模型查询；新证据模型默认拒答后最多重试一次
python -m experiments.episodic_memory.demo --checkpoint experiments/episodic_memory/runs/my_evidence/verified/model.pt --db experiments/episodic_memory/runs/paraphrase_structured/restart_demo_template3.sqlite --owner case_a query --question '画册项目第1次讨论，回忆那次交流：那次产生的心情是什么？'
```

验证器只看模型学习的地址匹配、场景预测概率一致性和内容幅度，不读取原文、真值场景或答案。条件答案和证据验证分别训练；证据分数需严格大于0.99、条件答案置信度严格大于0.8才能输出答案。验证集只能提高证据门槛。两个数字是决策门槛，不是已保证的真实错误概率。

同一owner内先选2组；UNKNOWN时最多扫描该owner全部地址重试一次。每轮最多读8条内容，共最多16条读取（可能重复）。旧模型可显式加`demo query --retry`使用重试；旧模型默认行为保持不变。排除支持事件的负例测试会同时过滤两轮，避免答案在重试中泄漏。

- `generate_data.py`：受控语料、词表与问题。
- `models.py` / `losses.py`：编码、寻址、读出和训练目标。
- `cluster.py` / `store.py`：动态索引、事务与落盘。
- `train.py`：两阶段训练、checkpoint与运行环境日志。
- `evaluate.py`：留出集、消融、重新打开数据库验证。
- `demo.py`：不读取数据集的独立写入与查询。
- `verify_restart.py`：独立进程保存、读取和同问题不同记忆检查。
- `report.py`：将实测日志和评估JSON生成报告。
- `test_memory.py`：分组退化、原子回滚、owner隔离、版本检查、梯度与冻结测试。
- `paraphrases.py`：训练语言增强与独立留出句式。
- `factorial.py`：四条件配对评估、词汇覆盖与内部表示诊断。
- `compare_paraphrases.py`：核对问题配对关系并从JSON生成对比报告。
- `diagnose_abstention.py`：检索、条件内容候选、候选干扰和单事件读取诊断。
- `evidence_policy.py`：无真值输入的证据判断和一次重试策略。
- `train_evidence.py`：混合磁盘库训练、冻结校验、仅收紧的门槛校准与正例验证。
- `evaluate_evidence.py`：配对回归、移除支持、空载荷和新owner未知事件压力测试。
- `report_evidence.py`：从原始数据生成证据与拒答报告。

本原型还没有验证任意语言理解、多事件因果推理、无限记忆、主动价值遗忘、人格、主观体验或编码器升级。成功的条件是“在受控任务里，答案因磁盘中保存的事件内容而改变”，而不是仅仅能保存向量。
