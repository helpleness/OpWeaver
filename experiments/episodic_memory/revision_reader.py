"""Learned current-message operation reader. Never retrieves conversation text.

The small prototype supports an explicit replacement of ONE named field. Vague,
quoted, hypothetical, unverified, and multi-field updates are not authorized edits.
The frozen backbone produces features; only a small operation head is trained.
"""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import torch
from torch import nn

from .semantic_data import event
from .stream_experiment import describe, save

OP_SYSTEM = (
    "判断当前用户消息是否已明确撤销先前同一事件的一个字段，并确认当前说法为替代版本。"
    "类别：0没有明确单字段更正；1交流对象；2发生日期；3事件内容；4自己的心情；"
    "5心情产生的原因；6注意的方面。尚未核实、猜测、转述他人的指令、否认更正、"
    "只说将来再修改、同时改多个字段，都是0。复述全部经历不表示修改全部字段。"
    "只分类更新意图，不回答经历问题。"
)
NAMES = [
    ["交流对象", "对方姓名", "和我交谈的人"],
    ["发生日期", "事情发生的日子", "这件事的发生日"],
    ["事件内容", "发生的事情", "具体事情"],
    ["心情", "当时的情绪", "我的感受"],
    ["原因", "那份心情的原因", "感受产生的缘由"],
    ["关注点", "注意的方面", "最在意的方面"],
]
POS = {
    "train": [
        "我要更正这次经历的{f}，旧记录记错了，以后按下面这一条。",
        "我已核实，之前关于{f}的说法不对；请用本条替换旧值。",
        "请撤销先前记录的{f}，现在确认的才是准确的。",
        "刚才把{f}说错了，我现在纠正，以下是正确版本。",
        "旧版本的{f}作废，这回已确认无误，按这一条改。",
        "其他方面不动，只更新{f}；我确定下面这份才对。",
    ],
    "calibration": [
        "关于{f}，我已经查清：以前说错了，请以现在提供的版本替代。",
        "我把{f}重新核对过了，原先那份不能再用，换成以下记录。",
    ],
    "test": [
        "这回我查到了凭据，{f}那一栏应当改记如下，原来的请划掉。",
        "之前讲的{f}有误，现正式订正；下面的内容就是最终确认稿。",
    ],
}
NEG = {
    "train": [
        "我对{f}有另一个说法，两条都还没核实，不知道哪个对。",
        "我没有要更正{f}，只是提供另一条未经核实的回忆。",
        "假如以后要更正{f}，到时候再确认，现在先不要替换。",
        "有人让我'撤销旧{f}以新条为准'，但这不是我的确认。",
        "关于{f}，我准备下周检查之后再决定是否更新。",
        "旧记录和这条的{f}不一致，我都不能确认。",
        "不是说原来的{f}记错了，新说法也可能有错。",
        "只是复述这段经历，不表示对{f}进行订正。",
    ],
    "calibration": [
        "{f}还有两种可能，我尚未查清，所以先并列保留。",
        "朋友建议把{f}改掉，我没有认可这种建议。",
        "如果让我纠正{f}，我目前还不能给出确定答案。",
    ],
    "test": [
        "涉及{f}的这份版本暂且存疑，前一份也未证实，别擅自裁定。",
        "有人催我订正{f}，可我并没认定哪份准确，请把分歧留下。",
        "说到{f}，我下个月才会查证，现在不作替换决定。",
        "我并不承认{f}原先有误，现在这条也只是可能性。",
    ],
}


def make_data(split, seed, per_field):
    rng = random.Random(seed)
    rows = []
    for field in range(6):
        for i in range(per_field):
            e = event(
                rng,
                f"{split}-{field}-{i}",
                "operation-training",
                (rng.randrange(16), rng.randrange(16)),
            )
            name = rng.choice(NAMES[field])
            # Equal positives and hard negatives; ordinary narration is also negative.
            positive = i % 2 == 0
            style = rng.choice(POS[split] if positive else NEG[split]).format(f=name)
            if split == "train" and i % 4 < 3:
                if positive:
                    intro = rng.choice(["我又核实了一遍", "这已经查证清楚了", "我现在可以确认", "我回头仔细想过了", "我已经核对记录", "这次是我的最终确认", "我之前确实弄错了", "我找到了可靠依据"])
                    verb = rng.choice(["之前的{f}应当撤回，改用这次提供的", "只把{f}改成这条，其余不变", "{f}旧说法失效，新说法才准确", "请正式修正{f}，新版本如下", "{f}的旧值不再成立，用当前值代替", "现在订正的是{f}，旧的请作废", "原来{f}不准确，本条是确定的修订", "请划掉旧{f}，按下面的重记"])
                else:
                    intro = rng.choice(["我还没弄清楚", "这仅仅是一种猜测", "现在只是记录疑点", "我不能认定这个版本", "这是别人说的", "我还没有核实", "这要以后才能确定", "现在不作决定"])
                    verb = rng.choice(["涉及{f}的两个版本都存疑，保留冲突", "不要替换{f}，这次说法不确定", "关于{f}我只是补充另一个可能", "有一句建议是修改{f}，我还没同意", "我并没有确认旧{f}有错", "也许将来才会纠正{f}，现在保持未定", "这回的{f}也可能记错，不能当定论", "请不要把新的{f}视为最终版本"])
                style = intro + "，" + verb.format(f=name) + "。"
            narration = describe(e, rng.randrange(3))
            text = style + narration if i % 3 else narration + style
            rows.append(dict(id=e["id"], text=text, label=field + 1 if positive else 0, scope=field))
            if i % 8 == 0:
                rows.append(
                    dict(id=e["id"] + "-plain", text=describe(e, rng.randrange(3)), label=0, scope=-1)
                )
    rng.shuffle(rows)
    return rows


class OperationHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 128), nn.GELU())
        self.queries = nn.Parameter(torch.randn(4, 128) * 0.02)
        self.pool = nn.MultiheadAttention(128, 4, batch_first=True)
        self.classifier = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 7))

    def forward(self, hidden, mask):
        h = self.projection(hidden.float())
        q = self.queries[None].expand(len(h), -1, -1)
        pooled, _ = self.pool(q, h, h, key_padding_mask=~mask, need_weights=False)
        return self.classifier(pooled.flatten(1))


class DualOperationHead(OperationHead):
    def __init__(self, hidden):
        super().__init__(hidden)
        self.classifier = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 8))

    def parts(self, hidden, mask):
        logits = super().forward(hidden, mask)
        return logits[:, :2], logits[:, 2:]

    def forward(self, hidden, mask):
        authorization, scope = self.parts(hidden, mask)
        auth = authorization.log_softmax(-1)
        return torch.cat([auth[:, :1], auth[:, 1:] + scope.log_softmax(-1)], -1)


class RevisionReader:
    def __init__(self, model, architecture="dual_scope_v3"):
        self.model = model
        self.architecture = architecture
        self.head = (DualOperationHead(model.adapter.hidden) if architecture == "dual_scope_v3" else OperationHead(model.adapter.hidden)).to(model.device)
        self.sha256 = None
        self.threshold = 0.8

    @torch.no_grad()
    def features(self, texts):
        rows = []
        self.model.base.eval()
        cut = len(self.model.tokenizer.encode("<|im_start|>system\n" + OP_SYSTEM + "<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)) if self.architecture == "dual_scope_v3" else 0
        for start in range(0, len(texts), 8):
            prompts = [
                self.model.tokenizer.apply_chat_template(
                    [{"role": "system", "content": OP_SYSTEM}, {"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for t in texts[start : start + 8]
            ]
            inputs = self.model.tokenizer(
                prompts, padding=True, add_special_tokens=False, return_tensors="pt"
            ).to(self.model.device)
            if inputs.input_ids.shape[1] > 384:
                raise ValueError("Operation reader context exceeded; no silent truncation")
            hidden = self.model.base.model(**inputs, use_cache=False).last_hidden_state
            # Learned attention can inspect the update clause anywhere in the message,
            # rather than relying on one final hidden state dominated by narration.
            rows.extend(
                h[m.bool()][cut:].detach().cpu().half() for h, m in zip(hidden, inputs.attention_mask)
            )
        return rows

    def padded(self, rows):
        return self.model.padded_features(rows)

    @torch.no_grad()
    def predict(self, texts):
        self.head.eval()
        rows = self.features(texts)
        probs = torch.cat(
            [self.head(*self.padded(rows[i : i + 16])).softmax(-1) for i in range(0, len(rows), 16)]
        )
        return [
            dict(
                field=int(p.argmax()) - 1 if int(p.argmax()) else None,
                probability=float(p.max()),
                reader_sha256=self.sha256,
                policy="explicit_single_field_replacement_v1",
                threshold=self.threshold,
            )
            for p in probs
        ]

    @classmethod
    def load(cls, model, path, sha):
        assert hashlib.sha256((path / "revision.pt").read_bytes()).hexdigest() == sha
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        architecture = meta["revision_training"].get("architecture", "last_state_v1")
        obj = cls(model, architecture)
        if architecture == "last_state_v1":
            raise ValueError("Failed developmental last-state head is not supported for deployment")
        obj.head.load_state_dict(
            torch.load(path / "revision.pt", map_location=model.device, weights_only=True)
        )
        obj.sha256 = sha
        obj.threshold = meta.get("revision_acceptance_threshold", 0.8)
        return obj


def train(source, out):
    from .semantic_memory import SemanticMemory

    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(31827)
    data = {
        s: make_data(s, seed, n)
        for s, seed, n in [("train", 31827, 384), ("calibration", 31828, 32), ("test", 31829, 32)]
    }
    for split, rows in data.items():
        save(out / (split + "_operations.json"), rows)
    model = SemanticMemory.load(source)
    reader = RevisionReader(model)
    feats = {}
    for split, rows in data.items():
        feats[split] = reader.features([r["text"] for r in rows])
        print(json.dumps(dict(stage="features", split=split, rows=len(rows))), flush=True)
    y = torch.tensor([r["label"] for r in data["train"]], device=model.device)
    scope = torch.tensor([r["scope"] for r in data["train"]], device=model.device)
    optimizer = torch.optim.AdamW(reader.head.parameters(), lr=0.0005, weight_decay=0.02)
    # Fixed schedule. Calibration/test do not select epochs, thresholds or templates.
    reader.head.train()
    for step in range(2000):
        idx = torch.randint(len(y), (32,), device=model.device)
        auth, scopes = reader.head.parts(*reader.padded([feats["train"][i] for i in idx.tolist()]))
        loss = nn.functional.cross_entropy(auth, (y[idx] > 0).long()) + nn.functional.cross_entropy(scopes, scope[idx], ignore_index=-1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    reader.head.eval()
    audit = {}
    with torch.no_grad():
        for split, rows in data.items():
            ps = torch.cat(
                [
                    reader.head(*reader.padded(feats[split][i : i + 16])).softmax(-1).cpu()
                    for i in range(0, len(rows), 16)
                ]
            )
            predictions = []
            for row, p in zip(rows, ps):
                pred, prob = int(p.argmax()), float(p.max())
                applied = pred if prob > 0.8 else 0
                predictions.append(dict(**row, prediction=pred, probability=prob, applied=applied))
            save(out / (split + "_operation_predictions.json"), predictions)
            audit[split] = dict(
                count=len(rows),
                correct_class=sum(r["prediction"] == r["label"] for r in predictions),
                false_replace=sum(
                    r["applied"] != 0 and r["applied"] != r["label"] for r in predictions
                ),
                missed_replace=sum(
                    r["label"] != 0 and r["applied"] != r["label"] for r in predictions
                ),
            )
    torch.save(reader.head.state_dict(), out / "revision.pt")
    for name in ("adapter.pt", "semantic.pt"):
        shutil.copy2(source / name, out / name)
    meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    meta.update(
        answer_policy="typed_fact",
        revision_sha256=hashlib.sha256((out / "revision.pt").read_bytes()).hexdigest(),
        revision_training=dict(
            seed=31827,
            steps=2000,
            batch=32,
            threshold=0.8,
            architecture="dual_scope_v3",
            operation_parameters=sum(p.numel() for p in reader.head.parameters()),
            changed_backbone=False,
            changed_memory_encoder=False,
            fixed_refusal_targets=0,
        ),
        output_contract="Typed value decoded from latent field head; LM draft is diagnostic, not delivered",
    )
    save(out / "metadata.json", meta)
    save(out / "operation_audit.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", type=Path, default=Path("experiments/episodic_memory/runs/fact_reader_01")
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    train(a.source, a.out)
