"""Learned field routing and latent support verification; no historical text reads."""

import hashlib
import json

import numpy as np
import torch
from torch import nn

from .generate_data import FIELDS, TASKS
from .qwen_memory import UNKNOWN, QwenMemory
from .semantic_data import ANSWERS, QUESTIONS

QUERY_SYSTEM = (
    "判断用户问题需要的唯一记忆字段：人物（交流对象姓名）、时间（第几天）、事件（发生的事）、"
    "情绪（我当时的心情）、原因（我的感受产生的缘由）、关注（我注意的方面）。"
    "不支持：未指明项目和第几次讨论、要求多个字段、是非确认、问其他信息。"
    "排除一个无关字段不等于要求多个字段。只判断问题类别，不回答事实。"
)

FACT_READER_SYSTEM = (
    "你是回答经历问题的助手。只根据内部记忆中与问题有关的事实简短作答。"
    "不要添加记忆未提供的人物、日期、动作主体或因果关系。表达可以自然变化。"
)

DECISION_MEANINGS = {
    "unsupported_query": "当前原型未能把问题识别为可处理的单字段查询；不代表没有这段经历。",
    "uncertain_query": "对问题所问字段的识别不够确定，需要澄清；不代表没有记忆。",
    "address_unverified": "检索没有确认与问题匹配的可用记忆；不能断言事情从未发生。",
    "missing_field": "已匹配事件，但目标字段缺失或读取不确定；其他内容可能仍然存在。",
    "conflicting_memories": "匹配记录对所问字段给出不一致的结果，需要核实，不能任选一个。",
    "damaged_memory": "匹配到的记忆数据异常，暂时不能读取。",
    "ambiguous_address": "没有唯一确认可读取的事件记录，需要进一步定位。",
    "uncertain_fact": "读取出的目标事实可信度不足，暂时不能确定。",
    "unsupported_generation": "读取器的表述尚未通过输出检查；不能由此断言记忆不存在。",
}


def status_prompt(question, reason):
    # No gold fact, recalled prose, or rejected candidate is given to this renderer.
    return (
        "请向用户简短解释下面的系统处理状态，可自然措辞，不需要固定句子。"
        "这不是请你回答经历事实。只解释状态，不推测人物、日期、事件、情绪或原因。"
        "不要将识别或读取失败说成用户没有经历过该事件。最多两句话。\n"
        "系统状态：" + DECISION_MEANINGS[reason] + "\n用户原问题（仅作为话题）：" + question
    )


def parse_answer(text, field):
    """Finite declared grammar: rejects negation, extra claims, and wrong-field answers."""
    for value, word in enumerate(FIELDS[TASKS[field]]):
        if text.strip() in {style.format(word) for style in ANSWERS[field]}:
            return value
    return None


class SemanticHeads(nn.Module):
    def __init__(self, hidden, dim=256):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 128), nn.GELU())
        self.queries = nn.Parameter(torch.randn(4, 128) * 0.02)
        self.pool = nn.MultiheadAttention(128, 4, batch_first=True)
        self.intent = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 7))
        self.values = nn.ModuleList([nn.Linear(dim, len(FIELDS[t]) + 1) for t in TASKS])

    def query(self, h, mask):
        h = self.projection(h.float())
        q = self.queries[None].expand(len(h), -1, -1)
        p, _ = self.pool(q, h, h, key_padding_mask=~mask, need_weights=False)
        return self.intent((p + q).flatten(1))


class SemanticMemory(QwenMemory):
    def __init__(
        self,
        backbone,
        device="cuda",
        adapter_path=None,
        semantic_path=None,
        canonical_reader=False,
        contextual_query=False,
    ):
        super().__init__(backbone, device, adapter_path, structured=True)
        self.semantic = SemanticHeads(self.adapter.hidden).to(self.device)
        self.contextual_query = contextual_query
        if contextual_query:
            self.semantic.context_head = nn.Sequential(
                nn.LayerNorm(self.adapter.hidden), nn.Linear(self.adapter.hidden, 7)
            ).to(self.device)
        with torch.no_grad():
            for new, old in zip(self.semantic.values, self.adapter.value_heads):
                new.weight[:-1].copy_(old.weight)
                new.bias[:-1].copy_(old.bias)
        if semantic_path:
            state = torch.load(semantic_path, map_location=self.device, weights_only=True)
            missing, unexpected = self.semantic.load_state_dict(state, strict=False)
            assert not unexpected and all(k.startswith("context_head.") for k in missing)
        self.semantic.eval()
        self.canonical_reader = canonical_reader
        self.route_tasks = None
        self.answer_policy = "legacy_template_gate"
        self.revision_reader = None
        self.label_ids = []
        dot = self.tokenizer.encode("。", add_special_tokens=False)[0]
        for label in ("人物", "时间", "事件", "感受", "原因", "关注"):
            ids = self.tokenizer.encode("\n" + label + "：", add_special_tokens=False)
            assert len(ids) <= 5
            self.label_ids.append(ids + [dot] * (5 - len(ids)))
        self.label_ids = torch.tensor(self.label_ids, device=self.device)

    @classmethod
    def load(cls, run):
        meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        for name in ("adapter", "semantic"):
            assert (
                hashlib.sha256((run / f"{name}.pt").read_bytes()).hexdigest()
                == meta[f"{name}_sha256"]
            )
        model = cls(
            meta["backbone"],
            adapter_path=run / "adapter.pt",
            semantic_path=run / "semantic.pt",
            canonical_reader=meta.get("canonical_reader", False),
            contextual_query=meta.get("contextual_query", False),
        )
        if meta.get("reader_objective") == "available_facts_only":
            model.use_fact_reader_prompt()
        model.answer_policy = meta.get("answer_policy", "legacy_template_gate")
        if meta.get("revision_sha256"):
            from .revision_reader import RevisionReader

            model.revision_reader = RevisionReader.load(model, run, meta["revision_sha256"])
        return model

    @torch.no_grad()
    def write_events(self, store, events, features=None):
        if self.revision_reader is None:
            return super().write_events(store, events, features)
        # The operation reader sees only the current user message, never eval labels.
        store.enable_revisions()
        operations = self.revision_reader.predict([e["text"] for e in events])
        rows = features if features is not None else self.features([e["text"] for e in events])
        encoded = []
        for start in range(0, len(events), 16):
            h, mask = self.padded_features(rows[start:start + 16])
            zs, keys = self.adapter.write(h, mask)
            encoded.extend((z[None], key[None]) for z, key in zip(zs, keys))
        for event, (z, key), op in zip(events, encoded, operations):
            supersedes = []
            field = op["field"]
            if field is not None and op["probability"] > op["threshold"]:
                vp = self.semantic.values[field](z[:, field]).softmax(-1)[0]
                op["value_probability"] = float(vp.max())
                if int(vp.argmax()) < len(FIELDS[TASKS[field]]) and float(vp.max()) > 0.8:
                    index = store._index(event["owner"])
                    ids = list(index.keys)
                    if ids:
                        keys = torch.tensor(
                            np.stack([index.keys[i] for i in ids]), device=self.device
                        )
                        agreement = self.adapter.agreement(key.expand(len(keys), -1), keys)
                        matches = [eid for eid, p in zip(ids, agreement.tolist()) if p > 0.99]
                        supersedes = store.active_ids(event["owner"], matches, field)
                        op["matched_ids"] = matches
            op["applied"] = bool(supersedes)
            store.add_many(
                [
                    dict(
                        id=event["id"],
                        owner=event["owner"],
                        timestamp=event.get("timestamp", 0),
                        memory=z[0].cpu().numpy(),
                        key=key[0].cpu().numpy(),
                        operation=op,
                        supersedes=supersedes,
                    )
                ]
            )

    def use_fact_reader_prompt(self):
        self.before = self.tokenizer.encode(
            "<|im_start|>system\n"
            + FACT_READER_SYSTEM
            + "<|im_end|>\n<|im_start|>user\n内部记忆：",
            add_special_tokens=False,
        )

    @torch.no_grad()
    def respond(self, store, owner, question, threshold=0.8, natural_status=False):
        """Expose controller status separately; no refusal sentence is a training target.

        Existing gates remain conservative. Passing them is NOT a semantic gold label.
        Rejections are structured states, not model answers. Optional natural
        explanation is an experimental draft; the pilot found status hallucinations.
        """
        checked = self.ask_checked(store, owner, question, threshold)
        decision = {k: v for k, v in checked.items() if k not in ("answer", "raw")}
        candidate = checked["raw"] if checked["model_generated"] else None
        if checked["eligible"]:
            answer = checked["answer"]
            source = "typed_latent_fact" if checked.get("fact") else "latent_reader"
            ended = True
        elif natural_status:
            rendered = self.chat(
                status_prompt(question, checked["reason"]), max_new_tokens=100, details=True
            )
            answer, ended = rendered["answer"], rendered["ended"]
            source = "frozen_base_status_explanation"
        else:
            answer, ended = None, None
            source = "controller_status_no_generated_answer"
        return dict(
            answer=answer,
            reader_raw=candidate,
            decision=decision,
            response_source=source,
            ended=ended,
            semantic_correctness="not_automatically_judged",
            status_explanation=DECISION_MEANINGS.get(checked["reason"]),
        )

    def memory_embeddings(self, z, present):
        if self.route_tasks is None:
            raise RuntimeError("Semantic reader requires an inferred or training field")
        all_tokens = self.adapter.soft_tokens(z, present).to(self.base.dtype)
        fields = self.route_tasks.to(self.device)
        idx = fields[:, None] * 4 + torch.arange(4, device=self.device)[None]
        selected = all_tokens.gather(1, idx[:, :, None].expand(-1, -1, all_tokens.shape[-1]))
        labels = self.base.get_input_embeddings()(self.label_ids[fields])
        return torch.cat([labels, selected], 1)

    def teacher_routed(self, z, present, questions, answers, fields):
        self.route_tasks = fields
        if self.canonical_reader:
            questions = [QUESTIONS["train"][int(field)][0] for field in fields]
        try:
            return self.teacher_loss(z, present, questions, answers)
        finally:
            self.route_tasks = None

    @torch.no_grad()
    def generate_routed(self, q, z, field, present=True):
        self.route_tasks = torch.tensor([field], device=self.device)
        if self.canonical_reader:
            q = QUESTIONS["train"][field][0]
        try:
            return self.generate(q, z, present, details=True)
        finally:
            self.route_tasks = None

    @torch.no_grad()
    def query_features(self, questions, batch=8):
        if not self.contextual_query:
            return self.features(questions, batch)
        self.base.eval()
        rows = []
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": QUERY_SYSTEM}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]
        for pos in range(0, len(texts), batch):
            inputs = self.tokenizer(
                texts[pos : pos + batch],
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.device)
            if inputs.input_ids.shape[1] > 320:
                raise ValueError("Query exceeds contextual budget")
            h = self.base.model(**inputs, use_cache=False).last_hidden_state
            for value, mask in zip(h, inputs.attention_mask):
                rows.append(value[mask.bool()][-1:].detach().cpu().half())
        return rows

    def query_distribution(self, h, mask):
        return (
            self.semantic.context_head(h[:, 0].float())
            if self.contextual_query
            else self.semantic.query(h, mask)
        )

    @torch.no_grad()
    def ask_checked(self, store, owner, question, threshold=0.8):
        self.semantic.eval()
        h, mask = self.padded_features(self.query_features([question]))
        qp = self.query_distribution(h, mask).softmax(-1)[0]
        field = int(qp.argmax())
        result = dict(
            answer=UNKNOWN,
            raw=UNKNOWN,
            reason="unsupported_query",
            field=field,
            intent_probability=float(qp[field]),
            support=0.0,
            evidence=0.0,
            ids=[],
            eligible=False,
            action="abstain",
            decision_source="controller",
            model_generated=False,
        )
        if field == 6:
            return result
        result["reason"] = "uncertain_query"
        if float(qp[field]) <= threshold:
            return result
        recalled = self.recall(store, owner, question)
        result.update(evidence=recalled["evidence"], reason="address_unverified")
        if not recalled["accepted"]:
            return result
        # Clustering locates the first event. An address-only scan discovers same-address conflicts.
        if self.contextual_query:
            h, mask = self.padded_features(self.features([question]))
        qkey = self.adapter.key(self.adapter.query_logits(h, mask))[0]
        index = store._index(owner)
        ids = list(index.keys)
        keys = torch.tensor(np.stack([index.keys[i] for i in ids]), device=self.device)
        agreement = self.adapter.agreement(qkey[None].expand(len(keys), -1), keys)
        matched = [eid for eid, p in zip(ids, agreement.tolist()) if p > 0.99]
        result.update(ids=matched, reason="ambiguous_address")
        if getattr(self, "answer_policy", "legacy_template_gate") == "typed_fact":
            active = store.active_ids(owner, matched, field)
            result["superseded_ids"] = [eid for eid in matched if eid not in active]
            matched = active
            result["active_ids"] = active
        if not matched or len(matched) > 8:
            return result
        zs, _ = store.fetch(owner, matched)
        z = torch.tensor(zs, device=self.device)
        result["reason"] = "damaged_memory"
        if not torch.isfinite(z).all() or (z.abs().amax((1, 2)) < 1e-6).any():
            return result
        vp = self.semantic.values[field](z[:, field]).softmax(-1)
        predicted = vp.argmax(-1)
        result.update(value_classes=predicted.tolist(), reason="missing_field")
        if (predicted == len(FIELDS[TASKS[field]])).any():
            return result
        result["reason"] = "conflicting_memories"
        if len(set(predicted.tolist())) > 1:
            return result
        support = min(float(qp[field]), float(vp.max(-1).values.min()))
        result.update(support=support, reason="uncertain_fact")
        if support <= threshold:
            return result
        # All matched records agree on the field. No decoded fact text is given to Qwen.
        chosen = matched.index(recalled["id"]) if recalled["id"] in matched else 0
        generated = self.generate_routed(question, z[chosen], field)
        if getattr(self, "answer_policy", "legacy_template_gate") == "typed_fact":
            # A typed fact is read from the latent value head, not parsed from prose.
            # The unconstrained LM draft is diagnostic only and cannot add assertions.
            value = FIELDS[TASKS[field]][int(predicted[chosen])]
            labels = ("交流对象", "发生日", "事件", "我的感受", "感受的原因", "关注点")
            result.update(
                answer=f"{labels[field]}：{value}。",
                raw=generated["raw"],
                token_min=generated["confidence"],
                model_generated=True,
                draft_ended=generated["ended"],
                draft_used_as_answer=False,
                fact=dict(field=TASKS[field], value=value, evidence_ids=matched),
                eligible=True,
                reason="supported",
                action="answer",
                decision_source="latent_value_head_typed_output",
            )
            return result
        candidate = parse_answer(generated["raw"], field)
        result.update(
            raw=generated["raw"],
            token_min=generated["confidence"],
            reason="unsupported_generation",
            model_generated=True,
        )
        if candidate != int(predicted[chosen]) or not generated["ended"]:
            return result
        result.update(
            answer=generated["raw"],
            eligible=True,
            reason="supported",
            action="answer",
            decision_source="generator_after_controller",
        )
        return result
