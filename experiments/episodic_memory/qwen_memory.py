"""Frozen Qwen language backbone with trainable continuous episodic memories.

The reader gets eight latent vectors, never historical token IDs or event text.
The structured variant decodes them into 32 continuous input embeddings.
Chat bypasses the adapter explicitly. Research prototype, not automatic mode routing.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .store import MemoryStore

UNKNOWN = "我没有这段经历的记忆。"
MEMORY_SYSTEM = "你是有个人经历记忆的助手。根据内部记忆回答问题，不能编造。没有对应记忆时回答：我没有这段经历的记忆。请简短作答。"


class MemoryAdapter(nn.Module):
    def __init__(self, hidden=896, slots=8, dim=256, structured=False):
        super().__init__()
        self.hidden, self.slots, self.dim = hidden, slots, dim
        self.structured = structured
        self.input_projection = nn.Linear(hidden, dim)
        self.slot_queries = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.event_address = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, 32)
        )
        self.query_address = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, 32)
        )
        self.prefix = nn.Linear(dim, hidden * (4 if structured else 1))
        nn.init.normal_(self.prefix.weight, std=0.02)
        nn.init.zeros_(self.prefix.bias)
        self.null_memory = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.prefix_position = nn.Parameter(
            torch.randn(slots * (4 if structured else 1), hidden) * 0.005
        )
        if structured:
            self.value_heads = nn.ModuleList(
                [nn.Linear(dim, n) for n in (8, 16, 8, 8, 8, 8, 16, 16)]
            )
            self.query_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 128))
            self.address_queries = nn.Parameter(torch.randn(2, 128) * 0.02)
            self.address_pool = nn.MultiheadAttention(128, 4, batch_first=True)
            self.address_norm = nn.LayerNorm(128)
            self.address_heads = nn.ModuleList([nn.Linear(128, 16), nn.Linear(128, 16)])

    def write(self, hidden, mask):
        projected = self.input_projection(hidden.float())
        q = self.slot_queries[None].expand(len(hidden), -1, -1)
        pooled, _ = self.pool(q, projected, projected, key_padding_mask=~mask, need_weights=False)
        z = self.norm(pooled + q)
        if self.structured:
            return z, self.key(torch.cat([self.value_heads[j](z[:, j]) for j in (6, 7)], -1))
        mean = (hidden.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return z, self.key(self.event_address(mean))

    @staticmethod
    def key(logits):
        probabilities = logits.reshape(-1, 2, 16).softmax(-1).flatten(1)
        return F.normalize(probabilities, dim=-1)

    @staticmethod
    def agreement(q, k):
        # Recover each normalized semantic distribution from the stored vector.
        q = q.reshape(-1, 2, 16)
        k = k.reshape(-1, 2, 16)
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-8)
        k = k / k.sum(-1, keepdim=True).clamp_min(1e-8)
        return (q * k).sum(-1).prod(-1)

    def soft_tokens(self, z, present):
        selected = torch.where(present[:, None, None], z, self.null_memory[None])
        return (
            0.1 * self.prefix(selected).reshape(len(z), -1, self.hidden)
            + self.prefix_position[None]
        )

    def query_logits(self, hidden, mask):
        if not self.structured:
            mean = (hidden.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            return self.query_address(mean)
        h = self.query_projection(hidden.float())
        q = self.address_queries[None].expand(len(h), -1, -1)
        p, _ = self.address_pool(q, h, h, key_padding_mask=~mask, need_weights=False)
        p = self.address_norm(p + q)
        return torch.cat([head(p[:, i]) for i, head in enumerate(self.address_heads)], -1)


class QwenMemory:
    def __init__(self, backbone, device="cuda", adapter_path=None, structured=False):
        self.backbone_path = Path(backbone)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=False)
        self.base = AutoModelForCausalLM.from_pretrained(
            backbone,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
            trust_remote_code=False,
        ).to(device)
        self.base.requires_grad_(False)
        self.base.eval()
        state = (
            torch.load(adapter_path, map_location=device, weights_only=True)
            if adapter_path
            else None
        )
        structured = structured or (state is not None and "value_heads.0.weight" in state)
        self.adapter = MemoryAdapter(self.base.config.hidden_size, structured=structured).to(device)
        if adapter_path:
            self.adapter.load_state_dict(state)
        self.adapter.eval()
        self.before = self.tokenizer.encode(
            "<|im_start|>system\n" + MEMORY_SYSTEM + "<|im_end|>\n<|im_start|>user\n内部记忆：",
            add_special_tokens=False,
        )

    @torch.no_grad()
    def features(self, texts, batch=8):
        rows = []
        self.base.eval()
        for start in range(0, len(texts), batch):
            inputs = self.tokenizer(
                texts[start : start + batch], padding=True, return_tensors="pt", truncation=False
            ).to(self.device)
            if inputs.input_ids.shape[1] > 192:
                raise ValueError("Event/query exceeds the declared feature budget")
            h = self.base.model(**inputs, use_cache=False).last_hidden_state
            for value, mask in zip(h, inputs.attention_mask):
                rows.append(value[mask.bool()].detach().cpu().half())
        return rows

    def padded_features(self, rows):
        h = torch.zeros(
            len(rows), max(len(r) for r in rows), self.base.config.hidden_size, device=self.device
        )
        mask = torch.zeros(h.shape[:2], dtype=torch.bool, device=self.device)
        for i, row in enumerate(rows):
            h[i, : len(row)] = row.to(self.device).float()
            mask[i, : len(row)] = True
        return h, mask

    def teacher_loss(self, z, present, questions, answers):
        prefix = self.memory_embeddings(z, present)
        before = self.base.get_input_embeddings()(torch.tensor(self.before, device=self.device))
        rows, targets = [], []
        for i, (q, a) in enumerate(zip(questions, answers)):
            after = self.tokenizer.encode(
                "\n问题：" + q + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
            )
            answer = self.tokenizer.encode(a + "<|im_end|>", add_special_tokens=False)
            suffix = self.base.get_input_embeddings()(
                torch.tensor(after + answer, device=self.device)
            )
            rows.append(torch.cat([before, prefix[i], suffix], 0))
            targets.append([-100] * (len(before) + len(prefix[i]) + len(after)) + answer)
        length = max(len(r) for r in rows)
        embeds = torch.stack([F.pad(r, (0, 0, 0, length - len(r))) for r in rows])
        attention = torch.tensor(
            [[1] * len(r) + [0] * (length - len(r)) for r in rows], device=self.device
        )
        labels = torch.tensor([r + [-100] * (length - len(r)) for r in targets], device=self.device)
        h = self.base.model(
            inputs_embeds=embeds, attention_mask=attention, use_cache=False
        ).last_hidden_state
        # Only project the positions predicting answer tokens; no full-sequence vocab tensor.
        valid = labels[:, 1:] != -100
        logits = self.base.lm_head(h[:, :-1][valid])
        return F.cross_entropy(logits.float(), labels[:, 1:][valid])

    def memory_embeddings(self, z, present):
        prefix = self.adapter.soft_tokens(z, present).to(self.base.dtype)
        if not self.adapter.structured:
            return prefix
        pieces = []
        for i, label in enumerate(("人物", "时间", "事件", "感受", "原因", "关注", "场景", "次数")):
            ids = self.tokenizer.encode("\n" + label + "：", add_special_tokens=False)
            fixed = self.base.get_input_embeddings()(torch.tensor(ids, device=self.device))
            pieces.extend([fixed[None].expand(len(z), -1, -1), prefix[:, i * 4 : (i + 1) * 4]])
        return torch.cat(pieces, 1)

    @torch.no_grad()
    def generate(
        self, question, z=None, present=False, max_new_tokens=40, details=False, trace=False
    ):
        self.base.eval()
        self.adapter.eval()
        if z is None:
            z = torch.zeros(1, self.adapter.slots, self.adapter.dim, device=self.device)
        elif z.ndim == 2:
            z = z[None]
        prefix = self.memory_embeddings(
            z.to(self.device), torch.tensor([present], device=self.device)
        ).to(self.base.dtype)
        after = self.tokenizer.encode(
            "\n问题：" + question + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
        )
        embeds = torch.cat(
            [
                self.base.get_input_embeddings()(torch.tensor(self.before, device=self.device))[
                    None
                ],
                prefix,
                self.base.get_input_embeddings()(torch.tensor(after, device=self.device))[None],
            ],
            1,
        )
        output = self.base.generate(
            inputs_embeds=embeds,
            attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        ids = output.sequences[0, -len(output.scores) :]
        text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        probabilities = [
            float(score[0].float().softmax(-1)[token]) for score, token in zip(output.scores, ids)
        ]
        ended = bool(len(ids) and int(ids[-1]) == self.tokenizer.eos_token_id)
        confidence = min(probabilities, default=0.0) if ended else 0.0
        result = dict(
            raw=text,
            confidence=confidence,
            ended=ended,
            tokens=len(ids),
            answer=text if confidence > 0.8 else UNKNOWN,
        )
        if trace:
            # Diagnostics only: preserve production decoding and decision exactly.
            result["trace"] = []
            for i, (score, token) in enumerate(zip(output.scores, ids)):
                p = score[0].float().softmax(-1)
                values, indices = p.topk(5)
                result["trace"].append(
                    dict(
                        index=i,
                        token_id=int(token),
                        token=self.tokenizer.decode([int(token)]),
                        probability=float(p[token]),
                        prefix=self.tokenizer.decode(ids[:i], skip_special_tokens=True),
                        through=self.tokenizer.decode(ids[: i + 1], skip_special_tokens=True),
                        alternatives=[
                            dict(
                                token_id=int(t),
                                token=self.tokenizer.decode([int(t)]),
                                probability=float(v),
                            )
                            for t, v in zip(indices, values)
                        ],
                    )
                )
        return result if details else text

    @torch.no_grad()
    def chat(self, text, max_new_tokens=100, details=False):
        self.base.eval()
        ids = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "你是一个乐于助人的助手。"},
                {"role": "user", "content": text},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        output = self.base.generate(
            **ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        generated = output[0, ids.input_ids.shape[1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if details:
            return dict(
                answer=answer,
                tokens=len(generated),
                ended=bool(len(generated) and int(generated[-1]) == self.tokenizer.eos_token_id),
            )
        return answer

    def version(self):
        # Retain the legacy namespace for old databases; the revision below distinguishes backbones.
        digest = hashlib.sha256(b"qwen2.5-0.5b-latent-memory-v1")
        metadata = self.backbone_path / ".cache/huggingface/download/model.safetensors.metadata"
        digest.update(metadata.read_text().splitlines()[0].encode())
        for name, t in self.adapter.state_dict().items():
            if name.startswith(
                (
                    "input_projection.",
                    "slot_queries",
                    "pool.",
                    "norm.",
                    "event_address.",
                    "value_heads.",
                )
            ):
                digest.update(name.encode())
                digest.update(t.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def store(self, path, create=False):
        return MemoryStore(
            path,
            self.version(),
            (self.adapter.slots, self.adapter.dim),
            32,
            create=create,
            threshold=0.45,
            max_members=8,
        )

    @torch.no_grad()
    def write_events(self, store, events, features=None):
        rows = features if features is not None else self.features([e["text"] for e in events])
        for start in range(0, len(events), 16):
            h, mask = self.padded_features(rows[start : start + 16])
            z, k = self.adapter.write(h, mask)
            store.add_many(
                [
                    dict(
                        id=e["id"],
                        owner=e["owner"],
                        timestamp=e.get("timestamp", i),
                        memory=z[i].cpu().numpy(),
                        key=k[i].cpu().numpy(),
                    )
                    for i, e in enumerate(events[start : start + 16])
                ]
            )

    @torch.no_grad()
    def recall(self, store, owner, question, threshold=0.99, exclude=()):
        h, mask = self.padded_features(self.features([question]))
        q = self.adapter.key(self.adapter.query_logits(h, mask))[0]
        attempts = []
        best = None
        for flat in (False, True):
            ids, count = store.retrieve(
                owner, q.cpu().numpy(), top_k=4, top_clusters=2, flat=flat, exclude=exclude
            )
            z, k = store.fetch(owner, ids)
            scores = self.adapter.agreement(
                q[None].expand(len(k), -1), torch.tensor(k, device=self.device)
            )
            selected = int(scores.argmax())
            confidence = float(scores[selected]) if ids else 0.0
            attempts.append(dict(ids=ids, candidates=count, evidence=confidence, flat=flat))
            best = dict(
                id=ids[selected] if ids else None,
                memory=z[selected],
                evidence=confidence,
                accepted=bool(ids) and confidence > threshold,
                attempts=attempts,
            )
            if best["accepted"]:
                break
        return best

    def ask(self, store, owner, question):
        recalled = self.recall(store, owner, question)
        z = torch.tensor(recalled.pop("memory"), device=self.device)
        generated = self.generate(question, z, recalled["accepted"], details=True)
        if not recalled["accepted"]:
            generated["answer"] = UNKNOWN
        return dict(**recalled, **generated)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
