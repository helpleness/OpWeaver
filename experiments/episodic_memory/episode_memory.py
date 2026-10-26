"""Ordered event-fragment latent memory: no field or revision classifiers.

This first fidelity experiment preserves sequence length. It compresses embedding
width, not duration; it is NOT a constant-size event summary or unlimited context.
SQLite stores learned float16 codes, addresses, speaker and insertion order only.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .semantic_memory import SemanticMemory
from .store import MemoryStore

EPISODE_SYSTEM = (
    "你是帮助用户回顾经历的助手。下面是同一事件按叙述先后排列的内部记忆片段。"
    "请理解完整内容及前后关系，再回答用户问题。用户明确纠正此前说法时采用纠正后的内容；"
    "只提出另一种可能、尚未确认或转述别人的建议，不等于推翻旧说法。"
    "无法确定的分歧要如实说明。未记录的内容不要猜测，不能把交流对象自动当事件行为者。"
    "自然简短作答，不需要固定句式。"
)


class SequenceCodec(nn.Module):
    def __init__(self, hidden, width=256):
        super().__init__()
        self.hidden, self.width = hidden, width
        self.encoder = nn.Linear(hidden, width, bias=False)
        self.decoder = nn.Linear(width, hidden, bias=False)
        # Context can affect codes while token order remains explicit.
        self.context = nn.Conv1d(width, width, 3, padding=1, groups=width, bias=False)
        nn.init.zeros_(self.context.weight)

    def encode(self, embeddings):
        z = self.encoder(embeddings.float())
        return z + self.context(z.transpose(1, 2)).transpose(1, 2)

    def decode(self, z):
        return self.decoder(z.float())


class EpisodeStore(MemoryStore):
    def __init__(self, path, version, width, create=False):
        super().__init__(path, version, (0, width), 32, create, threshold=0.45, max_members=8)
        with self.db:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS fragment_meta(event_id TEXT PRIMARY KEY REFERENCES events(event_id), speaker TEXT NOT NULL)"
            )

    def encode_memory(self, z):
        if z.ndim != 2 or z.shape[1] != self.shape[1] or not 0 < len(z) <= 192:
            raise ValueError("Invalid variable-length memory")
        return z.tobytes()

    def decode_memory(self, raw):
        return np.frombuffer(raw, dtype="<f2").reshape(-1, self.shape[1]).astype(np.float32)

    def collate_memories(self, rows):
        return rows

    def add_metadata(self, record):
        speaker = record.get("speaker", "user")
        if speaker not in ("user", "assistant"):
            raise ValueError("Unsupported speaker provenance")
        self.db.execute("INSERT INTO fragment_meta VALUES (?,?)", (record["id"], speaker))

    def ordered(self, owner, ids):
        wanted = set(ids)
        return [
            (eid, speaker)
            for eid, speaker in self.db.execute(
                "SELECT e.event_id,m.speaker FROM events e JOIN fragment_meta m USING(event_id) WHERE e.owner=? ORDER BY e.rowid",
                (owner,),
            )
            if eid in wanted
        ]


class EpisodeMemory:
    def __init__(self, backbone_model, codec=None):
        self.model = backbone_model
        self.base, self.tokenizer, self.device = (
            self.model.base,
            self.model.tokenizer,
            self.model.device,
        )
        self.codec = codec or SequenceCodec(self.base.config.hidden_size).to(self.device)
        self.address_heads = None

    @classmethod
    def load(cls, run, backbone_model=None):
        meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        model = backbone_model or SemanticMemory.load(Path(meta["source_run"]))
        codec = SequenceCodec(model.base.config.hidden_size, meta["width"]).to(model.device)
        assert hashlib.sha256((run / "codec.pt").read_bytes()).hexdigest() == meta["codec_sha256"]
        codec.load_state_dict(
            torch.load(run / "codec.pt", map_location=model.device, weights_only=True)
        )
        result = cls(model, codec)
        if meta.get("address_sha256"):
            assert (
                hashlib.sha256((run / "address.pt").read_bytes()).hexdigest()
                == meta["address_sha256"]
            )
            result.address_heads = nn.ModuleList(
                [copy.deepcopy(model.adapter.value_heads[i]) for i in (6, 7)]
            )
            result.address_heads.load_state_dict(
                torch.load(run / "address.pt", map_location=model.device, weights_only=True)
            )
        return result

    def version(self):
        h = hashlib.sha256(("ordered_episode_v1:" + self.model.version()).encode())
        for name, tensor in self.codec.state_dict().items():
            if not name.startswith("decoder."):
                h.update(name.encode())
                h.update(tensor.detach().cpu().numpy().tobytes())
        if self.address_heads is not None:
            for tensor in self.address_heads.state_dict().values():
                h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def store(self, path, create=False):
        return EpisodeStore(path, self.version(), self.codec.width, create)

    def embed_text(self, text):
        ids = torch.tensor(
            self.tokenizer.encode(text, add_special_tokens=False), device=self.device
        )
        return self.base.get_input_embeddings()(ids).detach()

    def encode_text(self, text):
        embeds = self.embed_text(text)
        if not 0 < len(embeds) <= 192:
            raise ValueError("Fragment exceeds 192-token writer budget; split explicitly")
        return self.codec.encode(embeds[None])[0]

    @torch.no_grad()
    def write_events(self, store, events):
        self.codec.eval()
        features = self.model.features([e["text"] for e in events])
        for start in range(0, len(events), 16):
            h, mask = self.model.padded_features(features[start : start + 16])
            z, keys = self.model.adapter.write(h, mask)
            if self.address_heads is not None:
                keys = self.model.adapter.key(
                    torch.cat([head(z[:, i + 6]) for i, head in enumerate(self.address_heads)], -1)
                )
            store.add_many(
                [
                    dict(
                        id=e["id"],
                        owner=e["owner"],
                        timestamp=e.get("timestamp", 0),
                        speaker=e.get("speaker", "user"),
                        memory=self.encode_text(e["text"]).half().cpu().numpy(),
                        key=k.cpu().numpy(),
                    )
                    for e, k in zip(events[start : start + 16], keys)
                ]
            )

    def prefix(self, fragments, question):
        pieces = [
            self.embed_text(
                "<|im_start|>system\n" + EPISODE_SYSTEM + "<|im_end|>\n<|im_start|>user\n"
            )
        ]
        for n, (z, speaker) in enumerate(fragments):
            pieces.append(
                self.embed_text(f"\n片段{n + 1}（{'用户' if speaker == 'user' else '助手'}）：")
            )
            pieces.append(self.codec.decode(z).to(self.base.dtype))
        if not fragments:
            pieces.append(self.embed_text("没有提供与本次问题匹配的事件记忆。"))
        pieces.append(
            self.embed_text("\n问题：" + question + "<|im_end|>\n<|im_start|>assistant\n")
        )
        return torch.cat(pieces)

    def training_loss(self, fragments, question, answer):
        prefix = self.prefix(fragments, question)
        ids = torch.tensor(
            self.tokenizer.encode(answer + "<|im_end|>", add_special_tokens=False),
            device=self.device,
        )
        embeds = torch.cat([prefix, self.base.get_input_embeddings()(ids)], 0)
        h = self.base.model(inputs_embeds=embeds[None], use_cache=False).last_hidden_state[0]
        logits = self.base.lm_head(h[len(prefix) - 1 : -1])
        return F.cross_entropy(logits.float(), ids)

    @torch.no_grad()
    def generate(self, fragments, question, max_new_tokens=100):
        self.base.eval()
        self.codec.eval()
        embeds = self.prefix(fragments, question)
        if len(embeds) + max_new_tokens > min(4096, self.base.config.max_position_embeddings):
            return dict(
                answer=None,
                ended=None,
                status="event_budget_exceeded",
                input_tokens=len(embeds),
                status_explanation="该事件片段超出本轮读取预算，尚未进行压缩归纳。",
            )
        output = self.base.generate(
            inputs_embeds=embeds[None],
            attention_mask=torch.ones(1, len(embeds), dtype=torch.long, device=self.device),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        ids = output[0]
        return dict(
            answer=self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
            ended=bool(len(ids) and int(ids[-1]) == self.tokenizer.eos_token_id),
            input_tokens=len(embeds),
            output_tokens=len(ids),
            status="generated",
            response_source="ordered_episode_latents",
            semantic_correctness="pending_external_review",
        )

    @torch.no_grad()
    def respond(self, store, owner, question):
        h, mask = self.model.padded_features(self.model.features([question]))
        qkey = self.model.adapter.key(self.model.adapter.query_logits(h, mask))[0]
        index = store._index(owner)
        # Clusters nominate the event; same-address fragments are then gathered.
        candidates, _ = store.retrieve(owner, qkey.cpu().numpy(), top_k=8, top_clusters=4)

        def matches(ids):
            if not ids:
                return []
            keys = torch.tensor(np.stack([index.keys[eid] for eid in ids]), device=self.device)
            scores = self.model.adapter.agreement(qkey[None].expand(len(keys), -1), keys)
            return [eid for eid, score in zip(ids, scores.tolist()) if score > 0.99]

        first = matches(candidates)
        if not first:
            first = matches(list(index.keys))
        ids = matches(list(index.keys)) if first else []
        ordered = store.ordered(owner, ids)
        zs, _ = store.fetch(owner, [eid for eid, _ in ordered]) if ordered else ([], [])
        fragments = [
            (torch.tensor(z, device=self.device), speaker) for z, (_, speaker) in zip(zs, ordered)
        ]
        result = self.generate(fragments, question)
        result["retrieval"] = dict(
            ids=[eid for eid, _ in ordered],
            threshold=0.99,
            fragments=len(fragments),
            raw_text_read=False,
        )
        return result
