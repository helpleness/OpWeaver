"""Refine existing closed-vocabulary event-address heads, never update intent.

Only predicts the shared event reference. No modification/confirmation/field
labels are given. Content remains immutable, ordered latent fragments.
"""

import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .generate_data import ORDINALS, PROJECTS
from .semantic_memory import SemanticMemory
from .stream_experiment import save
from .train_episode_memory import data


def main():
    out = Path("experiments/episodic_memory/runs/episode_identity_01")
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(57341)
    model = SemanticMemory.load(Path("experiments/episodic_memory/runs/fact_reader_01"))
    all_data, features, targets = {}, {}, {}
    for split, seed, count in [("train", 57341, 768), ("validation", 57342, 192)]:
        cases = data(seed=seed, count=count)
        rows = []
        for c in cases:
            project = next(
                i for i, p in enumerate(PROJECTS) if c["question"].startswith(p + "项目")
            )
            ordinal = next(i for i, o in enumerate(ORDINALS) if o + "讨论" in c["question"])
            rows.extend(dict(text=t, address=[project, ordinal]) for t in c["texts"])
        # No event reference: a diffuse key cannot pass the unchanged .99 matcher.
        chatter = (
            [
                "我们稍后再聊，我先休息一下。",
                "今天想整理一下心情，不用马上给建议。",
                "你可以先听我慢慢讲。",
                "我还没有说到具体是哪件事情。",
                "谢谢，我先记下来，回头继续。",
                "这段时间事情很多，暂时还没理出顺序。",
            ]
            if split == "train"
            else ["话题先放一下，今天就聊到这儿。", "还没有指定要讨论哪个事件。"]
        )
        rows.extend(dict(text=t, address=[-1, -1]) for t in chatter * 8)
        all_data[split] = rows
        save(out / (split + ".json"), rows)
        hs = model.features([r["text"] for r in rows])
        zs = []
        with torch.no_grad():
            for i in range(0, len(hs), 32):
                h, mask = model.padded_features(hs[i : i + 32])
                z, _ = model.adapter.write(h, mask)
                zs.append(z[:, 6:8].detach())
        features[split] = torch.cat(zs)
        targets[split] = torch.tensor([r["address"] for r in rows], device=model.device)
        print(json.dumps(dict(stage="identity_features", split=split, rows=len(rows))), flush=True)
    heads = nn.ModuleList([copy.deepcopy(model.adapter.value_heads[i]) for i in (6, 7)])
    heads.requires_grad_(True)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=0.002, weight_decay=0.01)
    for step in range(800):
        idx = torch.randint(len(features["train"]), (64,), device=model.device)
        labels = targets["train"][idx]
        loss = 0
        for field, head in enumerate(heads):
            logits = head(features["train"][idx, field])
            known = labels[:, field] >= 0
            loss = loss + F.cross_entropy(logits[known], labels[known, field])
            if (~known).any():
                loss = loss + 0.2 * F.kl_div(
                    logits[~known].log_softmax(-1),
                    torch.full_like(logits[~known], 1 / 16),
                    reduction="batchmean",
                )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    audit = {}
    with torch.no_grad():
        for split in all_data:
            labels = targets[split]
            logits = torch.stack([head(features[split][:, i]) for i, head in enumerate(heads)], 1)
            ps = logits.softmax(-1)
            known = labels[:, 0] >= 0
            true_p = ps[known].gather(-1, labels[known, :, None]).squeeze(-1).prod(-1)
            confident_wrong = (ps.max(-1).values.prod(-1) > 0.99) & (ps.argmax(-1) != labels).any(
                -1
            )
            audit[split] = dict(
                records=len(labels),
                known=int(known.sum()),
                true_address_over_099=int((true_p > 0.99).sum()),
                confident_wrong=int(confident_wrong.sum()),
            )
    torch.save(heads.state_dict(), out / "address.pt")
    save(
        out / "metadata.json",
        dict(
            address_sha256=hashlib.sha256((out / "address.pt").read_bytes()).hexdigest(),
            steps=800,
            threshold=0.99,
            audit=audit,
            purpose="Only same-event address association; no update authority, no answer gating",
            limitations="Existing 16 projects x16 discussion references; not open-world coreference",
        ),
    )
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
