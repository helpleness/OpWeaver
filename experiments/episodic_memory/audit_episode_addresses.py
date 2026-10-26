"""Development-only comparison of existing writer vs query address encoders."""

import json
from pathlib import Path

import torch

from .generate_data import ORDINALS, PROJECTS
from .semantic_memory import SemanticMemory
from .stream_experiment import read, save


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    model = SemanticMemory.load(Path("experiments/episodic_memory/runs/fact_reader_01"))
    turns = read(Path("experiments/episodic_memory/runs/episode_eval_02/conversation.json"))
    rows = model.features([t["text"] for t in turns])
    results = []
    for start in range(0, len(rows), 16):
        h, mask = model.padded_features(rows[start : start + 16])
        _, writer = model.adapter.write(h, mask)
        query = model.adapter.key(model.adapter.query_logits(h, mask))
        for t, w, q in zip(turns[start : start + 16], writer, query):
            if t["anchor"] is None:
                continue
            p = next(i for i, name in enumerate(PROJECTS) if t["anchor"].startswith(name + "项目"))
            o = next(i for i, name in enumerate(ORDINALS) if t["anchor"].endswith(name + "讨论"))
            target = torch.zeros(2, 16, device=model.device)
            target[0, p] = target[1, o] = 1
            results.append(
                dict(
                    id=t["id"],
                    kind=t["kind"],
                    writer=float(model.adapter.agreement(w[None], target[None])),
                    symmetric=float(model.adapter.agreement(q[None], target[None])),
                )
            )
    save(Path("experiments/episodic_memory/runs/episode_address_audit.json"), results)
    print(
        json.dumps(
            dict(
                count=len(results),
                writer_pass=sum(r["writer"] > 0.99 for r in results),
                symmetric_pass=sum(r["symmetric"] > 0.99 for r in results),
                revisions=[r for r in results if r["kind"] != "event"],
            )
        )
    )


if __name__ == "__main__":
    main()
