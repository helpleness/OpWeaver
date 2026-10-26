"""Development probe of frozen instruction following, not an answer scorer."""

import json
from pathlib import Path

import torch

from .revision_reader import OP_SYSTEM
from .semantic_memory import SemanticMemory


def main():
    torch.set_num_threads(4)
    model = SemanticMemory.load(Path("experiments/episodic_memory/runs/fact_reader_01"))
    rows = json.loads(
        Path(
            "experiments/episodic_memory/runs/revision_reader_01/calibration_operations.json"
        ).read_text(encoding="utf-8")
    )
    token_ids = [model.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(7)]
    results = []
    with torch.no_grad():
        for start in range(0, len(rows), 8):
            batch = rows[start : start + 8]
            prompts = [
                model.tokenizer.apply_chat_template(
                    [
                        dict(role="system", content=OP_SYSTEM + "只输出类别数字0至6，不要解释。"),
                        dict(role="user", content=r["text"]),
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in batch
            ]
            inputs = model.tokenizer(
                prompts, padding=True, add_special_tokens=False, return_tensors="pt"
            ).to(model.device)
            logits = model.base(**inputs, use_cache=False).logits
            last = torch.stack([h[m.bool()][-1] for h, m in zip(logits, inputs.attention_mask)])
            ps = last.float().softmax(-1)[:, token_ids].cpu()
            for r, p in zip(batch, ps):
                prediction, probability = int(p.argmax()), float(p.max())
                results.append(
                    dict(
                        **r,
                        prediction=prediction,
                        probability=probability,
                        applied=prediction if probability > 0.8 else 0,
                    )
                )
    out = Path("experiments/episodic_memory/runs/operation_frozen_probe_unpadded.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                count=len(results),
                correct=sum(r["prediction"] == r["label"] for r in results),
                false_replace=sum(r["applied"] not in (0, r["label"]) for r in results),
                missed=sum(r["label"] != 0 and r["applied"] != r["label"] for r in results),
            )
        )
    )


if __name__ == "__main__":
    main()
