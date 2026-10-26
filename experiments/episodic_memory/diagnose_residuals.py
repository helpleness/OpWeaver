"""Truth-assisted interventions on the PREVIOUS regression set, never inference inputs."""

import json
from collections import Counter
from pathlib import Path

import torch

from .generate_data import ANSWERS, OFFSETS, TASKS, load_events, question
from .models import pad_tokens
from .train import load_checkpoint, seed_everything
from .train_evidence import open_store


@torch.no_grad()
def main():
    root = Path(__file__).parent
    run = root / "runs/evidence_round_03/verified"
    previous = json.loads((root / "runs/evidence_comparison_03/comparison.json").read_text())
    records = previous["conditions"]["both"]["semantic_verified_retry"]["records"]
    old = previous["conditions"]["both"]["old_cluster"]["records"]
    seed_everything(43)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load_checkpoint(run / "model.pt", device)
    model.eval()
    events = {e["id"]: e for e in load_events(run / "data", "test")}
    store = open_store(root / "runs/paraphrase_structured_fresh/changed_events.sqlite", model, tok)
    rows, groups = [], {name: Counter() for name in ("refused", "wrong", "correct")}
    for i, r in enumerate(records):
        if not r["supported"]:
            continue
        e, task = events[r["id"]], r["task"]
        j, target = TASKS.index(task), r["target"]
        h, q = model.encode_questions(pad_tokens([tok.encode(question(e, task, 3))], device))
        a = r["attempts"][-1]
        z, k = store.fetch(e["owner"], a["ids"])
        z, k = torch.tensor(z[None], device=device), torch.tensor(k[None], device=device)
        raw, _, scores, _ = model.read_parts(h, q, z, k)
        selected = int(scores.argmax(-1))
        true_z, true_k = store.fetch(e["owner"], [e["id"]])
        true_z = torch.tensor(true_z[None], device=device)
        true_k = torch.tensor(true_k[None], device=device)
        single, _, _, _ = model.read_parts(h, q, true_z, true_k)
        aux = model.auxiliary[j](true_z[:, 0, model.slot_for_task[j]]).softmax(-1)
        intent = model.intent_head(h).softmax(-1)
        disagreements = [
            float((head(10 * q).softmax(-1) * head(10 * true_k[:, 0]).softmax(-1)).sum())
            for head in model.context_heads
        ]
        bucket = (
            "refused"
            if r["prediction"] == 0
            else ("correct" if r["prediction"] == target else "wrong")
        )
        row = dict(
            index=i,
            id=e["id"],
            task=task,
            bucket=bucket,
            target=ANSWERS[target],
            answer=ANSWERS[r["prediction"]],
            old_answer=ANSWERS[old[i]["prediction"]],
            old_bucket="refused"
            if old[i]["prediction"] == 0
            else ("correct" if old[i]["prediction"] == target else "wrong"),
            conditional=ANSWERS[int(raw[:, 1:].argmax()) + 1],
            intent=TASKS[int(intent.argmax())],
            intent_probability=float(intent.max()),
            selected_correct=a["ids"][selected] == e["id"],
            oracle_single=ANSWERS[int(single[:, 1:].argmax()) + 1],
            auxiliary=ANSWERS[OFFSETS[task] + int(aux.argmax())],
            auxiliary_probability=float(aux.max()),
            evidence=a["evidence"],
            value_probability=a["value_probability"],
            context_agreements=disagreements,
        )
        if row["intent"] != task:
            cause = "intent_error"
        elif not row["selected_correct"]:
            cause = "selection_error"
        elif row["oracle_single"] != row["target"]:
            cause = (
                "reader_error_with_decodable_slot"
                if row["auxiliary"] == row["target"]
                else "reader_and_slot_error"
            )
        elif row["conditional"] != row["target"]:
            cause = "candidate_interference"
        else:
            cause = (
                "evidence_veto"
                if row["evidence"] <= model.config.evidence_threshold
                else "value_veto"
            )
        row["diagnosis"] = cause if bucket != "correct" else "correct"
        groups[bucket][row["diagnosis"]] += 1
        groups[bucket]["task:" + task] += 1
        groups[bucket]["aux_correct"] += row["auxiliary"] == row["target"]
        rows.append(row)
    store.close()
    out = root / "runs/residual_diagnosis.json"
    out.write_text(
        json.dumps({"groups": groups, "records": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(groups, ensure_ascii=False, indent=2))
    print(json.dumps([r for r in rows if r["bucket"] == "wrong"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
