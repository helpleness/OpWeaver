"""Four same-question controls. Writes evidence/outputs, never scores answer prose.

Use stream_review afterwards for independent per-item model judging. Thresholds
remain 0.8 (field/value/update) and 0.99 (address); no reference text enters memory.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .bounded_attention import install
from .semantic_memory import SemanticMemory
from .stream_experiment import generate, prepare, read, save, tokenize, window


@torch.no_grad()
def run(out, new_run, source=None, seed=41968):
    torch.set_num_threads(4)
    install()
    if source is None:
        prepare(out, seed=seed, checkpoints=[8, 32, 64], fresh=True)
    else:
        import shutil

        out.mkdir(parents=True, exist_ok=False)
        for name in ("conversation.json", "probes.json", "protocol.json"):
            shutil.copy2(source / name, out / name)
    turns, probes = read(out / "conversation.json"), read(out / "probes.json")
    protocol = read(out / "protocol.json")
    modes = ["base_full_history", "base_recent_window", "memory_legacy", "memory_revised"]
    protocol.update(
        conditions=modes,
        planned_outputs=len(probes) * len(modes),
        scoring="No automatic semantic score. Blind per-item Luna review and evidence audit required.",
        new_run=str(new_run),
        legacy_run="runs/fact_reader_01",
        original_baselines_reused_from=str(source) if source else None,
    )
    save(out / "protocol.json", protocol)
    model = SemanticMemory.load(new_run)
    old_meta = read(Path("experiments/episodic_memory/runs/fact_reader_01/metadata.json"))
    new_meta = read(new_run / "metadata.json")
    for name in ("adapter_sha256", "semantic_sha256", "backbone"):
        assert old_meta[name] == new_meta[name], "Baseline must use the identical backbone and reader weights"
    revision_reader = model.revision_reader
    paths = {mode: out / (mode + ".sqlite") for mode in modes[2:]}
    stores = {mode: model.store(path, create=True) for mode, path in paths.items()}
    original = {}
    if source:
        original = {
            (r["id"], r["mode"]): r
            for r in [
                json.loads(line)
                for line in (source / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
        }
    start, prior, stats = time.perf_counter(), 0, []
    with (out / "outputs.jsonl").open("x", encoding="utf-8") as f:
        for cp in protocol["checkpoints"]:
            new = [
                dict(id=t["id"], owner="stream-user", timestamp=t["number"], text=t["text"])
                for t in turns[prior:cp]
            ]
            for mode in modes[2:]:
                model.revision_reader = revision_reader if mode == "memory_revised" else None
                model.write_events(stores[mode], new)
                stores[mode].close()
                stores[mode] = model.store(paths[mode])
            stats.append(
                dict(
                    checkpoint=cp,
                    stores={k: s.counts() for k, s in stores.items()},
                    tokens=tokenize(model, turns[:cp], "继续").input_ids.shape[1],
                )
            )
            for probe in [p for p in probes if p["checkpoint"] == cp]:
                for mode in modes:
                    visible = turns[:cp]
                    if mode == "base_recent_window":
                        visible = window(model, visible, probe["question"], 2048)
                    if mode.startswith("memory_"):
                        model.answer_policy = (
                            "typed_fact" if mode == "memory_revised" else "legacy_template_gate"
                        )
                        result = model.respond(
                            stores[mode], "stream-user", probe["question"], threshold=0.8
                        )
                        result["status"] = "memory_result"
                    elif source:
                        result = original[probe["id"], mode]["result"]
                    else:
                        result = generate(model, visible, probe["question"], 100)
                    row = dict(
                        id=probe["id"],
                        checkpoint=cp,
                        mode=mode,
                        question=probe["question"],
                        visible_turn_ids=[t["id"] for t in visible],
                        result=result,
                    )
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                print(
                    json.dumps(
                        dict(
                            checkpoint=cp,
                            probe=probe["probe"],
                            seconds=round(time.perf_counter() - start, 1),
                        )
                    ),
                    flush=True,
                )
            prior = cp
    for store in stores.values():
        store.close()
    save(
        out / "manifest.json",
        dict(
            seconds=time.perf_counter() - start,
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
            checkpoints=stats,
            model=read(new_run / "metadata.json"),
            threshold=0.8,
            address_threshold=0.99,
            hashes={
                n: hashlib.sha256((out / n).read_bytes()).hexdigest()
                for n in ("conversation.json", "probes.json", "outputs.jsonl")
            },
            output_judgements="pending independent evidence review",
            comparison="identical questions; base parameters unchanged",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--new-run", type=Path, required=True)
    p.add_argument("--regression-source", type=Path)
    p.add_argument("--seed", type=int, default=41968)
    a = p.parse_args()
    run(a.out, a.new_run, a.regression_source, a.seed)
