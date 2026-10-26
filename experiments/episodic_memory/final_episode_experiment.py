"""Fresh stream with data/association ablations and unchanged-Qwen controls."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

from .bounded_attention import install
from .episode_memory import EpisodeMemory
from .evaluate_episode_memory import event_text_control
from .stream_experiment import generate, prepare, read, save, window


def build_combined_run(public, identity, out):
    out.mkdir(exist_ok=False)
    shutil.copy2(public / "codec.pt", out / "codec.pt")
    shutil.copy2(identity / "address.pt", out / "address.pt")
    meta = read(public / "metadata.json")
    address = read(identity / "metadata.json")
    meta.update(
        address_sha256=address["address_sha256"],
        address_training=address,
        public_codec_source=str(public.resolve()),
    )
    save(out / "metadata.json", meta)


def fresh(out):
    prepare(out, seed=57351, checkpoints=[8, 32, 64, 128], fresh=True)
    turns, probes = read(out / "conversation.json"), read(out / "probes.json")
    prefixes = {
        20: "我把这段经历又核对了一遍，先前记下的缘故不准确。现在原因已经确认，请以这次叙述为准。",
        40: "我的记忆在情绪这一点上有分歧：下面这个版本也只是猜测，我无法确定它和之前哪个是真的。",
    }
    for n, prefix in prefixes.items():
        turns[n - 1]["text"] = prefix + "接着说说" + turns[n - 1]["text"].split("接着说说", 1)[1]
    questions = dict(
        early="我想找回那次聊天对象的名字，请告诉我。",
        middle="先别谈心情，那次经历中实际发生的是哪件事？",
        recent="回到当时，我把注意力主要放在了哪里？",
        missing="记录里有没有写那次交流对象叫什么？",
        correction="我后来核对过的那项归因，最后确认是什么？",
        conflict="关于那时的心情，现有这些叙述能得出确定结论吗？",
    )
    for p in probes:
        p["question"] = p["question"].split("，", 1)[0] + "，" + questions[p["probe"]]
    save(out / "conversation.json", turns)
    save(out / "probes.json", probes)
    return turns, probes


@torch.no_grad()
def run(out, small_path, public_path, routed_path):
    torch.set_num_threads(4)
    install()
    turns, probes = fresh(out)
    public = EpisodeMemory.load(public_path)
    old = public.model
    models = dict(
        episode_small=EpisodeMemory.load(small_path, old),
        episode_open=public,
        episode_open_grouped=EpisodeMemory.load(routed_path, old),
    )
    modes = [
        "base_full_history",
        "base_recent_window",
        "old_memory",
        *models,
        "base_same_event_text",
    ]
    stores = {
        name: model.store(out / (name + ".sqlite"), create=True) for name, model in models.items()
    }
    stores["old_memory"] = old.store(out / "old_memory.sqlite", create=True)
    protocol = read(out / "protocol.json")
    protocol.update(
        conditions=modes,
        planned_outputs=len(probes) * len(modes),
        seed=57351,
        models={"small": str(small_path), "public": str(public_path), "grouped": str(routed_path)},
        scoring="Luna per-item evidence review, primary independent review, quote/inconsistency audit, disagreement review and random agreement checks",
        address_threshold=0.99,
        new_model_field_or_revision_classifier=False,
        new_model_template_gate=False,
        training_on_test=False,
        hashes={
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in ("conversation.json", "probes.json")
        },
    )
    save(out / "protocol.json", protocol)
    start, prior, stats = time.perf_counter(), 0, []
    with (out / "outputs.jsonl").open("x", encoding="utf-8") as f:
        for cp in protocol["checkpoints"]:
            events = [
                dict(id=t["id"], owner="stream-user", timestamp=t["number"], text=t["text"])
                for t in turns[prior:cp]
            ]
            for name, model in models.items():
                model.write_events(stores[name], events)
                stores[name].close()
                stores[name] = model.store(out / (name + ".sqlite"))
            old.write_events(stores["old_memory"], events)
            stores["old_memory"].close()
            stores["old_memory"] = old.store(out / "old_memory.sqlite")
            stats.append(
                dict(checkpoint=cp, stores={name: store.counts() for name, store in stores.items()})
            )
            for probe in [p for p in probes if p["checkpoint"] == cp]:
                grouped_result = None
                for mode in modes:
                    visible = turns[:cp]
                    if mode == "base_recent_window":
                        visible = window(old, visible, probe["question"], 2048)
                    if mode == "old_memory":
                        result = old.respond(
                            stores[mode], "stream-user", probe["question"], threshold=0.8
                        )
                        result["status"] = "memory_result"
                    elif mode in models:
                        result = models[mode].respond(
                            stores[mode], "stream-user", probe["question"]
                        )
                        if mode == "episode_open_grouped":
                            grouped_result = result
                    elif mode == "base_same_event_text":
                        retrieved = set(grouped_result["retrieval"]["ids"])
                        visible = [t for t in turns[:cp] if t["id"] in retrieved]
                        result = event_text_control(old, visible, probe["question"])
                    else:
                        result = generate(old, visible, probe["question"], 100)
                    f.write(
                        json.dumps(
                            dict(
                                id=probe["id"],
                                checkpoint=cp,
                                mode=mode,
                                question=probe["question"],
                                visible_turn_ids=[t["id"] for t in visible],
                                result=result,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
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
            output_sha256=hashlib.sha256((out / "outputs.jsonl").read_bytes()).hexdigest(),
            model_metadata={
                n: read(p / "metadata.json")
                for n, p in [
                    ("small", small_path),
                    ("public", public_path),
                    ("grouped", routed_path),
                ]
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--small", type=Path, default=Path("experiments/episodic_memory/runs/episode_memory_02")
    )
    p.add_argument(
        "--public",
        type=Path,
        default=Path("experiments/episodic_memory/runs/episode_memory_open_01"),
    )
    p.add_argument(
        "--identity",
        type=Path,
        default=Path("experiments/episodic_memory/runs/episode_identity_01"),
    )
    p.add_argument(
        "--grouped",
        type=Path,
        default=Path("experiments/episodic_memory/runs/episode_memory_grouped_01"),
    )
    a = p.parse_args()
    build_combined_run(a.public, a.identity, a.grouped)
    run(a.out, a.small, a.public, a.grouped)
