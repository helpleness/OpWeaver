"""Sensitivity check of incremental versus one-shot prefill on actual long inputs."""

import argparse
import gc
from pathlib import Path

import torch

from .bounded_attention import install
from .episode_memory import EpisodeMemory
from .long_context_comparison import answer, extend, template
from .stream_experiment import read, save


@torch.no_grad()
def run(source):
    torch.set_num_threads(4)
    install()
    model = EpisodeMemory.load(
        Path("experiments/episodic_memory/runs/episode_memory_grouped_01")
    ).model
    out = source / "long_cache_sensitivity.json"
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for seed, budget in [(57511, 8192), (57511, 32000), (57512, 32000)]:
        folder = source / f"seed_{seed}"
        turns = read(folder / "conversation.json")
        probe = next(
            p
            for p in read(folder / "probes.json")
            if p["budget"] == budget and p["probe"] == "early"
        )
        visible = turns[: probe["checkpoint"]]
        full = template(model, visible, probe["question"])
        prefix = template(model, visible)
        cache, _ = extend(model, prefix)
        cached = answer(model, full, cache)
        del cache
        gc.collect()
        torch.cuda.empty_cache()
        try:
            direct = answer(model, full)
            rows.append(
                dict(
                    seed=seed,
                    budget=budget,
                    incremental=cached,
                    direct=direct,
                    identical=cached["answer"] == direct["answer"],
                )
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append(
                dict(
                    seed=seed,
                    budget=budget,
                    incremental=cached,
                    direct_status="oom_not_run",
                    identical=None,
                )
            )
        save(out, rows)
        print({"seed": seed, "budget": budget, "identical": rows[-1]["identical"]}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    run(p.parse_args().source)
