"""Prepare blinded per-item review packets for the two long-context streams."""

import argparse
import json
import random
from pathlib import Path

from .luna_review import digest, dump_rows, read_rows
from .stream_review import STREAM_RUBRIC, prepare


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in (57511, 57512):
        src = source / f"seed_{seed}"
        staging = out / f"source_packets_{seed}"
        prepare(src, staging)
        packets = read_rows(staging / "all_packets.jsonl")
        mapping = json.loads((staging / "private_mapping.json").read_text(encoding="utf-8"))
        probes = {p["id"]: p for p in json.loads((src / "probes.json").read_text(encoding="utf-8"))}
        for p in packets:
            m = mapping[p["review_id"]]
            m.update(seed=seed, budget=probes[m["output_id"]]["budget"])
            records.append((p, m))
    random.Random(57521).shuffle(records)
    packets, mapping = [], {}
    for i, (p, m) in enumerate(records):
        p["review_id"] = f"L{i:03d}"
        p.pop("binding_sha256")
        p["binding_sha256"] = digest(p)
        packets.append(p)
        mapping[p["review_id"]] = m
    assert len(packets) == 160
    dump_rows(out / "all_packets.jsonl", packets)
    splits = [packets[:54], packets[54:107], packets[107:]]
    for letter, part in zip("abc", splits):
        dump_rows(out / f"packets_{letter}.jsonl", part)
    (out / "private_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "rubric.txt").write_text(
        STREAM_RUBRIC
        + "\n本轮比较增长历史下的两种系统。证据为该时点可用的相关经历，不包含未来事件。请按问题实际要求判断；完整回答也可能明确说明原因已更正或情绪尚有分歧。不要将不知道等自然表述按固定句式打分。\n",
        encoding="utf-8",
    )
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                source=str(source),
                packets=160,
                parts=[54, 53, 53],
                blind="Model and length hidden; response style and execution logs may reveal condition",
                human_gold=False,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.out)
