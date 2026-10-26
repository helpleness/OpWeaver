"""Pinned, resumable open-corpus acquisition and source-separated preparation.

Corpus text is training/evaluation material only. Runtime memory never reads it.
No remote Python, HTML or dataset-provided instructions are executed.
"""

import argparse
import gzip
import hashlib
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SOURCES = [
    dict(
        name="fineweb2_zh",
        repo="HuggingFaceFW/fineweb-2",
        revision="af9c13333eb981300149d5ca60a8e9d659b276b9",
        file="data/cmn_Hani/train/000_00000.parquet",
        bytes=4841624662,
        sha256="3e43fefabc3ee500f9874655ece1776f96b81568cf33e0a6376835425ce42598",
        license="ODC-By-1.0",
    ),
    dict(
        name="oasst2",
        repo="OpenAssistant/oasst2",
        revision="179dd21fc55192153d94adb0e0ce8f69e222bf75",
        file="2023-11-05_oasst2_ready.trees.jsonl.gz",
        bytes=54370156,
        sha256="7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b",
        license="Apache-2.0",
    ),
]


def parallel_download(source, path):
    parts = path.parent / (path.name + ".parts")
    parts.mkdir(exist_ok=True)
    chunk_size = 64 * 1024**2
    jobs = [
        (i, offset, min(source["bytes"], offset + chunk_size) - 1)
        for i, offset in enumerate(range(0, source["bytes"], chunk_size))
    ]

    def fetch(job):
        index, start, end = job
        part = parts / f"{index:04d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part.stat().st_size
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['file']}?download=true&part={index}"
        for attempt in range(4):
            try:
                with requests.get(
                    url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 120)
                ) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or not response.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {start}-{end}/"):
                        raise ValueError("Unverified range response")
                    with part.open("wb") as f:
                        for chunk in response.iter_content(4 * 1024**2):
                            f.write(chunk)
                if part.stat().st_size != end - start + 1:
                    raise ValueError("Short range")
                return part.stat().st_size
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch, job) for job in jobs]):
            done += future.result()
            print(
                json.dumps(
                    dict(
                        stage="parallel_download",
                        source=source["name"],
                        completed_bytes=done,
                        total=source["bytes"],
                    )
                ),
                flush=True,
            )
    with path.open("xb") as dest:
        for index, _, _ in jobs:
            with (parts / f"{index:04d}.part").open("rb") as part:
                while chunk := part.read(8 * 1024**2):
                    dest.write(chunk)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def download(source, root):
    path = root / source["name"] / Path(source["file"]).name
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if not path.exists() and source["bytes"] > 100 * 1024**2:
        parallel_download(source, path)
    if not path.exists():
        for attempt in range(5):
            current = partial.stat().st_size if partial.exists() else 0
            try:
                url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['file']}?download=true"
                with requests.get(
                    url,
                    headers={"Range": f"bytes={current}-"} if current else {},
                    stream=True,
                    timeout=(30, 120),
                ) as response:
                    response.raise_for_status()
                    if current and response.status_code != 206:
                        raise RuntimeError("Server did not honor resume; preserving partial")
                    last = time.monotonic()
                    with partial.open("ab" if current else "wb") as f:
                        for chunk in response.iter_content(8 * 1024 * 1024):
                            f.write(chunk)
                            current += len(chunk)
                            if time.monotonic() - last > 20:
                                print(
                                    json.dumps(
                                        dict(
                                            stage="download",
                                            source=source["name"],
                                            bytes=current,
                                            total=source["bytes"],
                                        )
                                    ),
                                    flush=True,
                                )
                                last = time.monotonic()
                if current != source["bytes"]:
                    raise ValueError("Incomplete size")
                partial.rename(path)
                break
            except (requests.RequestException, ValueError):
                if attempt == 4:
                    raise
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != source["sha256"]:
        raise ValueError("Downloaded checksum mismatch: " + source["name"])
    readme = requests.get(
        f"https://huggingface.co/datasets/{source['repo']}/raw/{source['revision']}/README.md",
        timeout=30,
    )
    readme.raise_for_status()
    (path.parent / "SOURCE_README.md").write_text(readme.text, encoding="utf-8")
    write_json(
        path.parent / "source.json",
        dict(**source, local_file=str(path.resolve()), verified_sha256=actual),
    )
    print(
        json.dumps(
            dict(stage="download_verified", source=source["name"], bytes=path.stat().st_size)
        ),
        flush=True,
    )
    return path


def normalized(text):
    return "\n".join(line.strip() for line in text.replace("\r", "").splitlines() if line.strip())


def split_key(key):
    # Entire documents/conversation trees stay in one split.
    n = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return "test" if n < 2 else "validation" if n < 4 else "train"


def prepare(root, target_bytes=3 * 1024**3):
    import pyarrow.parquet as pq

    out = root / "prepared"
    out.mkdir(exist_ok=False)
    seen = set()
    counts = Counter()
    files = {
        split: (out / (split + ".jsonl")).open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }

    def emit(row):
        text = normalized(row["text"])
        if len(text) < 80 or len(text) > 60000 or "\ufffd" in text:
            counts["rejected_length_or_encoding"] += 1
            return
        # Avoid training on obvious direct contact details/IDs in public web text.
        has_email = "@" in text and re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
        has_numeric_contact = re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)|(?<!\d)\d{17}[\dXx](?!\d)", text)
        if has_email or has_numeric_contact:
            counts["rejected_contact_pattern"] += 1
            return
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            counts["exact_duplicates"] += 1
            return
        seen.add(digest)
        split = split_key(row["group_id"])
        row.update(text=text, text_sha256=digest, split=split)
        files[split].write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split + "_records"] += 1
        counts[split + "_text_bytes"] += len(text.encode())
        counts[row["source"] + "_records"] += 1
        counts["kept_text_bytes"] += len(text.encode())

    try:
        # Conversations first so the web-byte target cannot exclude this source.
        oasst = root / "oasst2" / Path(SOURCES[1]["file"]).name
        with gzip.open(oasst, "rt", encoding="utf-8") as f:
            for line in f:
                tree = json.loads(line)
                tree_id = tree.get("message_tree_id", tree.get("prompt", {}).get("message_id", ""))

                def visit(node, path):
                    if node.get("deleted") or node.get("review_result") is False:
                        return
                    if node.get("lang") not in ("en", "zh"):
                        return
                    role = "user" if node.get("role") == "prompter" else "assistant"
                    chain = path + [dict(role=role, text=node.get("text", ""))]
                    children = node.get("replies", [])
                    if not children:
                        emit(
                            dict(
                                source="oasst2",
                                group_id=tree_id,
                                source_id=node.get("message_id"),
                                language=node.get("lang"),
                                turns=chain,
                                text="\n".join(t["role"] + ": " + t["text"] for t in chain),
                            )
                        )
                    else:
                        for child in children:
                            visit(child, chain)

                visit(tree["prompt"], [])
        path = root / "fineweb2_zh" / Path(SOURCES[0]["file"]).name
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=2048, columns=["text", "id", "url", "language_score"]
        ):
            for row in batch.to_pylist():
                counts["web_rows_examined"] += 1
                if row["language_score"] < 0.9:
                    counts["rejected_language_score"] += 1
                    continue
                emit(
                    dict(
                        source="fineweb2_zh",
                        group_id=row["url"],
                        source_id=row["id"],
                        url=row["url"],
                        text=row["text"],
                        language="zh",
                    )
                )
            if counts["kept_text_bytes"] >= target_bytes:
                break
            if counts["web_rows_examined"] % 20480 == 0:
                print(json.dumps(dict(stage="prepare", **counts)), flush=True)
    finally:
        for f in files.values():
            f.close()
    write_json(
        out / "manifest.json",
        dict(
            sources=SOURCES,
            counts=dict(counts),
            target_clean_text_bytes=target_bytes,
            split_policy="96% train,2% validation,2% test by document URL or whole conversation tree SHA256",
            dedup="normalized exact-text hash only; not a near-duplicate guarantee",
            training_consumption="not yet trained; this is prepared corpus size",
        ),
    )
    print(json.dumps(dict(stage="prepared", **counts)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    if a.prepare:
        prepare(a.root)
    else:
        # Obtain small dialogues before the large web shard.
        for source in reversed(SOURCES):
            download(source, a.root)
