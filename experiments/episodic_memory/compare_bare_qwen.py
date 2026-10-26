"""Unmodified backbone controls: no adapter, no latent input, no confidence gate."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluate_semantic_memory import outcome
from .qwen_memory import UNKNOWN, save_json
from .train_qwen_memory import CHAT_PROMPTS

CHAT_SYSTEM = "你是一个乐于助人的助手。"
TASK_SYSTEM = (
    "你是回答个人经历问题的助手。只能依据提供的经历记录作答，不能猜测或补充细节。"
    "当前只支持指定项目和第几次讨论的单字段查询：交流对象姓名、发生在第几天、发生的事情、"
    "我当时的情绪、产生这种感受的原因、我关注的方面。"
    "如果没有对应事件、所问字段未记录、记录互相矛盾、没有指明具体事件，必须拒答。"
    "请求多个字段、是非确认或其他信息也不在当前支持范围。排除无关字段不等于请求多个字段。"
    "能回答时，只简短给出记录中的该项事实；需要拒答时只回答：" + UNKNOWN
)
MODES = ("chat_no_context", "task_no_context", "task_full_text", "task_oracle_text")


def context_text(case, oracle=False):
    records = [e for e in case["events"] if e["owner"] == case["owner"]]
    if oracle:
        records = [e for e in records if e["id"] == case["id"] + "-target"]
        if len(records) != 1:
            raise ValueError("Oracle positive control requires one target record")
    parts = []
    for e in records:
        if e["id"] == case["damage_id"]:
            parts.append("该事件内容不可用：" + e["anchor"] + "。")
        else:
            parts.append(e["text"])
    return "\n".join(parts) if parts else "（没有可用经历记录）"


def messages(case, mode):
    if mode == "chat_no_context":
        return [
            dict(role="system", content=CHAT_SYSTEM),
            dict(role="user", content=case["question"]),
        ]
    context = (
        context_text(case, oracle=mode == "task_oracle_text")
        if mode != "task_no_context"
        else "（没有可用经历记录）"
    )
    return [
        dict(role="system", content=TASK_SYSTEM),
        dict(role="user", content="经历记录：\n" + context + "\n问题：" + case["question"]),
    ]


@torch.no_grad()
def generate(base, tokenizer, msgs, max_new_tokens=40):
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(base.device)
    output = base.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    ids = output.sequences[0, inputs.input_ids.shape[1] :]
    raw = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return dict(
        raw=raw,
        answer=raw,
        ended=bool(len(ids) and int(ids[-1]) == tokenizer.eos_token_id),
        tokens=len(ids),
        input_tokens=inputs.input_ids.shape[1],
        token_min=min(float(s[0].float().softmax(-1)[t]) for s, t in zip(output.scores, ids)),
    )


@torch.no_grad()
def generate_batch(base, tokenizer, batch):
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch
    ]
    inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt").to(
        base.device
    )
    output = base.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=40,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    results = []
    for i, sequence in enumerate(output.sequences):
        ids = sequence[inputs.input_ids.shape[1] :]
        stops = (ids == tokenizer.eos_token_id).nonzero()
        ended = bool(len(stops))
        if ended:
            ids = ids[: int(stops[0]) + 1]
        raw = tokenizer.decode(ids, skip_special_tokens=True).strip()
        results.append(
            dict(
                raw=raw,
                answer=raw,
                ended=ended,
                tokens=len(ids),
                input_tokens=int(inputs.attention_mask[i].sum()),
                token_min=min(
                    float(s[i].float().softmax(-1)[t]) for s, t in zip(output.scores, ids)
                ),
            )
        )
    return results


@torch.no_grad()
def main(args):
    if not getattr(args, "legacy_protocol", False):
        raise ValueError(
            "Historical prompts/scoring require --legacy-protocol; see EVALUATION_CORRECTION.md"
        )
    torch.set_num_threads(4)
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    source_revision = (
        (Path(meta["backbone"]) / ".cache/huggingface/download/model.safetensors.metadata")
        .read_text()
        .splitlines()[0]
    )
    assert source_revision == meta["source_revision"]
    tokenizer = AutoTokenizer.from_pretrained(meta["backbone"], trust_remote_code=False)
    tokenizer.padding_side = "left"
    base = (
        AutoModelForCausalLM.from_pretrained(
            meta["backbone"],
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        .to("cuda")
        .eval()
    )
    base.requires_grad_(False)
    cases = json.loads((args.suite / "test.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((args.suite / "test.json").read_bytes()).hexdigest()
    prior = json.loads((args.run / "test/summary.json").read_text(encoding="utf-8"))
    assert digest == prior["cases_sha256"]
    save_json(
        args.out / "protocol.json",
        dict(
            case_sha256=digest,
            source_revision=source_revision,
            model_class=type(base).__name__,
            parameters=sum(p.numel() for p in base.parameters()),
            adapters_loaded=False,
            confidence_gating=False,
            max_new_tokens=40,
            do_sample=False,
            batch_size=args.batch,
            system_prompts=dict(chat=CHAT_SYSTEM, task=TASK_SYSTEM),
            modes=MODES,
            note="Post-hoc same-query controls; oracle supplies truth-selected text only for supported cases. No model changes.",
        ),
    )
    rows = []
    for mode in MODES:
        subset = (
            cases if mode != "task_oracle_text" else [c for c in cases if c["expected"] is not None]
        )
        for pos in range(0, len(subset), args.batch):
            batch = subset[pos : pos + args.batch]
            msgs_batch = [messages(case, mode) for case in batch]
            results = generate_batch(base, tokenizer, msgs_batch)
            for case, msgs, result in zip(batch, msgs_batch, results):
                strict = outcome(result["raw"], case["expected"], case["field"])
                if strict == "correct" and not result["ended"]:
                    strict = "truncated_match"
                row = dict(
                    id=case["id"],
                    kind=case["kind"],
                    mode=mode,
                    question=case["question"],
                    field=case["field"],
                    expected=case["expected"],
                    result=result,
                    strict_outcome=strict,
                    messages=msgs,
                )
                rows.append(row)
                with (args.out / "results.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done = pos + len(batch)
            if done % 40 == 0 or done == len(subset):
                print(
                    json.dumps(
                        dict(
                            mode=mode,
                            done=done,
                            total=len(subset),
                            seconds=time.perf_counter() - start,
                        )
                    ),
                    flush=True,
                )
    chat = []
    previous = json.loads((args.run / "chat_after.json").read_text(encoding="utf-8"))
    for q in CHAT_PROMPTS:
        result = generate(
            base,
            tokenizer,
            [dict(role="system", content=CHAT_SYSTEM), dict(role="user", content=q)],
            max_new_tokens=100,
        )
        chat.append(
            dict(
                question=q,
                base=result["raw"],
                memory_chat=previous[q],
                exact=result["raw"] == previous[q],
            )
        )
    save_json(args.out / "chat_comparison.json", chat)
    summary = dict(
        metric_status="legacy_protocol_only_not_semantic_accuracy",
        case_sha256=digest,
        source_revision=source_revision,
        chat_exact=sum(r["exact"] for r in chat),
        chat_count=len(chat),
        modes={},
        seconds=time.perf_counter() - start,
    )
    for mode in MODES:
        subset = [r for r in rows if r["mode"] == mode]
        summary["modes"][mode] = dict(
            count=len(subset),
            positive=dict(
                Counter(r["strict_outcome"] for r in subset if r["expected"] is not None)
            ),
            negative=dict(Counter(r["strict_outcome"] for r in subset if r["expected"] is None)),
            truncated=sum(not r["result"]["ended"] for r in subset),
            by_kind={
                k: dict(Counter(r["strict_outcome"] for r in subset if r["kind"] == k))
                for k in sorted({r["kind"] for r in subset})
            },
        )
    save_json(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run", type=Path, default=Path("experiments/episodic_memory/runs/semantic_memory_03")
    )
    p.add_argument(
        "--suite", type=Path, default=Path("experiments/episodic_memory/runs/semantic_suite_03")
    )
    p.add_argument(
        "--out", type=Path, default=Path("experiments/episodic_memory/runs/bare_qwen_comparison_02")
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument(
        "--legacy-protocol", action="store_true", help="Reproduce historical prompts and scoring"
    )
    main(p.parse_args())
