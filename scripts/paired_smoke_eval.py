#!/usr/bin/env python3
"""Paired MMLU-Pro smoke: ranks 0-1 FP8, ranks 2-3 AWQ."""

import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

BASELINE_MODEL = os.environ.get("EVAL_BASELINE_MODEL_ID", "Qwen/Qwen3.8-27B-FP8")
BASELINE_REVISION = os.environ.get(
    "EVAL_BASELINE_MODEL_REVISION", "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
)
AWQ_MODEL = Path(os.environ["OUTPUT_DIR"]).resolve()
EVAL_DIR = Path(os.environ.get("SMOKE_EVAL_DIR", AWQ_MODEL.parent / "smoke-eval"))
DATASET_ID = "TIGER-Lab/MMLU-Pro"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
SEED = 38027
PER_CATEGORY = int(os.environ.get("SMOKE_PER_CATEGORY", "8"))
CATEGORIES = ("math", "physics", "chemistry", "biology", "computer science", "engineering", "health", "business")
LETTERS = "ABCDEFGHIJ"


def choose_rows() -> list[dict]:
    dataset = load_dataset(DATASET_ID, split="test", revision=DATASET_REVISION)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        if row["category"] in CATEGORIES:
            buckets[row["category"]].append(dict(row))
    rng = random.Random(SEED)
    selected = []
    for category in CATEGORIES:
        rows = buckets[category]
        if len(rows) < PER_CATEGORY:
            raise RuntimeError(f"only {len(rows)} rows for {category}")
        rng.shuffle(rows)
        selected.extend(rows[:PER_CATEGORY])
    return sorted(selected, key=lambda row: (row["category"], int(row["question_id"])))


def prompt_for(row: dict) -> str:
    choices = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(row["options"]))
    return f"Answer with only the single letter of the best answer.\n\n{row['question']}\n\n{choices}"


def parse_answer(text: str) -> str | None:
    matches = re.findall(r"(?<![A-Z])([A-J])(?![A-Z])", text.upper())
    return matches[-1] if matches else None


def main() -> None:
    started = time.monotonic()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise RuntimeError("paired smoke requires exactly four ranks")
    torch.cuda.set_device(rank)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    selected_path = EVAL_DIR / "mmlu-pro-selected.jsonl"
    if rank == 0:
        selected_path.write_text("".join(json.dumps(r) + "\n" for r in choose_rows()))
    dist.barrier()
    rows = [json.loads(line) for line in selected_path.read_text().splitlines()]

    variant = "fp8" if rank < 2 else "awq"
    replica = rank % 2
    model_path = BASELINE_MODEL if variant == "fp8" else str(AWQ_MODEL)
    revision = BASELINE_REVISION if variant == "fp8" else None
    processor = AutoProcessor.from_pretrained(model_path, revision=revision, trust_remote_code=True)
    kwargs = {"dtype": torch.bfloat16, "device_map": {"": rank}, "trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    model.eval()

    results = []
    for index, row in enumerate(rows):
        if index % 2 != replica:
            continue
        text = processor.apply_chat_template(
            [{"role": "user", "content": prompt_for(row)}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False, reasoning_effort="low",
        )
        inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)
        completion = processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        predicted = parse_answer(completion)
        results.append({"id": str(row["question_id"]), "category": row["category"],
                        "gold": row["answer"], "predicted": predicted,
                        "correct": predicted == row["answer"], "malformed": predicted is None,
                        "completion": completion})
    (EVAL_DIR / f"{variant}-rank{rank}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )
    dist.barrier()

    if rank == 0:
        combined = {}
        for name, ranks in (("fp8", (0, 1)), ("awq", (2, 3))):
            combined[name] = []
            for result_rank in ranks:
                combined[name] += [json.loads(line) for line in (EVAL_DIR / f"{name}-rank{result_rank}.jsonl").read_text().splitlines()]
        if {r["id"] for r in combined["fp8"]} != {r["id"] for r in combined["awq"]}:
            raise RuntimeError("FP8 and AWQ item IDs differ")
        count = len(combined["fp8"])
        report = {"benchmark": DATASET_ID, "dataset_revision": DATASET_REVISION,
                  "samples": count, "categories": list(CATEGORIES), "seed": SEED,
                  "baseline_model": BASELINE_MODEL,
                  "baseline_revision": BASELINE_REVISION,
                  "fp8_accuracy": sum(r["correct"] for r in combined["fp8"]) / count,
                  "awq_accuracy": sum(r["correct"] for r in combined["awq"]) / count,
                  "fp8_malformed": sum(r["malformed"] for r in combined["fp8"]),
                  "awq_malformed": sum(r["malformed"] for r in combined["awq"]),
                  "elapsed_seconds": round(time.monotonic() - started, 2)}
        fp8_by_id = {row["id"]: row for row in combined["fp8"]}
        awq_by_id = {row["id"]: row for row in combined["awq"]}
        report["fp8_pass_awq_fail"] = sum(
            fp8_by_id[item_id]["correct"] and not awq_by_id[item_id]["correct"]
            for item_id in fp8_by_id
        )
        report["awq_pass_fp8_fail"] = sum(
            awq_by_id[item_id]["correct"] and not fp8_by_id[item_id]["correct"]
            for item_id in fp8_by_id
        )
        report["accuracy_delta"] = report["awq_accuracy"] - report["fp8_accuracy"]
        report["passed"] = (
            report["accuracy_delta"] >= -0.03
            and report["fp8_malformed"] == 0
            and report["awq_malformed"] == 0
        )
        (EVAL_DIR / "paired-smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise RuntimeError("paired scored smoke gate failed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
