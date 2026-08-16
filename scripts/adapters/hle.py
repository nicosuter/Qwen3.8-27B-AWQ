#!/usr/bin/env python3
"""Humanity's Last Exam adapter for scripts/run_eval_protocol.py.

HLE is deliberately unsaturated: frontier models score low, which is the
opposite of everything else in this protocol, where 83% of items score
identically on both checkpoints. Items near the decision boundary are where a
paired comparison has power, and this is where they are.

Grading runs in two steps. A normalized string match settles an item whenever
it fires, because string equality has no false positives, and 591 of the 2,500
items are single-letter multipleChoice where it is the whole story. The
remaining 1,909 exactMatch items are not safe to grade that way: their answers
run to a median of 8 tokens, 20% carry LaTeX such as `\\frac{2}{7}`, and 11% are
free text. A model writes `2/7` on one draw and `\\frac{2}{7}` on the next, so a
string match would manufacture a discordant pair out of formatting alone and
feed it to the comparator as signal. Those items go to a judge.

The judge is pinned and lives on its own endpoint, which may be a model served
here or a hosted one. It must not be either checkpoint under test, since a
model grading its own house style favours one arm. It sees the question, the
reference answer, and one submitted answer, never a transcript and never which
arm produced it. Verdicts are cached on (item, normalized answer), so two
responses that said the same thing are guaranteed the same grade rather than
merely likely to get one, and replicates are nearly free.

Grading and generation are split. `run --defer-judging` leaves the open items
unscored and `score` grades them afterwards, because a data-parallel serve of
the model under test already holds every GPU, and because a judge fault should
not cost the generation. The comparator refuses a file with a deferred row, so
a half-graded suite cannot be read as a result.

The absolute is still not comparable to a published HLE score, because that
leaderboard grades with `openai/o3-mini` and this does not. The delta is what
this suite is for.

Images arrive as data URLs and are passed through unchanged rather than decoded
and re-encoded, so both checkpoints are sent the same bytes by construction.
The judge is text-only: it compares answers against a reference rather than
re-solving the question, so it does not need the image.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

try:
    from _common import (
        AdapterError,
        base_row,
        build_payload,
        check_action,
        env_path,
        env_str,
        execute_order,
        has_repetition_loop,
        load_pins,
        module_pin,
        post_chat,
        raw_response_path as _raw_response_path,
        read_jsonl,
        reasoning_tokens,
        request_with_retries,
        require_pin,
        split_reasoning,
        timeout_row as _timeout_row,
        timing,
        unpack_choice,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # loading by file spec puts the repo root on sys.path
    from scripts.adapters._common import (  # type: ignore[no-redef]
        AdapterError,
        base_row,
        build_payload,
        check_action,
        env_path,
        env_str,
        execute_order,
        has_repetition_loop,
        load_pins,
        module_pin,
        post_chat,
        raw_response_path as _raw_response_path,
        read_jsonl,
        reasoning_tokens,
        request_with_retries,
        require_pin,
        split_reasoning,
        timeout_row as _timeout_row,
        timing,
        unpack_choice,
        write_json,
        write_jsonl,
    )


SUITE = "hle"
HARNESS_ID = "builtin-hle-judged-v1"
VERIFIER_ID = "judged-equivalence-v1"
DATASET_REPO = "cais/hle"
DATASET_FILE = "data/test-00000-of-00001.parquet"

# The judge may be served here or called over an API, so a pin has two forms:
#
#   hf:owner/name@<40-hex commit>     weights we hold, pinned exactly
#   api:provider/model-id@<snapshot>  a hosted model, pinned as far as it goes
#
# The second is weaker and honestly so: a hosted model cannot be hashed, and the
# most we can record is the dated snapshot we asked for. An alias like a bare
# `-latest` is refused, because it can move between the two arms of a run.
DEFAULT_JUDGE_REPO = "openai/gpt-oss-20b"
JUDGE_PIN_RE = re.compile(
    r"^(?:"
    r"hf:(?P<repo>[\w.-]+/[\w.-]+)@(?P<revision>[0-9a-f]{40})"
    r"|"
    r"api:(?P<provider>[\w.-]+)/(?P<model>[\w.:-]+)@(?P<snapshot>[\w.-]+)"
    r")$"
)
FLOATING_ALIAS_RE = re.compile(r"(^|[-_])(latest|preview|current)($|[-_])", re.IGNORECASE)


def judge_pin_parts(pin: str) -> dict[str, str]:
    """Split a judge pin into the model to request and the record to keep."""
    match = JUDGE_PIN_RE.match(pin or "")
    if not match:
        raise AdapterError(
            "pins.judge must be hf:owner/name@<40-hex> or "
            f"api:provider/model-id@<snapshot>; got {pin!r}. The judge decides "
            "scores, so an unpinned one makes the gate unreproducible."
        )
    if match.group("repo"):
        return {"scheme": "hf", "model": match.group("repo"),
                "snapshot": match.group("revision")}
    model, snapshot = match.group("model"), match.group("snapshot")
    for part in (model, snapshot):
        if FLOATING_ALIAS_RE.search(part):
            raise AdapterError(
                f"pins.judge names the moving alias {part!r}; pin the dated "
                "snapshot instead, or the two arms can be graded by different "
                "models"
            )
    return {"scheme": "api", "model": model, "snapshot": snapshot}

DEFAULT_MAX_TOKENS = 65536
DEFAULT_JUDGE_MAX_TOKENS = 2048

ANSWER_INSTRUCTION = (
    "Answer the question. End your reply with a final line of exactly this "
    "form:\n\nAnswer: <answer>\n\n"
    "giving only the answer itself, with no explanation. For a multiple-choice "
    "question give only the option letter."
)

JUDGE_INSTRUCTION = """You are grading one answer to an exam question.

[question]
{question}

[reference answer]
{reference}

[submitted answer]
{submitted}

Decide whether the submitted answer is equivalent to the reference answer.
Ignore differences in notation, formatting and wording: `\\frac{{2}}{{7}}`, `2/7`
and `$\\frac{{2}}{{7}}$` are the same answer, and so are `1.4E-14` and
`1.4*10^-14`. A partially correct answer, a different value, or an answer that
omits part of what the reference gives is not correct.

Reply with exactly two lines and nothing else:
reasoning: <one sentence>
correct: yes|no"""

ANSWER_LINE_RE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
VERDICT_RE = re.compile(r"^\s*correct\s*:\s*(yes|no)\b", re.IGNORECASE | re.MULTILINE)
OPTION_LETTER_RE = re.compile(r"^\(?([A-Za-z])\)?\s*(?:[.):]|$)")
WHITESPACE_RE = re.compile(r"\s+")
TRAILING_RE = re.compile(r"[.,;:]+$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("prepare", help="materialize questions, images and the answer key")

    run = sub.add_parser("run", help="score the frozen task order against the server")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--request-timeout", type=float, default=1800.0)
    run.add_argument("--retries", type=int, default=2)
    # The judge wants a card of its own, and a data-parallel serve of the model
    # under test already holds every one. Deferring lets generation finish, the
    # server come down, and the judge grade both arms afterwards.
    run.add_argument(
        "--defer-judging",
        action="store_true",
        help="write rows needing a judge unscored, for `score` to grade later",
    )

    score = sub.add_parser("score", help="judge the items a deferred run left open")
    score.add_argument("--generations", required=True, type=Path)
    score.add_argument("--key", required=True, type=Path)
    score.add_argument("--results", required=True, type=Path)
    score.add_argument("--metadata", type=Path)
    score.add_argument("--concurrency", type=int, default=8)

    for parser_with_judge in (run, score):
        parser_with_judge.add_argument(
            "--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS
        )
        parser_with_judge.add_argument("--judge-timeout", type=float, default=300.0)
        # A judge fault is infrastructure, not model behavior, and must never
        # land in the score as a zero, so it retries harder than the model.
        parser_with_judge.add_argument("--judge-retries", type=int, default=4)

    pin = sub.add_parser("pin", help="print the pins object to paste into protocol.json")
    pin.add_argument("--dataset", help="the 40-character dataset commit")
    pin.add_argument("--judge", help="the judge as repo@40-character-revision")
    pin.add_argument("--judge-repo", default=DEFAULT_JUDGE_REPO)
    pin.add_argument("--resolve", action="store_true", help="look the commits up on the Hub")
    return parser.parse_args(argv)


def self_pin() -> str:
    return module_pin([Path(__file__), Path(__file__).resolve().parent / "_common.py"])


def raw_response_path(run_dir: Path, variant: str, replicate: int, item_id: str) -> Path:
    return _raw_response_path(run_dir, SUITE, variant, replicate, item_id)


def validate_pins(pins: dict[str, str]) -> None:
    dataset = pins.get("dataset", "")
    if not re.fullmatch(r"[0-9a-f]{40}", dataset):
        raise AdapterError(
            "pins.dataset must be the 40-character HLE dataset commit; "
            f"got {dataset!r}. A branch or tag is not an immutable pin."
        )
    judge_pin_parts(pins.get("judge", ""))
    require_pin(pins, "harness", HARNESS_ID)
    require_pin(pins, "verifier", VERIFIER_ID)
    require_pin(pins, "adapter", self_pin())


def load_split(revision: str) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment guard
        raise AdapterError("huggingface_hub and pyarrow are required for prepare") from error
    path = hf_hub_download(
        DATASET_REPO, DATASET_FILE, repo_type="dataset", revision=revision
    )
    return pq.read_table(path).to_pylist()


def normalize(value: str) -> str:
    text = WHITESPACE_RE.sub(" ", str(value)).strip().casefold()
    return TRAILING_RE.sub("", text)


def image_dir(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}-images"


def materialize(
    rows: list[dict[str, Any]], run_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts, key = [], {}
    for row in rows:
        item_id = str(row.get("id") or "").strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not item_id or not question or not answer:
            raise AdapterError(f"row {row.get('id')} is incomplete")
        if item_id in key:
            raise AdapterError(f"duplicate item id {item_id}")

        image = row.get("image")
        image_ref = None
        if image:
            data_url = str(image)
            if not data_url.startswith("data:"):
                raise AdapterError(f"{item_id}: image is not a data URL")
            # Written out so the frozen set is inspectable and its bytes fixed,
            # but sent from this stored copy verbatim rather than re-encoded.
            path = image_dir(run_dir) / f"{item_id}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data_url, encoding="utf-8")
            image_ref = {
                "path": str(path.relative_to(run_dir)),
                "sha256": hashlib.sha256(data_url.encode()).hexdigest(),
            }

        prompts.append(
            {
                "id": item_id,
                "suite": SUITE,
                "text": question,
                "category": str(row.get("category") or "unknown"),
            }
        )
        key[item_id] = {
            # The judge needs the question to rule on equivalence, so the key
            # carries it and grading stays independent of the prompt file.
            "question": question,
            "answer": answer,
            "normalized": normalize(answer),
            "answer_type": str(row.get("answer_type") or "unknown"),
            "category": str(row.get("category") or "unknown"),
            "raw_subject": str(row.get("raw_subject") or "unknown"),
            "image": image_ref,
        }
    if not prompts:
        raise AdapterError("no items materialized")
    return prompts, key


def key_path(run_dir: Path) -> Path:
    return run_dir / "materialized" / f"{SUITE}.key.json"


def command_prepare(args: argparse.Namespace) -> int:
    check_action("prepare", SUITE)
    pins = load_pins()
    validate_pins(pins)
    run_dir = env_path("EVAL_RUN_DIR")
    prompts, key = materialize(load_split(pins["dataset"]), run_dir)
    prompts_path = env_path("EVAL_PROMPTS_JSONL")
    write_jsonl(prompts_path, prompts)
    write_json(
        key_path(run_dir),
        {"dataset": f"{DATASET_REPO}@{pins['dataset']}", "items": key},
    )
    print(f"materialized {len(prompts)} {SUITE} items to {prompts_path}", flush=True)
    return 0


def read_image(run_dir: Path, image: dict[str, Any]) -> str:
    path = run_dir / image["path"]
    data_url = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(data_url.encode()).hexdigest()
    if digest != image["sha256"]:
        raise AdapterError(
            f"{path}: sha256 {digest} does not match the materialized {image['sha256']}"
        )
    return data_url


def extract_answer(text: str) -> str | None:
    matches = ANSWER_LINE_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else None


def option_letter(value: str) -> str | None:
    """The chosen option in `D`, `(d)`, `D.` or `D. Kappa`, if there is one."""
    match = OPTION_LETTER_RE.match(value.strip())
    return match.group(1).upper() if match else None


class JudgeError(AdapterError):
    """The judge could not be reached or did not answer in the pinned format."""


class JudgeCache:
    """Verdicts keyed on (item, normalized answer), shared across arms.

    Two responses that reduce to the same string therefore get one verdict
    rather than two independent ones. That is what stops the judge from
    inventing a discordant pair out of responses that agreed, which is the
    failure mode a per-response judge would reintroduce.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for record in read_jsonl(path):
                self._entries[str(record["key"])] = record

    @staticmethod
    def key_for(item_id: str, normalized: str) -> str:
        digest = hashlib.sha256(f"{item_id}\x00{normalized}".encode("utf-8"))
        return digest.hexdigest()[:32]

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._entries.get(key)

    def put(self, key: str, record: dict[str, Any]) -> None:
        stored = dict(record, key=key)
        with self._lock:
            if key in self._entries:
                return
            self._entries[key] = stored
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stored, ensure_ascii=False) + "\n")
                handle.flush()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class Judge:
    """A pinned open-weights grader, blind to which arm it is grading."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        pin: str,
        cache: JudgeCache,
        max_tokens: int,
        timeout: float,
        retries: int,
        client: Callable[..., dict[str, Any]],
        scheme: str = "hf",
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.pin = pin
        self.scheme = scheme
        self.cache = cache
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.client = client
        self.calls = 0
        self.hits = 0

    def payload(self, question: str, reference: str, submitted: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": JUDGE_INSTRUCTION.format(
                    question=question, reference=reference, submitted=submitted
                ),
            }],
            "max_tokens": self.max_tokens,
            # As close to deterministic as the endpoint allows: the same three
            # strings should grade the same way whichever arm they came from.
            # Where it cannot be guaranteed, the verdict cache does the work
            # instead, since a repeated string is never re-asked.
            "temperature": 0.0,
        }
        if self.scheme == "hf":
            # vLLM-only, and a hosted endpoint rejects unknown fields.
            payload["top_p"] = 1.0
            payload["seed"] = 0
            payload["chat_template_kwargs"] = {"reasoning_effort": "low"}
        return payload

    def verdict(
        self, item_id: str, question: str, reference: str, submitted: str
    ) -> dict[str, Any]:
        key = self.cache.key_for(item_id, normalize(submitted))
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return dict(cached, cached=True)

        response, _ = request_with_retries(
            item_id, self.payload(question, reference, submitted),
            base_url=self.base_url, api_key=self.api_key,
            timeout=self.timeout, retries=self.retries, client=self.client,
        )
        if response is None:
            raise JudgeError(f"{item_id}: judge timed out after {self.retries} retries")

        content, raw_reasoning, _, _ = unpack_choice(item_id, response)
        _, text = split_reasoning(content, raw_reasoning)
        matches = VERDICT_RE.findall(text or "")
        if not matches:
            # Guessing "no" here would turn a malformed judge reply into a
            # scored failure for the model, which is not what happened.
            raise JudgeError(
                f"{item_id}: judge reply has no `correct: yes|no` line: {(text or '')[:200]!r}"
            )

        record = {
            "correct": matches[-1].lower() == "yes",
            "reply": (text or "").strip()[:500],
            "judge": self.pin,
            "normalized": normalize(submitted),
            "item_id": item_id,
        }
        self.cache.put(key, record)
        self.calls += 1
        return dict(record, cached=False)


def settled(score: float, graded_by: str) -> dict[str, Any]:
    return {"score": score, "graded_by": graded_by,
            "judge_cached": False, "deferred": False}


def grade(
    item_id: str,
    predicted: str | None,
    entry: dict[str, Any],
    judge: Judge | None,
    defer: bool = False,
) -> dict[str, Any]:
    """Return the score and how it was reached.

    A normalized string match settles an item outright, because string equality
    has no false positives. Only a mismatch is worth a judge call.
    """
    if predicted is None:
        return settled(0.0, "no-answer")

    if normalize(predicted) == entry["normalized"]:
        return settled(1.0, "exact")

    if entry["answer_type"] == "multipleChoice":
        # A named option is unambiguous once the letter is read off, whether the
        # reply says `D` or `D. Kappa`. Only an unreadable one needs a judge.
        chosen = option_letter(predicted)
        if chosen is not None:
            correct = chosen == option_letter(entry["answer"])
            return settled(1.0 if correct else 0.0, "option-letter")

    if defer:
        # Carried whole rather than through the truncated `predicted` field,
        # because this string is what the judge will actually rule on.
        return {"score": 0.0, "graded_by": "deferred", "judge_cached": False,
                "deferred": True, "submitted": predicted}

    if judge is None:
        raise JudgeError(
            f"{item_id}: needs a judge but none is configured; set "
            "EVAL_JUDGE_BASE_URL and EVAL_JUDGE_MODEL, or defer with "
            "--defer-judging and grade later with `score`"
        )
    ruling = judge.verdict(item_id, entry["question"], entry["answer"], predicted)
    return {
        "score": 1.0 if ruling["correct"] else 0.0,
        "graded_by": "judge",
        "judge_cached": bool(ruling["cached"]),
        "judge_reply": ruling["reply"],
        "deferred": False,
    }


def score_response(
    item_id: str,
    response: dict[str, Any],
    *,
    entry: dict[str, Any],
    replicate: int,
    thinking: bool,
    judge: Judge | None = None,
    defer: bool = False,
) -> dict[str, Any]:
    content, raw_reasoning, finish_reason, usage = unpack_choice(item_id, response)
    reasoning, answer_text = split_reasoning(content, raw_reasoning)
    predicted = extract_answer(answer_text)
    thought = reasoning_tokens(usage, reasoning, answer_text)
    verdict = grade(item_id, predicted, entry, judge, defer=defer)

    row = base_row(SUITE, item_id, replicate)
    row.update(
        {
            **verdict,
            "empty_answer": predicted is None,
            "repetition_loop": has_repetition_loop(answer_text or reasoning),
            # vLLM discards an unterminated think block, so a reply that ran to
            # the cap arrives with no text at all -- exactly where a loop is most
            # likely. Record whether there was anything to inspect, so a False
            # here is not read as "checked, and clean".
            "repetition_assessed": bool(answer_text or reasoning),
            "malformed_tool_call": False,
            "premature_final_answer": bool(thinking and predicted is not None and thought == 0),
            "context_failure": finish_reason == "length",
            "predicted": (predicted or "")[:200],
            "expected": entry["answer"][:200],
            "category": entry["category"],
            # Kept apart because the letter half is settled mechanically and the
            # free-form half goes to the judge, and the two should be readable
            # separately.
            "answer_type": entry["answer_type"],
            "has_image": entry["image"] is not None,
            "finish_reason": finish_reason,
            "output_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "reasoning_tokens": thought,
        }
    )
    return row


def run_item(
    item_id: str,
    text: str,
    entry: dict[str, Any],
    *,
    generation: dict[str, Any],
    model: str,
    seed: int,
    replicate: int,
    variant: str,
    run_dir: Path,
    base_url: str,
    api_key: str,
    args: argparse.Namespace,
    client: Callable[..., dict[str, Any]],
    judge: Judge | None = None,
    defer: bool = False,
) -> dict[str, Any]:
    payload = build_payload(
        text, generation, model=model, seed=seed,
        max_tokens=args.max_tokens, instruction=ANSWER_INSTRUCTION,
    )
    if entry["image"]:
        payload["messages"][0]["content"] = [
            {"type": "image_url",
             "image_url": {"url": read_image(run_dir, entry["image"])}},
            {"type": "text", "text": f"{text}\n\n{ANSWER_INSTRUCTION}"},
        ]

    started_wall, started = time.time(), time.monotonic()
    response, attempts = request_with_retries(
        item_id, payload, base_url=base_url, api_key=api_key,
        timeout=args.request_timeout, retries=args.retries, client=client,
    )
    if response is None:
        row = _timeout_row(SUITE, item_id, replicate)
        row.update(
            {
                "category": entry["category"],
                "answer_type": entry["answer_type"],
                "has_image": entry["image"] is not None,
                "graded_by": "timeout",
                "judge_cached": False,
                "deferred": False,
            }
        )
    else:
        row = score_response(
            item_id, response, entry=entry, replicate=replicate,
            thinking=bool(generation["enable_thinking"]), judge=judge, defer=defer,
        )
        path = raw_response_path(run_dir, variant, replicate, item_id)
        write_json(path, response)
        row["raw_response"] = str(path)
    row.update(timing(started_wall, started))
    row["attempts"] = attempts
    return row


def build_judge(
    pins: dict[str, str],
    run_dir: Path,
    args: argparse.Namespace,
    client: Callable[..., dict[str, Any]],
) -> Judge:
    pin = pins["judge"]
    parts = judge_pin_parts(pin)
    model = env_str("EVAL_JUDGE_MODEL")
    # An endpoint reports the name it serves, never the revision, so this is the
    # one link between the pin and the process actually answering.
    if model != parts["model"]:
        raise JudgeError(
            f"EVAL_JUDGE_MODEL is {model!r} but pins.judge names "
            f"{parts['model']!r}; point the run at the pinned judge"
        )
    return Judge(
        base_url=env_str("EVAL_JUDGE_BASE_URL"),
        api_key=os.environ.get("EVAL_JUDGE_API_KEY", "EMPTY"),
        model=model,
        pin=pin,
        scheme=parts["scheme"],
        cache=JudgeCache(run_dir / "judgements" / f"{SUITE}.jsonl"),
        max_tokens=args.judge_max_tokens,
        timeout=args.judge_timeout,
        retries=args.judge_retries,
        client=client,
    )


def deferred_path(run_dir: Path, variant: str, replicate: int) -> Path:
    return run_dir / "generations" / f"{SUITE}-{variant}-r{replicate}.jsonl"


def command_run(
    args: argparse.Namespace,
    client: Callable[..., dict[str, Any]] = post_chat,
    judge_client: Callable[..., dict[str, Any]] | None = None,
) -> int:
    check_action("run", SUITE)
    pins = load_pins()
    validate_pins(pins)

    run_dir = env_path("EVAL_RUN_DIR")
    judge = None if args.defer_judging else build_judge(
        pins, run_dir, args, judge_client or client
    )
    prompts = {str(row["id"]): row["text"] for row in read_jsonl(env_path("EVAL_PROMPTS_JSONL"))}
    order = json.loads(env_path("EVAL_TASK_ORDER_JSON").read_text(encoding="utf-8"))
    stored = json.loads(key_path(run_dir).read_text(encoding="utf-8"))
    key = stored["items"]
    generation = json.loads(env_str("EVAL_GENERATION_JSON"))

    missing = [item_id for item_id in order if item_id not in prompts or item_id not in key]
    if missing:
        raise AdapterError(f"task order references unmaterialized items: {missing[:10]}")

    replicate = int(env_str("EVAL_REPLICATE"))
    seed = int(env_str("EVAL_SEED"))
    variant = env_str("EVAL_VARIANT")
    model = env_str("EVAL_SERVED_MODEL")
    base_url = env_str("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    results_path = env_path("EVAL_RESULTS_JSONL")

    started = time.monotonic()
    rows = execute_order(
        order,
        lambda item_id: run_item(
            item_id, prompts[item_id], key[item_id],
            generation=generation, model=model, seed=seed, replicate=replicate,
            variant=variant, run_dir=run_dir, base_url=base_url, api_key=api_key,
            args=args, client=client, judge=judge, defer=args.defer_judging,
        ),
        args.concurrency,
    )

    open_items = sum(1 for row in rows if row["deferred"])
    if args.defer_judging:
        # Deliberately not EVAL_RESULTS_JSONL. These rows are not scores yet,
        # and the comparator refuses a file containing a deferred row, so a
        # half-graded suite cannot be read as a result.
        path = deferred_path(run_dir, variant, replicate)
        write_jsonl(path, rows)
        write_json(
            path.with_suffix(".meta.json"),
            {
                "suite": SUITE, "variant": variant, "replicate": replicate,
                "seed": seed, "served_model": model, "items": len(rows),
                "concurrency": args.concurrency, "max_tokens": args.max_tokens,
                "dataset": stored.get("dataset"), "generation": generation,
                "adapter": self_pin(), "judge": pins["judge"],
                "deferred_items": open_items, "deferred": True,
                "wall_clock_seconds": round(time.monotonic() - started, 3),
            },
        )
        print(
            f"generated {len(rows)} {SUITE} items to {path}; "
            f"{open_items} await judging",
            flush=True,
        )
        return 0

    write_jsonl(results_path, rows)

    by_category: dict[str, list[float]] = {}
    by_type: dict[str, list[float]] = {}
    by_grader: dict[str, int] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["score"])
        by_type.setdefault(row["answer_type"], []).append(row["score"])
        by_grader[row["graded_by"]] = by_grader.get(row["graded_by"], 0) + 1
    write_json(
        run_dir / "metadata" / f"{SUITE}-{variant}-r{replicate}.json",
        {
            "suite": SUITE,
            "variant": variant,
            "replicate": replicate,
            "seed": seed,
            "served_model": model,
            "items": len(rows),
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "dataset": stored.get("dataset"),
            "generation": generation,
            "generation_overrides": {},
            "adapter": self_pin(),
            "wall_clock_seconds": round(time.monotonic() - started, 3),
            "score": round(sum(row["score"] for row in rows) / len(rows), 6),
            "score_by_category": {
                name: round(sum(scores) / len(scores), 6)
                for name, scores in sorted(by_category.items())
            },
            "score_by_answer_type": {
                name: round(sum(scores) / len(scores), 6)
                for name, scores in sorted(by_type.items())
            },
            "items_by_grader": dict(sorted(by_grader.items())),
            "judge": pins["judge"],
            "judge_model": judge.model,
            "judge_calls": judge.calls,
            "judge_cache_hits": judge.hits,
            "judge_cache_size": len(judge.cache),
            "judge_max_tokens": args.judge_max_tokens,
            # Not the published protocol: HLE grades with openai/o3-mini and
            # this uses a pinned open-weights judge, so absolute numbers are not
            # comparable to published HLE scores.
            "grading": "string match then pinned open-weights judge, not model_graded_fact",
        },
    )
    print(f"scored {len(rows)} {SUITE} items to {results_path}", flush=True)
    return 0


def command_score(
    args: argparse.Namespace, client: Callable[..., dict[str, Any]] = post_chat
) -> int:
    """Judge the rows a deferred run left open and emit real results.

    Free of the EVAL_* contract, like the LiveCodeBench split it follows: it
    takes a deferred file, an answer key and somewhere to write, so it can run
    after the served model has released its GPUs.
    """
    stored = json.loads(args.key.read_text(encoding="utf-8"))
    key = stored["items"]
    rows = read_jsonl(args.generations)
    if not rows:
        raise AdapterError(f"{args.generations}: nothing to score")

    meta_path = args.generations.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    pin = meta.get("judge") or ""
    try:
        parts = judge_pin_parts(pin)
    except AdapterError as error:
        raise JudgeError(
            f"{meta_path}: no usable pinned judge recorded, so these rows "
            f"cannot be graded reproducibly: {error}"
        ) from error

    missing = [row["id"] for row in rows if row["id"] not in key]
    if missing:
        raise AdapterError(f"deferred rows reference unkeyed items: {missing[:10]}")

    # One cache for the whole file, and it is the same file both arms write to,
    # so a string judged for one arm is never re-judged for the other.
    judge = Judge(
        base_url=env_str("EVAL_JUDGE_BASE_URL"),
        api_key=os.environ.get("EVAL_JUDGE_API_KEY", "EMPTY"),
        model=parts["model"],
        pin=pin,
        scheme=parts["scheme"],
        cache=JudgeCache(args.generations.parent / f"{SUITE}-judgements.jsonl"),
        max_tokens=args.judge_max_tokens,
        timeout=args.judge_timeout,
        retries=args.judge_retries,
        client=client,
    )

    started = time.monotonic()
    # Indexed, not keyed on the item id: one file may hold several replicates of
    # the same item, and keying on the id would silently drop all but the last.
    positions = [str(index) for index in range(len(rows))]

    def judge_one(position: str) -> dict[str, Any]:
        row = dict(rows[int(position)])
        if not row.get("deferred"):
            return row
        entry = key[row["id"]]
        ruling = judge.verdict(
            row["id"], entry["question"], entry["answer"], row["submitted"]
        )
        row.update(
            {
                "score": 1.0 if ruling["correct"] else 0.0,
                "graded_by": "judge",
                "judge_cached": bool(ruling["cached"]),
                "judge_reply": ruling["reply"],
                "deferred": False,
            }
        )
        row.pop("submitted", None)
        return row

    graded = execute_order(positions, judge_one, args.concurrency)
    write_jsonl(args.results, graded)

    by_grader: dict[str, int] = {}
    for row in graded:
        by_grader[row["graded_by"]] = by_grader.get(row["graded_by"], 0) + 1

    if args.metadata:
        # Merge, so the generation step's record of which checkpoint produced
        # these responses survives grading.
        existing = (
            json.loads(args.metadata.read_text(encoding="utf-8"))
            if args.metadata.is_file()
            else {}
        )
        existing.update(
            {
                "suite": SUITE,
                "variant": meta.get("variant"),
                "replicate": meta.get("replicate"),
                "seed": meta.get("seed"),
                "served_model": meta.get("served_model"),
                "items": len(graded),
                "max_tokens": meta.get("max_tokens"),
                "dataset": stored.get("dataset"),
                "generation": meta.get("generation"),
                "generation_overrides": {},
                "adapter": self_pin(),
                "judge": pin,
                "judge_calls": judge.calls,
                "judge_cache_hits": judge.hits,
                "judge_cache_size": len(judge.cache),
                "judge_deferred": True,
                "items_by_grader": dict(sorted(by_grader.items())),
                "wall_clock_seconds": round(time.monotonic() - started, 3),
                "score": round(sum(row["score"] for row in graded) / len(graded), 6),
                "grading": (
                    "string match then pinned open-weights judge, "
                    "not model_graded_fact"
                ),
                "deferred": False,
            }
        )
        write_json(args.metadata, existing)

    print(
        f"scored {len(graded)} {SUITE} items to {args.results} "
        f"({judge.calls} judge calls, {judge.hits} cache hits)",
        flush=True,
    )
    return 0


def command_pin(args: argparse.Namespace) -> int:
    dataset, judge = args.dataset, args.judge
    if args.resolve:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise AdapterError("huggingface_hub is required for --resolve") from error
        api = HfApi()
        dataset = str(api.dataset_info(DATASET_REPO).sha)
        # Only an `hf:` judge can be resolved here. An `api:` one is pinned by
        # hand, because there is no hash to look up.
        judge = judge or f"hf:{args.judge_repo}@{api.model_info(args.judge_repo).sha}"
    print(
        json.dumps(
            {
                "dataset": dataset or "REPLACE_WITH_HLE_REVISION",
                "judge": judge or "hf:OWNER/NAME@REVISION or api:PROVIDER/MODEL@SNAPSHOT",
                "harness": HARNESS_ID,
                "verifier": VERIFIER_ID,
                "adapter": self_pin(),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "pin":
        return command_pin(args)
    if args.action == "prepare":
        return command_prepare(args)
    if args.action == "score":
        return command_score(args)
    return command_run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AdapterError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
