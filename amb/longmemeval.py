"""LongMemEval (ICLR 2025) as an outside check on the synthetic world.

Each question has its own haystack of ~50 chat sessions. Sessions arrive as
inbox items in date order, then the question. Answers are graded by an LLM
judge with the benchmark's own per-type prompts (the paper used GPT-4o as the
judge; here it is a Claude model, so scores are not directly comparable to
published leaderboards).
"""

from __future__ import annotations

import json
import random
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from .agent import InboxItem
from .llm import openai_usage_record, provider, supports_effort, usage_record, with_rate_limit_retry

URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
DEFAULT_TYPES = ("knowledge-update", "temporal-reasoning", "multi-session", "single-session-user")

_BASE = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
    "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
    "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
    "the response only contains a subset of the information required by the answer, answer no. "
)
JUDGE = {
    "single-session-user": _BASE,
    "single-session-assistant": _BASE,
    "multi-session": _BASE,
    "temporal-reasoning": _BASE + (
        "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the "
        "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
        "the answer is 18), the model's response is still correct. "
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
        "response contains the correct answer. Otherwise, answer no. If the response contains some previous "
        "information along with an updated answer, the response should be considered as correct as long as the "
        "updated answer is the required answer."
    ),
}
ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if "
    "the model correctly identifies the question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\n"
    "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)


def load(data_dir: Path) -> list[dict]:
    path = data_dir / "longmemeval_s_cleaned.json"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return json.loads(path.read_text())


def select(data: list[dict], n_per_type: int, seed: int = 0, types=DEFAULT_TYPES) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for t in types:
        pool = [d for d in data if d["question_type"] == t and not d["question_id"].endswith("_abs")]
        out += rng.sample(pool, min(n_per_type, len(pool)))
    return out


def _date(s: str) -> datetime | None:
    try:
        return datetime.strptime(s.split(" (")[0] + " " + s.split(") ")[-1], "%Y/%m/%d %H:%M")
    except (ValueError, IndexError):
        return None


def items(entry: dict) -> list[InboxItem]:
    out = []
    for i, (date, session) in enumerate(zip(entry["haystack_dates"], entry["haystack_sessions"]), start=1):
        turns = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in session)
        out.append(InboxItem(kind="event", step=i, time=_date(date), text=f"Chat session from {date}:\n{turns}"))
    q_time = _date(entry["question_date"])
    out.append(InboxItem(
        kind="question", step=len(out), time=q_time,
        text=f"(asked on {entry['question_date']}) {entry['question']}",
        qid=entry["question_id"], category=entry["question_type"], answer=[str(entry["answer"])],
        extra={"question": entry["question"]},
    ))
    return out


class Judge:
    def __init__(self, client, model: str) -> None:
        self.client, self.model = client, model

    def __call__(self, item: InboxItem, given: list[str]) -> dict:
        response = " ".join(given)
        q, truth = item.extra["question"], item.answer[0]
        if item.qid.endswith("_abs"):
            prompt = ABSTENTION.format(q, truth, response)
        else:
            prompt = JUDGE[item.category] + f"\n\nQuestion: {q}\n\nCorrect Answer: {truth}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."
        t0 = time.perf_counter()
        if provider(self.model) == "openai":
            resp = with_rate_limit_retry(self.client.responses.create, model=self.model, input=prompt, reasoning={"effort": "low"},
                                                store=False, max_output_tokens=2000)
            text, usage = resp.output_text, openai_usage_record(resp.usage, self.model)
        else:
            kw = {"output_config": {"effort": "low"}} if supports_effort(self.model) else {}
            resp = with_rate_limit_retry(self.client.messages.create, model=self.model, max_tokens=2000, messages=[{"role": "user", "content": prompt}], **kw)
            text, usage = next((b.text for b in resp.content if b.type == "text"), ""), usage_record(resp.usage, self.model)
        text = text.strip().lower()
        ok = text.startswith("yes")
        return {"correct": ok, "f1": float(ok), "judge_text": text[:50], "judge_usage": usage,
                "judge_s": time.perf_counter() - t0}
