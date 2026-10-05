"""The cold path: turn each incoming message into fact assertions and retractions.

Extraction runs once per (dataset, model, prompt version) and is cached on disk,
so the SQLite, graph and vector conditions all replay identical facts. Its cost
is reported separately from the agent's own (hot path) cost.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .llm import openai_usage_record, provider, supports_effort, usage_record, with_rate_limit_retry
from .stores import FactLog

PROMPT_VERSION = "v1"

OPS_HINT = (
    "Prefer these predicates when they fit: owned_by (service -> team), "
    "depends_on (service -> the service it calls), member_of (person -> team), "
    "on_call (team -> person currently holding the pager), incident (incident id -> service). "
    "Entity types: service, team, person, incident."
)
GENERIC_HINT = (
    "Use short snake_case predicates. Capture durable facts about the user and the world: "
    "preferences, possessions, relationships, plans, events with their dates, and changes."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "assert": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "subject_type": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "object_type": {"type": "string"},
                },
                "required": ["subject", "subject_type", "predicate", "object", "object_type"],
                "additionalProperties": False,
            },
        },
        "retract": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["assert", "retract"],
    "additionalProperties": False,
}


def _prompt(text: str, when: str, known: list[str], current: list[str], hint: str) -> str:
    return (
        "You maintain a knowledge graph from a stream of messages. Extract the durable facts "
        "this message states as (subject, predicate, object) triples.\n\n"
        f"{hint}\n\n"
        "Rules:\n"
        "- Ignore log lines, chit-chat, and anything that is not a fact worth remembering.\n"
        "- Reuse the exact spelling of a known entity when the message refers to it, even by a "
        "nickname or short form. Use full canonical names for new entities.\n"
        "- If the message says a fact is no longer true or has been replaced (a handoff, a "
        "transfer, a removal), put the old fact under retract, copied exactly from the current "
        "facts below, and put the new fact under assert.\n"
        "- Return empty lists if the message has no facts.\n\n"
        f"Message time: {when}\n"
        f"Known entities mentioned: {', '.join(known) if known else '(none)'}\n"
        "Current facts about them:\n"
        + ("\n".join(current) if current else "(none)")
        + f"\n\nMessage:\n<message>\n{text}\n</message>"
    )


class Extractor:
    def __init__(self, client, model: str, cache_dir: Path, dataset_key: str, generic: bool = False) -> None:
        self.client = client
        self.model = model
        self.generic = generic
        key = hashlib.sha256(f"{dataset_key}|{model}|{PROMPT_VERSION}|{generic}".encode()).hexdigest()[:16]
        self.path = cache_dir / f"{dataset_key}-{key}.jsonl"
        self.cache: dict[int, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                rec = json.loads(line)
                self.cache[rec["step"]] = rec

    def ingest(self, log: FactLog, step: int, when, text: str) -> dict:
        """Extract (or replay) facts for one message and apply them to the log."""
        if step in self.cache:
            rec = dict(self.cache[step], cached=True)
        else:
            rec = self._extract(log, step, when, text)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            self.cache[step] = rec
            rec = dict(rec, cached=False)
        log.apply(rec["ops"], step, when, text)
        return rec

    def _extract(self, log: FactLog, step: int, when, text: str) -> dict:
        body = text.split("\n\nattached logs:\n")[0]
        known = log.known_entities_in(body)
        current = [f"{f.subject} | {f.predicate} | {f.object}" for f in log.facts_about(known)]
        prompt = _prompt(body, when.strftime("%Y-%m-%d %H:%M"), known, current,
                         GENERIC_HINT if self.generic else OPS_HINT)
        t0 = time.perf_counter()
        if provider(self.model) == "openai":
            resp = with_rate_limit_retry(
                self.client.responses.create,
                model=self.model,
                input=prompt,
                text={"format": {"type": "json_schema", "name": "facts", "schema": SCHEMA, "strict": True}},
                reasoning={"effort": "low"},
                store=False,
                max_output_tokens=8000,
            )
            raw, stop, usage = resp.output_text, getattr(resp, "status", None), openai_usage_record(resp.usage, self.model)
        else:
            fmt = {"format": {"type": "json_schema", "schema": SCHEMA}}
            if supports_effort(self.model):
                fmt["effort"] = "low"
            resp = with_rate_limit_retry(
                self.client.messages.create,
                model=self.model,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
                output_config=fmt,
            )
            raw = next((b.text for b in resp.content if b.type == "text"), "")
            stop, usage = resp.stop_reason, usage_record(resp.usage, self.model)
        latency = time.perf_counter() - t0
        try:
            ops = json.loads(raw)
        except json.JSONDecodeError:
            ops = {"assert": [], "retract": [], "error": f"unparseable ({stop})"}
        return {
            "step": step,
            "ops": ops,
            "usage": usage,
            "latency_s": latency,
            "stop_reason": stop,
        }
