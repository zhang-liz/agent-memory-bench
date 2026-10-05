"""The agent loop. One long tool-use conversation per run, six memory setups.

The agent pulls inbox items one at a time with `read_inbox`. Items are either
messages (with attached logs) or questions it must answer with `submit_answer`.
Only the memory setup changes between conditions: model, prompts, inbox and
answer tools are identical.

    full     everything stays in context (prompt caching on)
    compact  server-side compaction at the 50k-token API minimum
    memtool  Anthropic memory tool + context editing
    vector   hybrid search over messages and extracted facts + context editing
    sqlite   read-only SQL over extracted facts + context editing
    graph    read-only Cypher over the same facts in FalkorDB + context editing
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import anthropic

from .llm import supports_effort, usage_record, with_rate_limit_retry
from .memory_tool import MemoryFiles

CONDITIONS = ("full", "compact", "memtool", "vector", "sqlite", "graph")

BASE_SYSTEM = """You are the operations agent for Acme's platform engineering org.

Work through your inbox by calling read_inbox, one item at a time, until it says the inbox is empty. Most items are chat messages, tickets, emails and attached logs. Read them and keep track of what matters: who owns which service, which services depend on which, who is on which team, who holds the on-call pager for each team, and incidents. These facts change over time; the latest message wins.

Some inbox items are questions. Answer each question with submit_answer as soon as you receive it, then keep reading. Answers are graded exactly: use the names as written in the messages, and for list questions include every correct item and nothing else. Never ask for clarification; give your best answer.

Do not stop until read_inbox says the inbox is empty."""

MEMORY_NOTES = {
    "full": "",
    "compact": "",
    "memtool": (
        "\n\nOlder inbox items are cleared from your context as you go. Use your memory directory "
        "to keep anything you may need to answer later questions."
    ),
    "vector": (
        "\n\nOlder inbox items are cleared from your context as you go. A memory system records every "
        "message you read and the facts extracted from it. Use search_memory to look things up when "
        "answering questions."
    ),
    "sqlite": (
        "\n\nOlder inbox items are cleared from your context as you go. A memory system extracts the facts "
        "from every message you read into a SQLite database. Use run_sql to look things up when answering "
        "questions."
    ),
    "graph": (
        "\n\nOlder inbox items are cleared from your context as you go. A memory system extracts the facts "
        "from every message you read into a graph database. Use run_cypher to look things up when answering "
        "questions."
    ),
}

NEXT_SYSTEM = """You are the operations agent for Acme's platform engineering org, starting a new shift.

Earlier shifts read the org's chat messages, tickets, emails and logs. You did not see them, and this conversation starts empty. Your inbox now holds only questions about the current state: who owns which service, which services depend on which, who is on which team, who holds the on-call pager for each team, and incidents.

Call read_inbox to get each question and answer it with submit_answer, then keep reading until read_inbox says the inbox is empty. Answers are graded exactly: use the names as the messages wrote them, and for list questions include every correct item and nothing else. Never ask for clarification; give your best answer."""

# What survives into the new shift, per setup.
NEXT_NOTES = {
    "none": "\n\nNothing from earlier shifts was kept.",
    "handoff": "\n\nThe previous shift left you this handoff note:\n\n<handoff>\n{note}\n</handoff>",
    "memtool": "\n\nEarlier shifts kept notes in your memory directory. Use your memory tool to read them.",
    "vector": (
        "\n\nA memory system recorded every message earlier shifts read and the facts extracted from them. "
        "Use search_memory to look things up."
    ),
    "sqlite": (
        "\n\nA memory system extracted the facts from every message earlier shifts read into a SQLite "
        "database. Use run_sql to look things up."
    ),
    "graph": (
        "\n\nA memory system extracted the facts from every message earlier shifts read into a graph "
        "database. Use run_cypher to look things up."
    ),
}
NEXT_CONDITIONS = tuple(NEXT_NOTES)

# A note long enough for any state this benchmark builds, so it is never cut off by a length cap.
HANDOFF_MAX_TOKENS = 32_000
HANDOFF_PROMPT = (
    "Your shift is over. The next operations agent starts with an empty conversation and will not see "
    "anything you read. Write the handoff note they will get: everything they may need to answer questions "
    "about the current state of services, owners, dependencies, teams, on-call and incidents. Reply with the "
    "note only."
)


def system_prompt(task: str, condition: str, handoff_note: str = "") -> str:
    if task == "next":
        return NEXT_SYSTEM + NEXT_NOTES[condition].format(note=handoff_note)
    note = MEMORY_NOTES[condition]
    if task == "lme":
        return LME_SYSTEM + note.replace("every message you read", "every session you read")
    return BASE_SYSTEM + note


INBOX_TOOLS = [
    {
        "name": "read_inbox",
        "description": "Return the next inbox item. Call it again after handling each item.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "submit_answer",
        "description": "Answer a question from the inbox. Call it once per question, as soon as you get the question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question_id": {"type": "string"},
                "answer": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One entry per item. A single-answer question has one entry.",
                },
            },
            "required": ["question_id", "answer"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

FACT_SHAPES = (
    "Predicates in use include owned_by (service -> team), depends_on (service -> service it calls), "
    "member_of (person -> team), on_call (team -> person), incident (incident id -> service); "
    "others may appear."
)

LME_SYSTEM = """You are a personal assistant with long-term memory.

Work through your inbox by calling read_inbox, one item at a time, until it says the inbox is empty. Most items are past chat sessions between you and the user, with their dates. Read them and keep track of what you learn about the user. Facts change over time; the latest session wins.

Some inbox items are questions from the user. Answer each with submit_answer as soon as you receive it, as one short, direct answer. If the sessions do not contain the information, say so. Never ask for clarification.

Do not stop until read_inbox says the inbox is empty."""

QUERY_TOOLS = {
    "vector": {
        "name": "search_memory",
        "description": (
            "Search memory: hybrid keyword + semantic search, reranked, over every inbox message read so far "
            "and every fact extracted from them. Facts that stopped being true are marked NO LONGER TRUE. "
            "Returns the top k results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "sqlite": {
        "name": "run_sql",
        "description": (
            "Run one read-only SQLite query against memory. Table facts(id, subject, subject_type, predicate, "
            "object, object_type, valid_from, valid_to, source_step); valid_to is NULL while a fact is still "
            "true. View current_facts has only facts that are still true. Recursive CTEs work for chains. "
            + FACT_SHAPES + " Returns up to 50 rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "graph": {
        "name": "run_cypher",
        "description": (
            "Run one read-only Cypher query against the memory graph (FalkorDB). Nodes are (:Entity {name, type}). "
            "Each fact is a relationship whose type is the predicate in UPPER_SNAKE_CASE, with properties "
            "valid_from, valid_to (null while the fact is still true) and source_step; filter on "
            "r.valid_to IS NULL for current facts. Variable-length patterns such as -[:DEPENDS_ON*1..8]-> work. "
            + FACT_SHAPES.replace("owned_by", "OWNED_BY").replace("depends_on", "DEPENDS_ON")
            .replace("member_of", "MEMBER_OF").replace("on_call", "ON_CALL").replace("incident (", "INCIDENT (")
            + " Returns up to 50 rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class InboxItem:
    kind: str  # "event" | "question"
    step: int
    time: datetime | None
    text: str
    qid: str | None = None
    category: str | None = None
    answer: list[str] | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class RunConfig:
    condition: str
    model: str
    effort: str = "medium"
    clear_trigger: int = 20_000
    clear_keep: int = 5
    clear_at_least: int = 8_000
    compact_trigger: int = 50_000
    max_tokens: int = 8_000
    task: str = "ops"  # "ops" (synthetic world), "next" (a new session after it) or "lme" (LongMemEval)
    handoff: bool = False  # after the inbox, ask for a handoff note for the next session
    handoff_note: str = ""  # the note a "handoff" session starts with


def query_tool(condition: str, task: str) -> dict:
    """The memory query tool, with schema hints that fit the task's facts."""
    tool = json.loads(json.dumps(QUERY_TOOLS[condition]))
    if task == "lme":
        how = {"sqlite": "Run SELECT DISTINCT predicate FROM facts", "graph": "Run CALL db.relationshipTypes()"}
        for shape in (FACT_SHAPES, _graph_shapes()):
            tool["description"] = tool["description"].replace(
                shape + " ", f"{how.get(condition, '')} to see which predicates exist. ")
    return tool


def _graph_shapes() -> str:
    return (FACT_SHAPES.replace("owned_by", "OWNED_BY").replace("depends_on", "DEPENDS_ON")
            .replace("member_of", "MEMBER_OF").replace("on_call", "ON_CALL").replace("incident (", "INCIDENT ("))


class Episode:
    """Drives one run and writes every call, tool use and answer to a JSONL log."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        cfg: RunConfig,
        items: list[InboxItem],
        grade: Callable[[InboxItem, list[str]], dict],
        log_path: Path,
        ingest: Callable[[InboxItem], dict] | None = None,
        query: Callable[[str, dict], tuple[str, bool, float]] | None = None,
        memory_dir: Path | None = None,
        keep_memory: bool = False,
    ) -> None:
        self.client, self.cfg, self.items, self.grade = client, cfg, items, grade
        self.ingest, self.query = ingest, query
        self.memory = MemoryFiles(memory_dir, fresh=not keep_memory) if cfg.condition == "memtool" else None
        self.log_path = log_path
        self.cursor = 0
        self.pending: InboxItem | None = None
        self.answered: set[str] = set()
        self.api_calls = 0
        self.nudges = 0

    # -- logging ---------------------------------------------------------
    def _log(self, rec: dict) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # -- request shape per condition --------------------------------------
    def _request_kwargs(self) -> dict:
        c = self.cfg.condition
        tools = list(INBOX_TOOLS)
        betas: list[str] = []
        extra: dict = {}
        if c in QUERY_TOOLS:
            tools.append(query_tool(c, self.cfg.task))
        if c == "memtool":
            tools.append({"type": "memory_20250818", "name": "memory"})
        if c == "compact":
            betas.append("compact-2026-01-12")
            extra["context_management"] = {
                "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": self.cfg.compact_trigger}}]
            }
        if c in ("memtool", "vector", "sqlite", "graph"):
            betas.append("context-management-2025-06-27")
            extra["context_management"] = {
                "edits": [{
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "input_tokens", "value": self.cfg.clear_trigger},
                    "keep": {"type": "tool_uses", "value": self.cfg.clear_keep},
                    "clear_at_least": {"type": "input_tokens", "value": self.cfg.clear_at_least},
                }]
            }
        output_config = {"effort": self.cfg.effort} if supports_effort(self.cfg.model) else None
        kw = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system_prompt(self.cfg.task, self.cfg.condition, self.cfg.handoff_note),
            tools=tools,
            cache_control={"type": "ephemeral"},
            **extra,
        )
        if betas:
            kw["betas"] = betas
        if output_config:
            kw["output_config"] = output_config
        return kw

    # -- tools -------------------------------------------------------------
    def _read_inbox(self) -> str:
        if self.pending is not None:
            return f"Answer question {self.pending.qid} with submit_answer before reading more."
        if self.cursor >= len(self.items):
            return "The inbox is empty. You are done."
        item = self.items[self.cursor]
        self.cursor += 1
        header = f"[Inbox item {self.cursor} of {len(self.items)}"
        if item.time:
            header += f" | {item.time:%Y-%m-%d %H:%M}"
        header += "]"
        if item.kind == "question":
            self.pending = item
            return f"{header}\nQUESTION {item.qid}: {item.text}\nAnswer it now with submit_answer."
        if self.ingest:
            rec = self.ingest(item)
            self._log({"type": "extract", "step": item.step, "cached": rec.get("cached"),
                       "usage": rec["usage"], "latency_s": rec["latency_s"],
                       "n_assert": len(rec["ops"].get("assert", [])), "n_retract": len(rec["ops"].get("retract", []))})
        return f"{header}\n{item.text}"

    def _submit(self, inp: dict) -> tuple[str, bool]:
        qid = inp.get("question_id", "")
        item = self.pending if self.pending and self.pending.qid == qid else None
        if item is None:
            return f"No open question with id {qid!r}.", True
        g = self.grade(item, inp.get("answer", []))
        self._log({"type": "answer", "qid": qid, "category": item.category, "step": item.step,
                   "given": inp.get("answer"), "truth": item.answer, **g})
        self.answered.add(qid)
        self.pending = None
        return "Answer recorded. Continue with read_inbox.", False

    def _run_tool(self, block) -> dict:
        t0 = time.perf_counter()
        is_error, engine_ms = False, None
        name, inp = block.name, block.input or {}
        if name == "read_inbox":
            out = self._read_inbox()
        elif name == "submit_answer":
            out, is_error = self._submit(inp)
        elif name == "memory" and self.memory:
            out, is_error = self.memory.run(inp)
        elif name in ("search_memory", "run_sql", "run_cypher") and self.query:
            out, is_error, engine_ms = self.query(name, inp)
        else:
            out, is_error = f"Unknown tool {name}", True
        wall_ms = (time.perf_counter() - t0) * 1000
        if name not in ("read_inbox", "submit_answer"):
            self._log({"type": "tool", "name": name, "wall_ms": wall_ms, "engine_ms": engine_ms,
                       "is_error": is_error, "chars": len(out), "cursor": self.cursor,
                       "input": json.dumps(inp)[:500]})
        res = {"type": "tool_result", "tool_use_id": block.id, "content": out}
        if is_error:
            res["is_error"] = True
        return res

    # -- main loop -------------------------------------------------------------
    def run(self) -> dict:
        kw = self._request_kwargs()
        messages: list[dict] = [{"role": "user", "content": "Start working through the inbox."}]
        max_calls = len(self.items) * 4 + 50
        refusals = 0
        t_start = time.perf_counter()
        done = False
        while not done and self.api_calls < max_calls:
            t0 = time.perf_counter()
            try:
                resp = with_rate_limit_retry(self.client.beta.messages.create, messages=messages, **kw)
            except anthropic.BadRequestError as e:
                self._log({"type": "error", "where": "api", "message": str(e)[:2000]})
                break
            latency = time.perf_counter() - t0
            self.api_calls += 1
            usage = usage_record(resp.usage, self.cfg.model)
            cm = getattr(resp, "context_management", None)
            self._log({
                "type": "call", "n": self.api_calls, "cursor": self.cursor, "latency_s": latency,
                "stop_reason": resp.stop_reason, "usage": usage,
                "applied_edits": [e.model_dump() for e in (getattr(cm, "applied_edits", None) or [])] if cm else [],
                "compacted": any(b.type == "compaction" for b in resp.content),
            })
            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if tool_uses:
                messages.append({"role": "user", "content": [self._run_tool(b) for b in tool_uses]})
                continue
            if resp.stop_reason == "refusal":
                refusals += 1
                if refusals >= 3:
                    self._log({"type": "error", "where": "refusal", "category": getattr(resp.stop_details, "category", None)})
                    break
            if self.cursor >= len(self.items) and self.pending is None:
                done = True
                break
            self.nudges += 1
            messages.append({"role": "user", "content": "Keep going: call read_inbox for the next item."})

        if done and self.cfg.handoff:
            messages.append({"role": "user", "content": HANDOFF_PROMPT})
            resp = with_rate_limit_retry(self.client.beta.messages.create, messages=messages,
                                         **{**kw, "tool_choice": {"type": "none"}, "max_tokens": HANDOFF_MAX_TOKENS})
            self.api_calls += 1
            self._log({"type": "call", "n": self.api_calls, "cursor": self.cursor, "latency_s": None,
                       "stop_reason": resp.stop_reason, "usage": usage_record(resp.usage, self.cfg.model),
                       "applied_edits": [], "compacted": False, "handoff": True})
            self._log({"type": "handoff", "text": "".join(b.text for b in resp.content if b.type == "text")})

        summary = {
            "type": "summary", "condition": self.cfg.condition, "model": self.cfg.model,
            "finished": done, "items_read": self.cursor, "items_total": len(self.items),
            "questions_total": sum(1 for i in self.items if i.kind == "question"),
            "questions_answered": len(self.answered), "api_calls": self.api_calls,
            "nudges": self.nudges, "wall_s": time.perf_counter() - t_start,
        }
        self._log(summary)
        return summary
