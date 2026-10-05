"""The same agent loop on the OpenAI Responses API.

What maps one to one: model, prompts, inbox and answer tools, the query tools,
and server-side compaction (`context_management` with `compact_threshold`).

What has no native OpenAI equivalent, and is implemented here instead:
- Clearing old tool results (Anthropic's `clear_tool_uses_20250919`): the
  harness replaces old `function_call_output` contents with a placeholder,
  with the same trigger / keep / clear-at-least settings.
- The memory tool (Anthropic's `memory_20250818`): a function tool with the
  same commands, served by the same file backend, plus the protocol text
  Anthropic injects into the system prompt.

Requests are stateless (`store=False`), with encrypted reasoning carried
forward, so the harness owns the full history just as it does for Claude.
"""

from __future__ import annotations

import json
import time

import openai

from .agent import HANDOFF_MAX_TOKENS, HANDOFF_PROMPT, INBOX_TOOLS, Episode, query_tool, system_prompt
from .llm import openai_usage_record, with_rate_limit_retry

CLEARED = "[This tool result was cleared to save context. Query memory if you need it again.]"

MEMORY_PROTOCOL = """

IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE.
MEMORY PROTOCOL:
1. Use the `view` command of your `memory` tool to check for earlier progress.
2. ... (work on the task) ...
   - As you make progress, record status / progress / thoughts etc in your memory.
ASSUME INTERRUPTION: Your context window might be reset at any moment, so you risk losing any progress that is not recorded in your memory directory."""

MEMORY_TOOL = {
    "type": "function",
    "name": "memory",
    "description": (
        "Read and write files in your /memories directory, which persists while older context is cleared. "
        "Commands: view (path, optional view_range [start, end]) shows a directory listing or a file with line "
        "numbers; create (path, file_text) creates or overwrites a file; str_replace (path, old_str, new_str) "
        "replaces one unique occurrence; insert (path, insert_line, insert_text) inserts after a line (0 = top); "
        "delete (path); rename (old_path, new_path). All paths start with /memories. Keep the directory organized."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "delete", "rename"]},
            "path": {"type": "string"},
            "view_range": {"type": "array", "items": {"type": "integer"}},
            "file_text": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
            "insert_line": {"type": "integer"},
            "insert_text": {"type": "string"},
            "old_path": {"type": "string"},
            "new_path": {"type": "string"},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
}


def _function_tool(t: dict) -> dict:
    out = {"type": "function", "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}
    if t.get("strict"):
        out["strict"] = True
    return out


def _as_dict(item) -> dict:
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    return {k: (_as_dict(v) if hasattr(v, "__dict__") else v) for k, v in vars(item).items()}


class _Call:
    """Adapts an OpenAI function_call item to the shape Episode._run_tool expects."""

    def __init__(self, item) -> None:
        self.id = item.call_id
        self.name = item.name
        try:
            self.input = json.loads(item.arguments or "{}")
        except json.JSONDecodeError:
            self.input = {}


class OpenAIEpisode(Episode):
    def _request_kwargs(self) -> dict:
        c = self.cfg.condition
        tools = [_function_tool(t) for t in INBOX_TOOLS]
        if c in ("vector", "sqlite", "graph"):
            tools.append(_function_tool(query_tool(c, self.cfg.task)))
        system = system_prompt(self.cfg.task, c, self.cfg.handoff_note)
        if c == "memtool":
            tools.append(MEMORY_TOOL)
            system += MEMORY_PROTOCOL
        kw = dict(
            model=self.cfg.model,
            instructions=system,
            tools=tools,
            store=False,
            include=["reasoning.encrypted_content"],
            reasoning={"effort": self.cfg.effort},
            max_output_tokens=self.cfg.max_tokens,
        )
        if c == "compact":
            kw["context_management"] = [{"type": "compaction", "compact_threshold": self.cfg.compact_trigger}]
        return kw

    def _clear_old_results(self, items: list[dict], last_input_tokens: int) -> dict | None:
        """Client-side equivalent of clear_tool_uses: same trigger, keep and clear_at_least."""
        if self.cfg.condition not in ("memtool", "vector", "sqlite", "graph"):
            return None
        if last_input_tokens < self.cfg.clear_trigger:
            return None
        outputs = [i for i, it in enumerate(items) if it.get("type") == "function_call_output" and it["output"] != CLEARED]
        targets = outputs[: max(0, len(outputs) - self.cfg.clear_keep)]
        est = sum(len(items[i]["output"]) for i in targets) // 4
        if not targets or est < self.cfg.clear_at_least:
            return None
        for i in targets:
            items[i] = dict(items[i], output=CLEARED)
        return {"type": "client_clear_tool_results", "cleared_tool_uses": len(targets), "cleared_input_tokens_est": est}

    def run(self) -> dict:
        kw = self._request_kwargs()
        items: list[dict] = [{"role": "user", "content": "Start working through the inbox."}]
        max_calls = len(self.items) * 4 + 50
        refusals, last_in = 0, 0
        t_start = time.perf_counter()
        done = False
        while not done and self.api_calls < max_calls:
            edit = self._clear_old_results(items, last_in)
            t0 = time.perf_counter()
            try:
                resp = with_rate_limit_retry(self.client.responses.create, input=items, **kw)
            except openai.BadRequestError as e:
                self._log({"type": "error", "where": "api", "message": str(e)[:2000]})
                break
            latency = time.perf_counter() - t0
            self.api_calls += 1
            usage = openai_usage_record(resp.usage, self.cfg.model)
            last_in = usage["context_tokens"]
            compacted = any(o.type == "compaction" for o in resp.output)
            self._log({
                "type": "call", "n": self.api_calls, "cursor": self.cursor, "latency_s": latency,
                "stop_reason": getattr(resp, "status", None), "usage": usage,
                "applied_edits": [edit] if edit else [], "compacted": compacted,
            })
            new = [_as_dict(o) for o in resp.output]
            if compacted:
                # Everything before the newest compaction item is summarized by it.
                last = max(i for i, o in enumerate(new) if o.get("type") == "compaction")
                items, new = [], new[last:]
            items.extend(new)
            calls = [o for o in resp.output if o.type == "function_call"]
            if calls:
                for o in calls:
                    res = self._run_tool(_Call(o))
                    items.append({"type": "function_call_output", "call_id": o.call_id, "output": res["content"]})
                continue
            if any(getattr(c, "type", "") == "refusal" for o in resp.output if o.type == "message" for c in o.content):
                refusals += 1
                if refusals >= 3:
                    self._log({"type": "error", "where": "refusal"})
                    break
            if self.cursor >= len(self.items) and self.pending is None:
                done = True
                break
            self.nudges += 1
            items.append({"role": "user", "content": "Keep going: call read_inbox for the next item."})

        if done and self.cfg.handoff:
            items.append({"role": "user", "content": HANDOFF_PROMPT})
            resp = with_rate_limit_retry(self.client.responses.create, input=items, **{**kw, "tool_choice": "none", "max_output_tokens": HANDOFF_MAX_TOKENS})
            self.api_calls += 1
            self._log({"type": "call", "n": self.api_calls, "cursor": self.cursor, "latency_s": None,
                       "stop_reason": getattr(resp, "status", None),
                       "usage": openai_usage_record(resp.usage, self.cfg.model),
                       "applied_edits": [], "compacted": False, "handoff": True})
            self._log({"type": "handoff", "text": resp.output_text})

        summary = {
            "type": "summary", "condition": self.cfg.condition, "model": self.cfg.model,
            "finished": done, "items_read": self.cursor, "items_total": len(self.items),
            "questions_total": sum(1 for i in self.items if i.kind == "question"),
            "questions_answered": len(self.answered), "api_calls": self.api_calls,
            "nudges": self.nudges, "wall_s": time.perf_counter() - t_start,
        }
        self._log(summary)
        return summary
