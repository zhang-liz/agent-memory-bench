"""A scripted stand-in for the API, so the whole harness can run offline.

It reads every inbox item, makes one memory query per question, and answers
with whatever the query returned first. Use it to test the plumbing, never to
produce results.
"""

from __future__ import annotations

import itertools
import json
import re
from types import SimpleNamespace as NS

_ids = itertools.count(1)

QUERIES = {
    "run_cypher": "MATCH (a)-[r]->(b) WHERE r.valid_to IS NULL RETURN a.name, type(r), b.name LIMIT 3",
    "run_sql": "SELECT subject, predicate, object FROM current_facts LIMIT 3",
    "search_memory": "who owns",
    "memory": None,
}


def _usage(messages) -> NS:
    chars = len(json.dumps(messages, default=str))
    return NS(input_tokens=200, cache_creation_input_tokens=300, cache_read_input_tokens=chars // 4,
              output_tokens=40, iterations=None)


def _tool_use(name: str, inp: dict) -> NS:
    return NS(type="tool_use", id=f"toolu_{next(_ids)}", name=name, input=inp)


def _text_of(block) -> str:
    if isinstance(block, dict):
        return str(block.get("content", ""))
    return str(getattr(block, "content", ""))


class _Agent:
    def create(self, messages, tools, **kw):
        if kw.get("tool_choice", {}).get("type") == "none":  # handoff note
            return NS(content=[NS(type="text", text="Handoff: nothing to report.")], stop_reason="end_turn",
                      usage=_usage(messages), context_management=None, stop_details=None)
        names = {t.get("name") for t in tools}
        query_tool = next((n for n in QUERIES if n in names), None)
        last = messages[-1]["content"]
        result = _text_of(last[-1]) if isinstance(last, list) else str(last)
        prev_assistant = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
        prev_tool = None
        if prev_assistant:
            prev_tool = next((b.name for b in prev_assistant["content"] if getattr(b, "type", "") == "tool_use"), None)

        if "inbox is empty" in result:
            content = [NS(type="text", text="Inbox done.")]
            stop = "end_turn"
        elif "QUESTION" in result and query_tool:
            inp = {"command": "view", "path": "/memories"} if query_tool == "memory" else {"query": QUERIES[query_tool]}
            content, stop = [_tool_use(query_tool, inp)], "tool_use"
        elif "QUESTION" in result or (prev_tool == query_tool and query_tool):
            qid = _latest_qid(messages)
            guess = re.findall(r'"(?:b\.name|object)": "([^"]+)"', result)[:1] or ["unknown"]
            content, stop = [_tool_use("submit_answer", {"question_id": qid, "answer": guess})], "tool_use"
        else:
            content, stop = [_tool_use("read_inbox", {})], "tool_use"
        return NS(content=content, stop_reason=stop, usage=_usage(messages), context_management=None, stop_details=None)


def _latest_qid(messages) -> str:
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                hit = re.search(r"QUESTION (\S+):", _text_of(b))
                if hit:
                    return hit.group(1)
    return "?"


class _Plain:
    def create(self, messages, **kw):
        prompt = messages[0]["content"]
        text = "yes" if "Answer yes or no" in prompt else json.dumps({"assert": [], "retract": []})
        return NS(content=[NS(type="text", text=text)], stop_reason="end_turn", usage=_usage(messages))


class FakeClient:
    def __init__(self) -> None:
        self.messages = _Plain()
        self.beta = NS(messages=_Agent())


def _oai_usage(items) -> NS:
    chars = len(json.dumps(items, default=str))
    return NS(input_tokens=chars // 4, input_tokens_details=NS(cached_tokens=chars // 5, cache_write_tokens=0),
              output_tokens=40, output_tokens_details=NS(reasoning_tokens=20))


def _fcall(name: str, args: dict) -> NS:
    return NS(type="function_call", call_id=f"call_{next(_ids)}", name=name, arguments=json.dumps(args))


class _Responses:
    """Scripted Responses API: same behavior as the Anthropic fake agent."""

    def create(self, input, tools=None, **kw):
        if not tools:  # extractor or judge
            text = "yes" if "Answer yes or no" in str(input) else json.dumps({"assert": [], "retract": []})
            return NS(output=[NS(type="message", content=[NS(type="output_text", text=text)])],
                      output_text=text, usage=_oai_usage(input), status="completed")
        if kw.get("tool_choice") == "none":  # handoff note
            text = "Handoff: nothing to report."
            return NS(output=[NS(type="message", content=[NS(type="output_text", text=text)])],
                      output_text=text, usage=_oai_usage(input), status="completed")
        names = {t["name"] for t in tools}
        query_tool = next((n for n in QUERIES if n in names), None)
        last = input[-1]
        result = str(last.get("output", last.get("content", "")))
        prev_call = next((it.get("name") for it in reversed(input) if it.get("type") == "function_call"), None)
        if "inbox is empty" in result:
            out = [NS(type="message", content=[NS(type="output_text", text="Inbox done.")])]
        elif "QUESTION" in result and query_tool:
            args = {"command": "view", "path": "/memories"} if query_tool == "memory" else {"query": QUERIES[query_tool]}
            out = [_fcall(query_tool, args)]
        elif "QUESTION" in result or (query_tool and prev_call == query_tool):
            qid = next((m.group(1) for it in reversed(input)
                        if (m := re.search(r"QUESTION (\S+):", str(it.get("output", ""))))), "?")
            guess = re.findall(r'"(?:b\.name|object)": "([^"]+)"', result)[:1] or ["unknown"]
            out = [_fcall("submit_answer", {"question_id": qid, "answer": guess})]
        else:
            out = [_fcall("read_inbox", {})]
        return NS(output=out, usage=_oai_usage(input), status="completed", output_text="")


class FakeOpenAI:
    def __init__(self) -> None:
        self.responses = _Responses()
