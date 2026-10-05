"""Model prices and usage accounting."""

from __future__ import annotations

# USD per million tokens. Anthropic first-party API checked 2026-09-25; OpenAI
# pricing page checked 2026-10-02. Cache writes bill at 1.25x input on both.
# OpenAI "long" rates apply when a request's input exceeds 272k tokens.
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10},
    "gpt-6.1-sol": {"input": 2.00, "output": 10.00, "cache_read": 0.10,
                    "long": {"input": 4.00, "output": 15.00, "cache_read": 0.20}},
    "gpt-6-astra": {"input": 10.00, "output": 50.00, "cache_read": 1.00,
                    "long": {"input": 20.00, "output": 75.00, "cache_read": 2.00}},
    "gpt-6-luna": {"input": 0.10, "output": 0.50, "cache_read": 0.01,
                   "long": {"input": 0.20, "output": 0.75, "cache_read": 0.02}},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00, "cache_read": 0.20,
                      "long": {"input": 4.00, "output": 18.00, "cache_read": 0.40}},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50, "cache_read": 0.075},
}
LONG_CONTEXT = 272_000


def with_rate_limit_retry(fn, *args, **kwargs):
    """Call fn, waiting out rate limits. SDK retries are too short when several runs share one org quota."""
    import random
    import time

    for attempt in range(12):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # match both SDKs' RateLimitError without importing either here
            if type(e).__name__ != "RateLimitError" or attempt == 11:
                raise
            time.sleep(min(60, 2 ** attempt) + random.random())


def provider(model: str) -> str:
    return "openai" if model.startswith(("gpt-", "o1", "o3", "o4")) else "anthropic"


def supports_effort(model: str) -> bool:
    return not model.startswith("claude-haiku")


def openai_usage_record(usage, model: str) -> dict:
    """Billed tokens for one OpenAI Responses call. input_tokens includes cached and written tokens."""
    det = getattr(usage, "input_tokens_details", None)
    cached = getattr(det, "cached_tokens", 0) or 0
    written = getattr(det, "cache_write_tokens", 0) or 0
    total_in = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
    rec = {"input": total_in - cached - written, "cache_write": written, "cache_read": cached, "output": out,
           "reasoning": reasoning, "compaction_input": 0, "compaction_output": 0, "context_tokens": total_in}
    p = PRICES[model]
    if total_in > LONG_CONTEXT and "long" in p:
        p = p["long"]
    rec["usd"] = (rec["input"] * p["input"] + written * p["input"] * 1.25 + cached * p["cache_read"]
                  + out * p["output"]) / 1e6
    return rec


def usage_record(usage, model: str) -> dict:
    """Flatten an API usage object into billed token counts and dollars.

    With threshold compaction the top-level counts leave out the compaction
    iteration, so sum `usage.iterations` whenever it is present.
    """
    iters = getattr(usage, "iterations", None) or [usage]
    rec = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "compaction_input": 0, "compaction_output": 0}
    for it in iters:
        inp = getattr(it, "input_tokens", 0) or 0
        out = getattr(it, "output_tokens", 0) or 0
        rec["input"] += inp
        rec["output"] += out
        rec["cache_write"] += getattr(it, "cache_creation_input_tokens", 0) or 0
        rec["cache_read"] += getattr(it, "cache_read_input_tokens", 0) or 0
        if getattr(it, "type", None) == "compaction":
            rec["compaction_input"] += inp
            rec["compaction_output"] += out
    # Some responses report cache fields only at the top level.
    if iters is not None and len(iters) > 0 and iters[0] is not usage:
        if rec["cache_read"] == 0:
            rec["cache_read"] = getattr(usage, "cache_read_input_tokens", 0) or 0
        if rec["cache_write"] == 0:
            rec["cache_write"] = getattr(usage, "cache_creation_input_tokens", 0) or 0
    rec["context_tokens"] = rec["input"] + rec["cache_write"] + rec["cache_read"] - rec["compaction_input"]
    rec["usd"] = cost(rec, model)
    return rec


def cost(rec: dict, model: str) -> float:
    p = PRICES[model]
    return (
        rec["input"] * p["input"]
        + rec["cache_write"] * p["input"] * 1.25
        + rec["cache_read"] * p["cache_read"]
        + rec["output"] * p["output"]
    ) / 1e6
