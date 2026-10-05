"""Exact grading for the synthetic world. Names are matched through the alias map."""

from __future__ import annotations

import re


def normalize(name: str, aliases: dict[str, str]) -> str:
    s = name.strip().strip("`'\".,;:").strip()
    s = re.sub(r"\s+", " ", s)
    key = s.lower()
    if key in aliases:
        return aliases[key].lower()
    key = re.sub(r"^(team|the)\s+", "", key)
    key = re.sub(r"\s+(team|folks|service)$", "", key)
    return aliases.get(key, key).lower()


def grade_set(given: list[str], truth: list[str], aliases: dict[str, str]) -> dict:
    g = {normalize(x, aliases) for x in given if x.strip()}
    t = {normalize(x, aliases) for x in truth}
    hit = len(g & t)
    precision = hit / len(g) if g else 0.0
    recall = hit / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hit else 0.0
    return {"correct": g == t, "f1": f1}
