import json
from argparse import Namespace
from pathlib import Path

import pytest

from amb import world
from amb.fake import FakeOpenAI
from amb.grade import grade_set
from amb.openai_agent import CLEARED, OpenAIEpisode
from amb.agent import RunConfig
from amb.run import run_one, world_items


@pytest.mark.parametrize("condition", ["full", "compact", "memtool", "sqlite"])
def test_fake_openai_episode_runs_to_the_end(tmp_path: Path, condition: str):
    w = world.generate(1, 50)
    args = Namespace(model="gpt-6.1-sol", effort="medium", extract="oracle", extract_model=None)
    log = tmp_path / condition / "L50-w1-r1.jsonl"
    s = run_one(args, FakeOpenAI(), "ops", condition, world_items(w),
                lambda item, given: grade_set(given, item.answer, w.aliases), "ops-L50-w1", log, False)
    assert s["finished"] and s["questions_answered"] == 25
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    calls = [r for r in recs if r["type"] == "call"]
    assert all(r["usage"]["usd"] > 0 for r in calls)
    cleared = [e for r in calls for e in r["applied_edits"]]
    assert bool(cleared) == (condition in ("memtool", "sqlite")), "only memory setups clear old results"


def test_clearing_keeps_recent_results(tmp_path: Path):
    ep = OpenAIEpisode(None, RunConfig(condition="graph", model="gpt-6.1-sol"), [], None, tmp_path / "x.jsonl")
    items = [{"type": "function_call_output", "call_id": str(i), "output": "x" * 20_000} for i in range(10)]
    assert ep._clear_old_results(items, 10_000) is None, "below the trigger nothing changes"
    edit = ep._clear_old_results(items, 30_000)
    assert edit["cleared_tool_uses"] == 5
    assert [it["output"] == CLEARED for it in items] == [True] * 5 + [False] * 5
