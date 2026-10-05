import json
from argparse import Namespace
from pathlib import Path

import pytest

from amb import world
from amb.fake import FakeClient
from amb.grade import grade_set
from amb.run import run_one, world_items


@pytest.mark.parametrize("condition", ["full", "compact", "memtool", "sqlite"])
def test_fake_episode_runs_to_the_end(tmp_path: Path, condition: str):
    w = world.generate(1, 50)
    args = Namespace(model="claude-opus-5-5", effort="medium", extract="oracle", extract_model=None)
    log = tmp_path / condition / "L50-w1-r1.jsonl"
    s = run_one(args, FakeClient(), "ops", condition, world_items(w),
                lambda item, given: grade_set(given, item.answer, w.aliases), "ops-L50-w1", log, False)
    assert s["finished"] and s["questions_answered"] == s["questions_total"] == 25
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert recs[0]["type"] == "header" and recs[-1]["type"] == "summary"
    assert sum(r["type"] == "answer" for r in recs) == 25
