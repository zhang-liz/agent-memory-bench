from datetime import datetime
from pathlib import Path

import pytest

from amb import world
from amb.memory_tool import MemoryFiles
from amb.run import OracleExtractor, world_items
from amb.stores import FactLog, GraphSink, SqliteSink


def _load(sinks):
    log = FactLog(sinks)
    ex = OracleExtractor()
    w = world.generate(4, 100)
    for it in world_items(w):
        if it.kind == "event":
            ex.ingest(log, it.step, it.time, it.text, it)
    return w, log


def test_retractions_close_facts(tmp_path: Path):
    sq = SqliteSink(tmp_path / "m.sqlite")
    w, log = _load([sq])
    open_oncall = [f for f in log.open_facts() if f.predicate == "on_call"]
    teams = [f.subject for f in open_oncall]
    assert len(teams) == len(set(teams)), "a team should have one current on-call person"
    out, err, _ = sq.query("SELECT count(*) AS n FROM facts WHERE valid_to IS NOT NULL")
    assert not err and '"n": 0' not in out


def test_sqlite_is_read_only(tmp_path: Path):
    sq = SqliteSink(tmp_path / "m.sqlite")
    _load([sq])
    out, err, _ = sq.query("DELETE FROM facts")
    assert err and "readonly" in out.replace(" ", "").lower()


def test_graph_matches_sqlite(tmp_path: Path):
    try:
        g = GraphSink("amb_test_graph")
    except Exception:
        pytest.skip("FalkorDB is not running")
    sq = SqliteSink(tmp_path / "m.sqlite")
    _load([g, sq])
    gout, gerr, _ = g.query("MATCH ()-[r:DEPENDS_ON]->() WHERE r.valid_to IS NULL RETURN count(r) AS n")
    sout, serr, _ = sq.query("SELECT count(*) AS n FROM current_facts WHERE predicate='depends_on'")
    assert not gerr and not serr and gout == sout
    out, err, _ = g.query("CREATE (:Entity {name: 'x'})")
    assert err, "agent queries must be read-only"
    g.g.delete()


def test_memory_tool_blocks_traversal(tmp_path: Path):
    m = MemoryFiles(tmp_path / "mem")
    assert m.run({"command": "create", "path": "/memories/a.md", "file_text": "hi"}) == ("File created successfully at: /memories/a.md", False)
    out, err = m.run({"command": "view", "path": "/memories/../../etc/passwd"})
    assert err
    out, err = m.run({"command": "view", "path": "/memories/a.md"})
    assert not err and "     1\thi" in out
