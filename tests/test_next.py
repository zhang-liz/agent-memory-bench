import json

from amb.run import main
from amb.world import generate, followup_questions


def test_followup_questions_leave_session_a_unchanged():
    w = generate(1, 200)
    qs = followup_questions(w)
    assert len(qs) == 20
    assert not {q.text for q in qs} & {q.text for q in w.questions}
    assert [q.text for q in generate(1, 200).questions] == [q.text for q in w.questions]


def test_next_session_runs_from_what_session_a_left(tmp_path):
    out = str(tmp_path / "next-test")
    base = ["--fake", "--extract", "oracle", "--lengths", "50", "--worlds", "1", "--out", out]
    main(["ops", *base, "--conditions", "compact,memtool", "--handoff"])
    a_log = [json.loads(x) for x in open(f"{out}/ops/compact/L50-w1-r1.jsonl")]
    assert any(r["type"] == "handoff" and r["text"] for r in a_log)

    main(["next", *base, "--source", out, "--conditions", "none,handoff,memtool,sqlite"])
    for c in ("none", "handoff", "memtool", "sqlite"):
        recs = [json.loads(x) for x in open(f"{out}/next/{c}/L50-w1-r1.jsonl")]
        assert recs[-1]["type"] == "summary" and recs[-1]["finished"]
        assert sum(r["type"] == "answer" for r in recs) == recs[-1]["questions_total"] > 0
    header = json.loads(open(f"{out}/next/sqlite/L50-w1-r1.jsonl").readline())
    assert header["task"] == "next"
