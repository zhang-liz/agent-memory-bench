import random

from amb.scale import SQL_BLAST, build_sqlite, sql_blast_bfs


def test_tuned_blast_matches_recursive_cte(tmp_path):
    rng = random.Random(3)
    deps = [{"a": f"svc-{i}", "b": f"svc-{int(i * rng.random() ** 2)}"}
            for i in range(1, 400) for _ in range(rng.choice([1, 2, 3]))]
    sh = {"deps": deps, "owned": [], "teams": 0}
    db = build_sqlite(tmp_path / "s.sqlite", sh)
    for i in range(0, 60, 3):
        name = f"svc-{i}"
        assert sql_blast_bfs(db, name) == db.execute(SQL_BLAST, (name,)).fetchone()[0]
