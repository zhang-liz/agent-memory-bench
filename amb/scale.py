"""Fix 4: does memory recall stay fast as memory grows, and while agents write to it?

No LLM involved. Builds an ops-shaped graph (services, dependencies, owners,
on-call) at several sizes, then times the queries an agent actually issues:
a single lookup, the 3-hop "who do we page" question, and the variable-length
blast-radius question. Then it repeats the page query while writer threads
add and close facts, the way a fleet of agents shares one memory.
"""

from __future__ import annotations

import json
import random
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
from falkordb import FalkorDB

Q_LOOKUP = "MATCH (s:Entity {name: $s})-[r:OWNED_BY]->(t) WHERE r.valid_to IS NULL RETURN t.name"
Q_PAGE = (
    "MATCH (s:Entity {name: $s})-[d:DEPENDS_ON]->(dep)-[o:OWNED_BY]->(t)-[c:ON_CALL]->(p) "
    "WHERE d.valid_to IS NULL AND o.valid_to IS NULL AND c.valid_to IS NULL RETURN DISTINCT p.name"
)
Q_BLAST = (
    "MATCH (up:Entity)-[:DEPENDS_ON*1..4]->(s:Entity {name: $s}) RETURN count(DISTINCT up)"
)


def shape(n_edges: int) -> dict:
    """Pick entity counts so the total edge count lands near n_edges."""
    services = max(100, int(n_edges / 3.2))
    teams = max(10, services // 40)
    people = teams * 6
    return {"services": services, "teams": teams, "people": people}


def build_graph(g, n_edges: int, rng: random.Random) -> dict:
    sh = shape(n_edges)
    S, T, P = sh["services"], sh["teams"], sh["people"]
    g.query("CREATE INDEX FOR (e:Entity) ON (e.name)")
    batch = 5000

    def run(q, rows):
        for i in range(0, len(rows), batch):
            g.query(q, {"rows": rows[i:i + batch]})

    run("UNWIND $rows AS r CREATE (:Entity {name: r.name, type: r.type})",
        [{"name": f"svc-{i}", "type": "service"} for i in range(S)]
        + [{"name": f"team-{i}", "type": "team"} for i in range(T)]
        + [{"name": f"person-{i}", "type": "person"} for i in range(P)])
    owned = [{"a": f"svc-{i}", "b": f"team-{rng.randrange(T)}"} for i in range(S)]
    # Dependencies point to lower-numbered services, so the graph is a DAG with
    # a heavy head of popular shared services, like real infrastructure.
    deps = []
    for i in range(1, S):
        for _ in range(rng.choice([1, 2, 2, 3])):
            j = int(i * rng.random() ** 2)
            deps.append({"a": f"svc-{i}", "b": f"svc-{j}"})
    members = [{"a": f"person-{i}", "b": f"team-{i % T}"} for i in range(P)]
    oncall = [{"a": f"team-{i}", "b": f"person-{i}"} for i in range(T)]
    edge = ("UNWIND $rows AS r MATCH (a:Entity {name: r.a}), (b:Entity {name: r.b}) "
            "CREATE (a)-[:%s {valid_from: '2026-09-01', valid_to: null}]->(b)")
    run(edge % "OWNED_BY", owned)
    run(edge % "DEPENDS_ON", deps)
    run(edge % "MEMBER_OF", members)
    run(edge % "ON_CALL", oncall)
    sh["edges"] = len(owned) + len(deps) + len(members) + len(oncall)
    sh["deps"] = deps
    sh["owned"] = owned
    return sh


def build_sqlite(path: Path, sh: dict) -> sqlite3.Connection:
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path, check_same_thread=False)
    db.executescript("""
        CREATE TABLE facts (subject TEXT, predicate TEXT, object TEXT, valid_to TEXT);
        CREATE INDEX f_s ON facts(subject, predicate); CREATE INDEX f_o ON facts(object, predicate);
    """)
    rows = [(d["a"], "depends_on", d["b"], None) for d in sh["deps"]]
    rows += [(o["a"], "owned_by", o["b"], None) for o in sh["owned"]]
    rows += [(f"team-{i}", "on_call", f"person-{i}", None) for i in range(sh["teams"])]
    db.executemany("INSERT INTO facts VALUES (?,?,?,?)", rows)
    db.commit()
    return db


SQL_PAGE = """
SELECT DISTINCT c.object FROM facts d
JOIN facts o ON o.subject = d.object AND o.predicate = 'owned_by' AND o.valid_to IS NULL
JOIN facts c ON c.subject = o.object AND c.predicate = 'on_call' AND c.valid_to IS NULL
WHERE d.subject = ? AND d.predicate = 'depends_on' AND d.valid_to IS NULL
"""
SQL_BLAST = """
WITH RECURSIVE up(name, depth) AS (
  SELECT subject, 1 FROM facts WHERE object = ? AND predicate = 'depends_on'
  UNION SELECT f.subject, up.depth + 1 FROM facts f JOIN up ON f.object = up.name
  WHERE f.predicate = 'depends_on' AND up.depth < 4
) SELECT count(DISTINCT name) FROM up
"""




def sql_blast_bfs(db, name: str) -> int:
    """Tuned baseline: walk one hop at a time and skip names already seen.

    The recursive CTE dedupes on (name, depth), so a service reachable at
    several depths is expanded several times. This does what an engineer
    would write by hand, with one indexed query per hop.
    """
    seen, frontier = set(), [name]
    for _ in range(4):
        if not frontier:
            break
        rows = db.execute(
            "SELECT DISTINCT subject FROM facts WHERE predicate = 'depends_on' "
            "AND object IN (SELECT value FROM json_each(?))", (json.dumps(frontier),))
        frontier = [r[0] for r in rows if r[0] not in seen]
        seen.update(frontier)
    return len(seen)


def time_fn(fn, names: list[str]) -> dict:
    out = []
    for n in names:
        t0 = time.perf_counter()
        fn(n)
        out.append((time.perf_counter() - t0) * 1000)
    return {"client_ms": pct(out)}


def pct(xs: list[float]) -> dict:
    a = np.array(xs)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "n": len(xs)}


def time_queries(g, q: str, names: list[str]) -> dict:
    server, client = [], []
    for n in names:
        t0 = time.perf_counter()
        r = g.ro_query(q, {"s": n}, timeout=30000)
        client.append((time.perf_counter() - t0) * 1000)
        server.append(r.run_time_ms)
    return {"server_ms": pct(server), "client_ms": pct(client)}


def time_sql(db, q: str, names: list[str]) -> dict:
    out = []
    for n in names:
        t0 = time.perf_counter()
        db.execute(q, (n,)).fetchall()
        out.append((time.perf_counter() - t0) * 1000)
    return {"client_ms": pct(out)}


def writer_loop(stop: threading.Event, graph_name: str, S: int, T: int, seed: int, counter: list[int]) -> None:
    rng = random.Random(seed)
    g = FalkorDB().select_graph(graph_name)
    while not stop.is_set():
        a, b = f"svc-{rng.randrange(S)}", f"team-{rng.randrange(T)}"
        # A handoff: close the current owner fact and open a new one.
        g.query(
            "MATCH (s:Entity {name: $a})-[r:OWNED_BY]->() WHERE r.valid_to IS NULL SET r.valid_to = '2026-10-01' "
            "WITH s MATCH (t:Entity {name: $b}) CREATE (s)-[:OWNED_BY {valid_from: '2026-10-01', valid_to: null}]->(t)",
            {"a": a, "b": b},
        )
        counter[0] += 1


def main(args) -> None:
    out_dir = Path(args.out) / "scale"
    out_dir.mkdir(parents=True, exist_ok=True)
    db = FalkorDB()
    version = db.connection.execute_command("MODULE", "LIST")
    results = {"falkordb_modules": str(version), "sizes": []}
    rng = random.Random(0)
    for n in [int(x) for x in args.sizes.split(",")]:
        name = f"amb_scale_{n}"
        g = db.select_graph(name)
        try:
            g.delete()
        except Exception:
            pass
        g = db.select_graph(name)
        t0 = time.perf_counter()
        sh = build_graph(g, n, rng)
        build_s = time.perf_counter() - t0
        sample = [f"svc-{rng.randrange(sh['services'])}" for _ in range(args.queries)]
        # Blast radius matters most for shared core services, which have the
        # largest upstream trees, so sample those from the popular head.
        core = [f"svc-{int(sh['services'] * rng.random() ** 3)}" for _ in range(max(50, args.queries // 5))]
        row = {"target_edges": n, "edges": sh["edges"], "services": sh["services"], "build_s": build_s,
               "lookup": time_queries(g, Q_LOOKUP, sample),
               "page": time_queries(g, Q_PAGE, sample),
               "blast": time_queries(g, Q_BLAST, core)}
        if args.with_sqlite:
            sdb = build_sqlite(out_dir / f"scale_{n}.sqlite", sh)
            row["sqlite_page"] = time_sql(sdb, SQL_PAGE, sample)
            row["sqlite_blast"] = time_sql(sdb, SQL_BLAST, core)
            row["sqlite_blast_bfs"] = time_fn(lambda s: sql_blast_bfs(sdb, s), core)
        # Readers under concurrent writers.
        row["concurrency"] = []
        for w in [int(x) for x in args.writers.split(",")]:
            stop, counter = threading.Event(), [0]
            threads = [threading.Thread(target=writer_loop, args=(stop, name, sh["services"], sh["teams"], i, counter))
                       for i in range(w)]
            for t in threads:
                t.start()
            time.sleep(0.5 if w else 0)
            t0, before = time.perf_counter(), counter[0]
            res = time_queries(g, Q_PAGE, sample)
            elapsed, writes = time.perf_counter() - t0, counter[0] - before
            stop.set()
            for t in threads:
                t.join()
            row["concurrency"].append({"writers": w, "page": res, "writes_per_s": writes / elapsed})
            print(f"  {n:>9} edges, {w:>2} writers: page p50 {res['client_ms']['p50']:.2f} ms, "
                  f"p99 {res['client_ms']['p99']:.2f} ms, {writes / elapsed:.0f} writes/s during reads")
        print(f"{sh['edges']:>9} edges built in {build_s:.1f}s | page p50 {row['page']['client_ms']['p50']:.2f} ms "
              f"p99 {row['page']['client_ms']['p99']:.2f} ms | blast (core services) p50 {row['blast']['client_ms']['p50']:.2f} ms "
              f"p99 {row['blast']['client_ms']['p99']:.2f} ms")
        results["sizes"].append(row)
        g.delete()
    path = out_dir / "scale.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
