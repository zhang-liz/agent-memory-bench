"""Graph database against graph database, on the queries an agent's memory issues.

No LLM involved. The same generated ops graph and the same Cypher run on
FalkorDB, Neo4j Community and Memgraph, one engine at a time, each in Docker
on the same machine. Measured:

- load time and container memory after load,
- lookup, 3-hop page and 1-4 hop blast-radius latency (client side, p50/p99),
- page latency while writer threads hand services to new owners,
- one memory per user: many small per-user graphs, the way agent memory is
  usually partitioned. FalkorDB keeps each user in its own graph; Neo4j and
  Memgraph Community allow one database, so users share it with a user_id
  property and an index, which is what you would do there.

Every engine gets an index on the lookup key and a warm-up pass before timing.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

from .scale import Q_BLAST, Q_LOOKUP, Q_PAGE, SQL_PAGE, pct, sql_blast_bfs

ENGINES = {
    "falkordb": {"image": "falkordb/falkordb:6.0.1", "container": "amb-falkordb", "ports": ["6379:6379"], "env": {}},
    "neo4j": {"image": "neo4j:5-community", "container": "amb-neo4j", "ports": ["7687:7687"],
              # Room to hold the 1M-edge graph in memory so Neo4j is not disk-bound, but a small
              # starting heap, so container memory shows what Neo4j uses rather than what it reserved.
              "env": {"NEO4J_AUTH": "none", "NEO4J_server_memory_heap_initial__size": "256m",
                      "NEO4J_server_memory_heap_max__size": "2g", "NEO4J_server_memory_pagecache_size": "1g"}},
    "memgraph": {"image": "memgraph/memgraph:latest", "container": "amb-memgraph", "ports": ["7688:7687"], "env": {}},
}


# ---------------------------------------------------------------- data

def make_data(n_edges: int, rng: random.Random) -> dict:
    """Same shape as scale.build_graph, as plain rows every engine can load."""
    services = max(100, int(n_edges / 3.2))
    teams = max(10, services // 40)
    people = teams * 6
    nodes = ([{"name": f"svc-{i}", "type": "service"} for i in range(services)]
             + [{"name": f"team-{i}", "type": "team"} for i in range(teams)]
             + [{"name": f"person-{i}", "type": "person"} for i in range(people)])
    rels = {"OWNED_BY": [{"a": f"svc-{i}", "b": f"team-{rng.randrange(teams)}"} for i in range(services)]}
    deps = []
    for i in range(1, services):
        for _ in range(rng.choice([1, 2, 2, 3])):
            deps.append({"a": f"svc-{i}", "b": f"svc-{int(i * rng.random() ** 2)}"})
    rels["DEPENDS_ON"] = deps
    rels["MEMBER_OF"] = [{"a": f"person-{i}", "b": f"team-{i % teams}"} for i in range(people)]
    rels["ON_CALL"] = [{"a": f"team-{i}", "b": f"person-{i}"} for i in range(teams)]
    return {"services": services, "teams": teams, "nodes": nodes, "rels": rels,
            "edges": sum(len(v) for v in rels.values())}


# ---------------------------------------------------------------- engines

class Falkor:
    name = "falkordb"

    def __init__(self, graph: str = "amb_graphs", db=None):
        from falkordb import FalkorDB
        self.db = db or FalkorDB()
        self.graph = graph
        self.g = self.db.select_graph(graph)

    def new_client(self):
        return Falkor(self.graph)

    def ping(self):
        self.db.connection.ping()

    def select(self, graph: str):
        # Same connection pool, other graph key: how one app serves many users.
        return Falkor(graph, self.db)

    def write(self, q: str, params: dict | None = None):
        self.g.query(q, params or {})

    def read(self, q: str, params: dict | None = None):
        return self.g.ro_query(q, params or {}, timeout=60000).result_set

    def create_index(self, tenant: bool = False):
        self.write("CREATE INDEX FOR (e:Entity) ON (e.name)")

    def drop(self):
        try:
            self.g.delete()
        except Exception:
            pass


class Bolt:
    """Neo4j and Memgraph both speak Bolt and Cypher, so one adapter serves both."""

    def __init__(self, name: str, uri: str):
        from neo4j import GraphDatabase
        self.name = name
        self.uri = uri
        self.driver = GraphDatabase.driver(uri, auth=None, notifications_min_severity="OFF")

    def new_client(self):
        return Bolt(self.name, self.uri)

    def ping(self):
        self.read("RETURN 1")

    def _run(self, q: str, params: dict | None, retries: int = 20):
        for attempt in range(retries):
            try:
                with self.driver.session() as s:
                    return s.run(q, params or {}).values()
            except Exception as e:  # lock conflicts and deadlocks under concurrent writers
                if attempt == retries - 1 or not any(k in str(e) for k in ("Deadlock", "conflict", "Serialization", "TransientError", "Cannot resolve")):
                    raise
                time.sleep(0.005 * (attempt + 1))

    def write(self, q: str, params: dict | None = None):
        self._run(q, params)

    def read(self, q: str, params: dict | None = None):
        return self._run(q, params)

    def create_index(self, tenant: bool = False):
        if self.name == "memgraph":
            self.write("CREATE INDEX ON :Entity(name)")
            if tenant:
                self.write("CREATE INDEX ON :Entity(user_id)")
                try:
                    self.write("CREATE INDEX ON :Entity(user_id, name)")
                except Exception:
                    pass  # older Memgraph has no composite index; the user_id index still applies
        else:
            key = "(e.user_id, e.name)" if tenant else "(e.name)"
            self.write(f"CREATE INDEX entity_key IF NOT EXISTS FOR (e:Entity) ON {key}")
            self.write("CALL db.awaitIndexes(300)")

    def drop(self):
        if self.name == "memgraph":
            self.write("MATCH (n) DETACH DELETE n")
            for q in ("DROP INDEX ON :Entity(name)", "DROP INDEX ON :Entity(user_id)", "DROP INDEX ON :Entity(user_id, name)"):
                try:
                    self.write(q)
                except Exception:
                    pass
        else:
            self.write("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
            try:
                self.write("DROP INDEX entity_key IF EXISTS")
            except Exception:
                pass


class Lite:
    """SQLite as a facts table, the strongest non-graph store in the token benchmark.

    Same data and the same three questions, in the SQL an engineer would write:
    joins for the page, the tuned hop-by-hop walk for blast radius. WAL mode, so
    readers never wait for the writer, which is SQLite's best setup for this.
    In-process, so there is no network hop: the reads start with an advantage.
    """

    name = "sqlite"
    path = Path("results/graphs/sqlite.db")

    def __init__(self):
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=60, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")

    def new_client(self):
        return Lite()

    def ping(self):
        self.db.execute("SELECT 1")

    def create_index(self, tenant: bool = False):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS facts (subject TEXT, predicate TEXT, object TEXT, valid_to TEXT);
            CREATE INDEX IF NOT EXISTS f_s ON facts(subject, predicate);
            CREATE INDEX IF NOT EXISTS f_o ON facts(object, predicate);
        """)

    def bulk_load(self, data: dict) -> None:
        rows = [(r["a"], rel.lower(), r["b"], None) for rel, rs in data["rels"].items() for r in rs]
        self.db.execute("BEGIN")
        self.db.executemany("INSERT INTO facts VALUES (?,?,?,?)", rows)
        self.db.execute("COMMIT")

    def write(self, q: str, params: dict | None = None):
        assert q == HANDOFF, "SQLite runs only the handoff write"
        # Same change as the Cypher: retract the current owner, assert the new one, in one transaction.
        self.db.execute("BEGIN IMMEDIATE")
        self.db.execute("UPDATE facts SET valid_to = '2026-10-01' WHERE subject = ? AND predicate = 'owned_by' "
                        "AND valid_to IS NULL", (params["a"],))
        self.db.execute("INSERT INTO facts VALUES (?, 'owned_by', ?, NULL)", (params["a"], params["b"]))
        self.db.execute("COMMIT")

    def read(self, q: str, params: dict | None = None):
        s = params["s"]
        if q == Q_PAGE:
            return self.db.execute(SQL_PAGE, (s,)).fetchall()
        if q == Q_LOOKUP:
            return self.db.execute("SELECT object FROM facts WHERE subject = ? AND predicate = 'owned_by' "
                                   "AND valid_to IS NULL", (s,)).fetchall()
        if q == Q_BLAST:
            return sql_blast_bfs(self.db, s)
        raise ValueError(q)

    def drop(self):
        self.db.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.path) + suffix).unlink(missing_ok=True)
        self.__init__()


def connect(engine: str):
    if engine == "sqlite":
        return Lite()
    if engine == "falkordb":
        return Falkor()
    return Bolt(engine, "bolt://localhost:7687" if engine == "neo4j" else "bolt://localhost:7688")


# ---------------------------------------------------------------- docker

def start(engine: str) -> None:
    if engine == "sqlite":
        return
    e = ENGINES[engine]
    subprocess.run(["docker", "rm", "-f", e["container"]], capture_output=True)
    cmd = ["docker", "run", "-d", "--name", e["container"]]
    for p in e["ports"]:
        cmd += ["-p", p]
    for k, v in e["env"].items():
        cmd += ["-e", f"{k}={v}"]
    subprocess.run(cmd + [e["image"]], check=True, capture_output=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            connect(engine).ping()
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"{engine} did not start")


def stop(engine: str) -> None:
    if engine == "sqlite":
        return
    subprocess.run(["docker", "rm", "-f", ENGINES[engine]["container"]], capture_output=True)


def memory_mb(engine: str) -> float:
    if engine == "sqlite":
        return math.nan  # in-process, so there is no separate server to measure
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", ENGINES[engine]["container"]],
                         capture_output=True, text=True).stdout.split("/")[0].strip()
    num = float("".join(c for c in out if c.isdigit() or c == "."))
    return num * {"GiB": 1024, "MiB": 1, "KiB": 1 / 1024}[next(u for u in ("GiB", "MiB", "KiB") if u in out)]


# ---------------------------------------------------------------- load and time

def load(db, data: dict, user: str | None = None, batch: int = 5000) -> None:
    """Load nodes then each relationship type in batches. With user set, every node carries user_id."""
    if hasattr(db, "bulk_load"):
        db.bulk_load(data)
        return
    tag = ", user_id: $u" if user else ""
    match = "{name: r.a%s}), (b:Entity {name: r.b%s})" % ((", user_id: $u",) * 2 if user else ("", ""))
    nodes = data["nodes"]
    for i in range(0, len(nodes), batch):
        db.write(f"UNWIND $rows AS r CREATE (:Entity {{name: r.name, type: r.type{tag}}})", {"rows": nodes[i:i + batch], "u": user})
    for rel, rows in data["rels"].items():
        q = (f"UNWIND $rows AS r MATCH (a:Entity {match} "
             f"CREATE (a)-[:{rel} {{valid_from: '2026-09-01', valid_to: null}}]->(b)")
        for i in range(0, len(rows), batch):
            db.write(q, {"rows": rows[i:i + batch], "u": user})


def time_reads(db, q: str, params: list[dict]) -> dict:
    for p in params[:20]:  # warm-up
        db.read(q, p)
    out = []
    for p in params:
        t0 = time.perf_counter()
        db.read(q, p)
        out.append((time.perf_counter() - t0) * 1000)
    return pct(out)


HANDOFF = ("MATCH (s:Entity {name: $a})-[r:OWNED_BY]->() WHERE r.valid_to IS NULL SET r.valid_to = '2026-10-01' "
           "WITH s MATCH (t:Entity {name: $b}) CREATE (s)-[:OWNED_BY {valid_from: '2026-10-01', valid_to: null}]->(t)")


def writer(db, stop_ev: threading.Event, data: dict, seed: int, counter: list[int]) -> None:
    rng = random.Random(seed)
    while not stop_ev.is_set():
        db.write(HANDOFF, {"a": f"svc-{rng.randrange(data['services'])}", "b": f"team-{rng.randrange(data['teams'])}"})
        counter[0] += 1


def single_graph(engine: str, n_edges: int, n_queries: int, writers: list[int]) -> dict:
    rng = random.Random(0)
    data = make_data(n_edges, rng)
    db = connect(engine)
    db.drop()
    db.create_index()
    mem0 = memory_mb(engine)
    t0 = time.perf_counter()
    load(db, data)
    row = {"engine": engine, "edges": data["edges"], "load_s": time.perf_counter() - t0}
    time.sleep(2)
    row["memory_mb"] = memory_mb(engine)
    row["memory_idle_mb"] = mem0
    sample = [{"s": f"svc-{rng.randrange(data['services'])}"} for _ in range(n_queries)]
    core = [{"s": f"svc-{int(data['services'] * rng.random() ** 3)}"} for _ in range(max(50, n_queries // 5))]
    row["lookup"] = time_reads(db, Q_LOOKUP, sample)
    row["page"] = time_reads(db, Q_PAGE, sample)
    row["blast"] = time_reads(db, Q_BLAST, core)
    print(f"  {engine}: {data['edges']} edges loaded in {row['load_s']:.1f}s, {row['memory_mb']:.0f} MB | "
          f"page p50/p99 {row['page']['p50']:.2f}/{row['page']['p99']:.2f} ms | "
          f"blast p50/p99 {row['blast']['p50']:.1f}/{row['blast']['p99']:.1f} ms", flush=True)
    row["concurrency"] = []
    for w in writers:
        stop_ev, counter = threading.Event(), [0]
        threads = [threading.Thread(target=writer, args=(db.new_client(), stop_ev, data, i, counter)) for i in range(w)]
        for t in threads:
            t.start()
        time.sleep(0.5 if w else 0)
        t0, before = time.perf_counter(), counter[0]
        res = time_reads(db, Q_PAGE, sample)
        elapsed, writes = time.perf_counter() - t0, counter[0] - before
        stop_ev.set()
        for t in threads:
            t.join()
        row["concurrency"].append({"writers": w, "page": res, "writes_per_s": writes / elapsed})
        print(f"    {w:>2} writers: page p50/p99 {res['p50']:.2f}/{res['p99']:.2f} ms, {writes / elapsed:.0f} writes/s", flush=True)
    db.drop()
    return row


# Per-user page query: in a shared database every pattern must also pin the user.
Q_PAGE_USER = (
    "MATCH (s:Entity {user_id: $u, name: $s})-[d:DEPENDS_ON]->(dep)-[o:OWNED_BY]->(t)-[c:ON_CALL]->(p) "
    "WHERE d.valid_to IS NULL AND o.valid_to IS NULL AND c.valid_to IS NULL RETURN DISTINCT p.name"
)


def per_user(engine: str, users: int, edges_per_user: int, n_queries: int) -> dict:
    rng = random.Random(1)
    data = make_data(edges_per_user, rng)
    db = connect(engine)
    names = [f"user_{i}" for i in range(users)]
    if engine == "falkordb":
        for u in names:
            db.select(u).drop()
    else:
        db.drop()
        db.create_index(tenant=True)
    mem0 = memory_mb(engine)
    t0 = time.perf_counter()
    for u in names:
        if engine == "falkordb":
            g = db.select(u)
            g.create_index()
            load(g, data)
        else:
            load(db, data, user=u)
    row = {"engine": engine, "users": users, "edges_per_user": data["edges"], "load_s": time.perf_counter() - t0}
    time.sleep(2)
    row["memory_mb"], row["memory_idle_mb"] = memory_mb(engine), mem0
    picks = [(rng.choice(names), f"svc-{rng.randrange(data['services'])}") for _ in range(n_queries)]
    clients = {}

    def page(u, s):
        if engine == "falkordb":
            # One open client per user graph, as an app would keep; no new connection per query.
            if u not in clients:
                clients[u] = db.select(u)
            return clients[u].read(Q_PAGE, {"s": s})
        return db.read(Q_PAGE_USER, {"u": u, "s": s})

    for u, s in picks[:20]:
        page(u, s)
    out = []
    for u, s in picks:
        t0 = time.perf_counter()
        page(u, s)
        out.append((time.perf_counter() - t0) * 1000)
    row["page"] = pct(out)
    # Forgetting one user: the request a privacy rule or a "delete my data" button sends.
    victims = rng.sample(names, 10)
    out = []
    for u in victims:
        t0 = time.perf_counter()
        if engine == "falkordb":
            db.select(u).drop()
        else:
            db.write("MATCH (n:Entity {user_id: $u}) DETACH DELETE n", {"u": u})
        out.append((time.perf_counter() - t0) * 1000)
    row["delete_user_ms"] = pct(out)
    print(f"  {engine}: {users} users x {data['edges']} edges in {row['load_s']:.1f}s, {row['memory_mb']:.0f} MB | "
          f"page p50/p99 {row['page']['p50']:.2f}/{row['page']['p99']:.2f} ms | "
          f"delete a user p50 {row['delete_user_ms']['p50']:.1f} ms", flush=True)
    if engine == "falkordb":
        for u in names:
            db.select(u).drop()
    else:
        db.drop()
    return row


def main(args) -> None:
    out_dir = Path(args.out) / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "graphs.json"
    results = json.loads(path.read_text()) if path.exists() else {"single": [], "per_user": []}
    writers = [int(x) for x in args.writers.split(",")]
    # Only one engine runs at a time, so none competes with another for CPU or memory.
    subprocess.run(["docker", "compose", "stop"], capture_output=True)
    for engine in args.engines.split(","):
        print(f"== {engine}", flush=True)
        start(engine)
        try:
            if not args.skip_single:
                results["single"] = [r for r in results["single"] if r["engine"] != engine]
                results["single"].append(single_graph(engine, args.edges, args.queries, writers))
            # Per-user graphs are a graph-database question; SQLite runs the single-store test only.
            if args.users and engine != "sqlite":
                results["per_user"] = [r for r in results["per_user"] if r["engine"] != engine]
                results["per_user"].append(per_user(engine, args.users, args.edges_per_user, args.queries))
            path.write_text(json.dumps(results, indent=2))
        finally:
            stop(engine)
    subprocess.run(["docker", "compose", "up", "-d"], capture_output=True)
    print(f"wrote {path}")
