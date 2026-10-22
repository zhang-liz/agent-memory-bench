"""One graph per user: how many small memory graphs fit in one FalkorDB, and do reads stay fast?

No LLM. Starts a fresh FalkorDB container, creates N per-user memory graphs (each the
ops shape from amb/scale.py at about 400 edges, close to the agent memory graph in the
talk), then measures:
  - memory per graph (Redis used_memory delta / N, and container RSS)
  - 3-hop "who do we page" read latency on random tenants, idle and while writer
    threads hand off ownership on other random tenants
  - time to delete one tenant's whole graph ("forget me")

Run: uv run amb tenants --graphs 10000 --runs 3
"""

from __future__ import annotations

import json
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from falkordb import FalkorDB

from amb.scale import Q_PAGE, build_graph

IMAGE, NAME, PORT = "falkordb/falkordb:6.0.1", "amb-tenants", 6391
EDGES_PER_GRAPH = 370


def pct(xs):
    a = np.array(xs)
    return {"p50": round(float(np.percentile(a, 50)), 3), "p99": round(float(np.percentile(a, 99)), 3), "n": len(xs)}


def start():
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", NAME, "-p", f"{PORT}:6379", IMAGE], check=True, capture_output=True)
    for _ in range(60):
        try:
            FalkorDB(port=PORT).connection.ping()
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("FalkorDB did not start")


def rss_mb():
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", NAME],
                         capture_output=True, text=True).stdout.split("/")[0].strip()
    num = float(out[:-3])
    return num * 1024 if out.endswith("GiB") else num


def used_mb(db):
    return db.connection.info("memory")["used_memory"] / 2**20


def reads(db, n_graphs, count, rng):
    lat = []
    for _ in range(count):
        g = db.select_graph(f"user-{rng.randrange(n_graphs)}")
        t0 = time.perf_counter()
        g.ro_query(Q_PAGE, {"s": f"svc-{rng.randrange(100)}"})
        lat.append((time.perf_counter() - t0) * 1000)
    return pct(lat)


def writer(stop, n_graphs, seed, counter):
    rng, db = random.Random(seed), FalkorDB(port=PORT)
    while not stop.is_set():
        g = db.select_graph(f"user-{rng.randrange(n_graphs)}")
        g.query("MATCH (s:Entity {name: $a})-[r:OWNED_BY]->() WHERE r.valid_to IS NULL SET r.valid_to = '2026-10-01' "
                "WITH s MATCH (t:Entity {name: $b}) CREATE (s)-[:OWNED_BY {valid_from: '2026-10-01', valid_to: null}]->(t)",
                {"a": f"svc-{rng.randrange(100)}", "b": f"team-{rng.randrange(10)}"})
        counter[0] += 1


def one_run(n_graphs, writers, seed):
    start()
    db = FalkorDB(port=PORT)
    time.sleep(2)
    base_used, base_rss = used_mb(db), rss_mb()
    t0 = time.perf_counter()

    def build(i):
        build_graph(FalkorDB(port=PORT).select_graph(f"user-{i}"), EDGES_PER_GRAPH, random.Random(seed * 1_000_003 + i))

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(build, range(n_graphs)))
    build_s = time.perf_counter() - t0
    time.sleep(3)
    keys = db.connection.dbsize()
    used, rss = used_mb(db), rss_mb()
    rng = random.Random(seed)
    sample = db.select_graph("user-0").ro_query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
    idle = reads(db, n_graphs, 2000, rng)
    stop, counter = threading.Event(), [0]
    ths = [threading.Thread(target=writer, args=(stop, n_graphs, seed + k, counter)) for k in range(writers)]
    for t in ths:
        t.start()
    time.sleep(1)
    w0, t1 = counter[0], time.perf_counter()
    busy = reads(db, n_graphs, 2000, rng)
    writes_per_s = (counter[0] - w0) / (time.perf_counter() - t1)
    stop.set()
    for t in ths:
        t.join()
    dels = []
    for i in rng.sample(range(n_graphs), 200):
        t0 = time.perf_counter()
        db.select_graph(f"user-{i}").delete()
        dels.append((time.perf_counter() - t0) * 1000)
    row = {"graphs": n_graphs, "keys": keys, "edges_per_graph": sample, "build_s": round(build_s, 1),
           "used_mb": round(used - base_used, 1), "rss_mb": round(rss - base_rss, 1),
           "kb_per_graph_used": round((used - base_used) * 1024 / n_graphs, 1),
           "kb_per_graph_rss": round((rss - base_rss) * 1024 / n_graphs, 1),
           "read_idle_ms": idle, "read_with_writers_ms": busy, "writers": writers,
           "writes_per_s": round(writes_per_s), "delete_graph_ms": pct(dels)}
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    return row


def main(args) -> None:
    rows = []
    for r in range(args.runs):
        row = one_run(args.graphs, args.writers, seed=r + 1)
        print(json.dumps(row), flush=True)
        rows.append(row)
    out = Path(args.out) / "tenants.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"image": IMAGE, "runs": rows}, indent=2))
    print(f"wrote {out}")
