"""Memory stores. Every structured condition reads the same extracted facts.

`FactLog` is the canonical record of what the extractor said. The SQLite, graph
and vector stores are mirrors of it, so the only thing that differs between
those conditions is how the agent queries memory, not what memory holds.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROW_LIMIT = 50
RESULT_CHAR_LIMIT = 6000


@dataclass
class Fact:
    fid: int
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    valid_from_step: int
    valid_from: str
    valid_to_step: int | None = None
    valid_to: str | None = None

    def sentence(self) -> str:
        span = f"true from {self.valid_from}"
        if self.valid_to:
            span += f" until {self.valid_to} (NO LONGER TRUE)"
        return f"{self.subject} ({self.subject_type}) {self.predicate} {self.object} ({self.object_type}); {span}"


def norm_predicate(p: str) -> str:
    p = re.sub(r"[^a-zA-Z0-9]+", "_", p.strip()).strip("_").lower()
    return p or "related_to"


class FactLog:
    """Applies extractor output and fans each change out to the attached sinks."""

    def __init__(self, sinks: list["Sink"]) -> None:
        self.facts: list[Fact] = []
        self.entities: dict[str, str] = {}  # lowercase name -> canonical name
        self.entity_types: dict[str, str] = {}
        self.sinks = sinks

    def open_facts(self) -> list[Fact]:
        return [f for f in self.facts if f.valid_to is None]

    def known_entities_in(self, text: str) -> list[str]:
        """Entities the extractor should reuse, found by name or short name in text."""
        low = text.lower()
        hits = []
        for key, name in self.entities.items():
            short = re.sub(r"^team ", "", key)
            short = short.rsplit("-", 1)[0] if "-" in short else short
            if key in low or (len(short) >= 4 and short in low):
                hits.append(name)
        return sorted(set(hits))

    def facts_about(self, names: list[str], limit: int = 60) -> list[Fact]:
        wanted = {n.lower() for n in names}
        out = [f for f in self.open_facts() if f.subject.lower() in wanted or f.object.lower() in wanted]
        return out[:limit]

    def apply(self, ops: dict, step: int, when: datetime, source_text: str) -> None:
        stamp = when.strftime("%Y-%m-%d %H:%M")
        for r in ops.get("retract", []):
            pred = norm_predicate(r["predicate"])
            for f in self.open_facts():
                if (f.subject.lower(), f.predicate, f.object.lower()) == (r["subject"].lower(), pred, r["object"].lower()):
                    f.valid_to_step, f.valid_to = step, stamp
                    for s in self.sinks:
                        s.close_fact(f)
        for a in ops.get("assert", []):
            subj = self._canon(a["subject"], a.get("subject_type", "thing"))
            obj = self._canon(a["object"], a.get("object_type", "thing"))
            pred = norm_predicate(a["predicate"])
            if any((f.subject, f.predicate, f.object) == (subj, pred, obj) for f in self.open_facts()):
                continue
            f = Fact(len(self.facts) + 1, subj, self.entity_types[subj.lower()], pred, obj,
                     self.entity_types[obj.lower()], step, stamp)
            self.facts.append(f)
            for s in self.sinks:
                s.add_fact(f)
        for s in self.sinks:
            s.add_message(step, stamp, source_text)

    def _canon(self, name: str, etype: str) -> str:
        name = name.strip()
        key = name.lower()
        if key not in self.entities:
            self.entities[key] = name
            self.entity_types[key] = (etype or "thing").strip().lower()
        return self.entities[key]


class Sink:
    def add_fact(self, f: Fact) -> None: ...
    def close_fact(self, f: Fact) -> None: ...
    def add_message(self, step: int, stamp: str, text: str) -> None: ...
    def query(self, q: str) -> tuple[str, bool, float]:
        """Run an agent query. Returns (result text, is_error, engine ms)."""
        raise NotImplementedError


def _fmt_rows(header: list[str], rows: list[list]) -> str:
    shown = rows[:ROW_LIMIT]
    lines = [json.dumps(dict(zip(header, r)), default=str) for r in shown]
    out = "\n".join(lines) if lines else "(no rows)"
    if len(rows) > ROW_LIMIT:
        out += f"\n... {len(rows) - ROW_LIMIT} more rows not shown"
    if len(out) > RESULT_CHAR_LIMIT:
        out = out[:RESULT_CHAR_LIMIT] + "\n... truncated"
    return out


class SqliteSink(Sink):
    SCHEMA = """
    CREATE TABLE facts (
        id INTEGER PRIMARY KEY, subject TEXT, subject_type TEXT, predicate TEXT,
        object TEXT, object_type TEXT, valid_from TEXT, valid_to TEXT, source_step INTEGER
    );
    CREATE INDEX facts_s ON facts(subject, predicate);
    CREATE INDEX facts_o ON facts(object, predicate);
    CREATE VIEW current_facts AS SELECT * FROM facts WHERE valid_to IS NULL;
    """

    def __init__(self, path: Path) -> None:
        path.unlink(missing_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript(self.SCHEMA)
        self.ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)

    def add_fact(self, f: Fact) -> None:
        self.db.execute(
            "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?)",
            (f.fid, f.subject, f.subject_type, f.predicate, f.object, f.object_type, f.valid_from, None, f.valid_from_step),
        )
        self.db.commit()

    def close_fact(self, f: Fact) -> None:
        self.db.execute("UPDATE facts SET valid_to=? WHERE id=?", (f.valid_to, f.fid))
        self.db.commit()

    def query(self, q: str) -> tuple[str, bool, float]:
        deadline = time.perf_counter() + 5
        self.ro.set_progress_handler(lambda: int(time.perf_counter() > deadline), 10_000)
        t0 = time.perf_counter()
        try:
            cur = self.ro.execute(q)
            rows = [list(r) for r in cur.fetchall()]
            header = [d[0] for d in cur.description or []]
        except sqlite3.Error as e:
            return f"SQL error: {e}", True, (time.perf_counter() - t0) * 1000
        ms = (time.perf_counter() - t0) * 1000
        return _fmt_rows(header, rows), False, ms


class GraphSink(Sink):
    def __init__(self, graph_name: str, host: str = "localhost", port: int = 6379) -> None:
        from falkordb import FalkorDB

        self.g = FalkorDB(host=host, port=port).select_graph(graph_name)
        try:
            self.g.delete()
        except Exception:
            pass  # graph did not exist yet
        self.g = FalkorDB(host=host, port=port).select_graph(graph_name)
        self.g.query("CREATE (:Entity {name: '__init__'})")
        self.g.query("MATCH (n:Entity {name: '__init__'}) DELETE n")
        self.g.create_node_range_index("Entity", "name")

    @staticmethod
    def rel_type(predicate: str) -> str:
        return norm_predicate(predicate).upper()

    def add_fact(self, f: Fact) -> None:
        self.g.query(
            "MERGE (s:Entity {name: $s}) ON CREATE SET s.type = $st "
            "MERGE (o:Entity {name: $o}) ON CREATE SET o.type = $ot "
            f"CREATE (s)-[:{self.rel_type(f.predicate)} {{fid: $fid, valid_from: $vf, valid_to: null, source_step: $step}}]->(o)",
            {"s": f.subject, "st": f.subject_type, "o": f.object, "ot": f.object_type,
             "fid": f.fid, "vf": f.valid_from, "step": f.valid_from_step},
        )

    def close_fact(self, f: Fact) -> None:
        self.g.query("MATCH ()-[r]->() WHERE r.fid = $fid SET r.valid_to = $vt", {"fid": f.fid, "vt": f.valid_to})

    def query(self, q: str) -> tuple[str, bool, float]:
        t0 = time.perf_counter()
        try:
            res = self.g.ro_query(q, timeout=5000)
        except Exception as e:
            return f"Cypher error: {e}", True, (time.perf_counter() - t0) * 1000
        header = [h[1] for h in res.header]
        rows = [[_plain(v) for v in r] for r in res.result_set]
        return _fmt_rows(header, rows), False, res.run_time_ms


def _plain(v):
    """Turn FalkorDB node/edge objects into something JSON can show."""
    if hasattr(v, "properties"):
        props = dict(v.properties)
        if hasattr(v, "relation"):
            props["_type"] = v.relation
        elif getattr(v, "labels", None):
            props["_labels"] = v.labels
        return props
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if hasattr(v, "nodes") and callable(v.nodes):
        return {"path_nodes": [_plain(n) for n in v.nodes()]}
    return v


class VectorSink(Sink):
    """Hybrid BM25 + dense retrieval, fused with RRF, then cross-encoder reranked.

    Indexes both the raw messages and the extracted fact sentences, so it holds
    strictly more than the structured stores do.
    """

    _models: tuple | None = None

    def __init__(self) -> None:
        import numpy as np

        self.np = np
        self.docs: list[str] = []
        self.vecs: list = []
        self.fact_doc: dict[int, int] = {}
        if VectorSink._models is None:
            from fastembed import TextEmbedding
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            VectorSink._models = (
                TextEmbedding("BAAI/bge-small-en-v1.5"),
                TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2"),
            )
        self.embedder, self.reranker = VectorSink._models

    def _embed(self, text: str):
        return next(iter(self.embedder.embed([text])))

    def _add(self, text: str) -> int:
        self.docs.append(text)
        self.vecs.append(self._embed(text))
        return len(self.docs) - 1

    def add_fact(self, f: Fact) -> None:
        self.fact_doc[f.fid] = self._add("FACT: " + f.sentence())

    def close_fact(self, f: Fact) -> None:
        i = self.fact_doc[f.fid]
        self.docs[i] = "FACT: " + f.sentence()
        self.vecs[i] = self._embed(self.docs[i])

    def add_message(self, step: int, stamp: str, text: str) -> None:
        body, _, logs = text.partition("\n\nattached logs:\n")
        self._add(f"MESSAGE (inbox item {step}, {stamp}):\n{body}")
        if logs:
            self._add(f"LOGS attached to inbox item {step}:\n{logs}")

    def search(self, q: str, k: int = 8) -> list[str]:
        from rank_bm25 import BM25Okapi

        np = self.np
        if not self.docs:
            return []
        tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())  # noqa: E731
        bm = BM25Okapi([tok(d) for d in self.docs]).get_scores(tok(q))
        qv = self._embed(q)
        dense = np.array(self.vecs) @ qv
        fused: dict[int, float] = {}
        for ranking in (np.argsort(-bm)[:40], np.argsort(-dense)[:40]):
            for rank, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + rank)
        cands = sorted(fused, key=fused.get, reverse=True)[:30]
        scores = list(self.reranker.rerank(q, [self.docs[i] for i in cands]))
        order = [cands[i] for i in np.argsort(-np.array(scores))[:k]]
        return [self.docs[i] for i in order]

    def query(self, q: str, k: int = 8) -> tuple[str, bool, float]:
        t0 = time.perf_counter()
        hits = self.search(q, k)
        ms = (time.perf_counter() - t0) * 1000
        out = "\n\n---\n\n".join(hits) if hits else "(no results)"
        return out[:RESULT_CHAR_LIMIT], False, ms
