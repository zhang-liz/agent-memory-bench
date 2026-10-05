"""Command line entry point: run the matrix, LongMemEval, the scale test, or the analysis."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import world as worldmod
from .agent import CONDITIONS, NEXT_CONDITIONS, Episode, InboxItem, RunConfig
from .extract import Extractor
from .grade import grade_set
from .llm import PRICES, provider
from .stores import FactLog, GraphSink, SqliteSink, VectorSink

ROOT = Path(__file__).resolve().parent.parent
STORE_CONDITIONS = ("vector", "sqlite", "graph")


def load_env() -> None:
    """Read KEY=VALUE lines from .env without overriding the real environment."""
    import os

    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if k.lower().replace("-", "_") == "openai_api_key":
            k = "OPENAI_API_KEY"
        os.environ.setdefault(k, v)


def _client(fake: bool, model: str):
    if fake:
        from .fake import FakeClient, FakeOpenAI

        return FakeOpenAI() if provider(model) == "openai" else FakeClient()
    if provider(model) == "openai":
        import openai

        return openai.OpenAI(max_retries=6)
    import anthropic

    return anthropic.Anthropic(max_retries=6)


def _ints(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return out


def world_items(w: worldmod.World) -> list[InboxItem]:
    out = []
    for it in w.inbox():
        if isinstance(it, worldmod.Event):
            out.append(InboxItem("event", it.step, it.time, it.text, extra={"asserts": it.asserts, "retracts": it.retracts}))
        else:
            out.append(InboxItem("question", it.after_step, None, it.text, qid=it.qid, category=it.category, answer=it.answer))
    return out


class OracleExtractor:
    """Perfect extraction from the generator's ground truth. For ablations and dry runs only."""

    def ingest(self, log: FactLog, step, when, text, item: InboxItem) -> dict:
        types = {"owned_by": ("service", "team"), "depends_on": ("service", "service"),
                 "member_of": ("person", "team"), "on_call": ("team", "person")}
        ops = {
            "assert": [{"subject": s, "subject_type": types[p][0], "predicate": p, "object": o, "object_type": types[p][1]}
                       for s, p, o in item.extra.get("asserts", [])],
            "retract": [{"subject": s, "predicate": p, "object": o} for s, p, o in item.extra.get("retracts", [])],
        }
        log.apply(ops, step, when, text)
        return {"ops": ops, "usage": {"usd": 0.0}, "latency_s": 0.0, "cached": True}


def build_store(condition: str, name: str, run_dir: Path):
    if condition == "sqlite":
        return SqliteSink(run_dir / f"{name}.sqlite")
    if condition == "graph":
        return GraphSink(name)
    if condition == "vector":
        return VectorSink()
    return None


def memory_dir(out: str, task: str, name: str, condition: str) -> Path:
    # The output directory is part of the name, so separate experiments never share notes.
    return ROOT / "scratch" / "memories" / f"{Path(out).name}-{task}-{name}-{condition}"


def run_one(args, client, task: str, condition: str, items: list[InboxItem], grade, dataset_key: str,
            log_path: Path, generic_extract: bool, prefill: list[InboxItem] | None = None,
            memory_from: Path | None = None, handoff_note: str = "") -> dict:
    """One run. For a later session, `prefill` loads the store with what earlier
    sessions read, `memory_from` copies an earlier session's memory directory,
    and `handoff_note` is the note it left."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = log_path.with_suffix(".partial")
    tmp.unlink(missing_ok=True)
    name = log_path.stem
    store = build_store(condition, f"amb_{task}_{condition}_{name}".replace("-", "_"), log_path.parent)
    ingest = query = None
    if store is not None:
        log = FactLog([store])
        if args.extract == "oracle":
            ex = OracleExtractor()
            ingest = lambda it: ex.ingest(log, it.step, it.time, it.text, it)  # noqa: E731
        else:
            ex = Extractor(client, args.extract_model or args.model, _cache_dir(args), dataset_key, generic_extract)
            ingest = lambda it: ex.ingest(log, it.step, it.time, it.text)  # noqa: E731

        for it in prefill or []:
            if it.kind == "event":
                ingest(it)

        def query(tool: str, inp: dict):
            if tool == "search_memory":
                return store.query(inp.get("query", ""), int(inp.get("k") or 8))
            return store.query(inp.get("query", ""))

    cfg = RunConfig(condition=condition, model=args.model, effort=args.effort, task=task,
                    handoff=getattr(args, "handoff", False) and condition in ("full", "compact"),
                    handoff_note=handoff_note)
    with tmp.open("w") as f:
        f.write(json.dumps({"type": "header", "task": task, "condition": condition, "dataset": dataset_key,
                            "model": args.model, "effort": args.effort, "extract": args.extract,
                            "extract_model": args.extract_model or args.model, "git": _git_rev(),
                            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "cfg": cfg.__dict__}) + "\n")
    episode_cls = Episode
    if provider(args.model) == "openai":
        from .openai_agent import OpenAIEpisode

        episode_cls = OpenAIEpisode
    mem = memory_dir(getattr(args, "out", "results"), task, name, condition)
    if memory_from is not None:
        shutil.rmtree(mem, ignore_errors=True)
        shutil.copytree(memory_from, mem)
    ep = episode_cls(client, cfg, items, grade, tmp, ingest=ingest, query=query,
                     memory_dir=mem, keep_memory=memory_from is not None)
    summary = ep.run()
    tmp.rename(log_path)
    return summary


def prepare_extraction(args, client, dataset_key: str, items: list[InboxItem], generic: bool) -> None:
    """Fill the extraction cache for one dataset before any store condition reads it."""
    if args.extract == "oracle":
        return
    ex = Extractor(client, args.extract_model or args.model, _cache_dir(args), dataset_key, generic)
    log = FactLog([])
    for it in items:
        if it.kind == "event":
            ex.ingest(log, it.step, it.time, it.text)


def _cache_dir(args) -> Path:
    # Fake runs must never leave cached facts where a real run would replay them.
    return ROOT / ("scratch" if getattr(args, "fake", False) else "cache") / "extract"


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def dataset_key(L: int, seed: int, scale: int) -> str:
    # The key names the extraction cache, so a big world must never share one with a small world.
    return f"ops-L{L}-w{seed}" + (f"-x{scale}" if scale > 1 else "")


def cmd_ops(args) -> None:
    client = _client(args.fake, args.model)
    conds = args.conditions.split(",")
    jobs = []
    for L in _ints(args.lengths):
        for seed in _ints(args.worlds):
            w = worldmod.generate(seed, L, args.scale)
            items = world_items(w)
            grade = (lambda item, given, w=w: grade_set(given, item.answer, w.aliases))
            key = dataset_key(L, seed, args.scale)
            for rep in range(1, args.repeats + 1):
                for c in conds:
                    path = Path(args.out) / "ops" / c / f"L{L}-w{seed}-r{rep}.jsonl"
                    if path.exists():
                        continue
                    jobs.append((key, items, grade, c, path))
    _execute(args, client, "ops", jobs, generic=False)


def cmd_next(args) -> None:
    """A new session after the inbox: only questions, answered from what survived.

    Session A comes from `amb ops --handoff --out <source>` (compact writes the
    handoff note, memtool writes the memory directory). Stores are rebuilt from
    the same cached extraction, which is what a store would hold after session A.
    """
    client = _client(args.fake, args.model)
    jobs = []
    for L in _ints(args.lengths):
        for seed in _ints(args.worlds):
            w = worldmod.generate(seed, L, args.scale)
            events = [it for it in world_items(w) if it.kind == "event"]
            items = [InboxItem("question", L, None, q.text, qid=q.qid, category=q.category, answer=q.answer)
                     for q in worldmod.followup_questions(w, args.per_category)]
            grade = (lambda item, given, w=w: grade_set(given, item.answer, w.aliases))
            name = f"L{L}-w{seed}-r1"
            for c in args.conditions.split(","):
                path = Path(args.out) / "next" / c / f"{name}.jsonl"
                if path.exists():
                    continue
                extra = {}
                if c in STORE_CONDITIONS:
                    extra["prefill"] = events
                elif c == "memtool":
                    src = memory_dir(args.source, "ops", name, "memtool")
                    if not src.exists():
                        print(f"skip {c} {name}: no session A memory at {src}")
                        continue
                    extra["memory_from"] = src
                elif c == "handoff":
                    a_log = Path(args.source) / "ops" / "compact" / f"{name}.jsonl"
                    notes = [json.loads(x)["text"] for x in a_log.open() if '"type": "handoff"' in x] if a_log.exists() else []
                    if not notes:
                        print(f"skip {c} {name}: no handoff note in {a_log}")
                        continue
                    extra["handoff_note"] = notes[-1]
                jobs.append((dataset_key(L, seed, args.scale), items, grade, c, path, extra, events))
    if not jobs:
        print("nothing to run: every requested run already has a log")
        return
    print(f"{len(jobs)} runs to go")
    for key, _, _, _, _, _, events in {j[0]: j for j in jobs if j[3] in STORE_CONDITIONS}.values():
        prepare_extraction(args, client, key, events, False)
    with ThreadPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(run_one, args, client, "next", c, items, grade, key, path, False, **extra): path
                for key, items, grade, c, path, extra, _ in jobs}
        for f in as_completed(futs):
            path = futs[f]
            try:
                s = f.result()
                print(f"done {path.relative_to(args.out)}: answered {s['questions_answered']}/{s['questions_total']}, "
                      f"{s['api_calls']} calls, {s['wall_s']:.0f}s")
            except Exception as e:
                print(f"FAILED {path}: {e!r}", file=sys.stderr)


def cmd_lme(args) -> None:
    from . import longmemeval as lme

    client = _client(args.fake, args.model)
    judge = lme.Judge(client, args.judge_model or args.model)
    entries = lme.select(lme.load(ROOT / "data"), args.n_per_type, seed=args.sample_seed)
    jobs = []
    for e in entries:
        items = lme.items(e)
        for rep in range(1, args.repeats + 1):
            for c in args.conditions.split(","):
                path = Path(args.out) / "lme" / c / f"{e['question_id']}-r{rep}.jsonl"
                if not path.exists():
                    jobs.append((f"lme-{e['question_id']}", items, judge, c, path))
    _execute(args, client, "lme", jobs, generic=True)


def _execute(args, client, task: str, jobs: list, generic: bool) -> None:
    if not jobs:
        print("nothing to run: every requested run already has a log")
        return
    print(f"{len(jobs)} runs to go")
    needs_extract = {key: items for key, items, _, c, _ in jobs if c in STORE_CONDITIONS}
    if needs_extract:
        print(f"extracting facts for {len(needs_extract)} datasets (cached after the first time)")
        with ThreadPoolExecutor(args.jobs) as pool:
            futs = [pool.submit(prepare_extraction, args, client, k, items, generic) for k, items in needs_extract.items()]
            for f in as_completed(futs):
                f.result()
    with ThreadPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(run_one, args, client, task, c, items, grade, key, path, generic): path
                for key, items, grade, c, path in jobs}
        for f in as_completed(futs):
            path = futs[f]
            try:
                s = f.result()
                print(f"done {path.relative_to(args.out)}: answered {s['questions_answered']}/{s['questions_total']}, "
                      f"{s['api_calls']} calls, {s['wall_s']:.0f}s")
            except Exception as e:  # keep the rest of the matrix going
                print(f"FAILED {path}: {e!r}", file=sys.stderr)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="amb")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--conditions", default=",".join(CONDITIONS))
        sp.add_argument("--model", default="gpt-6.1-sol", choices=sorted(PRICES))
        sp.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
        sp.add_argument("--extract", default="llm", choices=["llm", "oracle"])
        sp.add_argument("--extract-model", default=None, choices=sorted(PRICES))
        sp.add_argument("--repeats", type=int, default=1)
        sp.add_argument("--jobs", type=int, default=3)
        sp.add_argument("--out", default=str(ROOT / "results"))
        sp.add_argument("--fake", action="store_true", help="offline dry run with a scripted fake model")

    ops = sub.add_parser("ops", help="run the synthetic on-call world")
    common(ops)
    ops.add_argument("--lengths", default="50,100,200")
    ops.add_argument("--worlds", default="1-5")
    ops.add_argument("--scale", type=int, default=1, help="50 x scale services; 1 is the standard world")
    ops.add_argument("--handoff", action="store_true",
                     help="full/compact runs end by writing a handoff note for `amb next`")
    ops.set_defaults(fn=cmd_ops)

    nxt = sub.add_parser("next", help="a new session after the inbox, answered from what survived")
    common(nxt)
    nxt.set_defaults(conditions=",".join(NEXT_CONDITIONS))
    nxt.add_argument("--lengths", default="200")
    nxt.add_argument("--worlds", default="1")
    nxt.add_argument("--per-category", type=int, default=4)
    nxt.add_argument("--scale", type=int, default=1, help="must match the session A run")
    nxt.add_argument("--source", default=str(ROOT / "results" / "next"),
                     help="--out of the `amb ops --handoff` run that played session A")
    nxt.set_defaults(fn=cmd_next)

    lme = sub.add_parser("lme", help="run the LongMemEval subset")
    common(lme)
    lme.add_argument("--n-per-type", type=int, default=5)
    lme.add_argument("--sample-seed", type=int, default=0)
    lme.add_argument("--judge-model", default=None, choices=sorted(PRICES))
    lme.set_defaults(fn=cmd_lme)

    scale = sub.add_parser("scale", help="graph latency vs size and concurrent writers (no LLM)")
    scale.add_argument("--sizes", default="10000,100000,1000000")
    scale.add_argument("--queries", type=int, default=500)
    scale.add_argument("--writers", default="0,5,10,20")
    scale.add_argument("--with-sqlite", action="store_true")
    scale.add_argument("--out", default=str(ROOT / "results"))
    scale.set_defaults(fn=lambda a: __import__("amb.scale", fromlist=["main"]).main(a))

    graphs = sub.add_parser("graphs", help="FalkorDB vs Neo4j vs Memgraph on the same Cypher (no LLM)")
    graphs.add_argument("--engines", default="falkordb,neo4j,memgraph")
    graphs.add_argument("--edges", type=int, default=1000000)
    graphs.add_argument("--queries", type=int, default=500)
    graphs.add_argument("--writers", default="0,5,10,20")
    graphs.add_argument("--users", type=int, default=0, help="also run the one-memory-per-user test with this many users")
    graphs.add_argument("--edges-per-user", type=int, default=2000)
    graphs.add_argument("--skip-single", action="store_true")
    graphs.add_argument("--out", default=str(ROOT / "results"))
    graphs.set_defaults(fn=lambda a: __import__("amb.graphs", fromlist=["main"]).main(a))

    an = sub.add_parser("analyze", help="tables and charts from result logs")
    an.add_argument("--out", default=str(ROOT / "results"))
    an.set_defaults(fn=lambda a: __import__("amb.analyze", fromlist=["main"]).main(a))

    args = p.parse_args(argv)
    for flag in ("extract_model", "judge_model"):
        other = getattr(args, flag, None)
        if other and provider(other) != provider(args.model):
            p.error(f"--{flag.replace('_', '-')} must use the same API as --model (one client per run)")
    load_env()
    args.fn(args)


if __name__ == "__main__":
    main()
