"""Turn result logs into tables (report.md) and slide charts (PNG).

Confidence intervals are 95% cluster bootstrap over worlds (ops) or questions
(LongMemEval): resampling whole worlds keeps correlated questions together.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .agent import CONDITIONS

# Categorical slots 1-6 in fixed order (dataviz reference palette, validated
# for adjacent pairs in light mode). Three slots are under 3:1 contrast, so
# every line carries a direct label and every chart has a table in report.md.
COLORS = dict(zip(CONDITIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
LABELS = {"full": "Full history", "compact": "Compaction", "memtool": "Memory tool",
          "vector": "Vector (hybrid)", "sqlite": "SQLite", "graph": "Graph (FalkorDB)"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CATS = ("lookup", "multihop", "transitive", "changed", "verbatim")
RNG = np.random.default_rng(0)


def load_runs(root: Path, task: str) -> list[dict]:
    runs = []
    for path in sorted((root / task).glob("*/*.jsonl")):
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        head = next(r for r in recs if r["type"] == "header")
        summ = next((r for r in recs if r["type"] == "summary"), {})
        calls = [r for r in recs if r["type"] == "call"]
        tools = [r for r in recs if r["type"] == "tool"]
        extracts = [r for r in recs if r["type"] == "extract"]
        answers = [r for r in recs if r["type"] == "answer"]
        stem = path.stem  # ops: L100-w3-r1, lme: <qid>-r1
        run = {
            "path": str(path), "condition": head["condition"], "model": head["model"],
            "unit": stem.rsplit("-r", 1)[0], "rep": int(stem.rsplit("-r", 1)[1]),
            "finished": summ.get("finished", False), "answered": summ.get("questions_answered", 0),
            "questions": summ.get("questions_total", 0), "api_calls": len(calls), "nudges": summ.get("nudges", 0),
            "wall_s": summ.get("wall_s", 0.0),
            "hot_usd": sum(c["usage"]["usd"] for c in calls),
            "cold_usd": sum(e["usage"].get("usd", 0.0) for e in extracts),
            "judge_usd": sum(a.get("judge_usage", {}).get("usd", 0.0) for a in answers),
            "tokens_billed": sum(c["usage"]["input"] + c["usage"]["cache_write"] + c["usage"]["cache_read"] + c["usage"]["output"] for c in calls),
            "cache_read": sum(c["usage"]["cache_read"] for c in calls),
            "llm_s": sum(c["latency_s"] for c in calls),
            "mem_s": sum(t["wall_ms"] for t in tools) / 1000,
            "mem_engine_ms": [t["engine_ms"] for t in tools if t.get("engine_ms") is not None],
            "mem_errors": sum(1 for t in tools if t["is_error"]),
            "mem_calls": len(tools),
            "context_curve": [(c["cursor"], c["usage"]["context_tokens"]) for c in calls],
            "answers": answers,
        }
        if task == "ops":
            run["L"] = int(stem.split("-")[0][1:])
            run["world"] = int(stem.split("-")[1][1:])
        runs.append(run)
    return runs


def boot_ci(groups: list[list[float]], n: int = 2000) -> tuple[float, float, float]:
    """Mean of all values, with a 95% CI from resampling whole groups."""
    groups = [g for g in groups if g]
    if not groups:
        return float("nan"), float("nan"), float("nan")
    flat = np.concatenate(groups)
    if len(groups) == 1:
        return float(flat.mean()), float("nan"), float("nan")
    stats = []
    for _ in range(n):
        pick = RNG.integers(0, len(groups), len(groups))
        stats.append(np.concatenate([groups[i] for i in pick]).mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(flat.mean()), float(lo), float(hi)


def fmt_ci(m, lo, hi, pct=False, money=False) -> str:
    if np.isnan(m):
        return "-"
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"${x:.3f}") if money else (lambda x: f"{x:.2f}")
    return f(m) if np.isnan(lo) else f"{f(m)} [{f(lo)}, {f(hi)}]"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _dodge(ys: list[float], gap: float) -> list[float]:
    """Spread end-of-line label positions so no two are closer than gap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def ops_report(runs: list[dict], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = ["## Synthetic on-call world", ""]
    conds = [c for c in CONDITIONS if any(r["condition"] == c for r in runs)]
    lengths = sorted({r["L"] for r in runs})
    by = defaultdict(list)
    for r in runs:
        by[(r["condition"], r["L"])].append(r)

    def per_world(rs, fn):
        g = defaultdict(list)
        for r in rs:
            g[r["world"]].extend(fn(r))
        return list(g.values())

    # Accuracy and cost table.
    lines += ["### Accuracy and cost per task", "",
              "| Length | Setup | Runs | Exact-match accuracy | Agent cost (hot path) | Extraction cost (cold path) | Total cost | API calls | Unfinished |",
              "|---|---|---|---|---|---|---|---|---|"]
    for L in lengths:
        for c in conds:
            rs = by[(c, L)]
            if not rs:
                continue
            acc = boot_ci(per_world(rs, lambda r: [float(a["correct"]) for a in r["answers"]] + [0.0] * (r["questions"] - len(r["answers"]))))
            hot = boot_ci(per_world(rs, lambda r: [r["hot_usd"]]))
            cold = boot_ci(per_world(rs, lambda r: [r["cold_usd"]]))
            tot = boot_ci(per_world(rs, lambda r: [r["hot_usd"] + r["cold_usd"]]))
            calls = np.mean([r["api_calls"] for r in rs])
            unfinished = sum(not r["finished"] for r in rs)
            lines.append(f"| {L} | {LABELS[c]} | {len(rs)} | {fmt_ci(*acc, pct=True)} | {fmt_ci(*hot, money=True)} | "
                         f"{fmt_ci(*cold, money=True)} | {fmt_ci(*tot, money=True)} | {calls:.0f} | {unfinished} |")
    lines += ["", "Unanswered questions count as wrong. Extraction is shared by the vector, SQLite and graph setups "
              "(same facts, extracted once per world), and its full cost is charged to each of them.", ""]

    # Accuracy by category.
    lines += ["### Accuracy by question type", "", "| Length | Setup | " + " | ".join(CATS) + " |",
              "|---|---|" + "---|" * len(CATS)]
    for L in lengths:
        for c in conds:
            rs = by[(c, L)]
            if not rs:
                continue
            cells = [fmt_ci(*boot_ci(per_world(rs, lambda r, k=k: [float(a["correct"]) for a in r["answers"] if a["category"] == k])), pct=True)
                     for k in CATS]
            lines.append(f"| {L} | {LABELS[c]} | " + " | ".join(cells) + " |")
    lines.append("")

    # Time per task.
    lines += ["### Where the time goes", "",
              "| Length | Setup | LLM time per task (s) | Memory tool time per task (s) | Memory calls | Median engine ms per memory call | Memory query errors |",
              "|---|---|---|---|---|---|---|"]
    for L in lengths:
        for c in conds:
            rs = by[(c, L)]
            if not rs:
                continue
            eng = [x for r in rs for x in r["mem_engine_ms"]]
            lines.append(f"| {L} | {LABELS[c]} | {np.mean([r['llm_s'] for r in rs]):.0f} | {np.mean([r['mem_s'] for r in rs]):.1f} | "
                         f"{np.mean([r['mem_calls'] for r in rs]):.0f} | {np.median(eng) if eng else float('nan'):.2f} | "
                         f"{np.mean([r['mem_errors'] for r in rs]):.1f} |")
    lines.append("")

    Lmax = max(lengths)

    # Chart 1: context size per call across the task.
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ends = []
    for c in conds:
        rs = by[(c, Lmax)]
        if not rs:
            continue
        bins = defaultdict(list)
        for r in rs:
            for cursor, tok in r["context_curve"]:
                bins[cursor].append(tok)
        xs = sorted(bins)
        ys = [np.mean(bins[x]) / 1000 for x in xs]
        ax.plot(xs, ys, color=COLORS[c], linewidth=2, label=LABELS[c])
        ends.append((c, xs[-1], ys[-1]))
    top = max(e[2] for e in ends)
    for (c, x, y), ly in zip(ends, _dodge([e[2] for e in ends], top * 0.045)):
        ax.annotate(LABELS[c], (x, y), xytext=(x + 2, ly), textcoords="data", color=INK, fontsize=9, va="center",
                    arrowprops={"arrowstyle": "-", "color": COLORS[c], "linewidth": 1} if abs(ly - y) > top * 0.01 else None)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlabel("Inbox items read (messages and questions)", color=INK2)
    ax.set_ylabel("Tokens in context per call (thousands)", color=INK2)
    ax.set_title(f"Tokens the model reads on every call, {Lmax}-item task", color=INK, loc="left", fontsize=12)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(out / "chart_context_per_call.png")
    plt.close(fig)

    # Chart 2: total cost per task by length (grouped bars, one group per length).
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    width = 0.8 / len(conds)
    for i, c in enumerate(conds):
        means, errs = [], [[], []]
        for L in lengths:
            m, lo, hi = boot_ci(per_world(by[(c, L)], lambda r: [r["hot_usd"] + r["cold_usd"]]))
            means.append(m)
            errs[0].append(0 if np.isnan(lo) else m - lo)
            errs[1].append(0 if np.isnan(hi) else hi - m)
        xs = np.arange(len(lengths)) + (i - (len(conds) - 1) / 2) * width
        ax.bar(xs, means, width * 0.9, color=COLORS[c], label=LABELS[c], edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.set_xticks(range(len(lengths)), [f"{L} items" for L in lengths])
    ax.set_ylabel("USD per task (agent + extraction)", color=INK2)
    ax.set_title("Cost per task", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "chart_cost_per_task.png")
    plt.close(fig)

    # Chart 3: accuracy by question type at the longest length.
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for i, c in enumerate(conds):
        rs = by[(c, Lmax)]
        means, errs = [], [[], []]
        for k in CATS:
            m, lo, hi = boot_ci(per_world(rs, lambda r, k=k: [float(a["correct"]) for a in r["answers"] if a["category"] == k]))
            means.append(m)
            errs[0].append(0 if np.isnan(lo) else m - lo)
            errs[1].append(0 if np.isnan(hi) else hi - m)
        xs = np.arange(len(CATS)) + (i - (len(conds) - 1) / 2) * width
        ax.bar(xs, means, width * 0.9, color=COLORS[c], label=LABELS[c], edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.set_xticks(range(len(CATS)), ["Lookup", "Multi-hop", "Transitive", "Changed fact", "Verbatim detail"])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"Accuracy by question type, {Lmax}-item task", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(out / "chart_accuracy_by_type.png")
    plt.close(fig)

    # Chart 4: time per task split into LLM time and memory time.
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ys = np.arange(len(conds))
    llm = [np.mean([r["llm_s"] for r in by[(c, Lmax)]]) for c in conds]
    mem = [np.mean([r["mem_s"] for r in by[(c, Lmax)]]) for c in conds]
    ax.barh(ys, llm, 0.6, color="#2a78d6", label="Model time", edgecolor=SURFACE, linewidth=2)
    ax.barh(ys, mem, 0.6, left=llm, color="#eb6834", label="Memory tool time", edgecolor=SURFACE, linewidth=2)
    for y, a, b in zip(ys, llm, mem):
        ax.annotate(f"memory {b:.1f}s of {a + b:.1f}s", (a + b, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(ys, [LABELS[c] for c in conds])
    ax.invert_yaxis()
    ax.set_xlabel("Seconds per task", color=INK2)
    ax.set_title(f"Where the time goes, {Lmax}-item task", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.margins(x=0.25)
    fig.tight_layout()
    fig.savefig(out / "chart_time_split.png")
    plt.close(fig)

    lines += ["Charts: chart_context_per_call.png, chart_cost_per_task.png, chart_accuracy_by_type.png, chart_time_split.png", ""]
    return lines


def lme_report(runs: list[dict]) -> list[str]:
    lines = ["## LongMemEval subset (outside check)", "",
             "| Setup | Questions | Accuracy | Agent cost per question | Extraction cost per question | Judge cost |",
             "|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        rs = [r for r in runs if r["condition"] == c]
        if not rs:
            continue
        acc = boot_ci([[float(a["correct"]) for a in r["answers"]] or [0.0] for r in rs])
        lines.append(f"| {LABELS[c]} | {len(rs)} | {fmt_ci(*acc, pct=True)} | ${np.mean([r['hot_usd'] for r in rs]):.3f} | "
                     f"${np.mean([r['cold_usd'] for r in rs]):.3f} | ${np.mean([r['judge_usd'] for r in rs]):.3f} |")
    types = sorted({a["category"] for r in runs for a in r["answers"]})
    lines += ["", "| Setup | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for c in CONDITIONS:
        rs = [r for r in runs if r["condition"] == c]
        if rs:
            cells = [f"{100 * np.mean([a['correct'] for r in rs for a in r['answers'] if a['category'] == t] or [np.nan]):.0f}%" for t in types]
            lines.append(f"| {LABELS[c]} | " + " | ".join(cells) + " |")
    lines += ["", "The judge is a Claude model using LongMemEval's own grading prompts; the paper used GPT-4o, so these "
              "numbers are not comparable to published leaderboards. Compare setups with each other only.", ""]
    return lines


def scale_report(path: Path, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(path.read_text())
    lines = ["## Memory fetch latency at scale (no LLM)", "",
             "| Edges | Lookup p50 / p99 (ms) | Page query p50 / p99 (ms) | Blast radius p50 / p99 (ms) |",
             "|---|---|---|---|"]
    for r in d["sizes"]:
        cell = lambda k: f"{r[k]['client_ms']['p50']:.2f} / {r[k]['client_ms']['p99']:.2f}"  # noqa: E731
        lines.append(f"| {r['edges']:,} | {cell('lookup')} | {cell('page')} | {cell('blast')} |")
    lines += ["", "| Edges | Writers | Page query p50 / p99 (ms) | Writes/s during reads |", "|---|---|---|---|"]
    for r in d["sizes"]:
        for w in r["concurrency"]:
            lines.append(f"| {r['edges']:,} | {w['writers']} | {w['page']['client_ms']['p50']:.2f} / "
                         f"{w['page']['client_ms']['p99']:.2f} | {w['writes_per_s']:.0f} |")
    if "sqlite_page" in d["sizes"][0]:
        lines += ["", "In-process SQLite on the same data, for reference (no network hop, so not a like-for-like "
                  "comparison with a database server):", "", "| Edges | SQLite page p50 / p99 (ms) | SQLite blast p50 / p99 (ms) |", "|---|---|---|"]
        for r in d["sizes"]:
            lines.append(f"| {r['edges']:,} | {r['sqlite_page']['client_ms']['p50']:.3f} / {r['sqlite_page']['client_ms']['p99']:.3f} | "
                         f"{r['sqlite_blast']['client_ms']['p50']:.3f} / {r['sqlite_blast']['client_ms']['p99']:.3f} |")
    lines.append("")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    edges = [r["edges"] for r in d["sizes"]]
    series = [("page", "Page query (3 hops)", "#2a78d6"), ("blast", "Blast radius (1-4 hops, core services)", "#eb6834")]
    ends = []
    for key, label, color in series:
        for stat, style in (("p50", "-"), ("p99", "--")):
            ys = [r[key]["client_ms"][stat] for r in d["sizes"]]
            ax.plot(edges, ys, style, color=color, linewidth=2, marker="o", markersize=5, label=f"{label} {stat}")
            ends.append((f"{label} {stat}", ys[-1]))
    ax.set_yscale("log")
    logs = _dodge([np.log10(y) for _, y in ends], 0.12)
    for (label, y), ly in zip(ends, logs):
        ax.annotate(label, (edges[-1], y), xytext=(edges[-1] * 1.25, 10 ** ly), textcoords="data",
                    fontsize=9, color=INK, va="center")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xscale("log")
    ax.set_xlabel("Edges in the memory graph", color=INK2)
    ax.set_ylabel("Milliseconds per query, client side (log scale)", color=INK2)
    ax.set_title("Recall latency as memory grows", color=INK, loc="left", fontsize=12)
    ax.margins(x=0.35)
    fig.tight_layout()
    fig.savefig(out / "chart_scale_latency.png")
    plt.close(fig)
    lines += ["Chart: chart_scale_latency.png", ""]
    return lines


def main(args) -> None:
    root = Path(args.out)
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# agent-memory-bench results", ""]
    ops = load_runs(root, "ops") if (root / "ops").exists() else []
    if ops:
        models = sorted({r["model"] for r in ops})
        lines += [f"Model(s): {', '.join(models)}", ""] + ops_report(ops, out)
    lme = load_runs(root, "lme") if (root / "lme").exists() else []
    if lme:
        lines += lme_report(lme)
    if (root / "scale" / "scale.json").exists():
        lines += scale_report(root / "scale" / "scale.json", out)
    (out / "report.md").write_text("\n".join(lines))
    print(f"wrote {out / 'report.md'}")
