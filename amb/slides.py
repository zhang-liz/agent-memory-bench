"""Dark slide charts that match the talk deck (dark surface, pink highlight).

One series is the story (graph memory, pink); the others are context (gray),
and every mark carries a direct label, so identity never rests on color alone.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze import LABELS, _dodge, load_runs

SURFACE, INK, INK2, GRID = "#1f1f1f", "#ffffff", "#c3c2b7", "#3a3a3a"
PINK, GRAY = "#ff6bb5", "#6e6e6e"


# Charts are drawn small and at high dpi, so the type is large once placed on a
# slide: about 30 px on a 1920 px slide, readable from the back of a big room.
FIG_W, FIG_H, DPI = 6.4, 3.6, 375
TICK, LABEL, NOTE = 13, 14, 15


def _fig(plt, w=FIG_W, h=FIG_H):
    fig, ax = plt.subplots(figsize=(w, h), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=TICK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _color(c: str) -> str:
    return PINK if c == "graph" else GRAY


def context_chart(runs, L: int, out: Path, plt) -> None:
    fig, ax = _fig(plt)
    ends = []
    for c in ("full", "compact", "vector", "sqlite", "graph"):
        rs = [r for r in runs if r["condition"] == c and r["L"] == L]
        if not rs:
            continue
        bins = defaultdict(list)
        for r in rs:
            for cursor, tok in r["context_curve"]:
                bins[cursor].append(tok)
        xs = sorted(bins)
        ys = [np.mean(bins[x]) / 1000 for x in xs]
        ax.plot(xs, ys, color=_color(c), linewidth=3 if c == "graph" else 2, zorder=3 if c == "graph" else 2)
        ends.append((c, xs[-1], ys[-1]))
    top = max(e[2] for e in ends)
    for (c, x, y), ly in zip(ends, _dodge([e[2] for e in ends], top * 0.11)):
        ax.annotate(LABELS[c].split(" (")[0], (x, y), xytext=(x + 4, ly), textcoords="data", color=PINK if c == "graph" else INK2,
                    fontsize=NOTE, va="center", fontweight="bold" if c == "graph" else "normal")
    ax.set_xlabel("Inbox items read", color=INK2, fontsize=LABEL)
    ax.set_ylabel("Tokens in context (K)", color=INK2, fontsize=LABEL)
    ax.margins(x=0.2)
    fig.tight_layout()
    fig.savefig(out / f"slide_context_L{L}.png", facecolor=SURFACE)
    plt.close(fig)


def cost_chart(runs, L: int, out: Path, plt) -> None:
    conds = [c for c in ("full", "compact", "vector", "sqlite", "graph") if any(r["condition"] == c and r["L"] == L for r in runs)]
    fig, ax = _fig(plt)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    hot = [np.mean([r["hot_usd"] for r in runs if r["condition"] == c and r["L"] == L]) for c in conds]
    cold = [np.mean([r["cold_usd"] for r in runs if r["condition"] == c and r["L"] == L]) for c in conds]
    ys = np.arange(len(conds))
    for y, c, h, k in zip(ys, conds, hot, cold):
        ax.barh(y, h, 0.62, color=_color(c), edgecolor=SURFACE, linewidth=2)
        if k:
            ax.barh(y, k, 0.62, left=h, color=_color(c), alpha=0.45, edgecolor=SURFACE, linewidth=2, hatch="//")
        # "\\$" keeps matplotlib from reading the dollar signs as math mode.
        ax.annotate(f"\\${h + k:.2f}" + (f" (\\${k:.2f} extraction)" if k else ""), (h + k, y), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=NOTE, color=PINK if c == "graph" else INK)
    ax.set_yticks(ys, [LABELS[c].split(" (")[0] for c in conds], fontsize=LABEL)
    for t, c in zip(ax.get_yticklabels(), conds):
        t.set_color(PINK if c == "graph" else INK2)
    ax.invert_yaxis()
    ax.set_xlabel("USD per task", color=INK2, fontsize=LABEL)
    ax.margins(x=0.75)
    fig.tight_layout()
    fig.savefig(out / f"slide_cost_L{L}.png", facecolor=SURFACE)
    plt.close(fig)


def accuracy_chart(runs, out: Path, plt) -> None:
    """Accuracy per setup at each length, as labeled dots (no colour-only encoding)."""
    lengths = sorted({r["L"] for r in runs})
    conds = [c for c in ("full", "compact", "vector", "sqlite", "graph") if any(r["condition"] == c for r in runs)]
    fig, ax = _fig(plt)
    width = 0.8 / len(conds)
    for i, c in enumerate(conds):
        xs, vals = [], []
        for j, L in enumerate(lengths):
            rs = [r for r in runs if r["condition"] == c and r["L"] == L]
            if not rs:
                continue
            got = sum(sum(a["correct"] for a in r["answers"]) for r in rs)
            total = sum(r["questions"] for r in rs)
            xs.append(j + (i - (len(conds) - 1) / 2) * width)
            vals.append(got / total)
        bars = ax.bar(xs, vals, width * 0.88, color=_color(c), edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.annotate(f"{100 * v:.0f}%", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=TICK - 2, color=PINK if c == "graph" else INK2)
            ax.annotate(LABELS[c].split(" (")[0], (b.get_x() + b.get_width() / 2, 0.02), ha="center", va="bottom",
                        rotation=90, fontsize=TICK - 3, color=SURFACE)
    ax.set_xticks(range(len(lengths)), [f"{L} messages" for L in lengths], fontsize=LABEL, color=INK2)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    fig.tight_layout()
    fig.savefig(out / "slide_accuracy.png", facecolor=SURFACE)
    plt.close(fig)


def scale_chart(path: Path, out: Path, plt) -> None:
    d = json.loads(path.read_text())
    edges = [r["edges"] for r in d["sizes"]]
    fig, ax = _fig(plt)
    series = [
        # The tuned traversal, not the plain recursive CTE: compare against the best SQLite version.
        ([r["sqlite_blast_bfs"]["client_ms"]["p99"] for r in d["sizes"]], "SQLite, tuned", GRAY, "--"),
        ([r["blast"]["client_ms"]["p99"] for r in d["sizes"]], "FalkorDB", PINK, "-"),
    ]
    for ys, label, color, style in series:
        ax.plot(edges, ys, style, color=color, linewidth=3, marker="o", markersize=8)
        for x, y in zip(edges, ys):
            # Baseline values sit above their points and FalkorDB's below, so the two never overlap where they meet.
            ax.annotate(f"{y:.0f} ms" if y >= 10 else f"{y:.1f} ms", (x, y), xytext=(0, -30 if color == PINK else 14),
                        textcoords="offset points", ha="center", fontsize=TICK, color=color if color == PINK else INK2)
        ax.annotate(label, (edges[-1], ys[-1]), xytext=(18, 0), textcoords="offset points", va="center",
                    fontsize=NOTE, color=PINK if color == PINK else INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(2, 1000)
    nominal = [r["target_edges"] for r in d["sizes"]]
    ax.set_xticks(edges, [f"{n / 1000:.0f}K" if n < 1e6 else f"{n / 1e6:.0f}M" for n in nominal])
    ax.minorticks_off()
    ticks = [t for t in (1, 3, 10, 30, 100, 300, 1000) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("Edges in agent memory", color=INK2, fontsize=LABEL)
    ax.set_ylabel("Slowest 1% (ms, log)", color=INK2, fontsize=LABEL)
    ax.margins(x=0.5)
    fig.tight_layout()
    fig.savefig(out / "slide_scale.png", facecolor=SURFACE)
    plt.close(fig)


def scale_compare_chart(scale: Path, graphs: Path, out: Path, plt) -> None:
    """Two panels on one slide: graph vs SQLite as memory grows, then graph engines as writers grow."""
    sc, gr = json.loads(scale.read_text()), json.loads(graphs.read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(2 * FIG_W, FIG_H + 0.4), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=TICK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_yscale("log")
        ax.minorticks_off()

    def draw(ax, xs, ys, label, color, style, below):
        ax.plot(xs, ys, style, color=color, linewidth=3, marker="o", markersize=8)
        ax.annotate(f"{ys[-1]:.0f} ms" if ys[-1] >= 10 else f"{ys[-1]:.1f} ms", (xs[-1], ys[-1]),
                    xytext=(0, -26 if below else 12), textcoords="offset points", ha="center",
                    fontsize=TICK, color=PINK if color == PINK else INK2)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(22, 0), textcoords="offset points", va="center",
                    fontsize=NOTE, color=PINK if color == PINK else INK2)

    # Left: the multi-hop query, against the tuned SQLite traversal.
    edges = [r["edges"] for r in sc["sizes"]]
    draw(left, edges, [r["sqlite_blast_bfs"]["client_ms"]["p99"] for r in sc["sizes"]], "SQLite", GRAY, "--", False)
    draw(left, edges, [r["blast"]["client_ms"]["p99"] for r in sc["sizes"]], "FalkorDB", PINK, "-", True)
    left.set_xscale("log")
    left.set_xticks(edges, [f"{n / 1000:.0f}K" if n < 1e6 else f"{n / 1e6:.0f}M" for n in (r["target_edges"] for r in sc["sizes"])])
    left.minorticks_off()
    left.set_xlabel("Edges in memory", color=INK2, fontsize=LABEL)
    left.set_title("Graph vs SQLite, as memory grows", color=INK, fontsize=LABEL, loc="left")
    left.margins(x=0.45)

    # Right: the page query on 1M edges while agents write.
    style = {"falkordb": ("FalkorDB", PINK, "-"), "memgraph": ("Memgraph", GRAY, "--"), "neo4j": ("Neo4j", GRAY, ":")}
    runs = {r["engine"]: r for r in gr["single"]}
    # Memgraph and Neo4j end within a few ms of each other, so they share one label.
    ends = []
    for eng in ("memgraph", "neo4j"):
        c = runs[eng]["concurrency"]
        _, color, ls = style[eng]
        right.plot([x["writers"] for x in c], [x["page"]["p99"] for x in c], ls, color=color, linewidth=3,
                   marker="o", markersize=8)
        ends.append((c[-1]["writers"], c[-1]["page"]["p99"]))
    lo, hi = sorted(y for _, y in ends)
    x_end, y_top = ends[0][0], max(y for _, y in ends)
    right.annotate(f"{lo:.0f}–{hi:.0f} ms", (x_end, y_top), xytext=(-8, 12), textcoords="offset points",
                   ha="right", fontsize=TICK, color=INK2)
    right.annotate("Neo4j,\nMemgraph", (x_end, (lo * hi) ** 0.5), xytext=(22, 0), textcoords="offset points",
                   va="center", fontsize=NOTE, color=INK2)
    c = runs["falkordb"]["concurrency"]
    draw(right, [x["writers"] for x in c], [x["page"]["p99"] for x in c], "FalkorDB", PINK, "-", True)
    right.set_xticks([x["writers"] for x in runs["falkordb"]["concurrency"]])
    right.set_xlabel("Agents writing at once (1M edges)", color=INK2, fontsize=LABEL)
    right.set_title("Graph databases, while agents write", color=INK, fontsize=LABEL, loc="left")
    right.margins(x=0.3)

    for ax, lo, hi in ((left, 2, 1000), (right, 0.1, 100)):
        ax.set_ylim(lo, hi)
        ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000) if lo <= t <= hi]
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    left.set_ylabel("Slowest 1% (ms, log)", color=INK2, fontsize=LABEL)
    fig.tight_layout(w_pad=3)
    fig.savefig(out / "slide_scale_compare.png", facecolor=SURFACE)
    plt.close(fig)


def _usd(path: Path, only_handoff: bool = False) -> float:
    calls = [r for r in map(json.loads, path.open()) if r["type"] == "call"]
    return sum(c["usage"]["usd"] for c in calls if c.get("handoff") or not only_handoff)


def next_session_slide(big: Path, cache: Path, out: Path, plt) -> None:
    """Full slide for the new-session test on the big world, drawn as one image so it keeps the deck's font."""
    name = "L300-w1-r1.jsonl"
    rows = [("none", "No memory"), ("handoff", "Handoff note"), ("memtool", "Memory files"),
            ("vector", "Vector"), ("sqlite", "SQLite"), ("graph", "Graph (FalkorDB)")]
    extraction = sum(json.loads(x)["usage"]["usd"] for x in next(cache.glob("ops-L300-w1-x10-*.jsonl")).open())
    a_compact, a_memtool = big / "ops" / "compact" / name, big / "ops" / "memtool" / name
    # Upkeep is what keeping the memory cost on top of the work itself: writing the note,
    # the note-taking agent's extra spend over the compaction agent, or background extraction.
    upkeep = {"none": 0.0, "handoff": _usd(a_compact, only_handoff=True),
              "memtool": _usd(a_memtool) - _usd(a_compact), "vector": extraction, "sqlite": extraction, "graph": extraction}
    data = []
    for c, label in rows:
        recs = [json.loads(x) for x in (big / "next" / c / name).open()]
        ans = [r for r in recs if r["type"] == "answer"]
        data.append((c, label, sum(a["correct"] for a in ans), len(ans), _usd(big / "next" / c / name), upkeep[c]))

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(108 / 1920, 1 - 70 / 1080, "The next session", fontsize=96, fontweight="light", color=INK, va="top")
    fig.text(108 / 1920, 1 - 245 / 1080,
             "A new agent with an empty context gets 20 questions\nabout a company with 500 services.",
             fontsize=28, color=INK2, va="top", linespacing=1.4)
    ax = fig.add_axes([440 / 1920, 250 / 1080, 1250 / 1920, 450 / 1080])
    ax.set_facecolor("#1e1e1e")
    for s in ax.spines.values():
        s.set_visible(False)
    ys = np.arange(len(data))
    for y, (c, label, got, total, per, up) in zip(ys, data):
        color = _color(c)
        ax.barh(y, 0.7 * got / total, 0.62, color=color if got else GRAY)
        ink = PINK if c == "graph" else INK
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=28, color=ink, transform=ax.get_yaxis_transform())
        ax.text(0.7 * got / total + 0.015, y, f"{got}/{total}", va="center", fontsize=28, color=ink)
        ax.text(1.1, y, f"\\${up:.2f}", va="center", ha="right", fontsize=26, color=INK2)
        ax.text(1.4, y, f"\\${per:.2f}", va="center", ha="right", fontsize=26, color=INK2)
    ax.text(1.1, -0.95, "Upkeep", ha="right", fontsize=24, color=INK2)
    ax.text(1.4, -0.95, "Per session", ha="right", fontsize=24, color=INK2)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(len(data) - 0.5, -1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.text(108 / 1920, 215 / 1080,
             "With 50 services, every setup with memory scored 20/20.\n"
             "Upkeep: writing the note, extra work to keep memory files, or background extraction.\n"
             "Graph, SQLite and vector missed the same 2: one name the extractor got wrong.",
             fontsize=22, color=INK2, va="top", linespacing=1.5)
    fig.savefig(out / "slide_next_session.png", facecolor="#1e1e1e")
    plt.close(fig)


TAKEAWAYS = [
    ("Get memory out of the context window.", "4.5x fewer tokens, 41% cheaper, same answers."),
    ("Inside one session, compaction is a strong free baseline.", "At 400 messages it tied the graph."),
    ("Across sessions, a store built in the background pays off.", "18/20 at 500 services, and the cheapest per new session."),
]


def takeaways_slide(out: Path, plt) -> None:
    """Closing summary, drawn as one image so it keeps the deck's font."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(108 / 1920, 1 - 90 / 1080, "Takeaways", fontsize=110, fontweight="light", color=INK, va="top")
    for i, (head, sub) in enumerate(TAKEAWAYS):
        y = 1 - (380 + i * 210) / 1080
        fig.text(108 / 1920, y, f"0{i + 1}", fontsize=44, fontweight="bold", color=PINK, va="top")
        fig.text(250 / 1920, y, head, fontsize=40, color=INK, va="top")
        fig.text(250 / 1920, y - 75 / 1080, sub, fontsize=30, color=INK2, va="top")
    fig.savefig(out / "slide_takeaways.png", facecolor="#1e1e1e")
    plt.close(fig)


def main(root: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib import font_manager

    # The deck uses Poppins; load it from scratch/fonts when present.
    for ttf in (Path(__file__).resolve().parent.parent / "scratch" / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    plt.rcParams["font.family"] = ["Poppins", "DejaVu Sans"]
    out = root / "slides"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root, "ops")
    for L in sorted({r["L"] for r in runs}):
        context_chart(runs, L, out, plt)
        cost_chart(runs, L, out, plt)
    accuracy_chart(runs, out, plt)
    scale = root.parent / "scale" / "scale.json"
    if scale.exists():
        scale_chart(scale, out, plt)
        graphs = root.parent / "graphs" / "graphs.json"
        if graphs.exists():
            scale_compare_chart(scale, graphs, out, plt)
    big = root.parent / "big"
    if (big / "next" / "graph").exists():
        next_session_slide(big, root.parent.parent / "cache" / "extract", out, plt)
    takeaways_slide(out, plt)
    return out


if __name__ == "__main__":
    import sys

    print(main(Path(sys.argv[1])))
