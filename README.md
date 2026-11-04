# agent-memory-bench

[![CI](https://github.com/zhang-liz/agent-memory-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/zhang-liz/agent-memory-bench/actions/workflows/ci.yml)

Does moving an agent's memory out of the context window and into a graph cut
its token bill without hurting accuracy? This repo runs a fair test of that
claim against five baselines, on a synthetic long-horizon task and on a public
benchmark, and reports where each memory setup wins and where it loses.

The short version: graph memory in FalkorDB read 4.5x fewer tokens than full
history for the same 25/25 score. At 1M edges with 20 agents writing, the
slowest 1% of FalkorDB reads stay under 1 ms; Neo4j and Memgraph sit at 14-36
ms. Deleting one user's memory takes 0.3-0.5 ms on FalkorDB and 150-190 ms on
Neo4j. 10,000 per-user memory graphs fit in one FalkorDB in about 3 GB.

This is the benchmark behind the FalkorDB talk and blog post "Agent Memory
for Long-Running AI Agents: Graph vs Vector" (Agent Memory Night, SF Tech
Week, 2026-10-08). The numbers quoted there are in [Results](#results), each
with the file it came from.

## Results

All runs: one Apple M5 laptop, `gpt-6.1-sol`, FalkorDB 6.0.1. One world
(`w1`), one repeat per setup. The raw logs are in `results/` and every table
below can be rebuilt with `uv run amb analyze`.

### Tokens and cost, 200 messages + 25 questions

Full history re-reads the whole inbox every turn. Graph memory sends a query
and reads back only the facts it needs. Prompt caching is on for both.

| Setup | Tokens read | Agent cost | Extraction | Total | Score |
|---|---|---|---|---|---|
| Full history | 20.0M | $2.38 | - | $2.38 | 25/25 |
| Compaction | 6.4M | $1.46 | - | $1.46 | 25/25 |
| Vector (hybrid) | 4.8M | $1.24 | $0.31 | $1.55 | 20/25 |
| SQLite | 4.4M | $1.11 | $0.31 | $1.42 | 25/25 |
| Graph (FalkorDB) | 4.4M | $1.10 | $0.31 | $1.41 | 25/25 |

Graph memory read 4.5x fewer tokens than full history for the same score.
Extraction (the slow loop) is one small LLM call per message, $0.31 for 200
messages, shared by vector, SQLite and graph. The graph's 25 queries took
0.1 s of engine time in total (the fast loop).

Per-turn latency with full history: about 1-2 s on the first turns, a median
of 13 s (max 23 s) after 100 messages. With graph memory the median after
100 messages is 2.4 s. Source: `results/main/ops/full/L200-w1-r1.jsonl`,
`results/main/ops/graph/L200-w1-r1.jsonl`; tables in
`results/main/report/report.md`.

### The write side sets the ceiling

At 400 messages graph, vector and compaction all scored 22/25 and all missed
the same `transitive` chain: extraction saved one service under two names,
which breaks the dependency chain for graph and SQL alike. Graph memory also
loses `verbatim` questions (an exact incident ID from raw text) to setups that
keep the raw text. A graph does not fix bad extraction; it makes good
extraction fast and cheap. Source: `results/main/report/report.md`.

### Graph database vs graph database, 1M edges

Same data, same queries, each engine in Docker one at a time. The query is the
3-hop "who do we page" read. p99 at 20 concurrent writers, three runs:

| Engine | Read p99, idle | Read p99, 20 writers | Delete one user's memory (p99, 1000 users) |
|---|---|---|---|
| FalkorDB 6.0.1 | 0.2-0.6 ms | 0.5-1.0 ms | 0.3-0.5 ms |
| Neo4j 5 Community | 1.1-3.0 ms | 14-36 ms | 150-190 ms |
| Memgraph | 0.3 ms | 15-34 ms | 1.6-2.6 ms |

FalkorDB reads never wait for writes (one write queue per graph, snapshot per
read), so the read tail stays flat under writers. On the same data and queries
Neo4j and Memgraph's tail grows 14-70x once agents write; FalkorDB's does not.
Forgetting a user is one graph delete on FalkorDB, 300-600x
faster than deleting the user's rows out of one shared Neo4j database. Neo4j
and Memgraph Community allow one database, so their per-user test uses a
`user_id` property and index; FalkorDB uses one graph per user. Source: `results/graphs/graphs.json`,
`results/rerun2/graphs/graphs.json`, `results/rerun3/graphs/graphs.json`.

### One graph per user, 10,000 graphs in one FalkorDB

About 400 edges per graph, 8 writer threads handing off ownership on random
tenants, three runs:

| Metric | Result |
|---|---|
| Memory per graph | about 300 KB |
| 3-hop read p50, idle | 0.6-0.7 ms |
| 3-hop read p50, 8 writers | 0.9-1.3 ms |
| Delete one user's graph p50 | 0.2-0.3 ms |

10,000 users fit in one FalkorDB process in about 3 GB, and one production
cluster runs 81,500 graphs this way, one per user. With one graph per user
there is no shared index to filter and nothing to leak. Source:
`results/main/tenants.json`. Rerun with `uv run amb tenants`.

### Recall latency vs graph size (FalkorDB alone)

| Edges | Lookup p50 | 3-hop p50 | 3-hop p99, 20 writers | Blast radius p99, FalkorDB | Blast radius p99, SQLite |
|---|---|---|---|---|---|
| 10k | 0.18 ms | 0.18 ms | 0.61 ms | 3 ms | 15 ms |
| 100k | 0.18 ms | 0.18 ms | 0.62 ms | 18 ms | 170 ms |
| 1M | 0.16 ms | 0.16 ms | 0.58 ms | 61 ms | 702 ms |

Read latency does not move between 10k and 1M edges. The variable-length
"what breaks if X goes down" query is where a graph pulls away from SQL: at 1M
edges SQLite's recursive CTE is 11x slower, and it runs in-process with no
network hop. Source: `results/scale/scale.json`, table in
`results/report/report.md`.

## Choosing or switching

Four rules for memory that lasts, and what each one needs from the store:

1. **Store what changed, not what was said.** Closed facts beat similar text.
   Emma, not Tom. GraphRAG-Bench: graph 71.5, vector 55.4.
2. **Reads must never wait for writes.** Snapshot per read, one write queue per
   graph. Reads stay 0.5-1 ms with 20 agents writing.
3. **One user, one memory. Not one filter.** Nothing to filter, nothing to
   leak. Delete a user: drop one graph, 0.3 ms.
4. **Pay for facts, not history.** The agent reads the facts, not the whole
   history. 4.4M tokens instead of 20M, caching on for both.

A graph does all four. Keep your memory layer (Cognee, Graphiti, Mem0,
LangChain, LlamaIndex all work with FalkorDB). Change the store under it.

```bash
docker run -p 6379:6379 -p 3000:3000 -it --rm falkordb/falkordb:latest
```

## The task

A seeded fake company: services, teams, people and on-call rotations. The
agent reads an inbox one item at a time (`read_inbox`). Items are Slack
messages, tickets and emails, each with 15-25 lines of attached logs, and facts
change over time (ownership transfers, on-call handoffs, dependencies added and
removed). Facts only ever arrive as free text, often by nickname ("the
payments service", "the Falcon folks"), so any structured memory has to
extract them first.

Five times per task the inbox asks questions, one of each type:

| Type | Example | What it tests |
|---|---|---|
| lookup | Which team currently owns eta-db? | one fact |
| multihop | X is down; page the on-call for every team that owns one of X's dependencies | 3 hops |
| transitive | If X goes down, which services break, directly or through a chain? | variable-length paths |
| changed | Who is the primary on-call for Team Harbor right now? | a fact that was replaced |
| verbatim | What was the incident ID for the TLS failures on upload-api? | an exact detail from raw text |

`verbatim` is there on purpose: it is the kind of question raw text answers
better than extracted facts. The generator knows the true state at every step,
so grading is exact (names match through an alias map).

## The six setups

Model, prompts, inbox tools and answer tool are identical. Only memory changes.
The harness runs on the OpenAI Responses API (default, `gpt-6.1-sol`) or the
Anthropic Messages API (any `claude-*` model).

| Setup | What the agent has |
|---|---|
| `full` | Everything stays in context. Prompt caching on. |
| `compact` | Server-side compaction at a 50k-token threshold (native on both APIs). |
| `memtool` | A memory directory tool + clearing of old inbox items from context. Native on Anthropic (`memory_20250818`, `clear_tool_uses_20250919`); on OpenAI the harness implements both with the same commands and settings. |
| `vector` | Hybrid BM25 + dense search (bge-small), RRF fusion, cross-encoder rerank, over raw messages and extracted facts. Same context editing. |
| `sqlite` | Read-only SQL over the extracted facts, recursive CTEs allowed. Same context editing. |
| `graph` | Read-only Cypher over the same facts in FalkorDB 6.0.1. Same context editing. |

`vector`, `sqlite` and `graph` replay identical extracted facts: extraction
runs once per world and is cached, so the comparison isolates how memory is
queried, not what it holds.

## Fairness rules

- Prompt caching is on for every setup, including full history.
- Baselines use the vendor's native features where they exist: compaction on
  both APIs; the memory tool and tool-result clearing on Anthropic. OpenAI has
  no native equivalent of the last two, so on OpenAI they are implemented in
  the harness (`amb/openai_agent.py`) with the same settings.
- Extraction cost (the cold path) is charged in full to each store setup and
  reported separately from the agent's own cost (the hot path).
- Every run logs every API call, tool call and answer, with token counts by type
  (uncached, cache write, cache read, output), latency, and context edits applied.
- Unanswered questions count as wrong. Unfinished runs are reported.
- Confidence intervals are 95% cluster bootstrap over worlds.

## Setup

```bash
docker compose up -d          # FalkorDB 6.0.1 on :6379
uv sync
# Keys go in .env (gitignored): OPENAI_API_KEY=... and/or ANTHROPIC_API_KEY=...
uv run pytest -q
```

## Running

```bash
# Offline dry run with a scripted fake model, to check the plumbing. Costs nothing.
uv run amb ops --fake --extract oracle --lengths 50 --worlds 1 --out scratch/dry

# Pilot: one world, shortest length, all six setups.
uv run amb ops --lengths 50 --worlds 1

# Full matrix: 3 lengths x 5 worlds x 6 setups = 90 runs.
uv run amb ops --lengths 50,100,200 --worlds 1-5 --jobs 6

# LongMemEval outside check: 5 questions per type x 4 types x 6 setups.
uv run amb lme --n-per-type 5

# Recall latency vs graph size and concurrent writers. No LLM.
uv run amb scale --sizes 10000,100000,1000000 --with-sqlite

# One memory graph per user: 10,000 graphs in one FalkorDB, read latency idle and
# under 8 writers, time to delete one user's graph. No LLM.
uv run amb tenants --graphs 10000 --runs 3

# A second session after the inbox: what survives in each memory setup.
# Needs an `amb ops --handoff` run first.
uv run amb next --lengths 200 --worlds 1

# Graph database vs graph database (and SQLite) on agent-memory queries. No LLM.
# Starts each engine in Docker one at a time: FalkorDB 6.0.1, Neo4j 5 Community, Memgraph.
uv run amb graphs --engines falkordb,neo4j,memgraph,sqlite --edges 1000000 \
  --writers 0,5,10,20 --users 1000 --edges-per-user 2000

# Tables (results/report/report.md) and slide charts (PNG).
uv run amb analyze
```

Runs that already have a log are skipped, so an interrupted matrix resumes
where it stopped. Useful flags: `--model` (default `gpt-6.1-sol`),
`--extract-model`, `--effort`, `--repeats`, `--conditions full,graph`.
`--extract oracle` feeds the stores perfect facts from the generator; use it
only as an ablation to separate extraction errors from query errors.

## Caveats

- The synthetic world is shaped like a graph (dependencies, ownership chains).
  The LongMemEval run and the `verbatim` questions are the checks against that bias.
- LongMemEval answers are graded by a Claude model with the benchmark's own
  prompts. The paper used GPT-4o, so scores are comparable across setups here,
  not to published leaderboards.
- In the scale and graphs tests, SQLite runs in-process (WAL mode) and the graph
  databases run as servers, so SQLite has no network hop. Even so it loses the
  multi-hop queries at 1M edges by 11x; it is ahead only on single-row lookups.
- In the graphs test, Neo4j and Memgraph Community allow one database, so the
  per-user test stores all users in one database with a `user_id` property and
  index. FalkorDB stores one graph per user. Each is the natural setup for that engine.
- Results in `results/` come from an Apple M5 laptop. Expect different
  absolute numbers on your machine; compare engines, not milliseconds.
- OpenAI's `input_tokens` is treated as including cached and cache-written
  tokens; the logs keep the raw split so this can be checked.
- Refusal fallbacks are off: a fallback would switch models mid-run and
  contaminate the comparison. Refusals are logged.
