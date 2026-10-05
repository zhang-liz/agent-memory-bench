# agent-memory-bench

> Disclosure: written by Liz Zhang, DevRel Lead at FalkorDB. Everything here is
> meant to be rerun by you: same data, same queries, every engine in its best
> setup. If you find a setup that is unfair to any engine, open an issue.

Does moving an agent's memory out of the context window and into a graph cut
its token bill without hurting accuracy? This repo runs a fair test of that
claim against five baselines, on a synthetic long-horizon task and on a public
benchmark, and reports where each memory setup wins and where it loses.

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
  databases run as servers, so SQLite has no network hop. It wins on simple
  lookups and under concurrent writes; it loses on multi-hop queries at 1M edges.
- In the graphs test, Neo4j and Memgraph Community allow one database, so the
  per-user test stores all users in one database with a `user_id` property and
  index. FalkorDB stores one graph per user. Each is the natural setup for that engine.
- Results in `results/` come from an Apple M5 laptop, three runs each. Expect
  different absolute numbers on your machine; compare engines, not milliseconds.
- OpenAI's `input_tokens` is treated as including cached and cache-written
  tokens; the logs keep the raw split so this can be checked.
- Refusal fallbacks are off: a fallback would switch models mid-run and
  contaminate the comparison. Refusals are logged.
