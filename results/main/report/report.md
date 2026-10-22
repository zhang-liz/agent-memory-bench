# agent-memory-bench results

Model(s): gpt-6.1-sol

## Synthetic on-call world

### Accuracy and cost per task

| Length | Setup | Runs | Exact-match accuracy | Agent cost (hot path) | Extraction cost (cold path) | Total cost | API calls | Unfinished |
|---|---|---|---|---|---|---|---|---|
| 200 | Full history | 1 | 100% | $2.379 | $0.000 | $2.379 | 252 | 0 |
| 200 | Compaction | 1 | 100% | $1.456 | $0.000 | $1.456 | 252 | 0 |
| 200 | Vector (hybrid) | 1 | 80% | $1.240 | $0.310 | $1.549 | 298 | 0 |
| 200 | SQLite | 1 | 100% | $1.107 | $0.310 | $1.417 | 274 | 0 |
| 200 | Graph (FalkorDB) | 1 | 100% | $1.102 | $0.310 | $1.411 | 275 | 0 |
| 400 | Compaction | 1 | 88% | $2.899 | $0.000 | $2.899 | 452 | 0 |
| 400 | Vector (hybrid) | 1 | 88% | $2.725 | $0.672 | $3.398 | 503 | 0 |
| 400 | SQLite | 1 | 80% | $2.529 | $0.672 | $3.201 | 481 | 0 |
| 400 | Graph (FalkorDB) | 1 | 88% | $2.549 | $0.672 | $3.221 | 481 | 0 |

Unanswered questions count as wrong. Extraction is shared by the vector, SQLite and graph setups (same facts, extracted once per world), and its full cost is charged to each of them.

### Accuracy by question type

| Length | Setup | lookup | multihop | transitive | changed | verbatim |
|---|---|---|---|---|---|---|
| 200 | Full history | 100% | 100% | 100% | 100% | 100% |
| 200 | Compaction | 100% | 100% | 100% | 100% | 100% |
| 200 | Vector (hybrid) | 100% | 60% | 60% | 80% | 100% |
| 200 | SQLite | 100% | 100% | 100% | 100% | 100% |
| 200 | Graph (FalkorDB) | 100% | 100% | 100% | 100% | 100% |
| 400 | Compaction | 100% | 100% | 60% | 100% | 80% |
| 400 | Vector (hybrid) | 100% | 100% | 60% | 100% | 80% |
| 400 | SQLite | 100% | 100% | 60% | 100% | 40% |
| 400 | Graph (FalkorDB) | 100% | 100% | 60% | 100% | 80% |

### Where the time goes

| Length | Setup | LLM time per task (s) | Memory tool time per task (s) | Memory calls | Median engine ms per memory call | Memory query errors |
|---|---|---|---|---|---|---|
| 200 | Full history | 4731 | 0.0 | 0 | nan | 0.0 |
| 200 | Compaction | 2675 | 0.0 | 0 | nan | 0.0 |
| 200 | Vector (hybrid) | 818 | 20.0 | 46 | 160.78 | 0.0 |
| 200 | SQLite | 583 | 0.0 | 22 | 0.52 | 0.0 |
| 200 | Graph (FalkorDB) | 731 | 0.1 | 23 | 1.56 | 0.0 |
| 400 | Compaction | 1445 | 0.0 | 0 | nan | 0.0 |
| 400 | Vector (hybrid) | 3513 | 27.9 | 51 | 181.85 | 0.0 |
| 400 | SQLite | 3603 | 0.0 | 29 | 0.34 | 0.0 |
| 400 | Graph (FalkorDB) | 3702 | 0.2 | 29 | 1.19 | 0.0 |

Charts: chart_context_per_call.png, chart_cost_per_task.png, chart_accuracy_by_type.png, chart_time_split.png
