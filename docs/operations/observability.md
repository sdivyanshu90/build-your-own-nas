# Observability

The structured event vocabulary, the fields every event carries, counters, and what is worth
alerting on.

## Why structured logging

A NAS run emits thousands of log lines across candidates, trials, and workers. Free-text
lines force downstream consumers to write brittle regexes. Structured events carry typed
key/value pairs, so `search_id`, `candidate_id`, `architecture_hash`, and
`duration_seconds` can be filtered and aggregated directly — and the same event stream
renders either as human-friendly console output or as newline-delimited JSON.

```bash
nas-engine search --config configs/my.yaml --set logging.format=console   # human
nas-engine search --config configs/my.yaml --set logging.format=json      # machine
```

```json
{"architecture_hash": "385cfb98d1da7ad7b5c1e202c730221f",
 "candidate_id": "7173d605bdf0413eb17d3581c621e8d9",
 "duration_seconds": 3.465662546999738,
 "event": "evaluation.completed", "level": "info",
 "objective_value": 0.2604166666666667,
 "search_id": "13131ab2c83a46b5b765f7f9e2534a10", "strategy": "random_search",
 "timestamp": "2026-07-27T09:00:18.555240Z",
 "trial_id": "b60a9a575f2a4030817f9ca68344d7de",
 "validation_accuracy": 0.2604166666666667, "worker_id": "main"}
```

Keys are sorted, which makes lines diffable.

## The event vocabulary

Event names are a **public interface** — dashboards and log queries are built on them — so
they are declared once as a closed enumeration rather than typed as free strings at call
sites. Adding one means adding a member; renaming one is a breaking change.

### Search lifecycle

| Event                | Level     | Emitted when          | Key fields                                                              |
| -------------------- | --------- | --------------------- | ----------------------------------------------------------------------- |
| `search.started`     | info      | A new search begins   | `strategy`, `max_evaluations`, `device`, `concurrency`, `seed`          |
| `search.resumed`     | info      | A search resumes      | `completed`, `requeued`                                                 |
| `search.completed`   | info      | A search finishes     | `stop_reason`, `completed`, `failed`, `duration_seconds`, `pareto_size` |
| `search.interrupted` | warning   | Ctrl-C                |                                                                         |
| `search.failed`      | **error** | An engine-level error | `error`, `error_type`                                                   |

### Candidate lifecycle

| Event                 | Level       | Emitted when              | Key fields                                        |
| --------------------- | ----------- | ------------------------- | ------------------------------------------------- |
| `candidate.proposed`  | info        | The strategy suggests one | `architecture_hash`, `rung`, `origin`, `mutation` |
| `candidate.duplicate` | info        | Rejected as already seen  | `existing_candidate_id`, `existing_status`        |
| `candidate.rejected`  | **warning** | Failed validation         | `reason`                                          |
| `candidate.pruned`    | info        | Exceeded a constraint     | `reason`                                          |
| `candidate.queued`    | info        | Accepted and queued       | `budget`                                          |
| `candidate.promoted`  | info        | Promoted to a higher rung |                                                   |
| `candidate.cancelled` | info        | Cancelled                 |                                                   |

### Evaluation

| Event                  | Level       | Emitted when             | Key fields                                                                |
| ---------------------- | ----------- | ------------------------ | ------------------------------------------------------------------------- |
| `evaluation.started`   | info        | An attempt is dispatched | `attempt`, `budget`                                                       |
| `evaluation.completed` | info        | An attempt succeeds      | `validation_accuracy`, `objective_value`, `duration_seconds`, `worker_id` |
| `evaluation.failed`    | **warning** | An attempt fails         | `failure_kind`, `error`                                                   |
| `evaluation.timeout`   | **warning** | An attempt times out     | `failure_kind`, `error`                                                   |

`evaluation.started` has no `worker_id`: it is emitted when the task is built, before any
worker has claimed it. The worker is known by the time the attempt completes.

### Search state

| Event                 | Level       | Emitted when                        | Key fields                            |
| --------------------- | ----------- | ----------------------------------- | ------------------------------------- |
| `population.updated`  | info        | The evolutionary population changes | `population_size`, `generation`       |
| `checkpoint.saved`    | info        | A checkpoint is written             | `sequence`                            |
| `checkpoint.restored` | info        | A checkpoint is loaded              | `created_at`, `proposed`, `completed` |
| `retry.scheduled`     | **warning** | A retry is queued                   | `attempt`, `delay_seconds`, `reason`  |
| `retry.exhausted`     | **error**   | Retries run out                     | `attempts`                            |
| `pareto.updated`      | info        | The front is recomputed             | `size`, `members`                     |
| `report.generated`    | info        | A report is written                 | `markdown`, `candidates`, `plots`     |

Severity is deliberate: **warning** means work was lost, **error** means the search or a
candidate is permanently damaged.

### Module diagnostics are a separate namespace

Alongside the enumeration, individual modules log their own diagnostics under a
`<module>.<thing>` name. These are **not** part of the `Event` vocabulary and may change:

| Name                                                             | Emitted by                            |
| ---------------------------------------------------------------- | ------------------------------------- |
| `engine.dataset_ready`                                           | The engine, once the dataset is built |
| `database.migrating`                                             | A schema migration                    |
| `repository.search_created`                                      | The repository, on insert             |
| `evaluator.started` / `evaluator.completed` / `evaluator.failed` | Inside the evaluator                  |

The evaluator's three are worth understanding. They report what an attempt did *from inside
the worker process*, where no search context is bound — `device`, `seed`,
`trainable_parameters`, and the evaluator's own measured duration. The engine separately
emits the corresponding `Event` for the same attempt.

They deliberately do **not** reuse the `evaluation.*` names. If they did, every attempt
would appear twice under one name with two different field sets, and anything counting
evaluations would silently report double. That is not hypothetical — it was a real defect,
and `tests/regression/test_event_vocabulary.py` now fails if any module logs a raw string
the `Event` enum already owns.

So: **count with `evaluation.*`, diagnose with `evaluator.*`.**

## Fields on every event

Ambient context is attached automatically, so no call site has to remember:

| Field               | Bound by            |
| ------------------- | ------------------- |
| `search_id`         | `search_context`    |
| `strategy`          | `search_context`    |
| `candidate_id`      | `candidate_context` |
| `architecture_hash` | `candidate_context` |
| `trial_id`          | `candidate_context` |
| `worker_id`         | `worker_context`    |

Plus `event`, `level`, and an ISO-8601 UTC `timestamp` on every line.

Under multiprocessing, interleaved output is unreadable without these. With them, filtering
is a `jq` away.

## Redaction

A processor walks every event dictionary and replaces the value of any key whose name
matches a sensitive fragment (`password`, `token`, `secret`, `api_key`, `auth`,
`credential`, `passwd`, `private_key`). Matching is case-insensitive and substring-based, so
`HF_API_TOKEN` and `db_password` are both caught. Nested mappings and sequences are
traversed, with a depth cap.

Defence in depth: the framework never logs credentials, but user configuration is untrusted.

## Useful queries

```bash
nas-engine search --config configs/my.yaml --set logging.format=json 2> run.jsonl
```

```bash
# accuracy of every completed evaluation
jq -c 'select(.event=="evaluation.completed") | {h:.architecture_hash[0:8], a:.validation_accuracy}' run.jsonl

# what failed, and why
jq -c 'select(.event=="evaluation.failed") | {h:.architecture_hash[0:8], k:.failure_kind, e:.error}' run.jsonl

# failure counts by kind
jq -r 'select(.event=="evaluation.failed") | .failure_kind' run.jsonl | sort | uniq -c

# one worker's timeline
jq -c 'select(.worker_id=="2") | {t:.timestamp, e:.event}' run.jsonl

# slowest evaluations
jq -r 'select(.event=="evaluation.completed") | "\(.duration_seconds)\t\(.architecture_hash[0:8])"' run.jsonl \
  | sort -rn | head -10

# size against accuracy, from the evaluator's inner line
jq -r 'select(.event=="evaluator.completed")
       | "\(.trainable_parameters)\t\(.validation_accuracy)"' run.jsonl

# running best
jq -r 'select(.event=="evaluation.completed") | .validation_accuracy' run.jsonl \
  | awk 'BEGIN{m=0} {if($1>m) m=$1; print NR"\t"$1"\t"m}'
```

## Counters

`CounterRegistry` provides monotonic counters (`increment`), last-value gauges
(`set_gauge`), and duration observations (`observe_duration`) that aggregate to count, sum,
min, max, and mean.

| Counter                 | Incremented when                              |
| ----------------------- | --------------------------------------------- |
| `candidates.accepted`   | A proposal becomes a queued candidate         |
| `candidates.duplicate`  | A proposal is rejected as a duplicate         |
| `candidates.invalid`    | A proposal fails validation                   |
| `candidates.pruned`     | A proposal violates a search-space constraint |
| `evaluations.completed` | An evaluation succeeds                        |
| `evaluations.failed`    | A candidate fails permanently                 |
| `evaluations.retried`   | A retry is scheduled                          |

| Observation  | Recorded                           |
| ------------ | ---------------------------------- |
| `evaluation` | Per-evaluation wall-clock duration |

The registry is thread-safe and deliberately **not** process-shared: each worker keeps its
own and returns a snapshot, which the parent merges via `MetricsSnapshot.merge`. That avoids
shared-memory complexity and its associated bugs.

### Reading the counters

The same numbers are on the result, as `EngineState` — the public path, and the one that
survives a resume because it is checkpointed:

```python
result = engine.run()
print(result.engine_state.to_dict())
# {'proposed': 47, 'accepted': 40, 'duplicates': 5, 'invalid': 0, 'pruned': 2,
#  'completed': 38, 'failed': 2, 'retried': 3, 'elapsed_seconds': 2471.4}
```

```bash
nas-engine search --config configs/my.yaml --json | jq '.engine_state'
```

### Exporting to a metrics system

Nothing here depends on one. To bridge:

```python
for name, value in result.engine_state.to_dict().items():
    if isinstance(value, int):
        PROM_COUNTERS[name].inc(value)
```

For a live feed rather than an end-of-run total, scrape the JSON event stream — one line per
`evaluation.completed` carries the accuracy and the duration, which is usually all a
dashboard needs.

## What is in the database

Structured logs go to stderr and are ephemeral. A **deliberately narrower** set of events is
also written to the `search_events` table — only the ones that explain a *candidate's* fate
after the fact, which is what you need when reading a result months later:

| Persisted event       | Payload          | Written by                              |
| --------------------- | ---------------- | --------------------------------------- |
| `candidate.claimed`   | `worker_id`      | A worker taking a queued candidate      |
| `candidate.pruned`    | `from`, `reason` | A terminal transition carrying a reason |
| `candidate.failed`    | `from`, `reason` | same                                    |
| `candidate.cancelled` | `from`, `reason` | same                                    |
| `candidate.recovered` | `new_status`     | The recovery sweep after a crash        |

Search-level events (`search.started`, `search.completed`, and the rest) are **not** in this
table — the `searches` row already carries `status`, `created_at`, `started_at`,
`completed_at`, `seed`, `config_hash`, and `environment_json`, which is the same information
in queryable form. Duplicating it as event rows would just be a second thing to keep
consistent.

The table is small by design: a handful of rows per candidate, never one per training step.

```bash
nas-engine status --json | jq '.counts'
```

```sql
SELECT event, COUNT(*) FROM search_events WHERE search_id = '…' GROUP BY event;

SELECT created_at, event, candidate_id, payload_json
FROM search_events WHERE search_id = '…' ORDER BY created_at DESC LIMIT 20;

-- why candidates were pruned
SELECT json_extract(payload_json, '$.reason') AS reason, COUNT(*)
FROM search_events WHERE search_id = '…' AND event = 'candidate.pruned'
GROUP BY reason ORDER BY 2 DESC;
```

`Repository.record_event` is public, so an application embedding the engine can append its
own audit rows to the same table.

## Monitoring a running search

```bash
watch -n 30 'nas-engine status --config configs/my.yaml'
```

Or programmatically:

```python
counts = repository.count_candidates_by_status(search_id)
completed = counts["completed"]
progress = completed / summary.planned_evaluations
```

## What to alert on

| Condition                                              | Severity      | Meaning                                   |
| ------------------------------------------------------ | ------------- | ----------------------------------------- |
| `search.failed`                                        | page          | An engine-level error; the search stopped |
| `retry.exhausted` rate > 10%                           | investigate   | Systematic failures, not bad luck         |
| `evaluation.failed` rate > 20%                         | investigate   | The recipe or the space is wrong          |
| No `evaluation.completed` for 3× the expected duration | investigate   | Stalled                                   |
| `candidate.duplicate` rate > 50%                       | informational | The space is nearly exhausted             |
| `candidate.pruned` rate > 50%                          | informational | A constraint is too tight                 |
| Disk usage above 80%                                   | investigate   | Artifacts accumulating                    |

The two "informational" rows are worth watching but are not failures — they mean the search
is doing something you should know about, not that something broke.

## Log levels

| Level     | Use                                                                  |
| --------- | -------------------------------------------------------------------- |
| `DEBUG`   | Per-epoch metrics, per-step logs, strategy internals. Diagnosis only |
| `INFO`    | The default. Lifecycle events                                        |
| `WARNING` | Failures, retries, recoveries, degraded fallbacks                    |
| `ERROR`   | Engine-level errors, retry exhaustion                                |

`DEBUG` on a long search produces a great deal of output. Use it on a short reproduction.

## See also

- [Production runbook](production-runbook.md) — acting on what you observe.
- [Troubleshooting](../guides/troubleshooting.md) — every error explained.
- [Concurrency](../architecture/concurrency.md) — why worker ids matter.
