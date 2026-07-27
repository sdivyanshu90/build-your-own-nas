# System overview

How the pieces fit together.

## The shape of the system

```mermaid
flowchart TB
    subgraph interface["Interfaces"]
        CLI["CLI<br/><i>cli.py</i>"]
        API["Python API<br/><i>__init__.py</i>"]
    end

    subgraph application["Application"]
        CONFIG["Configuration<br/><i>config/</i>"]
        ENGINE["Search engine<br/><i>orchestration/</i>"]
        REPORT["Reporting<br/><i>reporting/</i>"]
    end

    subgraph domain["Domain"]
        SPACE["Search space<br/><i>search_space/</i>"]
        ARCH["Architectures<br/><i>architectures/</i>"]
        STRAT["Strategies<br/><i>search/</i>"]
        OBJ["Objectives<br/><i>objectives/</i>"]
    end

    subgraph execution["Execution"]
        EVAL["Evaluation<br/><i>evaluation/</i>"]
        TRAIN["Training<br/><i>training/</i>"]
        MODELS["Models<br/><i>models/</i>"]
        DATA["Datasets<br/><i>datasets/</i>"]
    end

    subgraph infrastructure["Infrastructure"]
        DB["Persistence<br/><i>persistence/</i>"]
        OBS["Observability<br/><i>observability/</i>"]
        UTIL["Utilities<br/><i>utilities/</i>"]
    end

    CLI --> CONFIG
    API --> CONFIG
    CLI --> ENGINE
    API --> ENGINE
    CONFIG --> ENGINE
    ENGINE --> STRAT
    ENGINE --> EVAL
    ENGINE --> DB
    ENGINE --> OBJ
    STRAT --> SPACE
    SPACE --> ARCH
    EVAL --> MODELS
    EVAL --> TRAIN
    TRAIN --> DATA
    MODELS --> ARCH
    REPORT --> DB
    REPORT --> OBJ
    ENGINE -.-> OBS
    EVAL -.-> OBS
```

Arrows point from user to used. Nothing in the domain layer knows about the engine, the
database, or the CLI — which is what makes each of them testable in isolation.

## The domain model

Six nouns, and the relationships between them.

```mermaid
erDiagram
    SEARCH ||--o{ CANDIDATE : proposes
    CANDIDATE ||--o{ TRIAL : "is evaluated by"
    TRIAL ||--o{ METRIC : produces
    CANDIDATE ||--o{ ARTIFACT : produces
    SEARCH ||--o{ CHECKPOINT : saves
    CANDIDATE ||--o| CANDIDATE : "mutated from"

    SEARCH {
        string id PK
        string strategy
        string status
        int seed
        string config_hash
        json environment
    }
    CANDIDATE {
        string id PK
        string architecture_hash
        int rung
        json spec
        string status
        string parent_id FK
        string mutation
        float objective_value
        int retry_count
    }
    TRIAL {
        string id PK
        int attempt
        json budget
        string status
        float duration_seconds
        string worker_id
    }
    METRIC {
        string name
        float value
    }
    ARTIFACT {
        string kind
        string path
        int size_bytes
    }
    CHECKPOINT {
        int sequence
        json payload
    }
```

| Concept        | Definition                                                               |
| -------------- | ------------------------------------------------------------------------ |
| **Search**     | One run: a configuration, a seed, a strategy, and everything it produced |
| **Candidate**  | One architecture proposed within a search, at one fidelity rung          |
| **Trial**      | One evaluation attempt for a candidate. Retries create additional trials |
| **Metric**     | One measured scalar from one trial                                       |
| **Artifact**   | A file a trial produced — weights, a training checkpoint                 |
| **Checkpoint** | A snapshot of the strategy's state, so the search can resume             |

A candidate is identified within its search by `(architecture_hash, rung)`. The rung is
part of the key because multi-fidelity search deliberately re-evaluates the same
architecture at a larger budget, and those are genuinely different measurements.

## The search lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant E as SearchEngine
    participant S as Strategy
    participant R as Repository
    participant X as Executor
    participant V as Evaluator

    U->>E: run()
    E->>R: create_search(config, seed, environment)
    E->>S: propose(free_slots)
    S-->>E: [Proposal, …]

    loop for each proposal
        E->>E: hash → deduplicate → validate
        alt duplicate or invalid
            E->>R: record as DUPLICATE / FAILED / PRUNED
            E->>S: on_duplicate / on_rejected
        else accepted
            E->>R: add_candidate → VALIDATED → QUEUED
        end
    end

    E->>R: claim_next_queued() → RUNNING
    E->>R: start_trial()
    E->>X: run_batch(tasks)
    X->>V: evaluate(spec, budget, context)
    V-->>X: EvaluationResult
    X-->>E: [EvaluationResult, …]

    loop for each result
        alt succeeded
            E->>R: complete_trial(metrics, artifacts)
            E->>R: candidate → COMPLETED
        else retriable failure
            E->>R: fail_trial(error); candidate → QUEUED
        else permanent failure
            E->>R: fail_trial(error); candidate → FAILED
        end
        E->>S: observe(observation)
    end

    E->>R: save_checkpoint(strategy state + engine counters)
    E->>E: check stopping conditions
    E->>R: update_search_status(COMPLETED)
    E-->>U: SearchResult
```

Two things to notice.

**The engine, not the strategy, owns identity.** Hashing, duplicate detection, and
persistence live in the engine. A strategy that had to remember what it had already
proposed across a resume would need its own database, and every new strategy would
reimplement it.

**Failure is data, not control flow.** The evaluator never raises for a candidate-level
problem; it returns a failed result. The engine therefore has exactly one path through the
loop, and a failing candidate cannot abort the search.

## The candidate state machine

```mermaid
stateDiagram-v2
    [*] --> PROPOSED
    PROPOSED --> VALIDATED : passes validation
    PROPOSED --> PRUNED : exceeds a constraint
    PROPOSED --> FAILED : structurally invalid
    VALIDATED --> QUEUED : enqueued
    VALIDATED --> PRUNED : pruned by the strategy
    QUEUED --> RUNNING : claimed by a worker
    QUEUED --> FAILED : abandoned
    RUNNING --> COMPLETED : evaluation succeeded
    RUNNING --> QUEUED : retriable failure
    RUNNING --> FAILED : permanent failure or retries exhausted
    RUNNING --> PRUNED : eliminated by a rung

    PROPOSED --> CANCELLED
    VALIDATED --> CANCELLED
    QUEUED --> CANCELLED
    RUNNING --> CANCELLED

    COMPLETED --> [*]
    FAILED --> [*]
    PRUNED --> [*]
    CANCELLED --> [*]
```

Every transition is validated against a table that lives in exactly one place,
[`orchestration/lifecycle.py`](../../src/nas_engine/orchestration/lifecycle.py), so the
diagram and the code cannot drift apart. A golden fixture pins the table:
changing an edge is a deliberate act, not an accident.

**Why the states are distinct** is the recovery story. When a search is interrupted, the
database is the only record of what happened:

| State found on resume | Interpretation                | Action                                    |
| --------------------- | ----------------------------- | ----------------------------------------- |
| `QUEUED`              | Never started                 | Run it                                    |
| `RUNNING`             | A process died mid-evaluation | Requeue, or fail if retries are exhausted |
| `COMPLETED`           | Finished and recorded         | Leave alone                               |
| `FAILED`              | Permanently failed            | Leave alone                               |
| `PRUNED`              | Rejected by a constraint      | Leave alone                               |

Without distinct states these cases are indistinguishable, and resume either loses work or
repeats it.

`PRUNED` versus `FAILED` matters too: a pruned candidate is one where *nothing went wrong*
— a constraint did its job. Conflating them would make every report look like it had a
failure problem.

## Random-search flow

```mermaid
flowchart TD
    START([propose]) --> BUDGET{"budget<br/>remaining?"}
    BUDGET -->|no| EMPTY["return []"]
    BUDGET -->|yes| SAMPLE["sample from the space"]
    SAMPLE --> VALID{"valid?"}
    VALID -->|no| RETRY1{"attempts<br/>left?"}
    RETRY1 -->|yes| SAMPLE
    RETRY1 -->|no| FAIL["SearchSpaceError<br/>with rejection reasons"]
    VALID -->|yes| NOVEL{"hash<br/>seen?"}
    NOVEL -->|yes| RETRY2{"attempts<br/>left?"}
    RETRY2 -->|yes| SAMPLE
    RETRY2 -->|no| EXHAUST["mark exhausted"]
    NOVEL -->|no| EMIT["emit a Proposal"]
```

## Evolutionary-search flow

```mermaid
flowchart TD
    START([propose]) --> INIT{"population<br/>full?"}
    INIT -->|no| RANDOM["sample randomly"]
    RANDOM --> EMIT
    INIT -->|yes| EMPTY{"population<br/>empty?<br/><i>everything failed</i>"}
    EMPTY -->|yes| WARN["log and fall back<br/>to random sampling"]
    WARN --> RANDOM
    EMPTY -->|no| TOURN["tournament: sample S,<br/>take the best"]
    TOURN --> MUTATE["apply one mutation"]
    MUTATE --> REPAIR["repair global invariants"]
    REPAIR --> CHANGED{"different<br/>from the parent?"}
    CHANGED -->|no| RETRY{"attempts<br/>left?"}
    RETRY -->|yes| MUTATE
    RETRY -->|no| RANDOM
    CHANGED -->|yes| DUP{"already<br/>seen?"}
    DUP -->|yes| RETRY
    DUP -->|no| EMIT["emit a Proposal<br/>with parent and mutation"]

    OBS([observe]) --> OK{"succeeded<br/>and scored?"}
    OK -->|no| SKIP["skip: no fitness,<br/>no population entry"]
    OK -->|yes| APPEND["append to the population"]
    APPEND --> AGE["deque(maxlen=P) evicts<br/>the OLDEST member"]
```

## Successive-halving flow

```mermaid
flowchart TD
    START([propose]) --> RUNG{"current rung<br/>fully proposed?"}
    RUNG -->|no| WHICH{"rung 0?"}
    WHICH -->|yes| SAMPLE["sample a new architecture"]
    WHICH -->|no| PROMOTE["take the next survivor<br/>from the previous rung"]
    SAMPLE --> EMIT["emit at this rung's budget"]
    PROMOTE --> EMIT
    RUNG -->|yes| OUT{"results still<br/>outstanding?"}
    OUT -->|yes| WAIT["return [] — barrier"]
    OUT -->|no| LAST{"last rung?"}
    LAST -->|yes| DONE["finished"]
    LAST -->|no| SORT["sort by objective,<br/>keep the top n/η"]
    SORT --> SURV{"any<br/>survivors?"}
    SURV -->|no| EXHAUST["no survivors: stop"]
    SURV -->|yes| ADVANCE["advance to the next rung"]
    ADVANCE --> RUNG
```

## Training flow

```mermaid
flowchart TD
    FIT([fit]) --> SETUP["build optimiser, scheduler,<br/>gradient scaler, early stopper"]
    SETUP --> RESUME{"checkpoint<br/>exists?"}
    RESUME -->|yes| LOAD["restore weights, optimiser,<br/>scheduler, and counters"]
    RESUME -->|no| FRESH["start from epoch 0"]
    LOAD --> EPOCH
    FRESH --> EPOCH{"epochs<br/>remaining?"}

    EPOCH -->|yes| TRAIN["train one epoch"]
    TRAIN --> BATCH["forward → loss"]
    BATCH --> FINITE{"loss<br/>finite?"}
    FINITE -->|no| DIVERGE["NonFiniteLossError<br/><i>permanent, never retried</i>"]
    FINITE -->|yes| BACK["backward → clip → step → schedule"]
    BACK --> DEADLINE{"past the<br/>deadline?"}
    DEADLINE -->|yes| TIMEOUT["EvaluationTimeoutError<br/><i>retriable</i>"]
    DEADLINE -->|no| MORE{"more<br/>batches?"}
    MORE -->|yes| BATCH
    MORE -->|no| VAL["validate"]
    VAL --> IMPROVED{"improved?"}
    IMPROVED -->|yes| SNAP["snapshot the best weights"]
    IMPROVED -->|no| PATIENCE{"patience<br/>exhausted?"}
    PATIENCE -->|yes| STOP["stop early"]
    PATIENCE -->|no| EPOCH
    SNAP --> EPOCH

    EPOCH -->|no| RESTORE["restore the best weights"]
    STOP --> RESTORE
    RESTORE --> SAVE["write the final checkpoint"]
    SAVE --> OUTCOME([TrainingOutcome])
```

## Persistence flow

```mermaid
flowchart LR
    subgraph engine["Engine"]
        OP["a domain operation"]
    end
    subgraph repo["SearchRepository"]
        M["one method = one transaction"]
    end
    subgraph db["Database"]
        S["session()"]
        T[("SQLite<br/>WAL mode")]
    end

    OP --> M
    M --> S
    S -->|success| COMMIT["commit"]
    S -->|exception| ROLLBACK["rollback"]
    COMMIT --> T
    ROLLBACK --> T
    M -->|detached dataclass| OP
```

Every repository method runs inside exactly one transaction and returns a frozen dataclass,
never an ORM instance. Multi-step operations that must be atomic — claiming a queued
candidate, recording a completed trial with its metrics and artifacts — are single methods
for exactly that reason.

## Resume and recovery flow

```mermaid
flowchart TD
    RESUME([resume]) --> FIND{"search id<br/>given?"}
    FIND -->|no| LATEST["find the most recent search<br/>matching the project name"]
    FIND -->|yes| LOAD["load the search record"]
    LATEST --> LOAD
    LOAD --> COMPAT["compare the stored configuration<br/>with the current one"]
    COMPAT --> VER{"config version<br/>supported?"}
    VER -->|no| ABORT["ConfigVersionError"]
    VER -->|yes| SWEEP["recovery sweep"]

    SWEEP --> RUN{"any candidates<br/>in RUNNING?"}
    RUN -->|yes| MARK["mark their trials INTERRUPTED"]
    MARK --> RETRIES{"retries<br/>left?"}
    RETRIES -->|yes| REQUEUE["→ QUEUED, retry_count += 1"]
    RETRIES -->|no| ABANDON["→ FAILED<br/>retry_exhausted_error"]
    RUN -->|no| CKPT
    REQUEUE --> CKPT
    ABANDON --> CKPT

    CKPT{"checkpoint<br/>found?"} -->|no| WARN["warn: the strategy restarts<br/>its plan; duplicates will be skipped"]
    CKPT -->|yes| VALIDATE["validate strategy name<br/>and configuration hash"]
    VALIDATE --> RESTORE["restore strategy state<br/>and engine counters"]
    RESTORE --> RECONCILE["reconcile the completed count<br/>against the database"]
    WARN --> RECONCILE
    RECONCILE --> LOOP["continue the search loop"]
```

The **reconciliation** step is subtle and necessary. Recovery can *undo* a completion: a
candidate that a crashed process had already finished, but whose result was never
persisted, goes back to the queue. The checkpoint's counter still includes it, so without
reconciliation the engine would believe its budget was spent and would leave the recovered
candidate queued forever — silently returning fewer results than requested. The database is
authoritative because it is what the recovery sweep just updated.

## Module dependency graph

```mermaid
flowchart BT
    UTIL["utilities"]
    EXC["exceptions"]
    OBS["observability"]
    ARCH["architectures"]
    SPACE["search_space"]
    MODELS["models"]
    DATA["datasets"]
    TRAIN["training"]
    OBJ["objectives"]
    EVAL["evaluation"]
    SEARCH["search"]
    LIFE["orchestration.lifecycle"]
    PERSIST["persistence"]
    CONFIG["config"]
    ORCH["orchestration"]
    REPORT["reporting"]
    CLI["cli"]

    OBS --> EXC
    OBS --> UTIL
    ARCH --> UTIL
    ARCH --> EXC
    SPACE --> ARCH
    MODELS --> ARCH
    DATA --> UTIL
    TRAIN --> DATA
    OBJ --> EXC
    EVAL --> MODELS
    EVAL --> TRAIN
    SEARCH --> SPACE
    SEARCH --> EVAL
    LIFE --> EXC
    PERSIST --> LIFE
    PERSIST --> ARCH
    CONFIG --> SPACE
    CONFIG --> TRAIN
    CONFIG --> EVAL
    CONFIG --> OBJ
    ORCH --> CONFIG
    ORCH --> SEARCH
    ORCH --> PERSIST
    ORCH --> OBJ
    REPORT --> PERSIST
    REPORT --> OBJ
    CLI --> ORCH
    CLI --> REPORT
```

The graph is acyclic. The one place it could have become cyclic is
`persistence → orchestration.lifecycle`: the candidate state machine is a domain concept
that both persistence and the engine need. It lives in a leaf module that imports nothing
but the exception taxonomy, so importing it from persistence creates no cycle.

That property is enforced by a test, not by convention:
`test_the_domain_does_not_import_the_orchestrator` walks the AST of every module in the
leaf packages and fails if any of them imports from a higher layer.

## Execution modes

```mermaid
flowchart LR
    subgraph seq["Sequential"]
        E1["engine"] --> V1["evaluator"] --> R1["result"]
    end
    subgraph mp["Multiprocessing"]
        E2["engine"] --> POOL["ProcessPoolExecutor"]
        POOL --> W1["worker 1<br/><i>own evaluator</i>"]
        POOL --> W2["worker 2<br/><i>own evaluator</i>"]
        POOL --> W3["worker N"]
        W1 --> R2["results"]
        W2 --> R2
        W3 --> R2
    end
```

Both satisfy the same `EvaluationExecutor` interface, so the engine's logic is identical
either way and can be tested entirely sequentially. See
[concurrency](concurrency.md) for what parallelism does and does not change.

## Where to go next

| To understand                                           | Read                                    |
| ------------------------------------------------------- | --------------------------------------- |
| Each package's job and the public boundary              | [Component design](component-design.md) |
| What moves between components and where it is validated | [Data flow](data-flow.md)               |
| The database schema and the repository pattern          | [Persistence](persistence.md)           |
| Worker isolation and determinism under parallelism      | [Concurrency](concurrency.md)           |
| The trust boundary and the threat model                 | [Security](security.md)                 |
