# ADR 0002 — SQLite via SQLAlchemy ORM, behind a repository, with hand-written migrations

**Status:** Accepted
**Date:** 2026-07-27
**Supersedes:** none

## Context

A search that cannot be interrupted is not usable. Real runs take hours; laptops sleep,
schedulers pre-empt, and processes get killed. Everything the search learns must survive
that, and must still be queryable afterwards — "which architecture won, what did it score,
what was it mutated from, and under what configuration" is the whole point of running one.

Concretely, persistence has to provide:

- **Durability per unit of work.** A completed evaluation must be safe the moment it
  completes, not at the end of the run.
- **Queryability.** Ranking, Pareto fronts, and reports are computed by querying completed
  candidates. Recomputing from a log file does not scale and is easy to get subtly wrong.
- **Identity enforcement.** "One candidate per (search, architecture, rung)" must be a
  constraint the storage layer enforces, not a convention the engine remembers.
- **Concurrent access.** Workers write; `nas-engine status` reads while they do.
- **Zero operational overhead.** Nobody should have to run a database server to try a NAS
  framework.
- **Schema evolution.** The schema will change, and existing databases must not become
  unreadable.

## Decision

**SQLite**, accessed through the **SQLAlchemy ORM**, wrapped in a **repository** that
returns detached frozen dataclasses, with **hand-written sequential migrations**.

### Four pragmas, set on every connection

| Pragma | Value | Why |
| --- | --- | --- |
| `journal_mode` | `WAL` | Readers do not block the writer. `nas-engine status` during a search works because of this line |
| `foreign_keys` | `ON` | SQLite disables them by default, per connection. Without it, `ON DELETE CASCADE` is decoration |
| `synchronous` | `NORMAL` | Full fsync per commit costs more than it buys here: a power-loss window of one evaluation is acceptable, and WAL already protects against process crashes |
| `busy_timeout` | configurable ms | Turns a transient `database is locked` into a wait. Without it, a concurrent writer fails immediately |

`foreign_keys=ON` is the one most often missed, and its absence is silent — the schema
looks correct and the constraints simply do not run.

### The repository boundary

The ORM does not leak. `SearchRepository` returns `SearchSummary`, `CandidateSummary`,
and plain dictionaries — frozen dataclasses detached from any session.

This is not ceremony. A live ORM object lazy-loads on attribute access, so passing one past
its session produces `DetachedInstanceError` at an unpredictable place. Worse, in this
system a candidate crosses a **process** boundary, where an ORM object cannot go at all. A
detached dataclass is picklable, immutable, and cannot surprise anyone with a query.

**One public method is one transaction.** `complete_trial` writes the trial row, the
metrics rows, the artifact rows, and the candidate's state in a single commit. Either the
whole evaluation is recorded or none of it is; there is no state where metrics exist for a
candidate still marked running.

### Migrations are hand-written and sequential

A `schema_version` table holds one row. On connect, the current version is compared to the
version the build requires:

- older → apply each migration in order, each inside a transaction;
- equal → proceed;
- **newer → refuse**, with a message naming both versions.

Refusing is the important half. An older build silently operating on a newer schema
corrupts data in ways that surface much later.

## Alternatives considered

### PostgreSQL or MySQL

*Rejected.* They solve problems this project does not have — multi-host writers, tens of
thousands of writes per second, role-based access — at the cost of the one property that
matters most for adoption: a new user must be able to `pip install` and run a search. A
server is an operational dependency, a connection string to configure, a container to
orchestrate.

SQLite's real limits are honestly stated: **one writer at a time**, and **do not put the
file on NFS**. Neither binds a single-host search whose write rate is one transaction per
evaluation — that is, per multi-second training run. The database is idle essentially all
of the time.

If a genuinely distributed search were needed, the repository interface is the seam to
swap. That is precisely why it exists.

### JSON or JSONL files

*Rejected.* Tempting, because a NAS run *looks* like an append-only event log.

It breaks on: concurrent writes from workers (no atomicity, interleaved lines); querying
("the best feasible candidate by weighted score" becomes a full scan and a hand-written
sort); uniqueness (nothing stops two rows for the same architecture); and partial writes (a
process killed mid-write leaves a truncated line that breaks every later read).

Every one of those is something a database already solves correctly.

### Raw `sqlite3` with hand-written SQL

*Rejected*, though it was close, and it is what a smaller project should do.

The ORM buys three things worth the dependency: the schema is declared once as typed
models rather than duplicated between `CREATE TABLE` strings and row-unpacking code;
relationships and cascades are declarative; and `mypy --strict` checks column access, so a
renamed column is a type error rather than a runtime `KeyError`.

The cost is a dependency and some indirection. Note what is *not* a cost: the project uses
no string-interpolated SQL anywhere. Every query is parameterised, which the ORM makes the
path of least resistance.

### Alembic for migrations

*Rejected.* Alembic is the right answer for an application with many contributors and a long
schema history. Here it adds a dependency, a configuration file, a versions directory, and
autogenerate — which produces migrations that must be reviewed by hand anyway.

The schema is roughly seven tables. A list of `(version, description, statements)` applied
in order is ~150 lines, is readable in one sitting, and has no magic. **The point where this
becomes wrong is when migrations need branching or data backfills across many versions**;
at that point, adopt Alembic and treat the existing migrations as its baseline.

### Storing weights in the database

*Rejected.* Weight files are megabytes. BLOBs would bloat the file, make backups slow, and
make `VACUUM` expensive. Weights go to the filesystem; the database stores the **relative**
path and the size in bytes. Paths are validated against the artifact root before any write,
so a hostile architecture hash cannot escape the directory.

### Caching the Pareto front

*Rejected.* A cached front goes stale the instant another candidate completes, and the
staleness is invisible. `completed_metrics` returns the population and ranking is
recomputed. For the hundreds-of-candidates scale this project targets, that is
microseconds.

## Consequences

### Good

- `pip install` and run. No server, no credentials, no container.
- The database file *is* the experiment record: results, lineage, configuration, seeds, and
  the environment snapshot, in one file that can be copied, attached to a paper, or diffed.
- Crash recovery is a query — find candidates left `RUNNING`, requeue them.
- `UNIQUE(search_id, architecture_hash, rung)` makes duplicate evaluation impossible even
  under a race between two workers; the engine catches the integrity error and treats it as
  a duplicate.
- The whole thing is inspectable with the `sqlite3` CLI, or the `sqlite3` module if the CLI
  is not installed.

### Bad

- **One writer.** Multiprocessing writes results through the parent process, not from
  workers. That is a real constraint on the concurrency design — see
  [ADR 0004](0004-concurrency-model.md).
- **Not for network filesystems.** SQLite's locking is unreliable on NFS. Keep `nas.db` on
  local disk and point only the artifact root at a share.
- **A live backup needs the backup API**, not `cp` — WAL means the main file alone is
  incomplete. See [backup and recovery](../operations/backup-and-recovery.md).
- **Migrations are a manual discipline.** Adding a column means writing the migration and
  bumping the version. Forgetting produces "no such column" on an existing database. The
  version check catches the reverse direction, not this one.
- SQLAlchemy is a non-trivial dependency for what is, at this scale, modest usage.

## Verification

| Property                                           | Test                                                         |
| -------------------------------------------------- | ------------------------------------------------------------ |
| Foreign keys are actually enforced                 | `test_foreign_keys_are_enforced`                             |
| A failed transaction rolls back whole              | `test_transactions_roll_back_on_error`                       |
| A newer database is refused                        | `test_newer_database_is_refused`                             |
| Migration is idempotent                            | `test_ensure_schema_is_idempotent`                           |
| Identity is unique per rung                        | `test_identity_is_unique_per_rung`                           |
| A claimed candidate cannot be double-claimed       | `test_a_claimed_candidate_cannot_be_claimed_again`           |
| State transitions are validated at the boundary    | `test_state_transitions_are_validated`                       |
| A stored spec is re-validated on read              | `test_specification_is_revalidated_on_read`                  |
| A corrupt stored spec is rejected, not returned    | `test_corrupt_specification_is_rejected_on_read`             |
| Re-evaluation updates artifacts instead of failing | `test_rewriting_the_same_artifact_updates_rather_than_fails` |
| Timestamps are timezone-aware                      | `test_timestamps_are_timezone_aware`                         |
| Deleting a search cascades                         | `test_deleting_a_search_cascades`                            |
| Recovery requeues running candidates               | `test_running_candidates_are_requeued`                       |
| Recovery fails candidates out of retries           | `test_candidates_out_of_retries_are_failed`                  |
| Recovery leaves completed candidates alone         | `test_completed_candidates_are_untouched`                    |
| Checkpoint pruning keeps the newest                | `test_pruning_keeps_the_newest`                              |

`test_corrupt_specification_is_rejected_on_read` deserves a note: the database is a trust
boundary too. A row can be edited outside the application, so a stored architecture is
re-validated on the way out rather than trusted because it was valid on the way in.

## See also

- [Persistence](../architecture/persistence.md) — the schema, table by table.
- [Resuming a search](../guides/resuming-a-search.md) — checkpoint restore and the recovery sweep.
- [Backup and recovery](../operations/backup-and-recovery.md)
- [ADR 0004](0004-concurrency-model.md) — what the single-writer limit implies.
