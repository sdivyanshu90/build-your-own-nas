# Backup and recovery

What is precious, what is cheap to recreate, and how to get a search back.

## What is worth backing up

| Artefact             | Value                                                         | Recreatable?                              |
| -------------------- | ------------------------------------------------------------- | ----------------------------------------- |
| `nas.db`             | **High** — every result, every checkpoint, every lineage edge | No. Only by re-running the whole search   |
| `configs/*.yaml`     | **High** — the definition of the experiment                   | Only from memory. Keep in version control |
| `candidates/**/*.pt` | Medium — trained weights                                      | Yes, by re-training that architecture     |
| `reports/**`         | Low                                                           | Yes, in seconds: `nas-engine report`      |

The database is the thing. It holds the architecture specs, so weights can always be
regenerated from it; the reverse is not true. **Back up `nas.db`; everything else is a
convenience.**

## Backing up a live database

SQLite in WAL mode keeps recent writes in `nas.db-wal`, not yet folded into `nas.db`.
Copying `nas.db` alone while a search is running yields a **torn backup**: the main file
without the writes that complete it.

### The right way — `.backup`

```bash
sqlite3 artifacts/nas.db ".backup 'backups/nas-$(date +%Y%m%d-%H%M%S).db'"
```

The online backup API takes a consistent snapshot while writers are active. This is safe on
a running search.

### Also fine — `VACUUM INTO`

```bash
sqlite3 artifacts/nas.db "VACUUM INTO 'backups/nas-$(date +%Y%m%d).db'"
```

Consistent *and* compacted — usually noticeably smaller, because deleted checkpoint rows
leave free pages behind.

### Without the `sqlite3` CLI

The command-line tool is not always installed, but the same online backup API is in
Python's standard library:

```python
import sqlite3
from pathlib import Path

source = sqlite3.connect("artifacts/nas.db")
target = sqlite3.connect("backups/nas-20260727.db")
with target:
    source.backup(target)          # consistent snapshot, writers welcome
target.close()
source.close()

check = sqlite3.connect("backups/nas-20260727.db")
assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
check.close()
```

No extra dependency, and it works identically on every platform.

### Wrong — plain `cp`

```bash
cp artifacts/nas.db backups/          # torn if anything is writing
```

Only safe when nothing has the database open. If you must, copy all three files
(`nas.db`, `nas.db-wal`, `nas.db-shm`) together, and accept the race.

### Verify the backup

A backup you have not restored is a hypothesis.

```bash
sqlite3 backups/nas-20260727.db "PRAGMA integrity_check;"      # expect: ok
sqlite3 backups/nas-20260727.db "SELECT COUNT(*) FROM candidates;"
```

Better still, point the CLI at it:

```bash
mkdir -p /tmp/restore-test && cp backups/nas-20260727.db /tmp/restore-test/nas.db
nas-engine status --config configs/my.yaml --set project.output_dir=/tmp/restore-test
```

If `status` prints the expected counts, the backup is real.

## A backup script

```bash
#!/usr/bin/env bash
# backup-nas.sh — consistent snapshot with retention
set -euo pipefail

SOURCE="${1:-artifacts/nas.db}"
DEST="${2:-backups}"
KEEP="${KEEP:-14}"

[[ -f "$SOURCE" ]] || { echo "no database at $SOURCE" >&2; exit 1; }
mkdir -p "$DEST"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
target="$DEST/nas-$stamp.db"

sqlite3 "$SOURCE" ".backup '$target'"
sqlite3 "$target" "PRAGMA integrity_check;" | grep -qx ok \
  || { echo "integrity check FAILED for $target" >&2; rm -f "$target"; exit 1; }

gzip -9 "$target"
echo "wrote $target.gz ($(du -h "$target.gz" | cut -f1))"

# retention: keep the newest $KEEP
ls -1t "$DEST"/nas-*.db.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm --
```

The integrity check runs **before** the old backups are pruned, and a failed snapshot is
deleted rather than kept. A retention policy that rotates a corrupt backup into the archive
while deleting a good one is worse than no policy.

Schedule it:

```cron
0 * * * * /opt/nas-engine/backup-nas.sh /opt/nas-engine/artifacts/nas.db /var/backups/nas
```

## Restoring

```bash
# 1. Stop anything using the database.
# 2. Move the current file aside — do not delete it; it may still be readable.
mv artifacts/nas.db artifacts/nas.db.broken
rm -f artifacts/nas.db-wal artifacts/nas.db-shm

# 3. Restore.
gunzip -c /var/backups/nas/nas-20260727T120000Z.db.gz > artifacts/nas.db

# 4. Verify, then resume.
nas-engine status --config configs/my.yaml
nas-engine resume --config configs/my.yaml
```

Resuming after a restore re-runs whatever completed between the backup and the failure.
That is wasted compute, not lost correctness: candidate identity is content-addressed, so
re-evaluated architectures land on the same rows.

## Recovery scenarios

### The process was killed mid-search

Nothing special is required.

```bash
nas-engine resume --config configs/my.yaml
```

On resume the engine runs a **recovery sweep**: any candidate left in `RUNNING` is moved
back to `QUEUED` and its interrupted trial is marked failed. Then it reconciles the
checkpoint against the database — if the database says a candidate completed but the
checkpoint predates that, the database wins.

Verify:

```bash
nas-engine status --config configs/my.yaml     # no candidates should be RUNNING
```

### The database is corrupt

```bash
sqlite3 artifacts/nas.db "PRAGMA integrity_check;"
```

If it does not print `ok`:

```bash
# 1. Try to salvage into a fresh file.
sqlite3 artifacts/nas.db ".recover" | sqlite3 artifacts/nas-recovered.db
sqlite3 artifacts/nas-recovered.db "PRAGMA integrity_check;"
sqlite3 artifacts/nas-recovered.db "SELECT COUNT(*) FROM candidates;"

# 2. If that works, use it.
mv artifacts/nas.db artifacts/nas.db.corrupt
mv artifacts/nas-recovered.db artifacts/nas.db

# 3. Otherwise restore from backup.
```

`.recover` reconstructs what it can from the b-tree pages; it is more forgiving than
`.dump`, which stops at the first bad page.

Corruption on a local disk usually means hardware, a full filesystem, or a network
filesystem. **Do not put the database on NFS** — SQLite's locking is unreliable there. If
the artifacts must live on a network share, keep `nas.db` on local disk and point only the
artifact root at the share.

### The database was deleted, but the artifacts survive

The weights are still there, but the metadata that explains them is gone. There is no
automatic reconstruction — a `.pt` file does not carry its search id, its budget, or its
metrics.

What you can do: the filenames encode the architecture hash and the budget, so
`candidates/<hash>/weights_e5_f1_rnative_rung0.pt` tells you which architecture was trained
under which budget. Re-run the search with the same configuration and seed; the same
architectures will be proposed, and you can compare.

Restore from a backup instead. This is the scenario backups exist for.

### The disk filled up

```bash
df -h artifacts/
du -sh artifacts/candidates artifacts/reports
```

Reclaim, in order of preference:

```bash
# 1. Reports — free, regenerable.
rm -rf artifacts/reports && nas-engine report --config configs/my.yaml

# 2. Weights for candidates that are not on the Pareto front.
nas-engine pareto --config configs/my.yaml --json \
  | jq -r '.pareto_front[].architecture_hash' > keep.txt
find artifacts/candidates -mindepth 1 -maxdepth 1 -type d \
  | grep -vFf keep.txt | xargs -r rm -rf

# 3. Compact the database.
sqlite3 artifacts/nas.db "VACUUM;"
```

Then prevent a recurrence: lower `persistence.keep_checkpoints`, set
`evaluation.save_training_checkpoints: false`, or set `evaluation.save_weights: false`
entirely if you only care about which architecture won.

A search that hits ENOSPC mid-write records the failure and continues to the next candidate
rather than dying. The recovery sweep cleans up whatever was in flight.

### A migration failed

```bash
nas-engine doctor --config configs/my.yaml
```

Migrations run inside a transaction, so a failure leaves the schema at its previous version
rather than half-applied. Restore from the pre-upgrade backup and report the failure — the
error message names the migration and the statement.

### The checkpoint and the database disagree

Reconciliation is automatic: `_reconcile_completed` trusts the database. Rows are committed
transactionally per candidate; a checkpoint is a periodic snapshot and can lag.

Nothing is required of you. `search.resumed` logs the reconciled counts.

## Retention

| Data                         | Suggested retention |
| ---------------------------- | ------------------- |
| Hourly backups               | 24 hours            |
| Daily backups                | 14 days             |
| Weekly backups               | 3 months            |
| Backups of published results | Forever             |

The last row matters more than it looks. When a result is published — in a paper, a model
card, a decision record — its database is the evidence. Keep it, alongside the configuration
and the environment snapshot the search recorded. Together they are what makes the result
checkable a year later.

## What backups cannot give you

They restore the *record* of a search, not its *conclusion*. If the configuration was wrong,
restoring reproduces the wrong search faithfully. Keeping the configuration in version
control, and keeping the environment snapshot with the results, is what lets you tell those
two situations apart.

## See also

- [Resuming a search](../guides/resuming-a-search.md) — how resume works, step by step.
- [Persistence](../architecture/persistence.md) — the schema and the pragmas.
- [Production runbook](production-runbook.md) — the procedures around these.
