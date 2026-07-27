# Deployment

Running searches locally, in a container, and on a schedule.

## Local installation

### From source

```bash
git clone https://github.com/example/neural-architecture-search
cd neural-architecture-search
python -m venv .venv && source .venv/bin/activate
make install
nas-engine doctor
```

### From a built wheel

```bash
make build
pip install dist/nas_engine-0.1.0-py3-none-any.whl
```

`make verify-package` builds the wheel, installs it into a throwaway virtualenv, and runs
it — which catches a missing runtime dependency or a package-data omission that an editable
install would hide.

### CPU-only PyTorch

The default `pip install torch` pulls the CUDA build, several gigabytes. For a CPU-only
machine:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install nas-engine
```

### Pinning for a deployment

`pyproject.toml` declares lower bounds, which is right for a library. For a deployment, pin:

```bash
pip install pip-tools
pip-compile pyproject.toml --output-file requirements.lock
pip install --require-hashes -r requirements.lock
```

`--require-hashes` verifies artefact integrity, which is the part that actually defends
against a compromised index.

---

## Containers

### Build and run

```bash
docker build -t nas-engine .
docker run --rm nas-engine doctor
docker run --rm -v "$PWD/artifacts:/data/artifacts" nas-engine smoke
```

The image:

- is built in two stages, so the runtime carries no compiler toolchain, no test suite, and
  no build cache;
- runs as **uid 1000, not root** — a container escape starts unprivileged, and files written
  to a mounted volume are not owned by root on the host;
- is CPU-only by default (the CUDA build is roughly ten times larger);
- sets `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` to avoid thread thrash;
- has a `HEALTHCHECK` that runs `nas-engine doctor`.

### Running a real search

```bash
docker run --rm \
  -v "$PWD/artifacts:/data/artifacts" \
  -v "$PWD/configs:/app/configs:ro" \
  nas-engine search --config /app/configs/random_search.yaml \
                    --set project.output_dir=/data/artifacts
```

Mount `configs` read-only. Nothing in a search should write to it.

### Compose

```bash
docker compose run --rm doctor
docker compose run --rm smoke
docker compose run --rm search
docker compose run --rm shell
```

Every service mounts `./artifacts`, sets `NAS_ENGINE__PROJECT__OUTPUT_DIR=/data/artifacts`,
enables `no-new-privileges`, and — for the `search` service — caps CPU and memory. That cap
matters: a misconfigured search space can otherwise propose architectures large enough to
exhaust host memory.

### GPU

The default image is CPU-only. For CUDA:

1. Base both stages on `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`, or an official
   PyTorch CUDA image.
2. Drop `--index-url https://download.pytorch.org/whl/cpu` so pip installs the CUDA build.
3. Install the NVIDIA Container Toolkit on the host.

```bash
docker run --rm --gpus all -v "$PWD/artifacts:/data/artifacts" \
  nas-engine:cuda search --config /app/configs/random_search.yaml \
    --set hardware.device=cuda --set project.output_dir=/data/artifacts
```

Verify with `docker run --rm --gpus all nas-engine:cuda doctor`, which reports the detected
devices and the CUDA version.

---

## Directory layout

```text
<output_dir>/
├── nas.db                     the results database
├── nas.db-wal, nas.db-shm     WAL sidecars (present while open)
├── candidates/
│   └── <architecture_hash>/
│       ├── weights_e5_f1_rnative_rung0.pt
│       └── training_e5_f1_rnative_rung0.pt   (if enabled)
└── reports/
    ├── <search_id>_report.md
    ├── <search_id>_results.json
    ├── <search_id>_candidates.csv
    └── plots/<search_id>_*.png
```

### Sizing

| Item                  | Typical              | Notes                                                            |
| --------------------- | -------------------- | ---------------------------------------------------------------- |
| Database              | 1–50 MB              | Grows with checkpoints; `persistence.keep_checkpoints` bounds it |
| Weights per candidate | 4 bytes × parameters | 500 k parameters ≈ 2 MB                                          |
| Training checkpoint   | ~3× the weights      | Includes optimiser state                                         |
| Report and plots      | ~1 MB                |                                                                  |

A 100-candidate search over 500 k-parameter models needs roughly 200 MB of weights. With
`evaluation.save_training_checkpoints: true`, budget 800 MB.

**Every artifact path is validated against the artifact root** before writing, so a hostile
architecture hash cannot escape it.

---

## Long-running searches

### With `tmux` or `screen`

```bash
tmux new -s nas
nas-engine search --config configs/evolution.yaml --report
# Ctrl-B then D to detach
tmux attach -t nas
```

### As a systemd service

```ini
# /etc/systemd/system/nas-search.service
[Unit]
Description=nas-engine architecture search
After=network.target

[Service]
Type=oneshot
User=nas
WorkingDirectory=/opt/nas-engine
Environment="OMP_NUM_THREADS=2"
Environment="MKL_NUM_THREADS=2"
Environment="NAS_ENGINE__LOGGING__FORMAT=json"
ExecStartPre=/opt/nas-engine/.venv/bin/nas-engine doctor --config configs/evolution.yaml
ExecStart=/opt/nas-engine/.venv/bin/nas-engine resume --config configs/evolution.yaml
TimeoutStartSec=86400

# The search writes only to its output directory.
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/nas-engine/artifacts
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

`ExecStart` uses `resume`, not `search`. Restarting the unit then continues rather than
starting a new run — which is what you want after a reboot.

`ExecStartPre` runs `doctor` as a pre-flight gate: a misconfiguration fails before any
compute is spent.

### On a scheduler (Slurm)

```bash
#!/bin/bash
#SBATCH --job-name=nas-search
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

source /opt/nas-engine/.venv/bin/activate
nas-engine doctor --config configs/evolution.yaml

# `resume` handles both the first run and every requeue after pre-emption.
nas-engine resume --config configs/evolution.yaml \
  --set budget.max_seconds=13800          # stop before the 4h wall clock
```

Setting `budget.max_seconds` below the scheduler's limit lets the search stop cleanly and
checkpoint, rather than being killed mid-evaluation. It will still recover if killed — that
is what the recovery sweep is for — but a clean stop wastes nothing.

---

## Configuration management

Keep configurations in version control, alongside the code:

```text
configs/
├── smoke_test.yaml          committed
├── random_search.yaml       committed
├── evolution.yaml           committed
└── production.yaml          committed
```

Deployment-specific values go in the **environment**, not the file:

```bash
export NAS_ENGINE__PROJECT__OUTPUT_DIR=/mnt/fast/nas-artifacts
export NAS_ENGINE__HARDWARE__DEVICE=cuda
export NAS_ENGINE__CONCURRENCY__WORKERS=4
export NAS_ENGINE__LOGGING__FORMAT=json
```

That keeps the committed file identical across environments, which is what makes a result
reproducible somewhere else.

Validate every configuration in CI:

```bash
for config in configs/*.yaml; do
  nas-engine validate-config --config "$config"
done
```

The shipped CI workflow does exactly this.

---

## Upgrading

1. **Back up the database** — see
   [backup and recovery](backup-and-recovery.md).
2. Read the changelog for schema or hash changes.
3. Install the new version.
4. `nas-engine doctor` — checks the schema version among other things.
5. `nas-engine status` on an existing search.

The schema migrates automatically on connect. A database written by a *newer* version is
refused, with an explanation.

**If architecture hashes changed** (the changelog will say so), existing databases hold
hashes from the old scheme. Resuming such a search re-proposes everything as novel. Finish
in-flight searches before upgrading, or accept the restart.

---

## A pre-flight checklist

- [ ] `nas-engine doctor` passes
- [ ] Every configuration validates
- [ ] The output directory exists and is writable, with enough space
- [ ] `OMP_NUM_THREADS` and `MKL_NUM_THREADS` set for multi-worker runs
- [ ] `budget.max_seconds` set below any scheduler limit
- [ ] `budget.max_seconds_per_evaluation` set
- [ ] `logging.format: json` for machine-consumed logs
- [ ] The database is backed up if it holds anything you care about
- [ ] The smoke search passes on this machine

## See also

- [Observability](observability.md)
- [Backup and recovery](backup-and-recovery.md)
- [Production runbook](production-runbook.md)
- [Security](../architecture/security.md)
