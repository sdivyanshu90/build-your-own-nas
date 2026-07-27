#!/usr/bin/env bash
#
# End-to-end smoke test: the shortest sequence that proves the whole system works.
#
# Runs a four-candidate search on synthetic data, then exercises every inspection command
# against the result. No network access, no GPU, no configuration beyond the checked-in
# smoke config. Finishes in well under a minute on one CPU core.
#
# This is what `make smoke` runs and what CI uses as its end-to-end gate.
#
# Usage:
#   scripts/run_smoke_search.sh [output-directory]

set -euo pipefail

CONFIG="${NAS_SMOKE_CONFIG:-configs/smoke_test.yaml}"
OUTPUT="${1:-${NAS_SMOKE_OUTPUT:-artifacts/smoke}}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "error: configuration not found: ${CONFIG}" >&2
  echo "run this script from the repository root, or set NAS_SMOKE_CONFIG" >&2
  exit 2
fi

# `--set` overrides win over the file, so the output directory can be redirected without
# editing the committed configuration.
COMMON=(--config "${CONFIG}" --set "project.output_dir=${OUTPUT}")

step() {
  printf '\n\033[1m==> %s\033[0m\n' "$1"
}

step "Environment diagnostics"
nas-engine doctor "${COMMON[@]}"

step "Configuration validation"
nas-engine validate-config "${COMMON[@]}"

step "Running the search"
nas-engine search "${COMMON[@]}"

step "Search status"
nas-engine status "${COMMON[@]}"

step "Candidates"
nas-engine list-candidates "${COMMON[@]}" --limit 10

step "Best candidate"
nas-engine best "${COMMON[@]}"

step "Pareto front"
nas-engine pareto "${COMMON[@]}"

step "Held-out test evaluation"
nas-engine evaluate "${COMMON[@]}"

step "Exports"
nas-engine export "${COMMON[@]}" --format csv
nas-engine export "${COMMON[@]}" --format json

step "Report"
nas-engine report "${COMMON[@]}"

step "Resume (should be a no-op: the budget is already spent)"
nas-engine resume "${COMMON[@]}"

printf '\n\033[1;32mSmoke test passed.\033[0m Results are in %s\n' "${OUTPUT}"
