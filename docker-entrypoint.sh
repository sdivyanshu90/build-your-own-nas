#!/usr/bin/env bash
#
# Container entrypoint.
#
# Accepts three shapes of invocation:
#   docker run nas-engine search --config ...   -> runs `nas-engine search --config ...`
#   docker run nas-engine smoke                 -> runs the bundled smoke search
#   docker run nas-engine bash                  -> drops into a shell
#
# Dispatching here rather than making `nas-engine` the entrypoint directly keeps `smoke`
# and `bash` available without remembering `--entrypoint`.

set -euo pipefail

case "${1:-}" in
  smoke)
    shift
    exec /app/scripts/run_smoke_search.sh "${1:-/data/artifacts/smoke}"
    ;;
  bash | sh)
    exec "$@"
    ;;
  python)
    exec "$@"
    ;;
  "")
    exec nas-engine --help
    ;;
  *)
    exec nas-engine "$@"
    ;;
esac
