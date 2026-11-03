#!/usr/bin/env bash
# Self-contained R97W execution. Writes outputs only to caller-supplied directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 1 ]]; then
  printf 'Usage: bash research/benchmarks/r97w/portable/RUN_PORTABLE.sh OUTPUT_DIR\n' >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python3}"
"$PY" "$HERE/test_portable.py"
"$PY" "$HERE/experiment.py" --output "$1" --compare-archive
