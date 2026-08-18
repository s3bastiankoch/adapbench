#!/usr/bin/env bash
# Pack the local trace corpus into the release archive that scripts/fetch_data.sh fetches.
#
# Reads the suite directories from ${ADABENCH_DATA_ROOT:-./data}/traces (symlinks are
# dereferenced) and writes a .tar.zst whose top level is `tcp8/` and `goduet/`, so that
# fetch_data.sh can extract it straight into data/traces/. Upload the result and set
# DATA_URL to its URL.
#
#   scripts/pack_data.sh                       # -> adabench-traces.tar.zst
#   OUT=/tmp/traces.tar.zst scripts/pack_data.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ADABENCH_DATA_ROOT:-$REPO/data}/traces"
OUT="${OUT:-$REPO/adabench-traces.tar.zst}"

[[ -d "$SRC" ]] || { echo "no traces at $SRC (set ADABENCH_DATA_ROOT)"; exit 1; }

suites=()
for s in tcp8 goduet; do
  [[ -e "$SRC/$s" ]] && suites+=("$s") || echo "warning: $SRC/$s missing, skipping"
done
[[ ${#suites[@]} -gt 0 ]] || { echo "no suites found under $SRC"; exit 1; }

echo "Packing ${suites[*]} from $SRC -> $OUT"
# -h dereferences the (possibly symlinked) suite directories into real files.
tar -h --use-compress-program='zstd -19 -T0' -cf "$OUT" -C "$SRC" "${suites[@]}"
echo "Done: $(du -h "$OUT" | cut -f1)  ($OUT)"
echo "Upload it and run:  DATA_URL=<url> scripts/fetch_data.sh"
