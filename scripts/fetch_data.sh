#!/usr/bin/env bash
# Fetch the recorded duet-trace corpus into ./data/traces.
#
# The traces are released separately from the code (~378 MB tcp8 + ~10 MB goduet).
# Set DATA_URL to the release archive, or ADAPBENCH_DATA_ROOT to a local copy.
#
# Expected layout after fetching:
#   data/traces/tcp8/<trace>/{trace.meta.json,trace.data.npz}     # 118 Java traces
#   data/traces/goduet/<corpus>/{trace.meta.json,trace.data.npz}  # 5 Go duet corpora
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ADAPBENCH_DATA_ROOT:-$REPO/data}/traces"
DATA_URL="${DATA_URL:-https://github.com/s3bastiankoch/adapbench/releases/download/v1.0.0/adapbench-traces.tar.zst}"

if [[ -z "$DATA_URL" ]]; then
  echo "DATA_URL is not set."
  echo "Point it at the released trace archive, e.g.:"
  echo "  DATA_URL=https://github.com/s3bastiankoch/adapbench/releases/download/v1.0.0/adapbench-traces.tar.zst $0"
  echo "or set ADAPBENCH_DATA_ROOT to a directory that already contains traces/."
  exit 1
fi

mkdir -p "$DEST"
echo "Fetching trace corpus from $DATA_URL -> $DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fL "$DATA_URL" -o "$tmp/traces.tar.zst"
tar --use-compress-program=unzstd -xf "$tmp/traces.tar.zst" -C "$DEST"

echo "Done. tcp8 traces: $(ls "$DEST/tcp8" 2>/dev/null | wc -l), goduet corpora: $(ls "$DEST/goduet" 2>/dev/null | wc -l)"
