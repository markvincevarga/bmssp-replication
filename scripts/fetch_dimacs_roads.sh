#!/usr/bin/env bash
# Fetch the 12 USA road networks from the 9th DIMACS Implementation Challenge.
# These are the exact instances used by Castro et al. (2025) — arXiv:2511.03007.
#
# Each *.gr.gz is a DIMACS shortest-path file with travel-time edge weights.
# Total download size: ~2.5 GB (compressed). Decompressed: ~10 GB.
#
# Usage:
#   scripts/fetch_dimacs_roads.sh                # all 12
#   scripts/fetch_dimacs_roads.sh NY BAY         # only listed regions
#   KEEP_GZ=1 scripts/fetch_dimacs_roads.sh      # keep gzipped copy after decompression
#   FORCE=1 scripts/fetch_dimacs_roads.sh        # re-download even if file exists
#
# The Rust loader (graph_store::resolve_dimacs_path) accepts either *.gr or *.gr.gz.
# Default behavior leaves *.gr on disk (decompressed) for fastest reads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST_DIR="${DEST_DIR:-$ROOT_DIR/graphs/dimacs}"
BASE_URL="https://www.diag.uniroma1.it/~challenge9/data/USA-road-t"

ALL_REGIONS=(NY BAY COL FLA NW NE CAL LKS E W CTR USA)
FORCE="${FORCE:-0}"
KEEP_GZ="${KEEP_GZ:-0}"

if [ "$#" -gt 0 ]; then
    REGIONS=("$@")
else
    REGIONS=("${ALL_REGIONS[@]}")
fi

mkdir -p "$DEST_DIR"

# Pick a downloader once. curl is preferred (broader macOS availability).
if command -v curl >/dev/null 2>&1; then
    DL_CMD="curl"
elif command -v wget >/dev/null 2>&1; then
    DL_CMD="wget"
else
    echo "error: need either curl or wget on PATH" >&2
    exit 1
fi

download() {
    local url="$1" out="$2"
    if [ "$DL_CMD" = "curl" ]; then
        curl -fL --retry 3 --retry-delay 2 -o "$out" "$url"
    else
        wget -O "$out" "$url"
    fi
}

declare -i ok=0 fail=0 skipped=0
for region in "${REGIONS[@]}"; do
    name="USA-road-t.${region}"
    url="${BASE_URL}/${name}.gr.gz"
    gz_path="${DEST_DIR}/${name}.gr.gz"
    gr_path="${DEST_DIR}/${name}.gr"

    if [ "$FORCE" != "1" ] && [ -f "$gr_path" ]; then
        echo "[skip] ${name}.gr already present"
        skipped+=1
        continue
    fi

    echo "[get ] ${url}"
    if ! download "$url" "$gz_path"; then
        echo "[fail] ${name} download failed" >&2
        rm -f "$gz_path"
        fail+=1
        continue
    fi

    echo "[unzip] ${name}.gr.gz"
    if ! gunzip -kf "$gz_path"; then
        echo "[fail] ${name} gunzip failed" >&2
        rm -f "$gz_path"
        fail+=1
        continue
    fi

    if [ "$KEEP_GZ" != "1" ]; then
        rm -f "$gz_path"
    fi

    size=$(du -h "$gr_path" | cut -f1)
    echo "[ok  ] ${name}.gr (${size})"
    ok+=1
done

echo
echo "summary: ${ok} downloaded, ${skipped} skipped, ${fail} failed"
echo "destination: ${DEST_DIR}"
[ "$fail" -eq 0 ]
