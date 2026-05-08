#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <algorithm>"
    echo "  algorithm: dijkstra | bmssp_base"
    exit 1
fi

algo="$1"
outdir="target/profile/$algo"
mkdir -p "$outdir"

cargo build --release --bin profile --features hotpath,hotpath-alloc

for nodes in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304; do
    for edge_factor in 4 8 12 16 20 24 28 32; do
        outfile="$outdir/${nodes}_ef${edge_factor}.txt"
        echo "=== $algo | nodes=$nodes edge_factor=$edge_factor ==="
        cargo run --release --bin profile --features hotpath,hotpath-alloc -- "$algo" "$nodes" "$edge_factor" \
            > "$outfile" 2>&1
        echo "  -> saved to $outfile"
    done
done

echo "Done. Results in $outdir/"
