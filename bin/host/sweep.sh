#!/usr/bin/env bash
set -euo pipefail

export RUST_BACKTRACE=full

RUST_LOG_LEVEL=debug

CACHE_DIR="cache_dir"
RPC_URL=

BLOCK_NUMBERS=(
  23424730
)

for BLOCK_NUMBER in "${BLOCK_NUMBERS[@]}"; do
  # RUST_LOG="$RUST_LOG_LEVEL" cargo run -r -- \
  RUST_LOG="$RUST_LOG_LEVEL" cargo run -r --features execution-witness -- \
    --rpc-url "$RPC_URL" \
    --cache-dir "$CACHE_DIR" \
    --block-number "$BLOCK_NUMBER"
done
