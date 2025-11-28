#!/usr/bin/env bash

set -euo pipefail

export RUST_BACKTRACE=full
export RUST_LOG=info

export RPC_URL=
export RPC_WS_URL=

CACHE_DIR="cache_dir"

cargo run -r -- \
  --cache-dir "$CACHE_DIR" \
  --is-input-emulated
