#!/usr/bin/env bash
set -euo pipefail

export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_REGION=us-west-2
export AWS_S3_BUCKET=pico-proofs

BEGIN_BLOCK=23264565
END_BLOCK=23264965
STEP_SIZE=1
BATCH_SIZE=400
PAR_SIZE=1

export RUST_BACKTRACE=full
RUST_LOG_LEVEL=debug
CACHE_DIR="cache_dir"

RPC_URL=

RUST_LOG="$RUST_LOG_LEVEL" cargo run -r -- \
  --begin-block "$BEGIN_BLOCK" \
  --end-block "$END_BLOCK" \
  --step-size "$STEP_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --par-size "$PAR_SIZE" \
  --rpc-url "$RPC_URL" \
  --cache-dir "$CACHE_DIR"
