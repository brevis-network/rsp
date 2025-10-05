#!/usr/bin/env bash
set -euo pipefail

export RUST_BACKTRACE=full
RUST_LOG_LEVEL="${RUST_LOG_LEVEL:-info}"
: "${RPC_URL:?Please export RPC_URL, e.g. export RPC_URL=https://... }"

START_BLOCK=23265765
WORKERS=12
SIZE=100             # block number for each worker
END_BLOCK=$((START_BLOCK + WORKERS*SIZE - 1))

# ===== build once =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN="$WS_ROOT/target/release/rsp"

if [[ ! -x "$BIN" ]]; then
  (cargo build --manifest-path "$WS_ROOT/Cargo.toml" --release -p rsp >/dev/null)
  [[ -x "$BIN" ]] || { echo "Binary not found: $BIN"; exit 1; }
fi

RUN_ROOT="${RUN_ROOT:-run_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

pids=()
cleanup() {
  if ((${#pids[@]})); then
    echo "Stopping ${#pids[@]} workers..."
    kill "${pids[@]}" 2>/dev/null || true
    wait || true
  fi
}
trap cleanup INT TERM EXIT

run_group() {
  local from=$1
  local to=$2
  ((from > END_BLOCK)) && return 0
  ((to > END_BLOCK)) && to=$END_BLOCK

  local cache_dir="$RUN_ROOT/cache_dir_${from}_${to}"
  mkdir -p "$cache_dir"
  local log_file="${LOG_DIR}/blocks_${from}_${to}.log"

  echo "[worker ${from}-${to}] starting -> ${cache_dir}"
  {
    for ((n=from; n<=to; n++)); do
      echo "==> [${from}-${to}] block $n"
      RUST_LOG="$RUST_LOG_LEVEL" "$BIN" \
        --rpc-url "$RPC_URL" \
        --cache-dir "$cache_dir" \
        --block-number "$n"
    done
    echo "[worker ${from}-${to}] done."
  } | tee "$log_file"
}

echo "Range: $START_BLOCK .. $END_BLOCK (total=$((END_BLOCK-START_BLOCK+1)))"
echo "Launching $WORKERS workers, $SIZE blocks each"
for ((i=0; i<WORKERS; i++)); do
  from=$((START_BLOCK + i*SIZE))
  to=$((from + SIZE - 1))
  run_group "$from" "$to" &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
echo "All done. Outputs under: $RUN_ROOT"
exit "$fail"
