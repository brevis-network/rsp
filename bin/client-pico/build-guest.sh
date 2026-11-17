#!/usr/bin/env bash
# Builds the Pico guest ELF of the RSP client into ./elf/, with a record of what was built.
# The flags are the ones `cargo pico build` passes (brevis-vm sdk/cli/src/build/build.rs); none of
# our branch's guest-specific extras (--wrap=memset, -tail-dup-size) are here yet.
#
# The toolchain is rustup's `pico`, as `cargo pico` installs it. Where it is not linked into
# rustup, name the two programs instead: PICO_RUSTC, Pico's rustc, and PICO_CARGO, a nightly
# cargo to drive it (the build uses `-Z build-std`). TARGET_DIR moves cargo's target directory.
#
# usage: build-guest.sh [extra cargo args, e.g. --features arena]
set -euo pipefail
cd "$(dirname "$0")"

if [ -n "${PICO_RUSTC:-}" ]; then
    export RUSTC=$PICO_RUSTC
    CARGO=("${PICO_CARGO:-cargo}")
    rustc_pico() { "$PICO_RUSTC" "$@"; }
else
    CARGO=(cargo +pico)
    rustc_pico() { rustc +pico "$@"; }
fi
TD=${TARGET_DIR:-target}
US=$(printf '\037')
export GIT_TERMINAL_PROMPT=0
export CARGO_ENCODED_RUSTFLAGS="-Cpasses=lower-atomic${US}-Clink-arg=-Ttext=0x00200800${US}-Clink-arg=--fatal-warnings${US}-Cpanic=abort${US}--cfg${US}getrandom_backend=\"custom\""
"${CARGO[@]}" build --release \
    --target riscv64im-pico-zkvm-elf \
    -Z build-std=alloc,core,proc_macro,panic_abort,std \
    -Z build-std-features=compiler-builtins-mem \
    --target-dir "$TD" "$@"
BUILT=$TD/riscv64im-pico-zkvm-elf/release/rsp-client-pico

# What was built, in a file beside the ELF, so that a proof can be traced back to it: the tree
# tracks neither the ELF nor a digest of it.
mkdir -p elf
cp "$BUILT" elf/riscv64im-pico-zkvm-elf
if command -v sha256sum >/dev/null 2>&1; then
    DIGEST=$(sha256sum elf/riscv64im-pico-zkvm-elf | cut -d' ' -f1)
else
    DIGEST=$(shasum -a 256 elf/riscv64im-pico-zkvm-elf | cut -d' ' -f1)
fi
if COMMIT=$(git rev-parse HEAD 2>/dev/null); then
    [ -z "$(git status --porcelain 2>/dev/null)" ] || COMMIT="$COMMIT (with uncommitted changes)"
else
    COMMIT="(not a git checkout)"
fi
{
    echo "elf:       $(pwd)/elf/riscv64im-pico-zkvm-elf"
    echo "sha256:    $DIGEST"
    echo "built:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "toolchain: $(rustc_pico -vV | tr '\n' ' ' | tr -s ' ')"
    echo "commit:    $COMMIT"
    echo "arguments: $*"
    echo "rustflags: $(printf '%s' "$CARGO_ENCODED_RUSTFLAGS" | tr "$US" ' ')"
} > elf/riscv64im-pico-zkvm-elf.build-info
echo "guest ELF: $(pwd)/elf/riscv64im-pico-zkvm-elf"
echo "sha256:    $DIGEST"
echo "recorded:  $(pwd)/elf/riscv64im-pico-zkvm-elf.build-info"
