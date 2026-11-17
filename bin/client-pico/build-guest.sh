#!/usr/bin/env bash
# Builds the Pico guest ELF of the RSP client into ./elf/, with a record of what was built.
#
# Not `cargo pico build`: it sets CARGO_ENCODED_RUSTFLAGS itself and so cannot pass
# `--wrap=memset`. The flags it would pass (brevis-vm sdk/cli/src/build/build.rs) are reproduced
# below with that appended.
#
# `--wrap=memset` sends `memset` to `rsp-guest-mem`'s word-at-a-time `__wrap_memset`. Its
# `memcmp`/`bcmp` are picked up by defining the symbols, but `memset` cannot be:
# `compiler_builtins` references it internally, so its object is always pulled from the archive
# and a second definition fails the link.
#
# `-Cllvm-args=-tail-dup-size=12`, which our revm branch builds with for revm's dispatch loop, is
# left out: evm2 dispatches by tail call and the flag changes nothing (510.8 M against 510.9 M
# instructions a block over 24 Sepolia blocks).
#
# The toolchain is rustup's `pico`, as `cargo pico` installs it. Where it is not linked into
# rustup, name the two programs instead: PICO_RUSTC, Pico's rustc, and PICO_CARGO, a nightly
# cargo to drive it (the build uses `-Z build-std`). TARGET_DIR moves cargo's target directory.
#
# usage: build-guest.sh [extra cargo args, e.g. --features arena]
set -euo pipefail
cd "$(dirname "$0")"
./prepare-evm2.sh

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
export CARGO_ENCODED_RUSTFLAGS="-Cpasses=lower-atomic${US}-Clink-arg=-Ttext=0x00200800${US}-Clink-arg=--fatal-warnings${US}-Cpanic=abort${US}--cfg${US}getrandom_backend=\"custom\"${US}-Clink-arg=--wrap=memset"
"${CARGO[@]}" build --release \
    --target riscv64im-pico-zkvm-elf \
    -Z build-std=alloc,core,proc_macro,panic_abort,std \
    -Z build-std-features=compiler-builtins-mem \
    --target-dir "$TD" "$@"
BUILT=$TD/riscv64im-pico-zkvm-elf/release/rsp-client-pico

# Two symbol-table checks. Each failure leaves a correct guest that is quietly slower, which no
# test of the guest's output can notice. `llvm-nm` ships with the Pico toolchain.
NM="$(rustc_pico --print sysroot)/lib/rustlib/$(rustc_pico -vV | sed -n 's/^host: //p')/bin/llvm-nm"
[ -x "$NM" ] || { echo "ERROR: llvm-nm not found at $NM; cannot check the ELF's symbols" >&2; exit 1; }
SYMS="$("$NM" "$BUILT")"
# `--wrap=memset` took effect: the symbol is in the table only if it did (the link drops
# unreferenced sections).
case "$SYMS" in
    *__wrap_memset*) ;;
    *) echo "ERROR: __wrap_memset is not in the ELF, so --wrap=memset was dropped" >&2; exit 1 ;;
esac
# `memcpy` is pico-sdk's assembly one. `compiler_builtins` defines it too
# (`compiler-builtins-mem`), nothing pins which the link keeps, and the wrong one cost our
# `pico-rv64` branch about 5 % of a mainnet block's instructions for three weeks.
case "$SYMS" in
    *compiler_builtins*3mem*6memcpy* | *compiler_builtins::mem::memcpy*)
        echo "ERROR: memcpy resolved to compiler_builtins, not to pico-sdk's assembly" >&2; exit 1 ;;
    *) ;;
esac

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
    echo "evm2:      $(cat evm2-pico/.prepared) (revision, digest of the patches)"
    echo "arguments: $*"
    echo "rustflags: $(printf '%s' "$CARGO_ENCODED_RUSTFLAGS" | tr "$US" ' ')"
} > elf/riscv64im-pico-zkvm-elf.build-info
echo "guest ELF: $(pwd)/elf/riscv64im-pico-zkvm-elf"
echo "sha256:    $DIGEST"
echo "recorded:  $(pwd)/elf/riscv64im-pico-zkvm-elf.build-info"
