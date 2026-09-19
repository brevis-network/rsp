#!/usr/bin/env bash
# Builds the guest ELF.
#
# Not `cargo pico build`: its `--rustflags` option is ignored (it sets
# `CARGO_ENCODED_RUSTFLAGS` itself, unconditionally), so it cannot pass `--wrap=memset`. The
# invocation it would have made is reproduced below with that appended; the other flags are
# copied verbatim from its output and should be re-copied if it changes them.
#
# `rsp-guest-mem`'s word-at-a-time `memset` is ~7 M retired instructions cheaper per mainnet
# block than `compiler_builtins`'. Its `memcmp`/`bcmp` are picked up by defining the symbols,
# but `memset` cannot be: `compiler_builtins` references it internally so its object is always
# pulled from the archive, and linking that way fails with `duplicate symbol: memset`.
set -euo pipefail

cd "$(dirname "$0")"

US=$(printf '\037')
# `-tail-dup-size=12` raises LLVM's `TailDuplicator` per-block budget, which lets the
# interpreter's 256-way dispatch block be copied into the ~150 opcode arms instead of each arm
# jumping back to a shared header: -10.3 M retired instructions on block 24006677. It is also
# what keeps that block duplicated once `mload`/`mstore`/`sload` are `#[inline(always)]` into
# the loop -- without it the inlining costs more than it saves. 30 buys nothing.
#
# This is the *TailDuplicator* budget, not `-tail-dup-placement-threshold` /
# `-tail-dup-placement-aggressive-threshold` / `-tail-dup-succ-size`: those do nothing here
# (the dispatch block stays at one copy) and forcing duplication through them measured worse.
export CARGO_ENCODED_RUSTFLAGS="-Cpasses=lower-atomic${US}-Clink-arg=-Ttext=0x00200800${US}-Clink-arg=--fatal-warnings${US}-Cpanic=abort${US}-Clink-arg=--wrap=memset${US}-Cllvm-args=-tail-dup-size=12"

cargo +pico build --release \
    --target riscv64im-pico-zkvm-elf \
    -Z build-std=alloc,core,proc_macro,panic_abort,std \
    -Z build-std-features=compiler-builtins-mem

mkdir -p elf
cp target/riscv64im-pico-zkvm-elf/release/reth-pico elf/riscv64im-pico-zkvm-elf

# Assert `--wrap=memset` took: the symbol is in the table only if it did. `llvm-nm` ships with
# the pico toolchain, so this needs nothing installed. `memcmp`/`bcmp` need no check.
NM="$(rustc +pico --print sysroot)/lib/rustlib/$(rustc +pico -vV | sed -n 's/^host: //p')/bin/llvm-nm"
if [ -x "$NM" ]; then
    if ! "$NM" elf/riscv64im-pico-zkvm-elf | grep -q '__wrap_memset'; then
        echo "ERROR: __wrap_memset is not in the ELF, so --wrap=memset was dropped." >&2
        echo "       The guest is correct but ~7 M retired instructions a block slower." >&2
        exit 1
    fi
else
    echo "warning: llvm-nm not found at $NM; skipped the --wrap=memset check" >&2
fi

# Record what was built, so a proof can be traced back to an ELF: the tree tracks neither the
# ELF nor a digest of it, and two materially different valid ELFs build from this one source
# tree. To a file, not just stdout -- a scrolled terminal records nothing. `rustc +pico -vV`
# rather than `cargo +pico --version`, which prints cargo's version instead of the toolchain's.
if command -v shasum >/dev/null 2>&1; then
    DIGEST=$(shasum -a 256 elf/riscv64im-pico-zkvm-elf | cut -d' ' -f1)
elif command -v sha256sum >/dev/null 2>&1; then
    DIGEST=$(sha256sum elf/riscv64im-pico-zkvm-elf | cut -d' ' -f1)
else
    DIGEST="(no sha256 tool on PATH)"
fi
TOOLCHAIN=$(rustc +pico -vV 2>/dev/null | tr '\n' ' ' | tr -s ' ' || echo unknown)
if COMMIT=$(git rev-parse HEAD 2>/dev/null); then
    git diff --quiet HEAD 2>/dev/null || COMMIT="$COMMIT (dirty)"
else
    COMMIT="(not a git checkout)"
fi
INFO="elf/riscv64im-pico-zkvm-elf.build-info"
{
    echo "elf:       $(pwd)/elf/riscv64im-pico-zkvm-elf"
    echo "sha256:    $DIGEST"
    echo "built:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "toolchain: $TOOLCHAIN"
    echo "commit:    $COMMIT"
    echo "rustflags: $(printf '%s' "$CARGO_ENCODED_RUSTFLAGS" | tr "$US" ' ')"
} > "$INFO"

echo "guest ELF: $(pwd)/elf/riscv64im-pico-zkvm-elf"
echo "sha256:    $DIGEST"
echo "toolchain: $TOOLCHAIN"
echo "recorded:  $(pwd)/$INFO"
