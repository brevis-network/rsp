#!/usr/bin/env bash
# Builds the guest ELF.
#
# This exists instead of a plain `cargo pico build` because the guest needs one extra
# linker flag, and `cargo pico build` cannot pass it: its `--rustflags` option is ignored
# (it sets `CARGO_ENCODED_RUSTFLAGS` itself, unconditionally). So the invocation it would
# have made is reproduced here with `--wrap=memset` appended.
#
# What the flag is for: `rsp-guest-mem` provides a word-at-a-time `memset` that is ~7 M
# retired instructions cheaper per mainnet block than `compiler_builtins`'. Its `memcmp`
# and `bcmp` are picked up simply by defining the symbols, but `memset` cannot be —
# linking that way fails with `rust-lld: error: duplicate symbol: memset`, because
# `compiler_builtins` references `memset` internally and its object is always pulled out
# of the archive (nothing references `memcmp`, so that one never is). `--wrap` redirects
# the calls instead.
#
# Consequence worth knowing: building with a plain `cargo pico build` still succeeds and
# still produces a correct guest — just a ~1 % slower one, with no warning. If the guest's
# cycle count ever jumps by roughly that much, check this first.
#
# The flags below other than `--wrap` are copied verbatim from what `cargo pico build`
# prints; if that tool changes them, re-copy them from its output.
set -euo pipefail

cd "$(dirname "$0")"

US=$(printf '\037')
# `-tail-dup-size=12` raises LLVM's tail-duplication budget (`TailDuplicator`'s per-block
# instruction cutoff) from the target default to 12. It is what lets the interpreter's
# 256-way dispatch block (`lbu`/`slli`/`add`/`lw`/`ld`/`jr`) be copied into the ~150 opcode
# arms instead of every arm ending in a jump back to a shared header, and it does the same
# for merge blocks elsewhere in the guest. On its own: -10.3 M retired instructions on block
# 24006677. It is also what keeps the dispatch block duplicated once `mload`/`mstore`/`sload`
# are `#[inline(always)]` into the loop -- without it, inlining them pushes the block back
# out of the budget and costs more than the inlining saves. Raising it to 30 buys nothing.
#
# Note that this is the *TailDuplicator* budget, not `-tail-dup-placement-threshold` /
# `-tail-dup-placement-aggressive-threshold` / `-tail-dup-succ-size`: those were tried and
# do nothing here (the dispatch block stays at one copy), and forcing duplication through
# them was measured worse.
export CARGO_ENCODED_RUSTFLAGS="-Cpasses=lower-atomic${US}-Clink-arg=-Ttext=0x00200800${US}-Clink-arg=--fatal-warnings${US}-Cpanic=abort${US}-Clink-arg=--wrap=memset${US}-Cllvm-args=-tail-dup-size=12"

cargo +pico build --release \
    --target riscv64im-pico-zkvm-elf \
    -Z build-std=alloc,core,proc_macro,panic_abort,std \
    -Z build-std-features=compiler-builtins-mem

mkdir -p elf
cp target/riscv64im-pico-zkvm-elf/release/reth-pico elf/riscv64im-pico-zkvm-elf
echo "guest ELF: $(pwd)/elf/riscv64im-pico-zkvm-elf"
