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
# still produces a correct guest — just a ~1 % slower one, with no warning. The check at the
# bottom of this script is what turns "no warning" into a build failure: it looks for the
# `__wrap_memset` symbol in the linked ELF, which is only there if the flag took. A comment in
# `rsp-guest-mem` used to say "the cycle regression check exists to catch exactly that"; there
# has never been such a check anywhere in the repository, which is why this one is here.
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

# Assert the `--wrap=memset` above actually took, because losing it is otherwise silent:
# `cargo pico build` still succeeds, still produces a *correct* guest, and just costs ~7 M
# retired instructions a block. That is the failure this script's header warns about, and
# until now nothing checked for it.
#
# `llvm-nm` comes with the pico toolchain, so this needs nothing installed. With the flag,
# `rsp-guest-mem`'s `__wrap_memset` is in the symbol table and every `memset` call site has
# been redirected to it; without it, the symbol is absent and `compiler_builtins`' `memset`
# is what runs. `memcmp`/`bcmp` need no check -- they are picked up by defining the symbols.
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

# Record what was built, so a proof can be traced back to an ELF. The tree tracks neither the
# ELF nor a digest of it, and two materially different valid ELFs build from this one source
# tree, so without this line which one produced a given proof is not recoverable.
if command -v shasum >/dev/null 2>&1; then
    DIGEST=$(shasum -a 256 elf/riscv64im-pico-zkvm-elf | cut -d' ' -f1)
elif command -v sha256sum >/dev/null 2>&1; then
    DIGEST=$(sha256sum elf/riscv64im-pico-zkvm-elf | cut -d' ' -f1)
else
    DIGEST="(no sha256 tool on PATH)"
fi
echo "guest ELF: $(pwd)/elf/riscv64im-pico-zkvm-elf"
echo "sha256:    $DIGEST"
echo "toolchain: $(cargo +pico --version 2>/dev/null || echo unknown)"
