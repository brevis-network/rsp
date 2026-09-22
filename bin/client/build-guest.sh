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

# Two symbol-table assertions. Both failure modes produce a correct guest that is quietly slower,
# which is the only kind worth a check here -- an incorrect guest fails its own tests.
# `llvm-nm` ships with the pico toolchain, so this needs nothing installed.
NM="$(rustc +pico --print sysroot)/lib/rustlib/$(rustc +pico -vV | sed -n 's/^host: //p')/bin/llvm-nm"
if [ -x "$NM" ]; then
    # One snapshot, and `case` rather than `"$NM" ... | grep -q ...`. That pipeline is a trap under
    # the `set -o pipefail` above: `grep -q` closes the pipe at its first match, `llvm-nm` then dies
    # on the write, pipefail makes the *whole pipeline* fail, and an `if <pipeline>; then error`
    # therefore takes the else branch -- passing silently on exactly the input it exists to reject.
    # Measured, not reasoned: raw status 74 with a pattern that is present. The `__wrap_memset`
    # check below would have been safe either way only by accident, because `if !` turns that race
    # into a loud false failure instead of a silent pass.
    SYMS="$("$NM" elf/riscv64im-pico-zkvm-elf)"

    # `--wrap=memset` took: the symbol is in the table only if it did. `memcmp`/`bcmp` need no
    # check -- they are picked up by defining the symbols and nothing competes for them.
    #
    # This reads as a presence check on a symbol only because the link discards unreferenced
    # sections: `rustc +pico --target riscv64im-pico-zkvm-elf --print link-args` passes
    # `--gc-sections`. Without it the check would be vacuous, since `__wrap_memset` and `memcmp`
    # sit adjacent in one object and the former would ride in on the latter's references.
    case "$SYMS" in
        *__wrap_memset*) ;;
        *)
            echo "ERROR: __wrap_memset is not in the ELF, so --wrap=memset was dropped." >&2
            echo "       The guest is correct but ~7 M retired instructions a block slower." >&2
            exit 1
            ;;
    esac

    # `memcpy` resolved to `pico-sdk`'s assembly and not to `compiler_builtins`'. Two
    # implementations define it -- `pico-sdk` unconditionally through `global_asm!`, and
    # `compiler_builtins` through `-Z build-std-features=compiler-builtins-mem` -- and which one
    # the link keeps is not pinned by anything. A guest that keeps the wrong one carries an
    # 8-byte thunk into `compiler_builtins::mem::memcpy` and no assembly `memcpy` at all.
    #
    # Not hypothetical, and not cheap: the ELF shipped as brevis-vm's `reth-elf` between
    # 2026-09-02 and 2026-09-22 lost this race, and a rebuild of its own source retired
    # **4.97 % fewer instructions** over the thirteen bench blocks with byte-identical committed
    # values and the same event kinds -- i.e. the same program, 5 % cheaper. Nothing reported it:
    # the proofs were correct the whole time.
    #
    # Asserted as the absence of the loser rather than the presence of the winner, because that is
    # the exact fingerprint and it does not move when the assembly is edited. `memmove` is
    # deliberately not checked: `pico-sdk` supplies no assembly one, so it comes from
    # `compiler_builtins` in every build, winner or loser.
    # Both spellings: legacy mangling is the default and what `llvm-nm` prints today, but a
    # switch to v0 or a demangling `nm` would otherwise turn this check off without a word.
    case "$SYMS" in
        *compiler_builtins*3mem*6memcpy* | *compiler_builtins::mem::memcpy*)
            echo "ERROR: memcpy resolved to compiler_builtins, not to pico-sdk's assembly one." >&2
            echo "       The guest is correct but roughly 5 % slower. See the comment above." >&2
            exit 1
            ;;
        *) ;;
    esac
else
    # Fail closed. These two checks are the only thing standing between a silently 5 %-slower
    # guest and a shipped artefact, and a warning buried in an hour of build output is not a
    # signal. If a toolchain layout change moves `llvm-nm`, that is worth stopping for.
    echo "ERROR: llvm-nm not found at $NM, so the symbol-table checks could not run." >&2
    echo "       Set PICO_SKIP_SYMBOL_CHECKS=1 to build anyway." >&2
    [ "${PICO_SKIP_SYMBOL_CHECKS:-}" = 1 ] || exit 1
    echo "       PICO_SKIP_SYMBOL_CHECKS=1 -- continuing without them." >&2
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
