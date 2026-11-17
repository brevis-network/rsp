#!/usr/bin/env bash
# Checks out evm2 at the revision reth pins into ./evm2-pico and applies the changes in
# ./evm2-patches to it. The guest manifest patches evm2 to that directory.
#
# The changes are for this target only (riscv64im, no misaligned loads, no vector or byte-swap
# instructions, branches that compare two registers, no address scale), where a few of evm2's
# opcode handlers compile to very slow code and its dispatch pays on every opcode for a comparison
# and a shift the target does not need. They sit behind `target_arch = "riscv64"` and change no
# behaviour; the static gas charge relies on a gas counter below 2^63 and asserts it when a frame
# starts to run. The host keeps upstream evm2.
#
# 0005 changes nothing in the guest. It adds a cfg with which a host build of the patched evm2
# charges static gas as 0003 does, so that evm2's own unit tests and its EEST runner run that
# code (on a nightly rustc, which gives the tail-call dispatch the guest uses):
#   RUSTFLAGS="--cfg evm2_as_on_riscv64" cargo test -p evm2 --lib
#
# Re-run after changing REV or the patches: the checkout is rebuilt when either differs from what
# it was prepared from. (A patch that adds a file leaves that file behind when it is later
# removed from the series; delete ./evm2-pico by hand in that case.)
set -euo pipefail
cd "$(dirname "$0")"
# The revision in reth v2.11.0's manifest, and the repository its patch section names.
REV=66687c18b34d3bdbb899f5eb7605070615c03475
URL=https://github.com/DaniPopes/evm2
DIR=evm2-pico

want="$REV $(cat evm2-patches/*.patch | sha256sum | cut -c1-16)"
if [ -f "$DIR/.prepared" ] && [ "$(cat "$DIR/.prepared")" = "$want" ]; then
    exit 0
fi

export GIT_TERMINAL_PROMPT=0
[ -d "$DIR/.git" ] || git init -q "$DIR"
if ! git -C "$DIR" cat-file -e "$REV^{commit}" 2>/dev/null; then
    git -C "$DIR" fetch -q --depth 1 "$URL" "$REV"
fi
git -C "$DIR" reset -q --hard "$REV"
for patch in evm2-patches/*.patch; do
    git -C "$DIR" apply --whitespace=nowarn "../$patch"
done
echo "$want" > "$DIR/.prepared"
echo "evm2 $REV with $(ls evm2-patches/*.patch | wc -l) patches prepared in $(pwd)/$DIR"
