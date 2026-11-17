#!/usr/bin/env bash
# The workspace's lock file is what the client's tests run on; the guest's, in bin/client-pico,
# is what is proved. A git dependency that both take from the same branch has to be at the same
# commit in both, or the tests say nothing about the guest: bls12_381 was a commit behind in the
# guest's for a day, without the fix for MAP_FP2_TO_G2 at u = 0, while the tests passed on the
# fixed one.
set -euo pipefail
cd "$(dirname "$0")/../.."
same_commit() { # the repository, as both lock files name it
    local ws guest
    ws=$(grep -o "git+https://github.com/$1?[^\"]*" Cargo.lock | sort -u || true)
    guest=$(grep -o "git+https://github.com/$1?[^\"]*" bin/client-pico/Cargo.lock | sort -u || true)
    if [ -z "$ws" ] || [ "$ws" != "$guest" ]; then
        echo "ERROR: $1 is locked at [$guest] for the guest and at [$ws] for the workspace" >&2
        exit 1
    fi
    echo "$1: both lock files have ${ws##*#}"
}
same_commit brevis-network/bls12_381
same_commit brevis-network/bn
