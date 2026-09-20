#!/usr/bin/env bash
# The host and the guest resolve revm from two independent lockfiles. Between 2026-03-25 and
# 2026-09-20 they were ten months apart and nothing said so: both manifests named the same
# branch, both builds were green, and only the locks disagreed.
#
# A temporary disagreement is legitimate -- #23 deliberately ran the guest ahead of the host
# while its paired revm change was still open -- so this is not an equality check on every
# branch. It is an equality check at the point of merging into an integration branch, which
# is where #23's skew should have been caught, and where the state that #26 and #27 had to
# clean up afterwards would have been caught too.
set -euo pipefail

rev_of() {  # $1 = path to a Cargo.lock
    awk '
        /^name = "revm"$/    { in_revm = 1; next }
        in_revm && /^source/ { print; exit }
        /^\[\[package\]\]/   { in_revm = 0 }
    ' "$1" | grep -oE '#[0-9a-f]{40}' | tr -d '#'
}

HOST_LOCK=Cargo.lock
GUEST_LOCK=bin/client/Cargo.lock

host=$(rev_of "$HOST_LOCK")
guest=$(rev_of "$GUEST_LOCK")

if [[ -z "$host" || -z "$guest" ]]; then
    echo "could not read a revm rev out of both lockfiles" >&2
    echo "  $HOST_LOCK:  ${host:-<none>}" >&2
    echo "  $GUEST_LOCK: ${guest:-<none>}" >&2
    exit 2
fi

printf 'host  %-22s %s\n' "$HOST_LOCK" "$host"
printf 'guest %-22s %s\n' "$GUEST_LOCK" "$guest"

if [[ "$host" != "$guest" ]]; then
    cat >&2 <<'EOF'

The host and the guest would ship different revm revisions.

Both manifests can name the same branch and still resolve differently, because each lock
pins a revision and nothing re-resolves it. That is how the host spent six months on a
2025-12-01 revm while the guest moved on.

If this is a deliberate, temporary skew -- a guest change waiting on a revm PR -- it does
not belong on an integration branch yet. Land the revm side, then:

    cargo update -p revm                                       # host
    cargo update -p revm --manifest-path bin/client/Cargo.toml # guest

and rebuild the guest ELF with ./bin/client/build-guest.sh, not cargo pico build.
EOF
    exit 1
fi

echo "OK: both locks name the same revm revision"
