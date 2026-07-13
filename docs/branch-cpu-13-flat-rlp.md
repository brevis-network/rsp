# Branch report: `cpu-13-flat-rlp-arca`

_Base (main): `pico-rv64`. Merge-base: `cd3731529`. Compared as `pico-rv64...HEAD`._

## What this branch does

This branch implements **CPU-13 item #5**: it replaces the witness-trie wire
format used by the zkVM guest. Previously each block's parent state was shipped
as a bincode-serialized `MptNode` graph (`EthereumState`), which cost one heap
allocation per node to decode and a full RLP re-encode per node to verify inside
the guest. The branch replaces this with a **flat RLP blob format**
(`FlatEthereumState`): every trie's node encodings are concatenated in DFS
pre-order (root first) and borrowed zero-copy from the raw input buffer in the
guest.

The guest now:

- verifies the entire trie structure in a single linear keccak + frontier-match
  pass (every non-root blob must hash to a digest reference on the DFS frontier
  of already-accepted nodes, so the root hash commits to all of them);
- reads account/storage values directly off the raw blobs with no allocation;
- computes the post-state root with a single bottom-up **delta pass** over the
  blobs (changed nodes re-encoded/hashed once, unchanged children carried over
  as verbatim reference bytes) instead of materializing an overlay, updating it,
  and rehashing.

### Reported cycle impact (reth block 18884864)

Per the commit messages, on block 18884864 total cycles went **73,648,772
(baseline) → 58,186,978 (-21.0%)**, with `compute-state-root` dropping 21.5M →
8.33M and `initialize-witness-db` 11.58M → 10.55M. The openvm-eth reference is
cited at 63.6M.

> Note: `memory/cpu-13-rsp-cycle-optimization.md` records that trie flattening
> was previously measured as a *regression* due to revm caching. The commit
> messages on this branch report a net improvement, so the memory note likely
> predates the delta-pass work (commits `0a4eda8`/`c916fc5`). Worth
> re-validating against the measurement pipeline before trusting the -21% figure.

## Diffstat

```
 Cargo.lock                                |    2 +
 bin/client/Cargo.lock                     |    3 +-
 bin/client/src/main.rs                    |    7 +-
 bin/native-client/src/main.rs             |   24 +-
 bin/pico-processor/src/main.rs            |    2 +-
 crates/executor/client/Cargo.toml         |    1 +
 crates/executor/client/src/executor.rs    |   16 +-
 crates/executor/client/src/io.rs          |  170 ++-
 crates/executor/host/src/full_executor.rs |   18 +-
 crates/executor/host/src/host_executor.rs |    4 +-
 crates/mpt/Cargo.toml                     |    1 +
 crates/mpt/src/flat.rs                    | 1804 ++++++++++++++++++++++++
 crates/mpt/src/lib.rs                     |    6 +
 crates/mpt/src/mpt.rs                     |   15 +-
 14 files changed, 2000 insertions(+), 73 deletions(-)
```

## Commits (oldest → newest)

1. **`9bf0041` — Ship witness tries as flat RLP blobs (CPU-13 item 5)**
   Introduces `FlatEthereumState` and the guest-side linear verify + zero-copy
   reads. Post-state root initially computed on a copy-on-write `MptNode`
   overlay materialized only along updated paths.

2. **`5e94921` — Add legacy-input conversion path for existing fixtures**
   Adds `LegacyClientExecutorInput` (the old `MptNode`-graph wire format) plus a
   `native-client --convert-legacy-to` flag, so previously generated inputs and
   fixtures convert to the flat format offline without a `debug_executionWitness`
   RPC. All four rv64 bench fixtures convert and execute with matching roots.

3. **`2c1f925` — Lean node parsing via recorded kinds; drop no-op post-state
   filter** Records each node's kind during the verify pass so walks/materialize
   skip the full 17-item structural scan. Removes the post-state no-op filter
   (measured to filter nothing: 86 accounts / 191 slots, all real changes).

4. **`0a4eda8` — Compute post-state root via batched delta pass over the blobs**
   Replaces materialize-overlay + update + rehash with a single bottom-up delta
   pass. Branch collapses resolve surviving siblings through the edge table for
   `delete_internal` shape parity. Covered by fixed + randomized parity sweeps
   against `MptNode`. `compute-state-root` 21.5M → 13.6M.

5. **`c916fc5` — Single-pass RLP encoding in the delta engine; skip redundant
   verify scan** Encodes rebuilt branch nodes in one pass with arithmetic length
   computation (was allocating 16 intermediate buffers + double-copying payload)
   and drops the never-read edge-marker verify scan. `compute-state-root`
   13.65M → 8.33M.

## File-by-file changes

### `crates/mpt/src/flat.rs` (new, 1804 lines)
The whole flat format. Key public API:
- `FlatEthereumState<'a>` — wire type; `state_nodes: Cow<[u8]>` + per-storage
  `FlatStorageEntry` blobs. `from_state()` (host encode), `into_owned()`,
  `views()`.
- `FlatTrieView<'a>` — a parsed, linkage-verified trie: `parse_and_verify()`,
  `get()`, `materialize()`, `delta_root()` / `empty_delta_root()`.
- `FlatStateViews<'a>` — verified views over all tries: `post_state_root()`,
  `materialize_overlay()`.
- `flatten_trie()` — DFS pre-order RLP emitter (host side).
- A `Cow` serde helper that borrows from the input on the bincode-over-slice
  (guest) path and copies on the bincode-over-reader (host cache) path.
- `#[cfg(test)]` module (~930 lines) with 10 tests, incl. randomized parity
  sweeps against the existing `MptNode` implementation.

### `crates/mpt/src/mpt.rs`
Adds `Error::FlatTrie(&'static str)`. Makes `lcp`, `prefix_nibs`,
`node_from_digest` `pub(crate)`; adds `node_with_cached_reference()` (builds an
`MptNode` with a pre-computed reference cache for the overlay path).

### `crates/mpt/src/lib.rs`
Declares/exports the `flat` module: `flatten_trie`, `FlatEthereumState`,
`FlatStateViews`, `FlatStorageEntry`, `FlatTrieView`.

### `crates/executor/client/src/io.rs` (170 lines changed)
- `ClientExecutorInput` gains a lifetime `'a`; `parent_state` becomes
  `FlatEthereumState<'a>` (`#[serde(borrow)]`).
- `witness_db()` split into `verified_views()` (parse + verify tries against
  state/storage roots) and `witness_aux()` (bytecodes + ancestor-header chain →
  block-hash / bytecode lookup tables).
- `TrieDB` now holds `&FlatStateViews` instead of `&EthereumState`; `basic_ref`
  / `storage_ref` decode RLP off the borrowed blobs.
- Adds `into_owned()` and the `LegacyClientExecutorInput` → `ClientExecutorInput`
  conversion.

### `crates/executor/client/src/executor.rs`
`execute()` takes a borrowed input. `INIT_WITNESS_DB` now produces
`(views, block_hashes, bytecodes_by_hash)` and builds `TrieDB` from them.
`COMPUTE_STATE_ROOT` calls `views.post_state_root(&hashed_state)` instead of
`parent_state.update(...).state_root()`.

### Host / bin plumbing
- `full_executor.rs`, `host_executor.rs`: return types become
  `ClientExecutorInput<'static, _>`; host builds `parent_state` via
  `FlatEthereumState::from_state(&state)`; cache load reads into an in-memory
  buffer then `.into_owned()` (input borrows from the buffer).
- `bin/client/src/main.rs`: reads `raw` buffer *before* the profiled
  deserialize block so the zero-copy borrow outlives the input.
- `bin/native-client/src/main.rs`: `--convert-legacy-to <path>` flag.
- `bin/pico-processor/src/main.rs`: `write::<EthClientExecutorInput<'_>>`.
- Cargo: `alloy-rlp` added to client executor; `bincode` dev-dep added to mpt.

## Risks / things to watch
- **Zero-copy lifetime coupling**: the guest input borrows from the raw buffer;
  `bin/client/main.rs` had to hoist `raw` out of the profiled block. Any future
  refactor that drops the buffer early is a use-after-free-shaped bug (caught by
  the borrow checker, but worth flagging).
- **Cycle claim vs. memory note**: re-validate the -21% against the pipeline in
  `memory/cpu-13-rsp-cycle-optimization.md` before relying on it.
- **Delta-pass correctness**: post-state root now comes from a bespoke delta
  engine rather than the audited `MptNode` path; correctness rests on the
  randomized parity sweeps in `flat.rs` tests.
