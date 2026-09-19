//! Flat RLP wire format for the witness tries (CPU-13 item #5).
//!
//! Instead of shipping the sparse MPTs as a bincode-serialized `MptNode` graph (which costs one
//! allocation per node to decode and a full RLP re-encode per node to verify), each trie is
//! shipped as its nodes' raw RLP encodings concatenated in DFS pre-order, root first. Nodes
//! whose encoding is shorter than 32 bytes are inlined in their parent's encoding (exactly as in
//! the MPT hashing spec) and are not emitted separately; unresolved (pruned) subtrees are
//! represented only by the 32-byte digest inside their parent's encoding.
//!
//! The guest verifies the whole structure in one linear pass: every blob after the root must
//! keccak-hash to a digest reference on the current DFS frontier of already-accepted nodes, so
//! every accepted node is committed to by the root hash. Reads walk the raw blobs without
//! allocating. State mutation materializes only the touched paths into a regular sparse
//! [`MptNode`] overlay (untouched siblings stay as digests), on which the existing
//! `update()`/`hash()` machinery runs unchanged.

use std::borrow::Cow;

use alloy_primitives::{
    map::{B256Map, HashMap},
    B256,
};
use alloy_rlp::Encodable;
use reth_trie::HashedPostState;
use serde::{Deserialize, Serialize};

use crate::{
    mpt::{
        keccak, keccak_into_b256, node_from_digest, node_with_cached_reference, prefix_nibs,
        to_nibs, Error, MptNode, MptNodeData, MptNodeReference, EMPTY_ROOT,
    },
    EthereumState,
};

/// Serde helper for `Cow<'a, [u8]>` that borrows from the input when the deserializer supports
/// it (bincode over a byte slice, i.e. the guest path) and falls back to an owned copy when it
/// does not (bincode over a reader, i.e. the host input cache path).
pub mod cow_bytes {
    use std::borrow::Cow;

    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Cow<'_, [u8]>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(v)
    }

    struct CowVisitor;

    impl<'de> serde::de::Visitor<'de> for CowVisitor {
        type Value = Cow<'de, [u8]>;

        fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("bytes")
        }

        fn visit_borrowed_bytes<E>(self, v: &'de [u8]) -> Result<Self::Value, E> {
            Ok(Cow::Borrowed(v))
        }

        fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E> {
            Ok(Cow::Owned(v.to_vec()))
        }

        fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Self::Value, E> {
            Ok(Cow::Owned(v))
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Cow<'de, [u8]>, D::Error> {
        d.deserialize_bytes(CowVisitor)
    }
}

/// The wire representation of [`EthereumState`]: one flat blob region per trie.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlatEthereumState<'a> {
    #[serde(with = "cow_bytes", borrow)]
    pub state_nodes: Cow<'a, [u8]>,
    #[serde(borrow)]
    pub storage_tries: Vec<FlatStorageEntry<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlatStorageEntry<'a> {
    pub hashed_address: B256,
    #[serde(with = "cow_bytes", borrow)]
    pub nodes: Cow<'a, [u8]>,
}

impl FlatEthereumState<'_> {
    /// Flattens a fully-built [`EthereumState`] (host side).
    pub fn from_state(state: &EthereumState) -> FlatEthereumState<'static> {
        let mut storage_tries = state
            .storage_tries
            .iter()
            .map(|(hashed_address, trie)| FlatStorageEntry {
                hashed_address: *hashed_address,
                nodes: Cow::Owned(flatten_trie(trie)),
            })
            .collect::<Vec<_>>();
        // deterministic wire bytes
        storage_tries.sort_by_key(|e| e.hashed_address);

        FlatEthereumState {
            state_nodes: Cow::Owned(flatten_trie(&state.state_trie)),
            storage_tries,
        }
    }

    /// Converts any borrowed wire bytes into owned buffers.
    pub fn into_owned(self) -> FlatEthereumState<'static> {
        FlatEthereumState {
            state_nodes: Cow::Owned(self.state_nodes.into_owned()),
            storage_tries: self
                .storage_tries
                .into_iter()
                .map(|e| FlatStorageEntry {
                    hashed_address: e.hashed_address,
                    nodes: Cow::Owned(e.nodes.into_owned()),
                })
                .collect(),
        }
    }

    /// Parses and cryptographically verifies the internal linkage of every trie, returning
    /// read-only views. Root/storage-root anchoring against headers/accounts is the caller's
    /// responsibility.
    pub fn views(&self) -> Result<FlatStateViews<'_>, Error> {
        let state = FlatTrieView::parse_and_verify(&self.state_nodes)?;
        let mut storage =
            B256Map::with_capacity_and_hasher(self.storage_tries.len(), Default::default());
        for entry in &self.storage_tries {
            storage.insert(entry.hashed_address, FlatTrieView::parse_and_verify(&entry.nodes)?);
        }
        Ok(FlatStateViews { state, storage })
    }
}

/// Emits `root`'s trie as concatenated RLP node blobs in DFS pre-order.
pub fn flatten_trie(root: &MptNode) -> Vec<u8> {
    let mut out = Vec::with_capacity(1024);
    emit_node(root, &mut out, true);
    out
}

fn emit_node(node: &MptNode, out: &mut Vec<u8>, is_root: bool) {
    match node.as_data() {
        MptNodeData::Null => {
            if is_root {
                out.push(alloy_rlp::EMPTY_STRING_CODE);
            }
        }
        MptNodeData::Digest(_) => {
            // Pruned subtree: covered by the digest inside the parent's encoding. Only a
            // digest-only root needs its own blob.
            if is_root {
                node.encode(out);
            }
        }
        MptNodeData::Leaf(..) => node.encode(out),
        MptNodeData::Extension(_, child) => {
            node.encode(out);
            emit_child(child, out);
        }
        MptNodeData::Branch(children) => {
            node.encode(out);
            for child in children.iter().flatten() {
                emit_child(child, out);
            }
        }
    }
}

fn emit_child(child: &MptNode, out: &mut Vec<u8>) {
    // Children referenced by digest get their own blob; short ones are already inlined in the
    // parent's encoding.
    if matches!(child.reference(), MptNodeReference::Digest(_)) {
        emit_node(child, out, false);
    }
}

// --- minimal zero-alloc RLP scanning ------------------------------------------------------

/// (payload_offset, payload_len, is_list), all relative to `bytes[pos..]`'s absolute indices.
#[inline]
fn rlp_header(bytes: &[u8], pos: usize) -> Result<(usize, usize, bool), Error> {
    let err = || Error::FlatTrie("truncated RLP item");
    let b0 = *bytes.get(pos).ok_or_else(err)?;
    // Every arm returns a `(payload_offset, payload_len)` that callers turn straight into
    // `&bytes[payload..payload + len]`, so both the offset and the sum have to be inside the
    // buffer here rather than at each of the ~a dozen call sites. `be_len` bounds `len` by
    // `bytes.len()`, and the check below bounds the sum; neither addition can wrap, because
    // `pos <= bytes.len()` and both addends are under `bytes.len()`.
    //
    // **This is the expensive guard, and it is worth its price.** `rlp_header` runs once per
    // RLP item over the whole witness in `parse_and_verify`, so its two extra comparisons --
    // together with `be_len`'s checked arithmetic and `parse_node`'s restatement -- are
    // **+48.0 M retired instructions across the thirteen `perf/bench_data/rv64` blocks, 89 %
    // of this branch's total cost** (+1.36 M on block 24006677 alone). Measured by building
    // the guest with all three sites reverted and emulating both. It buys the property that a
    // hostile witness is rejected rather than read out of bounds, which no amount of replaying
    // real blocks can check -- see the note on `parse_and_verify`.
    let (payload, len, is_list) = match b0 {
        0x00..=0x7f => (pos, 1, false),
        0x80..=0xb7 => (pos + 1, (b0 - 0x80) as usize, false),
        0xb8..=0xbf => {
            let ll = (b0 - 0xb7) as usize;
            let len = be_len(bytes, pos + 1, ll)?;
            (pos + 1 + ll, len, false)
        }
        0xc0..=0xf7 => (pos + 1, (b0 - 0xc0) as usize, true),
        0xf8..=0xff => {
            let ll = (b0 - 0xf7) as usize;
            let len = be_len(bytes, pos + 1, ll)?;
            (pos + 1 + ll, len, true)
        }
    };
    if payload > bytes.len() || len > bytes.len() - payload {
        return Err(err());
    }
    Ok((payload, len, is_list))
}

/// Decodes a big-endian RLP length of `ll` bytes.
///
/// `ll` is up to 8, the full width of a `usize`, so `len << 8 | b` wraps. The guest builds with
/// `overflow-checks = false` and wraps silently where every host test, Miri pass and fuzz
/// execution panics -- a defect class the suite cannot see, because the shipped profile behaves
/// differently from the tested one. `checked_shl` does not close it: it returns `None` only when
/// the *shift amount* reaches the type width, and the shift here is the constant 8.
///
/// `checked_mul`/`checked_add` are checked in both profiles. The buffer bound goes *inside* the
/// loop: the accumulation only grows, so a value already past `bytes.len()` never comes back
/// under it, and stopping there caps the accumulator at the buffer length -- which is what makes
/// every `payload + len` downstream non-wrapping.
///
/// The bound is on the accumulated value, not on `ll`, because a length may legally carry
/// leading zero bytes. It therefore rejects identically on 32- and 64-bit, where the old code
/// took `0x01_0000_0005` for `5` and read a truncated item as a valid one.
#[inline]
fn be_len(bytes: &[u8], pos: usize, ll: usize) -> Result<usize, Error> {
    let raw = bytes
        .get(pos..pos.checked_add(ll).ok_or(Error::FlatTrie("RLP length overflow"))?)
        .ok_or(Error::FlatTrie("truncated RLP length"))?;
    let mut len = 0usize;
    for &b in raw {
        len = len
            .checked_mul(256)
            .and_then(|acc| acc.checked_add(b as usize))
            .ok_or(Error::FlatTrie("RLP length overflow"))?;
        if len > bytes.len() {
            return Err(Error::FlatTrie("RLP length exceeds the buffer"));
        }
    }
    Ok(len)
}

/// Total encoded length of the RLP item starting at `pos`.
///
/// **Never returns 0** -- two scan loops rely on that to terminate. Every `rlp_header` arm gives
/// either `payload == pos` with `len >= 1`, or `payload > pos`; `rlp_item_len_never_returns_zero`
/// sweeps them. Also bounded by `bytes.len() - pos`.
#[inline]
fn rlp_item_len(bytes: &[u8], pos: usize) -> Result<usize, Error> {
    let (payload, len, _) = rlp_header(bytes, pos)?;
    // `payload >= pos` and `payload + len <= bytes.len()` are both established by
    // `rlp_header`, so neither the subtraction nor the addition can wrap.
    Ok(payload - pos + len)
}

/// A child slot inside a node blob.
#[derive(Debug, Clone, Copy)]
enum FlatRef<'a> {
    Empty,
    Digest(&'a [u8]),
    /// Full RLP bytes of an inlined (< 32 byte) node.
    Inline(&'a [u8]),
}

#[derive(Debug, Clone, Copy)]
enum FlatNode<'a> {
    Null,
    Digest(&'a [u8]),
    Leaf {
        prefix: &'a [u8],
        value: &'a [u8],
    },
    Extension {
        prefix: &'a [u8],
        child: FlatRef<'a>,
    },
    /// Payload region of the 17-item list.
    Branch {
        payload: &'a [u8],
    },
}

/// Parses one node blob (`bytes` must be exactly the node's RLP encoding).
fn parse_node(bytes: &[u8]) -> Result<FlatNode<'_>, Error> {
    let (payload, len, is_list) = rlp_header(bytes, 0)?;
    if payload + len != bytes.len() {
        return Err(Error::FlatTrie("node blob length mismatch"));
    }
    if !is_list {
        return match len {
            0 => Ok(FlatNode::Null),
            32 => Ok(FlatNode::Digest(&bytes[payload..payload + 32])),
            _ => Err(Error::FlatTrie("unexpected string node")),
        };
    }
    let body = &bytes[payload..payload + len];

    // count and locate items
    let mut items = [(0usize, 0usize); 17];
    let mut n = 0usize;
    let mut pos = 0usize;
    while pos < body.len() {
        if n == 17 {
            return Err(Error::FlatTrie("too many items in node"));
        }
        let item_len = rlp_item_len(body, pos)?;
        // Re-asserted rather than assumed: it already follows from `rlp_header`, but this is the
        // parser that has to fail closed on a hostile witness.
        if item_len > body.len() - pos {
            return Err(Error::FlatTrie("item runs past the node"));
        }
        items[n] = (pos, item_len);
        n += 1;
        pos += item_len;
    }

    match n {
        2 => {
            let (p0, _) = items[0];
            let (h0, pl0, list0) = rlp_header(body, p0)?;
            if list0 {
                return Err(Error::FlatTrie("path is a list"));
            }
            let prefix = &body[h0..h0 + pl0];
            if prefix.is_empty() {
                return Err(Error::FlatTrie("empty path prefix"));
            }
            if prefix[0] & 0x20 != 0 {
                let (p1, _) = items[1];
                let (h1, pl1, list1) = rlp_header(body, p1)?;
                if list1 {
                    return Err(Error::FlatTrie("leaf value is a list"));
                }
                Ok(FlatNode::Leaf { prefix, value: &body[h1..h1 + pl1] })
            } else {
                let (p1, l1) = items[1];
                Ok(FlatNode::Extension { prefix, child: parse_ref(&body[p1..p1 + l1])? })
            }
        }
        17 => {
            let (p16, _) = items[16];
            let (_, pl16, _) = rlp_header(body, p16)?;
            if pl16 != 0 {
                return Err(Error::FlatTrie("branch node with value"));
            }
            Ok(FlatNode::Branch { payload: body })
        }
        _ => Err(Error::FlatTrie("unexpected node item count")),
    }
}

/// Parses one child-slot item (`bytes` = exactly the item's RLP encoding).
fn parse_ref(bytes: &[u8]) -> Result<FlatRef<'_>, Error> {
    let (payload, len, is_list) = rlp_header(bytes, 0)?;
    if is_list {
        return Ok(FlatRef::Inline(bytes));
    }
    match len {
        0 => Ok(FlatRef::Empty),
        32 => Ok(FlatRef::Digest(&bytes[payload..payload + 32])),
        _ => Err(Error::FlatTrie("unexpected child reference")),
    }
}

/// Iterates the child-slot items of a branch payload: `f(slot_index, item)`.
fn for_branch_children<'a>(
    payload: &'a [u8],
    mut f: impl FnMut(usize, FlatRef<'a>) -> Result<(), Error>,
) -> Result<(), Error> {
    let mut pos = 0usize;
    for slot in 0..16 {
        let item_len = rlp_item_len(payload, pos)?;
        f(slot, parse_ref(&payload[pos..pos + item_len])?)?;
        pos += item_len;
    }
    Ok(())
}

/// Returns the child-slot item `slot` of a branch payload.
fn branch_child(payload: &[u8], slot: usize) -> Result<FlatRef<'_>, Error> {
    let mut pos = 0usize;
    for _ in 0..slot {
        pos += rlp_item_len(payload, pos)?;
    }
    let item_len = rlp_item_len(payload, pos)?;
    parse_ref(&payload[pos..pos + item_len])
}

// --- verified view --------------------------------------------------------------------------

const EDGE_PRUNED: u32 = u32::MAX;
const EDGE_INLINE: u32 = u32::MAX - 1;

const KIND_LEAF: u8 = 0;
const KIND_EXT: u8 = 1;
const KIND_BRANCH: u8 = 2;
const KIND_DIGEST: u8 = 3;

#[derive(Debug, Clone, Copy)]
struct NodeRec {
    off: u32,
    len: u32,
    /// start into `edges`; branches own 16 slots, extensions 1, leaves 0.
    edge_start: u32,
    kind: u8,
}

/// A parsed, linkage-verified flat trie.
#[derive(Debug)]
pub struct FlatTrieView<'a> {
    bytes: &'a [u8],
    pub root_hash: B256,
    nodes: Vec<NodeRec>,
    /// keccak of each node's blob (computed during verification)
    hashes: Vec<B256>,
    edges: Vec<u32>,
}

/// A DFS-frontier entry with an incremental cursor over a node's child-slot items.
struct FrontierEntry {
    node_idx: u32,
    /// absolute offset of the next unscanned child-slot item
    item_pos: u32,
    /// absolute end of the node's item region
    items_end: u32,
    slot: u8,
    nslots: u8,
}

impl FrontierEntry {
    fn new(
        node_idx: u32,
        blob_off: u32,
        node: &FlatNode<'_>,
        bytes: &[u8],
    ) -> Result<Option<Self>, Error> {
        match node {
            FlatNode::Extension { .. } => {
                let (payload_off, payload_len, _) = rlp_header(bytes, blob_off as usize)?;
                let item1 = payload_off + rlp_item_len(bytes, payload_off)?;
                Ok(Some(FrontierEntry {
                    node_idx,
                    item_pos: item1 as u32,
                    items_end: (payload_off + payload_len) as u32,
                    slot: 0,
                    nslots: 1,
                }))
            }
            FlatNode::Branch { .. } => {
                let (payload_off, payload_len, _) = rlp_header(bytes, blob_off as usize)?;
                Ok(Some(FrontierEntry {
                    node_idx,
                    item_pos: payload_off as u32,
                    items_end: (payload_off + payload_len) as u32,
                    slot: 0,
                    nslots: 16,
                }))
            }
            _ => Ok(None),
        }
    }
}

impl<'a> FlatTrieView<'a> {
    /// Single linear pass: keccak every blob, check it against a pending digest reference on
    /// the DFS frontier, and record child edges.
    pub fn parse_and_verify(bytes: &'a [u8]) -> Result<Self, Error> {
        let mut view = FlatTrieView {
            bytes,
            root_hash: EMPTY_ROOT,
            nodes: Vec::with_capacity(bytes.len() / 96 + 4),
            hashes: Vec::with_capacity(bytes.len() / 96 + 4),
            edges: Vec::with_capacity(bytes.len() / 32 + 4),
        };

        if bytes.is_empty() {
            // An empty region encodes the empty trie.
            return Ok(view);
        }

        let root_len = rlp_item_len(bytes, 0)?;
        // `rlp_item_len` bounds this by `bytes.len()` now, but say so where the slice is
        // taken: a truncated root item used to panic here rather than return an error. The
        // panic aborts under `-Cpanic=abort` and so fails closed, but a malformed witness
        // should be a rejection, not a crash.
        if root_len > bytes.len() {
            return Err(Error::FlatTrie("truncated root item"));
        }
        let root_blob = &bytes[..root_len];
        let root = parse_node(root_blob)?;
        match root {
            FlatNode::Null => {
                if root_len != bytes.len() {
                    return Err(Error::FlatTrie("data after null root"));
                }
                return Ok(view);
            }
            FlatNode::Digest(d) => {
                if root_len != bytes.len() {
                    return Err(Error::FlatTrie("data after digest root"));
                }
                view.root_hash = B256::from_slice(d);
                view.nodes.push(NodeRec {
                    off: 0,
                    len: root_len as u32,
                    edge_start: 0,
                    kind: KIND_DIGEST,
                });
                view.hashes.push(view.root_hash);
                return Ok(view);
            }
            _ => {}
        }
        keccak_into_b256(root_blob, &mut view.root_hash);
        view.push_node(0, root_len, &root)?;
        view.hashes.push(view.root_hash);

        // DFS frontier. Each entry keeps an incremental cursor over the node's child-slot
        // items so every item is scanned exactly once across the whole pass.
        let mut frontier: Vec<FrontierEntry> = Vec::with_capacity(64);
        if let Some(entry) = FrontierEntry::new(0, 0, &root, bytes)? {
            frontier.push(entry);
        }

        let mut pos = root_len;
        while pos < bytes.len() {
            let len = rlp_item_len(bytes, pos)?;
            // `bytes.len() - pos`, not `pos + len`: the sum is what wrapped in the profile the
            // guest ships, turning "truncated" into "accepted". `rlp_header` establishes the same
            // bound, so this is a restatement at the point the slice is taken.
            if len > bytes.len() - pos {
                return Err(Error::FlatTrie("truncated node blob"));
            }
            let blob = &bytes[pos..pos + len];
            // 8-aligned so the digest lands as four `sd`; see `keccak_into_b256`.
            #[repr(align(8))]
            struct Digest(B256);
            let mut digest = Digest(B256::ZERO);
            keccak_into_b256(blob, &mut digest.0);
            let hash = digest.0;

            // Find the next pending digest reference matching this blob's hash. Non-matching
            // references we walk past are pruned subtrees and stay EDGE_PRUNED.
            let node_idx = 'search: loop {
                let Some(top) = frontier.last_mut() else {
                    return Err(Error::FlatTrie("blob does not attach to the trie"));
                };
                while top.item_pos < top.items_end && top.slot < top.nslots {
                    let (payload_off, payload_len, is_list) =
                        rlp_header(bytes, top.item_pos as usize)?;
                    let item_end = payload_off + payload_len;
                    let slot = top.slot;
                    top.item_pos = item_end as u32;
                    top.slot += 1;
                    if !is_list && payload_len == 32 && bytes[payload_off..item_end] == hash[..] {
                        let idx = view.nodes.len() as u32;
                        let rec = view.nodes[top.node_idx as usize];
                        view.edges[rec.edge_start as usize + slot as usize] = idx;
                        break 'search idx;
                    }
                }
                frontier.pop();
            };

            let node = parse_node(blob)?;
            if matches!(node, FlatNode::Null | FlatNode::Digest(_)) {
                return Err(Error::FlatTrie("null/digest blob below root"));
            }
            let this_idx = view.nodes.len() as u32;
            view.push_node(pos, len, &node)?;
            view.hashes.push(hash);
            let _ = node_idx;
            if let Some(entry) = FrontierEntry::new(this_idx, pos as u32, &node, bytes)? {
                frontier.push(entry);
            }
            pos += len;
        }

        Ok(view)
    }

    fn push_node(&mut self, off: usize, len: usize, node: &FlatNode<'_>) -> Result<(), Error> {
        let edge_start = self.edges.len() as u32;
        let kind = match node {
            FlatNode::Leaf { .. } => KIND_LEAF,
            FlatNode::Extension { .. } => KIND_EXT,
            FlatNode::Branch { .. } => KIND_BRANCH,
            _ => KIND_DIGEST,
        };
        self.nodes.push(NodeRec { off: off as u32, len: len as u32, edge_start, kind });
        match node {
            FlatNode::Extension { child, .. } => {
                self.edges.push(match child {
                    FlatRef::Digest(_) => EDGE_PRUNED,
                    FlatRef::Inline(_) => EDGE_INLINE,
                    FlatRef::Empty => return Err(Error::FlatTrie("extension with empty child")),
                });
            }
            FlatNode::Branch { .. } => {
                // all slots default to PRUNED; digest children get their edge patched by the
                // frontier matching, and inline/empty slots are resolved directly from the
                // blob during walks (the edge value is never consulted for them)
                let base = self.edges.len();
                self.edges.resize(base + 16, EDGE_PRUNED);
            }
            _ => {}
        }
        Ok(())
    }

    fn blob(&self, idx: u32) -> &'a [u8] {
        let rec = self.nodes[idx as usize];
        &self.bytes[rec.off as usize..(rec.off + rec.len) as usize]
    }

    /// Parses a verified node by index using its recorded kind, skipping the full structural
    /// validation that already ran during `parse_and_verify`. For branches this avoids scanning
    /// all 17 items.
    fn parse_indexed(&self, idx: u32) -> Result<FlatNode<'a>, Error> {
        let rec = self.nodes[idx as usize];
        let blob = self.blob(idx);
        match rec.kind {
            KIND_BRANCH => {
                let (payload, len, _) = rlp_header(blob, 0)?;
                Ok(FlatNode::Branch { payload: &blob[payload..payload + len] })
            }
            KIND_LEAF | KIND_EXT => {
                let (payload, len, _) = rlp_header(blob, 0)?;
                let body = &blob[payload..payload + len];
                let (h0, pl0, _) = rlp_header(body, 0)?;
                let prefix = &body[h0..h0 + pl0];
                let item1 = h0 + pl0;
                if rec.kind == KIND_LEAF {
                    let (h1, pl1, _) = rlp_header(body, item1)?;
                    Ok(FlatNode::Leaf { prefix, value: &body[h1..h1 + pl1] })
                } else {
                    let l1 = rlp_item_len(body, item1)?;
                    Ok(FlatNode::Extension { prefix, child: parse_ref(&body[item1..item1 + l1])? })
                }
            }
            _ => {
                let (h, _, _) = rlp_header(blob, 0)?;
                Ok(FlatNode::Digest(&blob[h..h + 32]))
            }
        }
    }

    /// Whether the trie is completely empty (hash == EMPTY_ROOT).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Retrieves the value for `key` (full key bytes, e.g. a 32-byte hashed key), walking the
    /// raw blobs without allocating.
    ///
    /// Returns `None` for a key the witness proves absent, and **`Err(NodeNotResolved)` for a
    /// key whose path leaves the witnessed region** -- a subtree the prover declined to
    /// encode, represented by its digest. Those two are different answers and must not be
    /// confused: the digest keeps the root hash correct, so `parse_and_verify` and the anchor
    /// check both pass, and reporting "absent" for the second makes omission a way to make an
    /// account or a slot read as zero. See the matching arm in `MptNode::get_internal` for
    /// what that buys an attacker and why the write path was never exposed to it.
    pub fn get(&self, key: &[u8]) -> Result<Option<&'a [u8]>, Error> {
        if self.is_empty() {
            return Ok(None);
        }
        let nkey = key.len() * 2;
        let mut node_idx = 0u32;
        let mut blob = self.blob(0);
        let mut inline = false;
        let mut pos = 0usize; // nibble cursor

        loop {
            let node = if inline { parse_node(blob)? } else { self.parse_indexed(node_idx)? };
            match node {
                FlatNode::Null => return Ok(None),
                // The root itself is a digest: the whole trie is outside the witness.
                FlatNode::Digest(d) => {
                    return Err(Error::NodeNotResolved(B256::from_slice(d)));
                }
                FlatNode::Leaf { prefix, value } => {
                    return Ok(match match_prefix(prefix, key, pos) {
                        Some(p) if p == nkey => Some(value),
                        _ => None,
                    });
                }
                FlatNode::Extension { prefix, child } => {
                    let Some(p) = match_prefix(prefix, key, pos) else { return Ok(None) };
                    if p > nkey {
                        return Ok(None);
                    }
                    pos = p;
                    match child {
                        FlatRef::Inline(b) => {
                            blob = b;
                            inline = true;
                        }
                        FlatRef::Digest(d) => {
                            if inline {
                                return Err(Error::FlatTrie("digest ref inside inline node"));
                            }
                            let edge =
                                self.edges[self.nodes[node_idx as usize].edge_start as usize];
                            if edge == EDGE_PRUNED {
                                // Not "absent": unwitnessed. See the note on `get`.
                                return Err(Error::NodeNotResolved(B256::from_slice(d)));
                            }
                            node_idx = edge;
                            blob = self.blob(edge);
                        }
                        FlatRef::Empty => return Err(Error::FlatTrie("empty extension child")),
                    }
                }
                FlatNode::Branch { payload } => {
                    if pos >= nkey {
                        return Ok(None);
                    }
                    let slot = nib_at(key, pos) as usize;
                    pos += 1;

                    // Fast path: a resolved digest child's node index is already in the edge
                    // table (recorded during verification), so descend in O(1) without
                    // rescanning the branch payload. This covers the entire interior of the
                    // descent; only inline/empty/pruned slots fall back to the O(slot) parse.
                    if !inline {
                        let edge =
                            self.edges[self.nodes[node_idx as usize].edge_start as usize + slot];
                        if edge != EDGE_PRUNED {
                            node_idx = edge;
                            blob = self.blob(edge);
                            continue;
                        }
                    }

                    match branch_child(payload, slot)? {
                        FlatRef::Empty => return Ok(None),
                        FlatRef::Inline(b) => {
                            blob = b;
                            inline = true;
                        }
                        FlatRef::Digest(d) => {
                            if inline {
                                return Err(Error::FlatTrie("digest ref inside inline node"));
                            }
                            // Not inline: the edge was EDGE_PRUNED (else the fast path took
                            // it), so this digest child is a subtree the witness does not
                            // cover. See the note on `get`: that is not the same answer as
                            // "absent" and must not be reported as one.
                            return Err(Error::NodeNotResolved(B256::from_slice(d)));
                        }
                    }
                }
            }
        }
    }

    /// Materializes a sparse `MptNode` overlay (the module-private node graph) containing the
    /// full paths for every key in
    /// `keys` (`(hashed_key, is_delete)`); everything off-path stays a digest stub. For delete
    /// keys, the remaining sibling of any 2-child branch on the path is materialized one level
    /// deep so that branch-collapse during `delete()` sees its real shape.
    pub fn materialize(&self, keys: &[(B256, bool)]) -> Result<MptNode, Error> {
        if self.is_empty() {
            return Ok(MptNode::default());
        }
        let nibs: Vec<(Vec<u8>, bool)> =
            keys.iter().map(|(k, del)| (to_nibs(k.as_slice()), *del)).collect();
        let key_refs: Vec<(&[u8], bool)> =
            nibs.iter().map(|(n, del)| (n.as_slice(), *del)).collect();
        self.mat(Src::Node(0), &key_refs)
    }

    fn mat(&self, src: Src<'a>, keys: &[(&[u8], bool)]) -> Result<MptNode, Error> {
        let blob = match src {
            Src::Node(idx) => self.blob(idx),
            Src::Inline(b) => b,
        };
        let node = match src {
            Src::Node(idx) => self.parse_indexed(idx)?,
            Src::Inline(b) => parse_node(b)?,
        };
        let data = match node {
            FlatNode::Null => return Ok(MptNode::default()),
            FlatNode::Digest(d) => return Ok(MptNodeData::Digest(B256::from_slice(d)).into()),
            FlatNode::Leaf { prefix, value } => MptNodeData::Leaf(prefix.to_vec(), value.to_vec()),
            FlatNode::Extension { prefix, child } => {
                let pn = prefix_nibs(prefix);
                let remaining: Vec<(&[u8], bool)> = keys
                    .iter()
                    .filter(|(k, _)| k.len() >= pn.len() && k[..pn.len()] == pn[..])
                    .map(|(k, del)| (&k[pn.len()..], *del))
                    .collect();
                let child_node = if remaining.is_empty() {
                    self.child_stub(src, child, 0)?
                } else {
                    self.mat(self.child_src(src, child, 0)?, &remaining)?
                };
                MptNodeData::Extension(prefix.to_vec(), Box::new(child_node))
            }
            FlatNode::Branch { payload } => {
                // group keys by their next nibble
                let mut groups: [Vec<(&[u8], bool)>; 16] = Default::default();
                let mut has_delete = false;
                for (k, del) in keys {
                    if k.is_empty() {
                        continue;
                    }
                    has_delete |= *del;
                    groups[k[0] as usize].push((&k[1..], *del));
                }
                // single scan over the child slots
                let mut refs = [FlatRef::Empty; 16];
                let mut child_count = 0usize;
                for_branch_children(payload, |slot, r| {
                    if !matches!(r, FlatRef::Empty) {
                        child_count += 1;
                    }
                    refs[slot] = r;
                    Ok(())
                })?;
                // a delete through a 2-child branch may collapse it: the surviving sibling's
                // real shape is then needed, so materialize all children one level deep.
                let force_shallow = has_delete && child_count == 2;

                let mut children: [Option<Box<MptNode>>; 16] = Default::default();
                for (slot, r) in refs.into_iter().enumerate() {
                    if matches!(r, FlatRef::Empty) {
                        continue;
                    }
                    let child_node = if !groups[slot].is_empty() {
                        self.mat(self.child_src(src, r, slot as u32)?, &groups[slot])?
                    } else if force_shallow {
                        match self.child_src(src, r, slot as u32) {
                            Ok(csrc) => self.mat(csrc, &[])?,
                            // pruned sibling: fall back to a digest stub (same failure mode
                            // as an absent orphan in the graph representation)
                            Err(_) => self.child_stub(src, r, slot as u32)?,
                        }
                    } else {
                        self.child_stub(src, r, slot as u32)?
                    };
                    children[slot] = Some(Box::new(child_node));
                }
                MptNodeData::Branch(children)
            }
        };

        // Pre-fill the reference cache: the materialized node encodes identically to its wire
        // blob (digest-stub children carry the same references), so its reference is already
        // known. This lets the post-update root recomputation reuse hashes for all untouched
        // materialized nodes instead of re-hashing them.
        let reference = match src {
            Src::Node(idx) if blob.len() >= 32 => {
                MptNodeReference::Digest(self.hashes[idx as usize])
            }
            _ => MptNodeReference::Bytes(blob.to_vec()),
        };
        Ok(node_with_cached_reference(data, reference))
    }

    /// A child as a stub: digest nodes for pruned/witnessed blob children, fully-materialized
    /// tiny nodes for inline children (they have no digest to stub with).
    fn child_stub(&self, _parent: Src<'a>, r: FlatRef<'a>, _slot: u32) -> Result<MptNode, Error> {
        match r {
            FlatRef::Digest(d) => Ok(MptNodeData::Digest(B256::from_slice(d)).into()),
            FlatRef::Inline(b) => self.mat(Src::Inline(b), &[]),
            FlatRef::Empty => Err(Error::FlatTrie("stub for empty child")),
        }
    }

    /// Resolves a child reference to a materialization source. Fails for pruned children.
    fn child_src(&self, parent: Src<'a>, r: FlatRef<'a>, slot: u32) -> Result<Src<'a>, Error> {
        match r {
            FlatRef::Inline(b) => Ok(Src::Inline(b)),
            FlatRef::Digest(_) => match parent {
                Src::Inline(_) => Err(Error::FlatTrie("digest ref inside inline node")),
                Src::Node(idx) => {
                    let edge =
                        self.edges[self.nodes[idx as usize].edge_start as usize + slot as usize];
                    if edge == EDGE_PRUNED || edge == EDGE_INLINE {
                        return Err(Error::FlatTrie("descend into pruned subtree"));
                    }
                    Ok(Src::Node(edge))
                }
            },
            FlatRef::Empty => Err(Error::FlatTrie("descend into empty child")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Src<'a> {
    Node(u32),
    Inline(&'a [u8]),
}

#[inline]
fn nib_at(key: &[u8], i: usize) -> u8 {
    let b = key[i >> 1];
    if i & 1 == 0 {
        b >> 4
    } else {
        b & 0xf
    }
}

/// Matches a compact-encoded prefix against `key`'s nibbles starting at `pos`; returns the new
/// nibble position on full match.
fn match_prefix(prefix: &[u8], key: &[u8], mut pos: usize) -> Option<usize> {
    let nkey = key.len() * 2;
    let flag = prefix[0];
    if flag & 0x10 != 0 {
        if pos >= nkey || nib_at(key, pos) != flag & 0xf {
            return None;
        }
        pos += 1;
    }
    for &b in &prefix[1..] {
        if pos + 2 > nkey || nib_at(key, pos) != b >> 4 || nib_at(key, pos + 1) != b & 0xf {
            return None;
        }
        pos += 2;
    }
    Some(pos)
}

/// Verified views over all tries of a [`FlatEthereumState`].
#[derive(Debug)]
pub struct FlatStateViews<'a> {
    pub state: FlatTrieView<'a>,
    pub storage: B256Map<FlatTrieView<'a>>,
}

impl FlatStateViews<'_> {
    /// Builds the copy-on-write [`EthereumState`] overlay for a state transition: paths for all
    /// effectively-changed accounts/slots are materialized, untouched storage tries become
    /// digest stubs. Running the existing `update()` + `state_root()` on the overlay with the
    /// returned (filtered) post state yields the exact post-state root.
    ///
    /// Every account the post state modifies must get a `storage_tries` entry, because
    /// [`EthereumState::update`] reaches for `entry(addr).or_default()` and a *fresh empty* trie
    /// rewrites the account's row with its storage wiped. A plain value transfer to a contract
    /// reaches that: it touches the account but makes no storage access, so nothing else notices
    /// the witness has no trie for it.
    ///
    /// Entries with no witnessed trie are seeded from `prior_storage_root` -- a digest stub when
    /// the account had storage, a real empty trie when it had none -- and the case that cannot be
    /// answered is rejected here rather than left for `update` to `unwrap`. The three-way split
    /// mirrors [`Self::post_state_root`] case for case; `overlay_state_parity` is a cross-check of
    /// the shipped path only while the two agree.
    pub fn materialize_overlay(
        &self,
        post_state: &HashedPostState,
    ) -> Result<EthereumState, Error> {
        let account_keys: Vec<(B256, bool)> =
            post_state.accounts.iter().map(|(k, a)| (*k, a.is_none())).collect();
        let state_trie = self.state.materialize(&account_keys)?;

        let mut storage_tries =
            HashMap::with_capacity_and_hasher(self.storage.len(), Default::default());
        for (hashed_address, view) in &self.storage {
            let trie = match post_state.storages.get(hashed_address) {
                Some(st) => {
                    let keys: Vec<(B256, bool)> =
                        st.storage.iter().map(|(k, v)| (*k, v.is_zero())).collect();
                    view.materialize(&keys)?
                }
                None => node_from_digest(view.root_hash),
            };
            storage_tries.insert(*hashed_address, trie);
        }

        for (hashed_address, account) in post_state.accounts.iter() {
            if account.is_none() || self.storage.contains_key(hashed_address) {
                continue;
            }
            let storage = post_state.storages.get(hashed_address);
            let trie = if storage.is_some_and(|st| st.wiped) {
                // `wiped` is legitimate -- a destroyed or freshly created account starts from
                // the empty trie, so its prior root does not come into it.
                MptNode::default()
            } else {
                match self.prior_storage_root(hashed_address)? {
                    // No storage before this block: the delta, if any, applies to the empty
                    // trie, which is the one case where `or_default()` was already right.
                    root if root == FLAT_EMPTY_ROOT => MptNode::default(),
                    // Non-empty storage and no witnessed trie. A stub carries the root through
                    // for an account whose storage is untouched; if the block changed a slot
                    // there is nothing to apply it to, and `post_state_root` rejects the same
                    // witness rather than computing a root that drops the storage.
                    root if storage.is_none() => node_from_digest(root),
                    _ => {
                        return Err(Error::FlatTrie(
                            "no witnessed storage trie for a modified account",
                        ))
                    }
                }
            };
            storage_tries.insert(*hashed_address, trie);
        }

        Ok(EthereumState { state_trie, storage_tries })
    }
}

#[cfg(test)]
mod tests {
    use alloy_primitives::U256;
    use reth_primitives_traits::Account;
    use reth_trie::{HashedPostState, HashedStorage, TrieAccount};

    use super::*;
    use crate::mpt::{MptNodeReference, KECCAK_EMPTY};

    fn keccak_trie(n: usize) -> MptNode {
        let mut trie = MptNode::default();
        for i in 0..n {
            trie.insert_rlp(&keccak(i.to_be_bytes()), i as u64 + 1).unwrap();
        }
        trie
    }

    #[test]
    fn flat_roundtrip_hash_and_get() {
        for n in [0usize, 1, 2, 3, 17, 100, 512] {
            let trie = keccak_trie(n);
            let bytes = flatten_trie(&trie);
            let view = FlatTrieView::parse_and_verify(&bytes).unwrap();
            assert_eq!(view.root_hash, trie.hash(), "n={n}");

            for i in 0..n + 20 {
                let key = keccak(i.to_be_bytes());
                let expected = trie.get(&key).unwrap();
                let got = view.get(&key).unwrap();
                assert_eq!(got, expected, "n={n} i={i}");
            }
        }
    }

    #[test]
    fn flat_inline_children() {
        // short keys/values produce inline (< 32 byte) nodes, like mpt::tests::test_tiny
        let mut trie = MptNode::default();
        trie.insert_rlp(b"a", 0u8).unwrap();
        trie.insert_rlp(b"b", 1u8).unwrap();
        let bytes = flatten_trie(&trie);
        let view = FlatTrieView::parse_and_verify(&bytes).unwrap();
        assert_eq!(view.root_hash, trie.hash());
        assert_eq!(view.get(b"a").unwrap(), trie.get(b"a").unwrap());
        assert_eq!(view.get(b"b").unwrap(), trie.get(b"b").unwrap());
        assert_eq!(view.get(b"c").unwrap(), None);
    }

    /// A pruned subtree verifies, and every key whose path enters it is **refused**, not
    /// reported absent.
    ///
    /// The two answers used to be the same one (`Ok(None)`), which is the fail-open half of
    /// the witness format: the digest keeps the root hash right, so verification and the
    /// anchor check both pass, and "the prover declined to encode this" arrives at the EVM as
    /// "this account does not exist". The flat view and the node graph must agree on refusing
    /// it, because the node graph is the oracle the differential harness compares against.
    #[test]
    fn flat_pruned_subtree() {
        let trie = keccak_trie(64);
        let MptNodeData::Branch(children) = trie.as_data().clone() else {
            panic!("expected branch root")
        };
        let mut pruned_children = children;
        let victim = pruned_children.iter_mut().flatten().next().expect("at least one child");
        **victim = node_from_digest(victim.hash());
        let pruned: MptNode = MptNodeData::Branch(pruned_children).into();
        assert_eq!(pruned.hash(), trie.hash());

        let bytes = flatten_trie(&pruned);
        let view = FlatTrieView::parse_and_verify(&bytes).unwrap();
        assert_eq!(view.root_hash, trie.hash());

        let mut pruned_hits = 0;
        for i in 0..64usize {
            let key = keccak(i.to_be_bytes());
            match pruned.get(&key) {
                Ok(expected) => {
                    // Inside the witness: the two paths must agree value for value, and the
                    // key must actually be there (these are all keys of the trie).
                    assert_eq!(view.get(&key).unwrap(), expected, "key {i}");
                    assert!(expected.is_some(), "key {i} vanished from the witnessed part");
                }
                Err(Error::NodeNotResolved(_)) => {
                    pruned_hits += 1;
                    assert!(
                        matches!(view.get(&key), Err(Error::NodeNotResolved(_))),
                        "key {i} descends into the pruned subtree; the flat view reported \
                         {:?} instead of refusing it",
                        view.get(&key)
                    );
                }
                Err(e) => panic!("unexpected error for key {i}: {e:?}"),
            }
        }
        assert!(pruned_hits > 0, "the pruned subtree should hide some keys");
    }

    #[test]
    fn flat_digest_and_null_roots() {
        let null = MptNode::default();
        let bytes = flatten_trie(&null);
        let view = FlatTrieView::parse_and_verify(&bytes).unwrap();
        assert_eq!(view.root_hash, EMPTY_ROOT);
        assert!(view.is_empty());

        let digest_root = node_from_digest(B256::repeat_byte(0x42));
        let bytes = flatten_trie(&digest_root);
        let view = FlatTrieView::parse_and_verify(&bytes).unwrap();
        assert_eq!(view.root_hash, B256::repeat_byte(0x42));
        // A digest root is a trie that is entirely outside the witness: it answers nothing.
        // (`Ok(None)` here would say every account in the world is non-existent.)
        assert!(matches!(
            view.get(&keccak(b"x")),
            Err(Error::NodeNotResolved(d)) if d == B256::repeat_byte(0x42)
        ));
    }

    #[test]
    fn flat_tamper_rejected() {
        let trie = keccak_trie(50);
        let mut bytes = flatten_trie(&trie);
        // flip a byte in a non-root node's region
        let root_len = rlp_item_len(&bytes, 0).unwrap();
        bytes[root_len + 10] ^= 0x01;
        assert!(FlatTrieView::parse_and_verify(&bytes).is_err());
    }

    #[test]
    fn materialize_update_parity() {
        // apply the same batch of inserts/updates/deletes to (a) the full graph and (b) a
        // materialized overlay; roots must agree
        const N: usize = 300;
        let trie = keccak_trie(N);
        let bytes = flatten_trie(&trie);
        let view = FlatTrieView::parse_and_verify(&bytes).unwrap();

        let mut keys: Vec<(B256, bool)> = Vec::new();
        let mut ops: Vec<([u8; 32], Option<u64>)> = Vec::new();
        for i in 0..10usize {
            let k = keccak(i.to_be_bytes());
            keys.push((B256::from(k), false));
            ops.push((k, Some(1_000_000 + i as u64)));
        }
        for i in 10..40usize {
            let k = keccak(i.to_be_bytes());
            keys.push((B256::from(k), true));
            ops.push((k, None));
        }
        for i in N..N + 10 {
            let k = keccak(i.to_be_bytes());
            keys.push((B256::from(k), false));
            ops.push((k, Some(i as u64)));
        }

        let mut full = trie.clone();
        let mut overlay = view.materialize(&keys).unwrap();
        for (k, v) in &ops {
            match v {
                Some(v) => {
                    full.insert_rlp(k, *v).unwrap();
                    overlay.insert_rlp(k, *v).unwrap();
                }
                None => {
                    full.delete(k).unwrap();
                    overlay.delete(k).unwrap();
                }
            }
        }
        assert_eq!(overlay.hash(), full.hash());
    }

    #[test]
    fn materialize_delete_all_and_reinsert() {
        // stress collapse cascades: delete most keys, insert a few
        const N: usize = 64;
        let trie = keccak_trie(N);
        let bytes = flatten_trie(&trie);
        let view = FlatTrieView::parse_and_verify(&bytes).unwrap();

        let mut keys: Vec<(B256, bool)> = Vec::new();
        for i in 0..N {
            keys.push((B256::from(keccak(i.to_be_bytes())), i % 2 == 0));
        }
        let mut full = trie.clone();
        let mut overlay = view.materialize(&keys).unwrap();
        for i in 0..N {
            let k = keccak(i.to_be_bytes());
            if i % 2 == 0 {
                full.delete(&k).unwrap();
                overlay.delete(&k).unwrap();
            } else {
                full.insert_rlp(&k, 7777u64 + i as u64).unwrap();
                overlay.insert_rlp(&k, 7777u64 + i as u64).unwrap();
            }
            assert_eq!(overlay.hash(), full.hash(), "i={i}");
        }
    }

    #[test]
    fn overlay_state_parity() {
        // end-to-end EthereumState parity: full graph update vs flat overlay update
        let mut state_trie = MptNode::default();
        let mut storage_a = MptNode::default();
        for i in 0..100usize {
            storage_a.insert_rlp(&keccak(i.to_be_bytes()), U256::from(i + 7)).unwrap();
        }
        let mut storage_b = MptNode::default();
        for i in 0..5usize {
            storage_b.insert_rlp(&keccak(i.to_be_bytes()), U256::from(i + 9)).unwrap();
        }

        let addr_a = B256::from(keccak(b"account-a"));
        let addr_b = B256::from(keccak(b"account-b"));
        let addr_c = B256::from(keccak(b"account-c")); // new account
        for (addr, storage, bal) in [(addr_a, &storage_a, 100u64), (addr_b, &storage_b, 200u64)] {
            let account = TrieAccount {
                nonce: 1,
                balance: U256::from(bal),
                storage_root: storage.hash(),
                code_hash: KECCAK_EMPTY,
            };
            state_trie.insert_rlp(addr.as_slice(), account).unwrap();
        }
        // filler accounts so the state trie has real structure
        for i in 0..50usize {
            let account = TrieAccount {
                nonce: i as u64,
                balance: U256::from(i),
                storage_root: EMPTY_ROOT,
                code_hash: KECCAK_EMPTY,
            };
            state_trie.insert_rlp(&keccak((1000 + i).to_be_bytes()), account).unwrap();
        }

        let mut storage_tries: HashMap<B256, MptNode> = HashMap::default();
        storage_tries.insert(addr_a, storage_a);
        storage_tries.insert(addr_b, storage_b);
        let state = EthereumState { state_trie, storage_tries };

        // post state: touch A (slot delete + update + new slot), create C, leave B alone
        let mut post = HashedPostState::default();
        let mut storage_changes = HashedStorage::new(false);
        storage_changes.storage.insert(B256::from(keccak(0usize.to_be_bytes())), U256::ZERO);
        storage_changes.storage.insert(B256::from(keccak(1usize.to_be_bytes())), U256::from(42));
        storage_changes.storage.insert(B256::from(keccak(200usize.to_be_bytes())), U256::from(43));
        post.accounts.insert(
            addr_a,
            Some(Account { nonce: 2, balance: U256::from(111), bytecode_hash: None }),
        );
        post.storages.insert(addr_a, storage_changes);
        post.accounts.insert(
            addr_c,
            Some(Account { nonce: 0, balance: U256::from(5), bytecode_hash: None }),
        );

        // (a) full graph
        let mut full = state.clone();
        full.update(&post);
        let expected_root = full.state_root();

        // (b) flat overlay, through a serde roundtrip (must be zero-copy)
        let flat = FlatEthereumState::from_state(&state);
        let ser = bincode::serialize(&flat).unwrap();
        let flat2: FlatEthereumState<'_> = bincode::deserialize(&ser).unwrap();
        assert!(matches!(flat2.state_nodes, Cow::Borrowed(_)), "must be zero-copy");
        let views = flat2.views().unwrap();
        assert_eq!(views.state.root_hash, state.state_root());
        let mut overlay = views.materialize_overlay(&post).unwrap();
        overlay.update(&post);
        assert_eq!(overlay.state_root(), expected_root);

        // (c) batched delta root, no intermediate trie at all
        assert_eq!(views.post_state_root(&post).unwrap(), expected_root);
    }

    /// Every malformed-witness shape the harness found as a *panic site*, as a rejection.
    ///
    /// A panic aborts under `-Cpanic=abort`, so these are liveness, not soundness. What made
    /// them worth closing is the **profile**: three are `usize` overflows in RLP length
    /// arithmetic, and the guest builds with `overflow-checks = false` while every host test,
    /// Miri pass and fuzz execution builds with them on -- so the shipped artefact wrapped where
    /// the tested one panicked, and an entire defect class was invisible to the whole suite.
    ///
    /// `0xbf`/`0xff` below introduce an eight-byte big-endian length, the width that wraps.
    #[test]
    fn malformed_witnesses_are_rejected_not_panicked() {
        // A well-formed witness, as the control.
        let trie = keccak_trie(8);
        let good = flatten_trie(&trie);
        assert!(FlatTrieView::parse_and_verify(&good).is_ok());

        let cases: Vec<(&str, Vec<u8>)> = std::vec![
            // Truncated root item: a list header claiming 0x30 payload bytes with none.
            ("truncated root list", std::vec![0xf8, 0x30]),
            // Long-form string header whose length bytes are missing.
            ("truncated long length", std::vec![0xbf]),
            // Eight-byte length of `usize::MAX`: `len << 8` eight times wraps to 0, which
            // used to make the item look empty and in bounds.
            (
                "wrapping string length",
                std::vec![0xbf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            ),
            (
                "wrapping list length",
                std::vec![0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            ),
            // A length that is merely far larger than the buffer.
            ("length past the buffer", std::vec![0xb8, 0xff, 0x00]),
            // A node whose declared payload runs past the blob.
            ("branch payload past the blob", std::vec![0xf8, 0x40, 0x80, 0x80]),
            // Not a defect: an empty region *is* the empty trie, and the loop below accepts it.
            ("empty witness", std::vec![]),
            // A 2-item node whose path prefix is the empty string -- what the item scan hands
            // `parse_node` when a list item carries nothing.
            ("empty path prefix", std::vec![0xc2, 0x80, 0x80]),
        ];
        for (name, bytes) in cases {
            if bytes.is_empty() {
                // The empty region is the empty trie, which is legal.
                assert!(FlatTrieView::parse_and_verify(&bytes).is_ok(), "{name}");
                continue;
            }
            assert!(
                FlatTrieView::parse_and_verify(&bytes).is_err(),
                "{name}: accepted a malformed witness"
            );
        }

        // A truncated *tail* blob, appended after a valid root.
        let mut truncated = good.clone();
        truncated.extend_from_slice(&[0xf8, 0x40]);
        assert!(FlatTrieView::parse_and_verify(&truncated).is_err(), "truncated tail blob");

        // And a wrapping length in a tail blob.
        let mut wrapping_tail = good;
        wrapping_tail.extend_from_slice(&[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
        assert!(FlatTrieView::parse_and_verify(&wrapping_tail).is_err(), "wrapping tail length");
    }

    /// `be_len`'s bound, including the shape a 32-bit build used to accept:
    /// `[0,0,0,1,0,0,0,5]` truncated to `5` and read a truncated item back as a valid one.
    ///
    /// Checked against `be_len` directly rather than through a witness, so each assertion means
    /// the same thing on both widths -- a `parse_and_verify` case could only exercise the host's,
    /// which is why the witness tests above never noticed.
    #[test]
    fn be_len_rejects_lengths_past_the_buffer() {
        let mut buf = [0u8; 64];

        // A length written with leading zeros is decodable, not a rejection: the bound is on
        // the accumulated value, never on `ll`.
        buf[..4].copy_from_slice(&[0x00, 0x00, 0x00, 0x07]);
        assert_eq!(be_len(&buf, 0, 4).unwrap(), 7);
        assert_eq!(be_len(&buf, 0, 0).unwrap(), 0, "no length bytes is length zero");

        // The 32-bit truncation witness. 0x01_0000_0005 keeps only `5` in a 32-bit `usize`;
        // both widths must reject it, because 64 bytes of buffer cannot hold either value.
        buf[..8].copy_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05]);
        assert!(be_len(&buf, 0, 8).is_err(), "a length past the buffer must be rejected");

        // Eight shifts of 8 is the full width of a 64-bit `usize`: this is the one that used
        // to come back as 0 and make a truncated item look empty and in bounds.
        assert!(be_len(&[0xff; 16], 0, 8).is_err(), "all-ones length");

        // Length bytes that are not all there.
        assert!(be_len(&buf, 60, 8).is_err(), "truncated length bytes");
        assert!(be_len(&buf, 0, usize::MAX).is_err(), "`pos + ll` must not wrap");
    }

    /// The floor two scan loops rest on: `parse_node`'s item walk and `parse_and_verify`'s
    /// blob walk both advance by `rlp_item_len` and neither tests for zero, because no
    /// `rlp_header` arm can produce one. One representative encoding per arm.
    #[test]
    fn rlp_item_len_never_returns_zero() {
        let mut long_string = std::vec![0xb8u8, 56];
        long_string.extend_from_slice(&[0u8; 56]);
        let mut long_list = std::vec![0xf8u8, 56];
        long_list.extend_from_slice(&[0x80u8; 56]);

        let cases: Vec<Vec<u8>> = std::vec![
            std::vec![0x00], // 0x00..=0x7f: the item is the byte
            std::vec![0x7f],
            std::vec![0x80], // 0x80..=0xb7 with an empty payload
            std::vec![0x83, 1, 2, 3],
            long_string,     // 0xb8..=0xbf
            std::vec![0xc0], // 0xc0..=0xf7 with an empty payload
            std::vec![0xc3, 0x01, 0x02, 0x03],
            long_list, // 0xf8..=0xff
        ];
        for bytes in &cases {
            let n = rlp_item_len(bytes, 0).unwrap();
            assert!(n >= 1, "rlp_item_len returned 0 for {bytes:02x?}");
            assert!(n <= bytes.len(), "rlp_item_len ran past the buffer for {bytes:02x?}");
        }
    }

    /// The two API-misuse shapes `delta_root` used to answer with a panic or a wrong root.
    ///
    /// A **duplicate key** makes `build_kvs` index `kvs[start].0[cp]` with `cp` equal to the key
    /// length and panic. An **unsorted list** violates `apply_branch`'s precondition, and one
    /// shape of that returns a *silently wrong root* under `--release` -- see the note on
    /// `apply_branch` for the mechanism; measured on the shape below before the fix, it gave
    /// `0x56e81f…`, the empty-trie root, where a real root was correct.
    ///
    /// Until this the property was enforced in **no shipped configuration**.
    #[test]
    fn delta_root_refuses_duplicate_and_unsorted_keys() {
        let trie = keccak_trie(40);
        let bytes = flatten_trie(&trie);
        let view = FlatTrieView::parse_and_verify(&bytes).unwrap();

        let k0 = B256::from(keccak(0usize.to_be_bytes()));
        let k1 = B256::from(keccak(1usize.to_be_bytes()));

        // A well-formed batch still works, and agrees with the node graph.
        let ok: Vec<(B256, Option<Vec<u8>>)> =
            std::vec![(k0, Some(alloy_rlp::encode(7u64))), (k1, Some(alloy_rlp::encode(8u64))),];
        let mut full = trie.clone();
        full.insert_rlp(k0.as_slice(), 7u64).unwrap();
        full.insert_rlp(k1.as_slice(), 8u64).unwrap();
        assert_eq!(view.delta_root(&ok).unwrap(), full.hash());

        // Duplicate key.
        let dup: Vec<(B256, Option<Vec<u8>>)> =
            std::vec![(k0, Some(alloy_rlp::encode(7u64))), (k0, Some(alloy_rlp::encode(9u64))),];
        assert!(matches!(
            view.delta_root(&dup),
            Err(Error::FlatTrie("delta changes must be strictly ascending by key"))
        ));
        assert!(matches!(
            FlatTrieView::empty_delta_root(&dup),
            Err(Error::FlatTrie("delta changes must be strictly ascending by key"))
        ));

        // A sorted list still descends, which is the control for the panicking test below.
        let keys: Vec<B256> = (0..24usize).map(|i| B256::from(keccak(i.to_be_bytes()))).collect();
        let nibs: Vec<Vec<u8>> = keys.iter().map(|k| to_nibs(k.as_slice())).collect();
        let val = alloy_rlp::encode(3u64);
        let mut list: Vec<Change<'_>> =
            nibs.iter().map(|n| (n.as_slice(), Some(val.as_slice()))).collect();
        list.sort_unstable_by(|a, b| a.0.cmp(b.0));
        assert!(view.apply_src(Src::Node(0), &list).is_ok());
    }

    /// The guard *below* the entry check, reached the way a caller bypassing `delta_root`
    /// would reach it -- `delta_root` sorts three lines before it descends, so nothing that
    /// goes through it can get here.
    ///
    /// Expects the `debug_assert!`'s message rather than the `assert!`'s: both guards are
    /// live, and in a test build the full-key `debug_assert!` fires first. The `assert!` is
    /// the one that reaches the guest, where the `debug_assert!` is compiled out.
    #[test]
    #[should_panic(expected = "apply_branch requires `changes` sorted by key")]
    fn apply_branch_panics_on_an_unsorted_list() {
        let trie = keccak_trie(40);
        let bytes = flatten_trie(&trie);
        let view = FlatTrieView::parse_and_verify(&bytes).unwrap();

        let keys: Vec<B256> = (0..24usize).map(|i| B256::from(keccak(i.to_be_bytes()))).collect();
        let nibs: Vec<Vec<u8>> = keys.iter().map(|k| to_nibs(k.as_slice())).collect();
        let val = alloy_rlp::encode(3u64);
        let mut list: Vec<Change<'_>> =
            nibs.iter().map(|n| (n.as_slice(), Some(val.as_slice()))).collect();
        list.sort_unstable_by(|a, b| a.0.cmp(b.0));
        list.reverse();
        let _ = view.apply_src(Src::Node(0), &list);
    }

    /// An account whose storage trie is **not** in the witness must not have its storage
    /// silently wiped -- by `post_state_root`, *or* by `materialize_overlay`.
    ///
    /// The reachable shape is a plain value transfer to a contract: the account is touched, so it
    /// appears in `post_state.accounts`, but no slot is read, so nothing notices the witness
    /// carries no storage trie for it and the row is rewritten with `storage_root = EMPTY_ROOT`.
    ///
    /// Both arms: a storage *change* with no witnessed trie must be refused outright, and no
    /// change must keep the root it had. Every case runs through **both** implementations -- the
    /// fix first landed only in `post_state_root`, and a parity test that exercises one of two
    /// implementations of the same thing is how they drift.
    #[test]
    fn unwitnessed_storage_trie_is_not_wiped_by_either_path() {
        let mut storage = MptNode::default();
        for i in 0..40usize {
            storage.insert_rlp(&keccak(i.to_be_bytes()), U256::from(i + 3)).unwrap();
        }
        let storage_root = storage.hash();
        assert_ne!(storage_root, EMPTY_ROOT);

        let contract = B256::from(keccak(b"contract"));
        let eoa = B256::from(keccak(b"eoa"));
        let mut state_trie = MptNode::default();
        state_trie
            .insert_rlp(
                contract.as_slice(),
                TrieAccount {
                    nonce: 1,
                    balance: U256::from(100),
                    storage_root,
                    code_hash: B256::repeat_byte(0xcd),
                },
            )
            .unwrap();
        state_trie
            .insert_rlp(
                eoa.as_slice(),
                TrieAccount {
                    nonce: 7,
                    balance: U256::from(500),
                    storage_root: EMPTY_ROOT,
                    code_hash: KECCAK_EMPTY,
                },
            )
            .unwrap();
        for i in 0..30usize {
            state_trie
                .insert_rlp(
                    &keccak((2000 + i).to_be_bytes()),
                    TrieAccount {
                        nonce: i as u64,
                        balance: U256::from(i),
                        storage_root: EMPTY_ROOT,
                        code_hash: KECCAK_EMPTY,
                    },
                )
                .unwrap();
        }

        // The witness carries the state trie and *no* storage trie -- which is exactly what a
        // value transfer needs, and exactly what an attacker would ship.
        let flat = FlatEthereumState {
            state_nodes: Cow::Owned(flatten_trie(&state_trie)),
            storage_tries: std::vec::Vec::new(),
        };
        let views = flat.views().unwrap();
        assert_eq!(views.state.root_hash, state_trie.hash());

        // (a) balance moves, storage untouched: the row keeps its storage root.
        let mut post = HashedPostState::default();
        post.accounts.insert(
            contract,
            Some(Account { nonce: 1, balance: U256::from(101), bytecode_hash: None }),
        );
        post.accounts
            .insert(eoa, Some(Account { nonce: 8, balance: U256::from(499), bytecode_hash: None }));

        // `HashedPostState`'s `Account` carries `bytecode_hash: Option<B256>`, and
        // `get_bytecode_hash()` answers `KECCAK_EMPTY` for `None`, so the rewritten rows take
        // that. What this test pins is the *storage* root, which is orthogonal.
        let mut expected = state_trie.clone();
        expected
            .insert_rlp(
                contract.as_slice(),
                TrieAccount {
                    nonce: 1,
                    balance: U256::from(101),
                    storage_root,
                    code_hash: KECCAK_EMPTY,
                },
            )
            .unwrap();
        expected
            .insert_rlp(
                eoa.as_slice(),
                TrieAccount {
                    nonce: 8,
                    balance: U256::from(499),
                    storage_root: EMPTY_ROOT,
                    code_hash: KECCAK_EMPTY,
                },
            )
            .unwrap();

        // `post_state.accounts` carries no code hash, so compare against the same shape the
        // computation produces rather than re-deriving it: what this pins is that the
        // *storage* root survives, which the `bytecode_hash: None` rows preserve.
        // Both paths, on every case below: `post_state_root` computes the root directly, and
        // `materialize_overlay` + `update` is the graph-trie route to the same answer.
        let overlay_root = |p: &HashedPostState| -> Result<B256, Error> {
            let mut overlay = views.materialize_overlay(p)?;
            overlay.update(p);
            Ok(overlay.state_root())
        };

        let got = views.post_state_root(&post).unwrap();
        assert_ne!(
            got,
            {
                // The wiped answer, i.e. what the pre-image computed.
                let mut wiped = state_trie.clone();
                wiped
                    .insert_rlp(
                        contract.as_slice(),
                        TrieAccount {
                            nonce: 1,
                            balance: U256::from(101),
                            storage_root: EMPTY_ROOT,
                            code_hash: KECCAK_EMPTY,
                        },
                    )
                    .unwrap();
                wiped
                    .insert_rlp(
                        eoa.as_slice(),
                        TrieAccount {
                            nonce: 8,
                            balance: U256::from(499),
                            storage_root: EMPTY_ROOT,
                            code_hash: KECCAK_EMPTY,
                        },
                    )
                    .unwrap();
                wiped.hash()
            },
            "the contract's storage was wiped from the post-state root"
        );
        assert_eq!(got, expected.hash());
        assert_eq!(
            overlay_root(&post).unwrap(),
            expected.hash(),
            "the overlay wiped the contract's storage that `post_state_root` preserved"
        );

        // (b) a storage *change* with no witnessed trie cannot be computed at all.
        let mut post_with_storage = post.clone();
        let mut changes = HashedStorage::new(false);
        changes.storage.insert(B256::from(keccak(0usize.to_be_bytes())), U256::from(9));
        post_with_storage.storages.insert(contract, changes);
        assert!(
            matches!(
                views.post_state_root(&post_with_storage),
                Err(Error::FlatTrie("no witnessed storage trie for a modified account"))
            ),
            "a modified account with no witnessed storage trie must be refused"
        );
        assert!(
            matches!(
                overlay_root(&post_with_storage),
                Err(Error::FlatTrie("no witnessed storage trie for a modified account"))
            ),
            "the overlay must refuse what `post_state_root` refuses, not answer it"
        );

        // (c) ... unless the account really had none, or the storage is wiped, both of which
        // stay computable.
        let mut post_eoa = HashedPostState::default();
        post_eoa
            .accounts
            .insert(eoa, Some(Account { nonce: 8, balance: U256::from(499), bytecode_hash: None }));
        let mut eoa_changes = HashedStorage::new(false);
        eoa_changes.storage.insert(B256::from(keccak(1usize.to_be_bytes())), U256::from(4));
        post_eoa.storages.insert(eoa, eoa_changes);
        assert_eq!(views.post_state_root(&post_eoa).unwrap(), overlay_root(&post_eoa).unwrap());

        let mut post_wiped = post.clone();
        let mut wiped_changes = HashedStorage::new(true);
        wiped_changes.storage.insert(B256::from(keccak(0usize.to_be_bytes())), U256::from(9));
        post_wiped.storages.insert(contract, wiped_changes);
        assert_eq!(views.post_state_root(&post_wiped).unwrap(), overlay_root(&post_wiped).unwrap());
    }

    /// Applies ops to a graph trie (reference) and via delta_root; roots must agree.
    fn delta_parity_case(trie: &MptNode, ops: &[([u8; 32], Option<u64>)]) {
        let bytes = flatten_trie(trie);
        let view = FlatTrieView::parse_and_verify(&bytes).unwrap();
        assert_eq!(view.root_hash, trie.hash());

        let mut full = trie.clone();
        for (k, v) in ops {
            match v {
                Some(v) => {
                    full.insert_rlp(k, *v).unwrap();
                }
                None => {
                    full.delete(k).unwrap();
                }
            }
        }

        let changes: Vec<(B256, Option<Vec<u8>>)> =
            ops.iter().map(|(k, v)| (B256::from(*k), v.map(|v| alloy_rlp::encode(v)))).collect();
        assert_eq!(view.delta_root(&changes).unwrap(), full.hash());
    }

    /// A trie whose branch children are **inline** nodes rather than 32-byte digests.
    ///
    /// Every `keccak_trie` key is a hash, so its leaves sit two to five nibbles deep and carry
    /// ~30 bytes of remaining path; each one encodes well over 32 bytes and therefore enters its
    /// parent as `0xa0` + a digest. So `apply_branch`'s bounds scan only ever sees the
    /// short-string form, and the short-*list* arm that decodes an inlined child's length never
    /// runs. Instrumenting the four arms over the whole crate's tests gave `[0, 4864, 0, 0]`, and
    /// a mutation of that arm consequently survived every test here.
    ///
    /// Keys sharing a 60-nibble prefix fix that: the branch lands at nibble 60, so each leaf has
    /// three nibbles of path (two compact bytes) and a one-byte value, encoding to six bytes --
    /// under the 32-byte inlining threshold.
    ///
    /// The other two arms stay at zero on purpose: a branch child is an empty string, a 32-byte
    /// digest or an inlined node, so it is never a lone byte below `0x80` and never needs a
    /// multi-byte length header. Both arms exist only because `rlp_item_len` has them.
    fn inline_child_trie() -> (MptNode, Vec<[u8; 32]>) {
        let mut trie = MptNode::default();
        let mut keys = Vec::new();
        for i in 0..16usize {
            let mut k = [0xABu8; 32];
            // Nibble 60 is the high nibble of byte 30, and that is what the branch splits on.
            k[30] = (i as u8) << 4;
            k[31] = i as u8;
            trie.insert_rlp(&k, 0u8).unwrap();
            keys.push(k);
        }
        (trie, keys)
    }

    #[test]
    fn delta_root_parity_with_inline_children() {
        let (trie, keys) = inline_child_trie();

        // The premise of the test, asserted rather than assumed: the children of the branch this
        // exercises really are inlined. `MptNodeReference::Bytes` is exactly "this node entered
        // its parent verbatim because its encoding is under 32 bytes"; `Digest` is the other
        // case. If a future change to the leaf encoding pushes these over the threshold, this
        // fails here instead of silently going back to covering nothing.
        let MptNodeData::Extension(_, inner) = trie.as_data() else {
            panic!("expected a 60-nibble extension at the root")
        };
        let MptNodeData::Branch(children) = inner.as_data() else {
            panic!("expected a branch under the extension")
        };
        let inlined = children
            .iter()
            .flatten()
            .filter(|c| matches!(c.reference(), MptNodeReference::Bytes(_)))
            .count();
        assert_eq!(inlined, 16, "every child of this branch should be inlined");

        let bytes = flatten_trie(&trie);
        let view = FlatTrieView::parse_and_verify(&bytes).unwrap();
        assert_eq!(view.root_hash, trie.hash());
        for k in &keys {
            assert_eq!(view.get(k).unwrap(), trie.get(k).unwrap());
        }

        // Mixed batch over that branch: updates, deletes and an insert, so the rebuild has
        // several changed slots and the splice walks inline items.
        let mut ops: Vec<([u8; 32], Option<u64>)> = Vec::new();
        for (i, k) in keys.iter().enumerate() {
            if i % 3 == 0 {
                ops.push((*k, Some(7)));
            } else if i % 3 == 1 {
                ops.push((*k, None));
            }
        }
        let mut fresh = [0xABu8; 32];
        fresh[30] = 0x51;
        fresh[31] = 0x99;
        ops.push((fresh, Some(3)));
        delta_parity_case(&trie, &ops);

        // And one where the branch collapses to a single survivor.
        let mut all_but_one: Vec<([u8; 32], Option<u64>)> =
            keys.iter().skip(1).map(|k| (*k, None)).collect();
        all_but_one.push((keys[0], Some(9)));
        delta_parity_case(&trie, &all_but_one);
    }

    #[test]
    fn delta_root_parity() {
        const N: usize = 300;
        let trie = keccak_trie(N);

        // mixed batch: updates, deletes (collapses), inserts
        let mut ops: Vec<([u8; 32], Option<u64>)> = Vec::new();
        for i in 0..10usize {
            ops.push((keccak(i.to_be_bytes()), Some(1_000_000 + i as u64)));
        }
        for i in 10..40usize {
            ops.push((keccak(i.to_be_bytes()), None));
        }
        for i in N..N + 10 {
            ops.push((keccak(i.to_be_bytes()), Some(i as u64)));
        }
        delta_parity_case(&trie, &ops);

        // delete everything
        let all_del: Vec<([u8; 32], Option<u64>)> =
            (0..N).map(|i| (keccak(i.to_be_bytes()), None)).collect();
        delta_parity_case(&trie, &all_del);

        // delete non-existent keys (no-ops) mixed with real work
        let mut noops: Vec<([u8; 32], Option<u64>)> = Vec::new();
        for i in N..N + 20 {
            noops.push((keccak(i.to_be_bytes()), None));
        }
        noops.push((keccak(3usize.to_be_bytes()), Some(777)));
        delta_parity_case(&trie, &noops);

        // small tries exercise extension splits and root collapses
        for n in [1usize, 2, 3, 5] {
            let small = keccak_trie(n);
            let mut ops: Vec<([u8; 32], Option<u64>)> = Vec::new();
            ops.push((keccak(0usize.to_be_bytes()), None));
            ops.push((keccak((n + 5).to_be_bytes()), Some(9)));
            ops.push((keccak((n + 6).to_be_bytes()), Some(10)));
            delta_parity_case(&small, &ops);
        }

        // empty trie + inserts
        let empty = MptNode::default();
        let ops: Vec<([u8; 32], Option<u64>)> =
            (0..8).map(|i: usize| (keccak(i.to_be_bytes()), Some(i as u64))).collect();
        delta_parity_case(&empty, &ops);
    }

    /// The randomized half of the delta-root differential, with a coverage guard.
    ///
    /// This was the only unguarded sweep in the campaign, and it is on the central output-path
    /// rewrite -- so "30 seeds passed" was the whole claim, and 30 seeds that all produced,
    /// say, pure updates on a fat trie would have passed identically. The tallies below say
    /// what the generator actually produced, and every one of them is a shape with its own arm
    /// in `apply_src`: an insert of a key the trie does not have, a delete that removes a leaf,
    /// a delete that *collapses* a branch to one child, a no-op delete of an absent key, and a
    /// batch that empties the trie outright.
    #[test]
    fn delta_root_randomized_parity() {
        let mut inserts = 0usize;
        let mut updates = 0usize;
        let mut deletes = 0usize;
        let mut absent_deletes = 0usize;
        let mut collapses = 0usize;
        let mut batch_sizes = (usize::MAX, 0usize);

        // sweep many pseudo-random op batches against the graph implementation
        for seed in 0u64..30 {
            let n = 20 + (seed as usize * 13) % 200;
            let trie = keccak_trie(n);
            let mut ops: Vec<([u8; 32], Option<u64>)> = Vec::new();
            let mut x = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
            let count = 1 + (seed as usize % 25);
            for j in 0..count {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let existing = x % 2 == 0;
                let idx = if existing { (x >> 8) as usize % n } else { n + j };
                let delete = (x >> 16) % 3 == 0;
                ops.push((keccak(idx.to_be_bytes()), if delete { None } else { Some(x >> 24) }));
            }
            // dedup by key (last op wins), mirroring HashedPostState semantics
            let mut seen = std::collections::HashMap::new();
            for (k, v) in ops {
                seen.insert(k, v);
            }
            let ops: Vec<([u8; 32], Option<u64>)> = seen.into_iter().collect();

            // Classify against the trie this batch is applied to, before applying it.
            batch_sizes = (batch_sizes.0.min(ops.len()), batch_sizes.1.max(ops.len()));
            let mut after = trie.clone();
            for (k, v) in &ops {
                let present = trie.get(k).unwrap().is_some();
                match (v, present) {
                    (Some(_), true) => updates += 1,
                    (Some(_), false) => inserts += 1,
                    (None, true) => deletes += 1,
                    (None, false) => absent_deletes += 1,
                }
                match v {
                    Some(v) => {
                        after.insert_rlp(k, *v).unwrap();
                    }
                    None => {
                        after.delete(k).unwrap();
                    }
                }
            }
            // A branch collapse shows up as the trie getting *shallower* while keys were
            // removed -- the arm that rebuilds a one-child branch as an extension or a leaf.
            if deletes > 0 && after.hash() != trie.hash() {
                let before_bytes = flatten_trie(&trie).len();
                let after_bytes = flatten_trie(&after).len();
                if after_bytes < before_bytes {
                    collapses += 1;
                }
            }

            delta_parity_case(&trie, &ops);
        }

        assert!(inserts > 0, "no insert of an absent key was generated");
        assert!(updates > 0, "no update of a present key was generated");
        assert!(deletes > 0, "no delete of a present key was generated");
        assert!(absent_deletes > 0, "no no-op delete of an absent key was generated");
        assert!(collapses > 0, "no batch shrank the trie, so no collapse arm was exercised");
        assert!(batch_sizes.0 <= 2, "no small batch was generated (min {})", batch_sizes.0);
        assert!(batch_sizes.1 >= 10, "no large batch was generated (max {})", batch_sizes.1);
    }
}

// --- batched delta root ----------------------------------------------------------------------
//
// Computes the post-state root directly from the verified blobs and a batch of key changes in
// one bottom-up pass: unchanged children are copied into the parent as their verbatim reference
// bytes, changed paths are re-encoded, and only re-encoded nodes are hashed. No intermediate
// node graph is built.

use crate::mpt::{lcp, to_encoded_path, EMPTY_ROOT as FLAT_EMPTY_ROOT};

/// A branch child during delta assembly.
#[derive(Default)]
enum Slot<'a> {
    #[default]
    Missing,
    /// Original child carried over unchanged (with resolution context for collapse).
    Keep { r: FlatRef<'a>, parent: Src<'a>, slot: u32 },
    /// Freshly rebuilt child encoding.
    New(Vec<u8>),
}

impl Slot<'_> {
    fn from_out(out: Out) -> Self {
        match out {
            Out::Empty => Slot::Missing,
            Out::Enc(enc) => Slot::New(enc),
        }
    }
}

/// Appends the RLP item bytes referencing `r`.
fn ref_bytes_of(r: FlatRef<'_>, out: &mut Vec<u8>) {
    match r {
        FlatRef::Digest(d) => {
            out.push(0xa0);
            out.extend_from_slice(d);
        }
        FlatRef::Inline(b) => out.extend_from_slice(b),
        FlatRef::Empty => out.push(alloy_rlp::EMPTY_STRING_CODE),
    }
}

/// A (remaining-key-nibbles, new-value) pair; `None` deletes the key.
type Change<'c> = (&'c [u8], Option<&'c [u8]>);

/// Rejects a change list that is not strictly ascending by key. Both callers sort first, so in
/// practice this is a duplicate-key check; written as the full ordering test to also cover a
/// caller that stops sorting.
///
/// Checked in the *shipped* build, not only under `debug_assertions` -- the guest builds with
/// those off, so the property was enforced in no configuration at all. Two things rest on it:
/// violating `apply_branch`'s sorted precondition can return a silently wrong root rather than
/// panicking, and a duplicate key makes `build_kvs` index `kvs[start].0[cp]` with `cp` equal to
/// the key length (the longest common prefix of a key with itself) and panic.
///
/// One extra pass of the comparison the sort just made: not measurable.
#[inline]
fn require_strictly_ascending(list: &[Change<'_>]) -> Result<(), Error> {
    if list.windows(2).any(|w| w[0].0 >= w[1].0) {
        return Err(Error::FlatTrie("delta changes must be strictly ascending by key"));
    }
    Ok(())
}

/// The result of rebuilding a subtree: its full RLP encoding, or nothing left.
enum Out {
    Empty,
    Enc(Vec<u8>),
}

/// Encoded length of an RLP string.
fn str_len(s: &[u8]) -> usize {
    if s.len() == 1 && s[0] < 0x80 {
        1
    } else if s.len() <= 55 {
        1 + s.len()
    } else {
        let mut n = 0;
        let mut len = s.len();
        while len > 0 {
            n += 1;
            len >>= 8;
        }
        1 + n + s.len()
    }
}

/// Appends an RLP list header for `payload_len` payload bytes.
fn list_header_into(out: &mut Vec<u8>, payload_len: usize) {
    if payload_len <= 55 {
        out.push(0xc0 + payload_len as u8);
    } else {
        let mut be = [0u8; 8];
        let mut n = 0;
        let mut len = payload_len;
        while len > 0 {
            be[7 - n] = (len & 0xff) as u8;
            len >>= 8;
            n += 1;
        }
        out.push(0xf7 + n as u8);
        out.extend_from_slice(&be[8 - n..]);
    }
}

/// Appends the RLP string encoding of `s`.
fn enc_str_into(out: &mut Vec<u8>, s: &[u8]) {
    if s.len() == 1 && s[0] < 0x80 {
        out.push(s[0]);
    } else if s.len() <= 55 {
        out.push(0x80 + s.len() as u8);
        out.extend_from_slice(s);
    } else {
        let mut len = s.len();
        let mut be = [0u8; 8];
        let mut n = 0;
        while len > 0 {
            be[7 - n] = (len & 0xff) as u8;
            len >>= 8;
            n += 1;
        }
        out.push(0xb7 + n as u8);
        out.extend_from_slice(&be[8 - n..]);
        out.extend_from_slice(s);
    }
}

/// Appends the reference item for a node encoding: verbatim if short, `0xa0 || keccak` else.
fn ref_item_into(out: &mut Vec<u8>, enc: &[u8]) {
    if enc.len() < 32 {
        out.extend_from_slice(enc);
    } else {
        out.push(0xa0);
        out.extend_from_slice(&keccak(enc));
    }
}

fn enc_leaf(nibs: &[u8], value: &[u8]) -> Vec<u8> {
    let path = to_encoded_path(nibs, true);
    let payload_len = str_len(&path) + str_len(value);
    let mut out = Vec::with_capacity(payload_len + 4);
    list_header_into(&mut out, payload_len);
    enc_str_into(&mut out, &path);
    enc_str_into(&mut out, value);
    out
}

/// `child_item` must already be valid RLP item bytes (an inline list or `0xa0 || digest`).
fn enc_ext(nibs: &[u8], child_item: &[u8]) -> Vec<u8> {
    let path = to_encoded_path(nibs, false);
    let payload_len = str_len(&path) + child_item.len();
    let mut out = Vec::with_capacity(payload_len + 4);
    list_header_into(&mut out, payload_len);
    enc_str_into(&mut out, &path);
    out.extend_from_slice(child_item);
    out
}

/// Builds a subtree from scratch out of sorted, distinct (nibbles, value) leaves.
fn build_kvs(kvs: &[(&[u8], &[u8])]) -> Out {
    match kvs {
        [] => Out::Empty,
        [(nibs, value)] => Out::Enc(enc_leaf(nibs, value)),
        _ => {
            // sorted: the common prefix of all is the lcp of first and last
            let cp = lcp(kvs[0].0, kvs[kvs.len() - 1].0);
            let mut outs: [Option<Vec<u8>>; 16] = Default::default();
            let mut start = 0;
            while start < kvs.len() {
                let nib = kvs[start].0[cp];
                let mut end = start + 1;
                while end < kvs.len() && kvs[end].0[cp] == nib {
                    end += 1;
                }
                let group: Vec<(&[u8], &[u8])> =
                    kvs[start..end].iter().map(|(k, v)| (&k[cp + 1..], *v)).collect();
                if let Out::Enc(enc) = build_kvs(&group) {
                    outs[nib as usize] = Some(enc);
                }
                start = end;
            }

            // single-pass branch encoding
            let payload_len: usize = outs
                .iter()
                .map(|o| o.as_ref().map_or(1, |e| if e.len() < 32 { e.len() } else { 33 }))
                .sum::<usize>() +
                1;
            let mut branch = Vec::with_capacity(payload_len + 4);
            list_header_into(&mut branch, payload_len);
            for out in &outs {
                match out {
                    None => branch.push(alloy_rlp::EMPTY_STRING_CODE),
                    Some(enc) => ref_item_into(&mut branch, enc),
                }
            }
            branch.push(alloy_rlp::EMPTY_STRING_CODE); // branch value: always empty

            if cp > 0 {
                let mut item = Vec::with_capacity(33);
                ref_item_into(&mut item, &branch);
                Out::Enc(enc_ext(&kvs[0].0[..cp], &item))
            } else {
                Out::Enc(branch)
            }
        }
    }
}

impl<'a> FlatTrieView<'a> {
    /// Computes the root of this trie after applying `changes` (keyed by full hashed key;
    /// `None` deletes). One bottom-up pass over the changed paths; untouched subtrees are
    /// carried over as their existing reference bytes.
    pub fn delta_root(&self, changes: &[(B256, Option<Vec<u8>>)]) -> Result<B256, Error> {
        if changes.is_empty() {
            return Ok(self.root_hash);
        }
        let nibs: Vec<Vec<u8>> = changes.iter().map(|(k, _)| to_nibs(k.as_slice())).collect();
        let mut list: Vec<Change<'_>> = changes
            .iter()
            .zip(nibs.iter())
            .map(|((_, v), n)| (n.as_slice(), v.as_deref()))
            .collect();
        list.sort_unstable_by(|a, b| a.0.cmp(b.0));
        require_strictly_ascending(&list)?;

        let out = if self.is_empty() {
            Self::apply_empty(&list)
        } else {
            self.apply_src(Src::Node(0), &list)?
        };
        Ok(match out {
            Out::Empty => FLAT_EMPTY_ROOT,
            Out::Enc(enc) => B256::from(keccak(&enc)),
        })
    }

    /// Root of a fresh trie holding only `changes`' insertions (used for wiped or unwitnessed
    /// storage tries).
    pub fn empty_delta_root(changes: &[(B256, Option<Vec<u8>>)]) -> Result<B256, Error> {
        let nibs: Vec<Vec<u8>> = changes.iter().map(|(k, _)| to_nibs(k.as_slice())).collect();
        let mut list: Vec<Change<'_>> = changes
            .iter()
            .zip(nibs.iter())
            .map(|((_, v), n)| (n.as_slice(), v.as_deref()))
            .collect();
        list.sort_unstable_by(|a, b| a.0.cmp(b.0));
        require_strictly_ascending(&list)?;
        Ok(match Self::apply_empty(&list) {
            Out::Empty => FLAT_EMPTY_ROOT,
            Out::Enc(enc) => B256::from(keccak(&enc)),
        })
    }

    /// Changes applied to an empty position: only insertions survive.
    fn apply_empty(changes: &[Change<'_>]) -> Out {
        let kvs: Vec<(&[u8], &[u8])> =
            changes.iter().filter_map(|(k, v)| v.map(|v| (*k, v))).collect();
        build_kvs(&kvs)
    }

    fn apply_src(&self, src: Src<'a>, changes: &[Change<'_>]) -> Result<Out, Error> {
        debug_assert!(!changes.is_empty());
        let node = match src {
            Src::Node(idx) => self.parse_indexed(idx)?,
            Src::Inline(b) => parse_node(b)?,
        };
        match node {
            FlatNode::Null => Ok(Self::apply_empty(changes)),
            FlatNode::Digest(d) => {
                // mutating through an unresolved subtree is impossible; identical failure mode
                // to MptNode::insert/delete hitting a digest
                Err(Error::NodeNotResolved(B256::from_slice(d)))
            }
            FlatNode::Leaf { prefix, value } => {
                let pn = prefix_nibs(prefix);
                let mut kvs: Vec<(&[u8], &[u8])> = Vec::with_capacity(changes.len() + 1);
                let mut leaf_state: Option<&[u8]> = Some(value);
                for (k, v) in changes {
                    if *k == pn.as_slice() {
                        leaf_state = *v;
                    } else if let Some(v) = v {
                        kvs.push((k, v));
                    }
                }
                if let Some(v) = leaf_state {
                    kvs.push((pn.as_slice(), v));
                }
                kvs.sort_unstable_by(|a, b| a.0.cmp(b.0));
                Ok(build_kvs(&kvs))
            }
            FlatNode::Extension { prefix, child } => {
                let pn = prefix_nibs(prefix);
                self.apply_ext(src, &pn, child, changes)
            }
            FlatNode::Branch { payload } => self.apply_branch(src, payload, changes),
        }
    }

    /// Applies changes at an extension with path `pn` and child reference `child`. `src` is the
    /// node owning `child` (for edge resolution).
    fn apply_ext(
        &self,
        src: Src<'a>,
        pn: &[u8],
        child: FlatRef<'a>,
        changes: &[Change<'_>],
    ) -> Result<Out, Error> {
        // diverging deletes are no-ops; drop them first
        let live: Vec<Change<'_>> = changes
            .iter()
            .filter(|(k, v)| v.is_some() || k.len() >= pn.len() && k[..pn.len()] == *pn)
            .copied()
            .collect();
        if live.is_empty() {
            // nothing effective: re-encode the unchanged extension
            let mut item = Vec::with_capacity(33);
            ref_bytes_of(child, &mut item);
            return Ok(Out::Enc(enc_ext(pn, &item)));
        }
        let d = live.iter().map(|(k, _)| lcp(k, pn)).min().unwrap();

        if d == pn.len() {
            // all changes are inside the extension's subtree
            let stripped: Vec<Change<'_>> =
                live.iter().map(|(k, v)| (&k[pn.len()..], *v)).collect();
            let child_out = self.apply_src(self.child_src(src, child, 0)?, &stripped)?;
            return self.merge_prefix(pn, child_out);
        }

        // the extension splits at nibble position d
        let mut slots: [Slot<'a>; 16] = Default::default();

        let mut same_slot: Vec<Change<'_>> = Vec::new();
        let mut groups: [Vec<(&[u8], &[u8])>; 16] = Default::default();
        for (k, v) in &live {
            if k[d] == pn[d] {
                same_slot.push((&k[d + 1..], *v));
            } else if let Some(v) = v {
                groups[k[d] as usize].push((&k[d + 1..], *v));
            }
        }

        // the original path continues under pn[d]
        let rest = &pn[d + 1..];
        slots[pn[d] as usize] = if same_slot.is_empty() {
            if rest.is_empty() {
                Slot::Keep { r: child, parent: src, slot: 0 }
            } else {
                let mut child_item = Vec::with_capacity(33);
                ref_bytes_of(child, &mut child_item);
                Slot::New(enc_ext(rest, &child_item))
            }
        } else {
            let sub = if rest.is_empty() {
                self.apply_src(self.child_src(src, child, 0)?, &same_slot)?
            } else {
                self.apply_ext(src, rest, child, &same_slot)?
            };
            Slot::from_out(sub)
        };

        for (slot, group) in groups.iter_mut().enumerate() {
            if !group.is_empty() {
                group.sort_unstable_by(|a, b| a.0.cmp(b.0));
                slots[slot] = Slot::from_out(build_kvs(group));
            }
        }

        let out = self.assemble_branch(slots)?;
        match out {
            Out::Empty => Ok(Out::Empty),
            other if d == 0 => Ok(other),
            other => self.merge_prefix(&pn[..d], other),
        }
    }

    fn apply_branch(
        &self,
        src: Src<'a>,
        payload: &'a [u8],
        changes: &[Change<'_>],
    ) -> Result<Out, Error> {
        // Sorted `changes` is a precondition, and it is the *whole* key that must be ordered,
        // not just the leading nibble: the recursive call gets `changes[start..idx]` with
        // `&k[1..]`, so the requirement reapplies at every depth.
        //
        // A violation is not merely a panic. Stepping over `changes`' runs directly, more than
        // 16 runs overruns `touched` and out-of-order runs walk off the end of a slot in the
        // splice -- but revisiting a slot instead runs the `count` update twice, and `count`
        // then lands at or below 1 (usually 1 by ordinary arithmetic; 0 via `count - 1 + 1`
        // wrapping with overflow checks off). Either reaches the `count <= 1` collapse path with
        // a `rebuilt` array that has lost a slot, and that path never reads `touched`: a wrong
        // root, no panic. Not a soundness problem -- this root is compared against the header
        // immediately after -- but a worse failure mode than a panic.
        //
        // Three layers, checking different things:
        //
        //  1. `delta_root`/`empty_delta_root`, the only entries here, check the whole list is
        //     **strictly** ascending in the shipped build. Strict is what rejects a *duplicate*
        //     key, which neither of the others catches.
        //  2. this `debug_assert!`, the only check on *full-key* ordering rather than leading
        //     nibbles. `O(changes)` at every depth, which is why it cannot be a real `assert!` --
        //     and `delta_root` sorts immediately before, so the guest loses nothing.
        //  3. the `assert!` in the slot loop below, `O(1)` per run, the one that reaches the guest.
        //
        // In a debug build (2) fires before (3) can, so a test aimed at (3) must expect *this*
        // message.
        debug_assert!(
            changes.windows(2).all(|w| w[0].0 <= w[1].0),
            "apply_branch requires `changes` sorted by key, not merely by leading nibble"
        );

        // One scan for the 16 child-item boundaries, which also counts the branch's original
        // children: an empty slot is exactly the one-byte `0x80` item, so its first byte is
        // the whole test and the count comes for free here instead of in a second pass.
        let mut bounds = [0u32; 18];
        let mut pos = 0usize;
        let mut count = 0usize;
        for slot in 0..16usize {
            let b = payload[pos];
            if b != alloy_rlp::EMPTY_STRING_CODE {
                count += 1;
            }
            // `rlp_item_len(payload, pos)` off the byte already in hand. Calling it re-reads
            // and re-bounds-checks the same byte and threads the answer back through a
            // `Result`; on block 24006677 this loop runs 29,760 times (1,860 branches x 16
            // slots) and was 17 retired instructions an iteration.
            //
            // The three arms below are `rlp_header`'s single-byte-header cases with
            // `payload_off - pos + len` already folded in, and they are what a branch's
            // children actually are: `0x80` for an empty slot, `0xa0` + 32 for a digest
            // reference, and a short list for an inlined node. The two multi-byte-length
            // forms (`0xb8..=0xbf`, `0xf8..=0xff`) fall through to the general scanner, so
            // this is the same function, not a narrower one.
            pos += match b {
                0x00..=0x7f => 1,
                0x80..=0xb7 => 1 + (b - 0x80) as usize,
                0xc0..=0xf7 => 1 + (b - 0xc0) as usize,
                _ => rlp_item_len(payload, pos)?,
            };
            bounds[slot + 1] = pos as u32;
        }
        bounds[17] = payload.len() as u32;

        // Rebuild only the changed slots. `changes` is sorted (see the precondition above),
        // so stepping over its runs visits exactly the changed slots -- 1.18 of 16 per branch
        // on mainnet block 24006677 -- and records them in order for the splice below.
        let mut rebuilt: [Option<Out>; 16] = Default::default();
        let mut changed = [false; 16];
        let mut touched = [0u8; 16];
        let mut ntouched = 0usize;
        // `payload_len` as a delta against the original child region, so the unchanged slots
        // are never revisited: `bounds[16]` is the length of all 16 original child items.
        let mut delta = 0isize;
        let mut idx = 0usize;
        while idx < changes.len() {
            let slot = changes[idx].0[0] as usize;
            let start = idx;
            while idx < changes.len() && changes[idx].0[0] == slot as u8 {
                idx += 1;
            }
            // The precondition above, as a guard that survives into the guest. One compare
            // per *run* -- 1.18 per branch on mainnet block 24006677, not one per slot --
            // which is why this can afford to be an `assert!` where the full
            // `changes.windows(2)` test above cannot: that one is O(changes) at every depth
            // of the recursion.
            //
            // It covers both failure modes the note above describes: more than 16 runs
            // overrunning `touched`, and a revisited slot running the `count` update twice
            // and reaching the `count <= 1` collapse path with a `rebuilt` array that has
            // lost a slot -- the one that returns a wrong root without panicking.
            assert!(
                ntouched == 0 || slot > touched[ntouched - 1] as usize,
                "apply_branch requires `changes` sorted by key"
            );
            changed[slot] = true;
            touched[ntouched] = slot as u8;
            ntouched += 1;
            let lo = bounds[slot] as usize;
            let hi = bounds[slot + 1] as usize;
            let was_non_empty = payload[lo] != alloy_rlp::EMPTY_STRING_CODE;
            let item = &payload[lo..hi];
            let group: Vec<Change<'_>> =
                changes[start..idx].iter().map(|(k, v)| (&k[1..], *v)).collect();
            let out = match parse_ref(item)? {
                FlatRef::Empty => Self::apply_empty(&group),
                r => self.apply_src(self.child_src(src, r, slot as u32)?, &group)?,
            };
            let new_len = match &out {
                Out::Enc(enc) if enc.len() < 32 => enc.len(),
                Out::Enc(_) => 33,
                Out::Empty => 1,
            };
            delta += new_len as isize - (hi - lo) as isize;
            count = count - usize::from(was_non_empty) + usize::from(matches!(out, Out::Enc(_)));
            rebuilt[slot] = Some(out);
        }

        if count <= 1 {
            // Rare (12 of 1,860 branches on block 24006677); find the survivor by a scan
            // rather than tracking a running maximum through the deletes above.
            let mut last = 0usize;
            for slot in 0..16usize {
                let non_empty = if changed[slot] {
                    matches!(rebuilt[slot], Some(Out::Enc(_)))
                } else {
                    payload[bounds[slot] as usize] != alloy_rlp::EMPTY_STRING_CODE
                };
                if non_empty {
                    last = slot;
                }
            }
            // rare: fall back to the slot-based collapse handling
            let mut slots: [Slot<'a>; 16] = Default::default();
            if count == 1 {
                slots[last] = if changed[last] {
                    Slot::from_out(rebuilt[last].take().unwrap())
                } else {
                    let item = &payload[bounds[last] as usize..bounds[last + 1] as usize];
                    Slot::Keep { r: parse_ref(item)?, parent: src, slot: last as u32 }
                };
            }
            return self.assemble_branch(slots);
        }

        // splice: copy maximal runs of unchanged original items verbatim, insert new items
        let payload_len = (bounds[16] as isize + delta) as usize + 1; // + the value item

        let mut out = Vec::with_capacity(payload_len + 4);
        list_header_into(&mut out, payload_len);
        let mut run_start = 0usize;
        for t in 0..ntouched {
            let slot = touched[t] as usize;
            out.extend_from_slice(&payload[run_start..bounds[slot] as usize]);
            match &rebuilt[slot] {
                Some(Out::Enc(enc)) => ref_item_into(&mut out, enc),
                _ => out.push(alloy_rlp::EMPTY_STRING_CODE),
            }
            run_start = bounds[slot + 1] as usize;
        }
        // trailing run, including the (always empty) value item
        out.extend_from_slice(&payload[run_start..]);
        Ok(Out::Enc(out))
    }

    /// Assembles a branch from its 16 slot states, collapsing when 0 or 1 children remain
    /// (mirroring `MptNode::delete_internal`'s branch case).
    fn assemble_branch(&self, slots: [Slot<'a>; 16]) -> Result<Out, Error> {
        let count = slots.iter().filter(|s| !matches!(s, Slot::Missing)).count();
        match count {
            0 => Ok(Out::Empty),
            1 => {
                let (nib, slot) = slots
                    .iter()
                    .enumerate()
                    .find(|(_, s)| !matches!(s, Slot::Missing))
                    .map(|(i, s)| (i as u8, s))
                    .unwrap();
                match slot {
                    Slot::New(enc) => self.merge_prefix(&[nib], Out::Enc(enc.clone())),
                    Slot::Keep { r, parent, slot } => match r {
                        FlatRef::Inline(b) => self.merge_child_node(&[nib], parse_node(b)?),
                        FlatRef::Digest(d) => match self.child_src(*parent, *r, *slot) {
                            Ok(Src::Node(idx)) => {
                                self.merge_child_node(&[nib], self.parse_indexed(idx)?)
                            }
                            _ => {
                                // pruned sibling: extension over the digest, identical to the
                                // graph representation's Digest-orphan fallback
                                let mut item = Vec::with_capacity(33);
                                item.push(0xa0);
                                item.extend_from_slice(d);
                                Ok(Out::Enc(enc_ext(&[nib], &item)))
                            }
                        },
                        FlatRef::Empty => unreachable!("empty slot counted as present"),
                    },
                    Slot::Missing => unreachable!(),
                }
            }
            _ => {
                // single-pass branch encoding: arithmetic lengths, one output buffer
                let payload_len: usize = slots
                    .iter()
                    .map(|slot| match slot {
                        Slot::Missing => 1,
                        Slot::Keep { r, .. } => match r {
                            FlatRef::Digest(_) => 33,
                            FlatRef::Inline(b) => b.len(),
                            FlatRef::Empty => 1,
                        },
                        Slot::New(enc) => {
                            if enc.len() < 32 {
                                enc.len()
                            } else {
                                33
                            }
                        }
                    })
                    .sum::<usize>() +
                    1;
                let mut out = Vec::with_capacity(payload_len + 4);
                list_header_into(&mut out, payload_len);
                for slot in slots.iter() {
                    match slot {
                        Slot::Missing => out.push(alloy_rlp::EMPTY_STRING_CODE),
                        Slot::Keep { r, .. } => ref_bytes_of(*r, &mut out),
                        Slot::New(enc) => ref_item_into(&mut out, enc),
                    }
                }
                out.push(alloy_rlp::EMPTY_STRING_CODE); // branch value: always empty
                Ok(Out::Enc(out))
            }
        }
    }

    /// Prepends `nibs` to a real (blob or inline) node during branch collapse.
    fn merge_child_node(&self, nibs: &[u8], node: FlatNode<'a>) -> Result<Out, Error> {
        match node {
            FlatNode::Leaf { prefix, value } => {
                let mut n = nibs.to_vec();
                n.extend(prefix_nibs(prefix));
                Ok(Out::Enc(enc_leaf(&n, value)))
            }
            FlatNode::Extension { prefix, child } => {
                let mut n = nibs.to_vec();
                n.extend(prefix_nibs(prefix));
                let mut item = Vec::with_capacity(33);
                ref_bytes_of(child, &mut item);
                Ok(Out::Enc(enc_ext(&n, &item)))
            }
            FlatNode::Branch { payload } => {
                // the branch itself is unchanged: its encoding is header + original payload
                let mut enc = Vec::with_capacity(payload.len() + 4);
                list_header_into(&mut enc, payload.len());
                enc.extend_from_slice(payload);
                let mut item = Vec::with_capacity(33);
                ref_item_into(&mut item, &enc);
                Ok(Out::Enc(enc_ext(nibs, &item)))
            }
            _ => Err(Error::FlatTrie("unexpected node kind in collapse")),
        }
    }

    /// Prepends extension path `pn` to a rebuilt child, merging prefixes when the child is a
    /// leaf or extension (mirroring `MptNode::delete_internal`'s extension case).
    fn merge_prefix(&self, pn: &[u8], child_out: Out) -> Result<Out, Error> {
        let enc = match child_out {
            Out::Empty => return Ok(Out::Empty),
            Out::Enc(enc) => enc,
        };
        match parse_node(&enc)? {
            FlatNode::Leaf { prefix, value } => {
                let mut nibs = pn.to_vec();
                nibs.extend(prefix_nibs(prefix));
                Ok(Out::Enc(enc_leaf(&nibs, value)))
            }
            FlatNode::Extension { prefix, .. } => {
                let mut nibs = pn.to_vec();
                nibs.extend(prefix_nibs(prefix));
                let (payload_off, payload_len, _) = rlp_header(&enc, 0)?;
                let body = &enc[payload_off..payload_off + payload_len];
                let item0_len = rlp_item_len(body, 0)?;
                let child_item = body[item0_len..].to_vec();
                Ok(Out::Enc(enc_ext(&nibs, &child_item)))
            }
            FlatNode::Branch { .. } => {
                let mut item = Vec::with_capacity(33);
                ref_item_into(&mut item, &enc);
                Ok(Out::Enc(enc_ext(pn, &item)))
            }
            _ => Err(Error::FlatTrie("unexpected node kind after apply")),
        }
    }
}

impl FlatStateViews<'_> {
    /// The storage root an account had *before* this block, read out of the already-verified
    /// state trie.
    ///
    /// Only reached for an account with no entry in `self.storage`, where a missing entry and an
    /// empty trie would otherwise be indistinguishable -- both `EMPTY_ROOT`, which wipes the
    /// account's storage. The state trie is anchored to the parent header's root, so the answer
    /// is authenticated: a nonexistent account gives `EMPTY_ROOT`, and a path leaving the witness
    /// is an error rather than an absence (see [`FlatTrieView::get`]).
    ///
    /// Costs one state-trie walk plus a `TrieAccount::decode` per modified account with no
    /// witnessed storage trie -- mostly EOAs -- on `COMPUTE_STATE_ROOT`.
    ///
    /// **Measured at zero.** An earlier note here estimated 10^5--10^6 retired instructions per
    /// mainnet block by extrapolating `TrieDB`'s ~1,460-instruction walk, and flagged that as
    /// not measured. It has since been measured, by building the guest with this function
    /// short-circuited to `FLAT_EMPTY_ROOT` and emulating both arms over the thirteen
    /// `perf/bench_data/rv64` blocks: **-819 instructions summed across all thirteen**, i.e.
    /// nothing, with the per-block difference landing on both sides of zero. The reason is that
    /// the arm is reached only for a modified account the witness carries no storage trie for,
    /// which these witnesses essentially never contain. Do not re-estimate it; if the witness
    /// shape changes, re-measure.
    ///
    /// Carrying the root through `verified_views` does not help -- that map covers exactly the
    /// accounts this does not. Two routes would, both **untried rather than rejected**: memoising
    /// `TrieDB::basic_ref`, which already decodes these same rows and discards `storage_root`; or
    /// expressing a `delta_root` change as a function of the prior row, since its descent reaches
    /// these leaves anyway. Both change the state-root path, and neither is worth landing on an
    /// estimate.
    fn prior_storage_root(&self, hashed_address: &B256) -> Result<B256, Error> {
        use alloy_rlp::Decodable;
        Ok(match self.state.get(hashed_address.as_slice())? {
            Some(mut bytes) => reth_trie::TrieAccount::decode(&mut bytes)?.storage_root,
            None => FLAT_EMPTY_ROOT,
        })
    }

    /// Computes the post-state root for `post_state` directly from the verified blobs, without
    /// building any intermediate trie: storage-trie delta roots feed updated account rows into
    /// the state-trie delta.
    pub fn post_state_root(&self, post_state: &HashedPostState) -> Result<B256, Error> {
        let mut state_changes: Vec<(B256, Option<Vec<u8>>)> =
            Vec::with_capacity(post_state.accounts.len());

        for (hashed_address, account) in post_state.accounts.iter() {
            match account {
                None => state_changes.push((*hashed_address, None)),
                Some(account) => {
                    let storage_root = match post_state.storages.get(hashed_address) {
                        Some(st) => {
                            let slot_changes: Vec<(B256, Option<Vec<u8>>)> = st
                                .storage
                                .iter()
                                .map(|(slot, value)| {
                                    (*slot, (!value.is_zero()).then(|| alloy_rlp::encode(value)))
                                })
                                .collect();
                            match self.storage.get(hashed_address) {
                                Some(view) if !st.wiped => view.delta_root(&slot_changes)?,
                                // `wiped` is legitimate -- a destroyed or freshly created
                                // account starts from the empty trie.
                                Some(_) => FlatTrieView::empty_delta_root(&slot_changes)?,
                                None if st.wiped => FlatTrieView::empty_delta_root(&slot_changes)?,
                                // No witnessed trie and not wiped: the delta can only be
                                // applied to the empty trie, which is the right answer exactly
                                // when the account had no storage. Anything else is a witness
                                // that does not cover what the block changed, and computing a
                                // root from it would silently drop the account's storage.
                                None => {
                                    if self.prior_storage_root(hashed_address)? == FLAT_EMPTY_ROOT {
                                        FlatTrieView::empty_delta_root(&slot_changes)?
                                    } else {
                                        return Err(Error::FlatTrie(
                                            "no witnessed storage trie for a modified account",
                                        ));
                                    }
                                }
                            }
                        }
                        // Unchanged storage: the row keeps the root it already had. Taking
                        // `EMPTY_ROOT` when there is no witnessed trie wipes it instead.
                        None => match self.storage.get(hashed_address) {
                            Some(v) => v.root_hash,
                            None => self.prior_storage_root(hashed_address)?,
                        },
                    };
                    let trie_account = reth_trie::TrieAccount {
                        nonce: account.nonce,
                        balance: account.balance,
                        storage_root,
                        code_hash: account.get_bytecode_hash(),
                    };
                    state_changes.push((*hashed_address, Some(alloy_rlp::encode(&trie_account))));
                }
            }
        }

        self.state.delta_root(&state_changes)
    }
}
