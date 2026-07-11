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

use alloy_primitives::{map::HashMap, B256};
use alloy_rlp::Encodable;
use reth_trie::HashedPostState;
use serde::{Deserialize, Serialize};

use crate::{
    mpt::{
        keccak, node_from_digest, node_with_cached_reference, prefix_nibs, to_nibs, Error,
        MptNode, MptNodeData, MptNodeReference, EMPTY_ROOT,
    },
    EthereumState,
};

/// Serde helper for `Cow<'a, [u8]>` that borrows from the input when the deserializer supports
/// it (bincode over a byte slice, i.e. the guest path) and falls back to an owned copy when it
/// does not (bincode over a reader, i.e. the host input cache path).
mod cow_bytes {
    use std::borrow::Cow;

    use serde::{Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(v: &Cow<'_, [u8]>, s: S) -> Result<S::Ok, S::Error> {
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

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Cow<'de, [u8]>, D::Error> {
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
            HashMap::with_capacity_and_hasher(self.storage_tries.len(), Default::default());
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
    match b0 {
        0x00..=0x7f => Ok((pos, 1, false)),
        0x80..=0xb7 => Ok((pos + 1, (b0 - 0x80) as usize, false)),
        0xb8..=0xbf => {
            let ll = (b0 - 0xb7) as usize;
            let len = be_len(bytes, pos + 1, ll)?;
            Ok((pos + 1 + ll, len, false))
        }
        0xc0..=0xf7 => Ok((pos + 1, (b0 - 0xc0) as usize, true)),
        0xf8..=0xff => {
            let ll = (b0 - 0xf7) as usize;
            let len = be_len(bytes, pos + 1, ll)?;
            Ok((pos + 1 + ll, len, true))
        }
    }
}

#[inline]
fn be_len(bytes: &[u8], pos: usize, ll: usize) -> Result<usize, Error> {
    let raw = bytes.get(pos..pos + ll).ok_or(Error::FlatTrie("truncated RLP length"))?;
    let mut len = 0usize;
    for &b in raw {
        len = (len << 8) | b as usize;
    }
    Ok(len)
}

/// Total encoded length of the RLP item starting at `pos`.
#[inline]
fn rlp_item_len(bytes: &[u8], pos: usize) -> Result<usize, Error> {
    let (payload, len, _) = rlp_header(bytes, pos)?;
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
    Leaf { prefix: &'a [u8], value: &'a [u8] },
    Extension { prefix: &'a [u8], child: FlatRef<'a> },
    /// Payload region of the 17-item list.
    Branch { payload: &'a [u8] },
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
                view.nodes.push(NodeRec { off: 0, len: root_len as u32, edge_start: 0, kind: KIND_DIGEST });
                view.hashes.push(view.root_hash);
                return Ok(view);
            }
            _ => {}
        }
        view.root_hash = B256::from(keccak(root_blob));
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
            if bytes.len() < pos + len {
                return Err(Error::FlatTrie("truncated node blob"));
            }
            let blob = &bytes[pos..pos + len];
            let hash = keccak(blob);

            // Find the next pending digest reference matching this blob's hash. Non-matching
            // references we walk past are pruned subtrees and stay EDGE_PRUNED.
            let node_idx = 'search: loop {
                let Some(top) = frontier.last_mut() else {
                    return Err(Error::FlatTrie("blob does not attach to the trie"));
                };
                while top.item_pos < top.items_end && top.slot < top.nslots {
                    let (payload_off, payload_len, is_list) = rlp_header(bytes, top.item_pos as usize)?;
                    let item_end = payload_off + payload_len;
                    let slot = top.slot;
                    top.item_pos = item_end as u32;
                    top.slot += 1;
                    if !is_list
                        && payload_len == 32
                        && bytes[payload_off..item_end] == hash[..]
                    {
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
            view.hashes.push(B256::from(hash));
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
    /// raw blobs without allocating. Returns `None` for absent keys *and* for keys that resolve
    /// into pruned subtrees (matching this fork's `MptNode::get_internal` behavior).
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
            let node =
                if inline { parse_node(blob)? } else { self.parse_indexed(node_idx)? };
            match node {
                FlatNode::Null | FlatNode::Digest(_) => return Ok(None),
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
                        FlatRef::Digest(_) => {
                            if inline {
                                return Err(Error::FlatTrie("digest ref inside inline node"));
                            }
                            let edge = self.edges
                                [self.nodes[node_idx as usize].edge_start as usize];
                            if edge == EDGE_PRUNED {
                                return Ok(None);
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
                    match branch_child(payload, slot)? {
                        FlatRef::Empty => return Ok(None),
                        FlatRef::Inline(b) => {
                            blob = b;
                            inline = true;
                        }
                        FlatRef::Digest(_) => {
                            if inline {
                                return Err(Error::FlatTrie("digest ref inside inline node"));
                            }
                            let edge = self.edges
                                [self.nodes[node_idx as usize].edge_start as usize + slot];
                            if edge == EDGE_PRUNED {
                                return Ok(None);
                            }
                            node_idx = edge;
                            blob = self.blob(edge);
                        }
                    }
                }
            }
        }
    }

    /// Materializes a sparse [`MptNode`] overlay containing the full paths for every key in
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
            FlatNode::Leaf { prefix, value } => {
                MptNodeData::Leaf(prefix.to_vec(), value.to_vec())
            }
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
                    let edge = self.edges
                        [self.nodes[idx as usize].edge_start as usize + slot as usize];
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
    pub storage: HashMap<B256, FlatTrieView<'a>>,
}

impl FlatStateViews<'_> {
    /// Builds the copy-on-write [`EthereumState`] overlay for a state transition: paths for all
    /// effectively-changed accounts/slots are materialized, untouched storage tries become
    /// digest stubs. Running the existing `update()` + `state_root()` on the overlay with the
    /// returned (filtered) post state yields the exact post-state root.
    pub fn materialize_overlay(&self, post_state: &HashedPostState) -> Result<EthereumState, Error> {
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

        Ok(EthereumState { state_trie, storage_tries })
    }
}

#[cfg(test)]
mod tests {
    use alloy_primitives::U256;
    use reth_primitives_traits::Account;
    use reth_trie::{HashedPostState, HashedStorage, TrieAccount};

    use super::*;
    use crate::mpt::KECCAK_EMPTY;

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

    #[test]
    fn flat_pruned_subtree() {
        // prune one subtree to a digest; the view must verify and treat it as absent
        let trie = keccak_trie(64);
        let MptNodeData::Branch(children) = trie.as_data().clone() else {
            panic!("expected branch root")
        };
        let mut pruned_children = children;
        let victim =
            pruned_children.iter_mut().flatten().next().expect("at least one child");
        **victim = node_from_digest(victim.hash());
        let pruned: MptNode = MptNodeData::Branch(pruned_children).into();
        assert_eq!(pruned.hash(), trie.hash());

        let bytes = flatten_trie(&pruned);
        let view = FlatTrieView::parse_and_verify(&bytes).unwrap();
        assert_eq!(view.root_hash, trie.hash());

        let mut pruned_hits = 0;
        for i in 0..64usize {
            let key = keccak(i.to_be_bytes());
            let got = view.get(&key).unwrap();
            let expected = pruned.get(&key).unwrap();
            assert_eq!(got, expected);
            if expected.is_none() {
                pruned_hits += 1;
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
        assert_eq!(view.get(&keccak(b"x")).unwrap(), None);
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
        for (addr, storage, bal) in
            [(addr_a, &storage_a, 100u64), (addr_b, &storage_b, 200u64)]
        {
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
        storage_changes
            .storage
            .insert(B256::from(keccak(200usize.to_be_bytes())), U256::from(43));
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

        let changes: Vec<(B256, Option<Vec<u8>>)> = ops
            .iter()
            .map(|(k, v)| (B256::from(*k), v.map(|v| alloy_rlp::encode(v))))
            .collect();
        assert_eq!(view.delta_root(&changes).unwrap(), full.hash());
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

    #[test]
    fn delta_root_randomized_parity() {
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
                ops.push((
                    keccak(idx.to_be_bytes()),
                    if delete { None } else { Some(x >> 24) },
                ));
            }
            // dedup by key (last op wins), mirroring HashedPostState semantics
            let mut seen = std::collections::HashMap::new();
            for (k, v) in ops {
                seen.insert(k, v);
            }
            let ops: Vec<([u8; 32], Option<u64>)> = seen.into_iter().collect();
            delta_parity_case(&trie, &ops);
        }
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
                .sum::<usize>()
                + 1;
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
        // collect original child refs in one scan
        let mut slots: [Slot<'a>; 16] = Default::default();
        let mut pos = 0usize;
        let mut orig: [FlatRef<'a>; 16] = [FlatRef::Empty; 16];
        for r in orig.iter_mut() {
            let item_len = rlp_item_len(payload, pos)?;
            *r = parse_ref(&payload[pos..pos + item_len])?;
            pos += item_len;
        }

        // group changes by leading nibble (they are sorted)
        let mut idx = 0usize;
        for slot in 0..16usize {
            let start = idx;
            while idx < changes.len() && changes[idx].0[0] == slot as u8 {
                idx += 1;
            }
            if start == idx {
                if !matches!(orig[slot], FlatRef::Empty) {
                    slots[slot] = Slot::Keep { r: orig[slot], parent: src, slot: slot as u32 };
                }
                continue;
            }
            let group: Vec<Change<'_>> =
                changes[start..idx].iter().map(|(k, v)| (&k[1..], *v)).collect();
            let out = match orig[slot] {
                FlatRef::Empty => Self::apply_empty(&group),
                r => self.apply_src(self.child_src(src, r, slot as u32)?, &group)?,
            };
            slots[slot] = Slot::from_out(out);
        }

        self.assemble_branch(slots)
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
                    .sum::<usize>()
                    + 1;
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
                                _ => FlatTrieView::empty_delta_root(&slot_changes)?,
                            }
                        }
                        None => self
                            .storage
                            .get(hashed_address)
                            .map(|v| v.root_hash)
                            .unwrap_or(FLAT_EMPTY_ROOT),
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
