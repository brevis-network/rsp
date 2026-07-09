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
        keccak, node_from_digest, prefix_nibs, to_nibs, Error, MptNode, MptNodeData,
        MptNodeReference, EMPTY_ROOT,
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

#[derive(Debug, Clone, Copy)]
struct NodeRec {
    off: u32,
    len: u32,
    /// start into `edges`; branches own 16 slots, extensions 1, leaves 0.
    edge_start: u32,
}

/// A parsed, linkage-verified flat trie.
#[derive(Debug)]
pub struct FlatTrieView<'a> {
    bytes: &'a [u8],
    pub root_hash: B256,
    nodes: Vec<NodeRec>,
    edges: Vec<u32>,
}

impl<'a> FlatTrieView<'a> {
    /// Single linear pass: keccak every blob, check it against a pending digest reference on
    /// the DFS frontier, and record child edges.
    pub fn parse_and_verify(bytes: &'a [u8]) -> Result<Self, Error> {
        let mut view = FlatTrieView {
            bytes,
            root_hash: EMPTY_ROOT,
            nodes: Vec::with_capacity(bytes.len() / 96 + 4),
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
                view.nodes.push(NodeRec { off: 0, len: root_len as u32, edge_start: 0 });
                return Ok(view);
            }
            _ => {}
        }
        view.root_hash = B256::from(keccak(root_blob));
        view.push_node(0, root_len, &root)?;

        // DFS frontier: (node index, next child slot to consider). A child slot is an index
        // into the node's edge range.
        let mut frontier: Vec<(u32, u32)> = Vec::with_capacity(64);
        frontier.push((0, 0));

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
                let Some((n_idx, slot)) = frontier.last_mut() else {
                    return Err(Error::FlatTrie("blob does not attach to the trie"));
                };
                let rec = view.nodes[*n_idx as usize];
                let n_edges = view.edge_count(*n_idx);
                while *slot < n_edges {
                    let s = *slot;
                    *slot += 1;
                    let edge = view.edges[rec.edge_start as usize + s as usize];
                    if edge != EDGE_PRUNED {
                        continue; // inline or already matched
                    }
                    let digest = view.child_digest(*n_idx, s)?;
                    if digest == hash {
                        let idx = view.nodes.len() as u32;
                        view.edges[rec.edge_start as usize + s as usize] = idx;
                        break 'search idx;
                    }
                }
                frontier.pop();
            };

            let node = parse_node(blob)?;
            if matches!(node, FlatNode::Null | FlatNode::Digest(_)) {
                return Err(Error::FlatTrie("null/digest blob below root"));
            }
            view.push_node(pos, len, &node)?;
            frontier.push((node_idx, 0));
            pos += len;
        }

        Ok(view)
    }

    fn push_node(&mut self, off: usize, len: usize, node: &FlatNode<'_>) -> Result<(), Error> {
        let edge_start = self.edges.len() as u32;
        self.nodes.push(NodeRec { off: off as u32, len: len as u32, edge_start });
        match node {
            FlatNode::Extension { child, .. } => {
                self.edges.push(match child {
                    FlatRef::Digest(_) => EDGE_PRUNED,
                    FlatRef::Inline(_) => EDGE_INLINE,
                    FlatRef::Empty => return Err(Error::FlatTrie("extension with empty child")),
                });
            }
            FlatNode::Branch { payload } => {
                let base = self.edges.len();
                self.edges.resize(base + 16, EDGE_PRUNED);
                for_branch_children(payload, |slot, r| {
                    self.edges[base + slot] = match r {
                        FlatRef::Digest(_) => EDGE_PRUNED,
                        FlatRef::Inline(_) => EDGE_INLINE,
                        FlatRef::Empty => EDGE_PRUNED, // never matched: digest lookup skips empties
                    };
                    Ok(())
                })?;
            }
            _ => {}
        }
        Ok(())
    }

    fn edge_count(&self, idx: u32) -> u32 {
        let start = self.nodes[idx as usize].edge_start;
        let end = self
            .nodes
            .get(idx as usize + 1)
            .map(|n| n.edge_start)
            .unwrap_or(self.edges.len() as u32);
        end - start
    }

    fn blob(&self, idx: u32) -> &'a [u8] {
        let rec = self.nodes[idx as usize];
        &self.bytes[rec.off as usize..(rec.off + rec.len) as usize]
    }

    /// The digest stored in child slot `slot` of node `idx` (empty slots return an
    /// impossible-to-match sentinel digest).
    fn child_digest(&self, idx: u32, slot: u32) -> Result<[u8; 32], Error> {
        let node = parse_node(self.blob(idx))?;
        let r = match node {
            FlatNode::Extension { child, .. } => {
                if slot != 0 {
                    return Err(Error::FlatTrie("bad extension slot"));
                }
                child
            }
            FlatNode::Branch { payload } => branch_child(payload, slot as usize)?,
            _ => return Err(Error::FlatTrie("edge on childless node")),
        };
        match r {
            FlatRef::Digest(d) => {
                let mut out = [0u8; 32];
                out.copy_from_slice(d);
                Ok(out)
            }
            // empty branch slots keep an EDGE_PRUNED edge but can never match a real keccak
            // preimage of a >= 32-byte blob; use an unmatchable sentinel.
            FlatRef::Empty => Ok([0u8; 32]),
            FlatRef::Inline(_) => Err(Error::FlatTrie("digest requested for inline child")),
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
            match parse_node(blob)? {
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
        let node = parse_node(blob)?;
        Ok(match node {
            FlatNode::Null => MptNode::default(),
            FlatNode::Digest(d) => MptNodeData::Digest(B256::from_slice(d)).into(),
            FlatNode::Leaf { prefix, value } => {
                MptNodeData::Leaf(prefix.to_vec(), value.to_vec()).into()
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
                MptNodeData::Extension(prefix.to_vec(), Box::new(child_node)).into()
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
                // a delete through a 2-child branch may collapse it: the surviving sibling's
                // real shape is then needed, so materialize all children one level deep.
                let mut child_count = 0usize;
                for_branch_children(payload, |_, r| {
                    if !matches!(r, FlatRef::Empty) {
                        child_count += 1;
                    }
                    Ok(())
                })?;
                let force_shallow = has_delete && child_count == 2;

                let mut children: [Option<Box<MptNode>>; 16] = Default::default();
                for slot in 0..16 {
                    let r = branch_child(payload, slot)?;
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
                MptNodeData::Branch(children).into()
            }
        })
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
    /// changed accounts/slots are materialized, untouched storage tries become digest stubs.
    /// Running the existing `update()` + `state_root()` on the overlay yields the exact
    /// post-state root.
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
    }
}
