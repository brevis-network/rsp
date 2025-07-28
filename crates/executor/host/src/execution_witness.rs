use alloy_provider::RootProvider;
use alloy_rpc_types::debug::ExecutionWitness;
use reth_trie::Nibbles;
use rsp_mpt::mpt::resolve_nodes;
use rsp_mpt::mpt::resolve_state_nodes;
use rsp_mpt::mpt::MptNode;
use rsp_mpt::mpt::MptNodeData;
use rsp_mpt::mpt::MptNodeReference;
use rsp_mpt::mpt::EMPTY_ROOT;
use rsp_mpt::EthereumState;

pub fn eth_state_from_execution_witness(
    witness: &ExecutionWitness,
    pre_state_root: alloy_primitives::FixedBytes<32>,
) -> EthereumState {
    let (state_trie, storage_tries) = build_validated_tries(witness, pre_state_root).unwrap();
    EthereumState { state_trie, storage_tries }
}

// Builds tries from the witness state.
// NOTE: This method should be called outside zkVM! In general you construct tries, then validate them inside zkVM.
pub fn build_validated_tries(
    witness: &ExecutionWitness,
    pre_state_root: alloy_primitives::FixedBytes<32>,
) -> Result<
    (
        MptNode,
        alloy_primitives::map::hash_map::HashMap<
            alloy_primitives::FixedBytes<32>,
            MptNode,
            alloy_primitives::map::foldhash::fast::RandomState,
        >,
    ),
    String,
> {
    // Step 1: Decode all RLP-encoded trie nodes and index by hash
    // IMPORTANT: Witness state contains both *state trie* nodes and *storage tries* nodes!
    let mut node_map: alloy_primitives::map::HashMap<MptNodeReference, MptNode> =
        alloy_primitives::map::HashMap::default();
    let mut node_by_hash: alloy_primitives::map::HashMap<alloy_primitives::B256, MptNode> =
        alloy_primitives::map::HashMap::default();
    let mut root_node: Option<MptNode> = None;

    for encoded in &witness.state {
        let node = MptNode::decode(encoded).expect("Valid MPT node in witness");
        let hash = alloy_primitives::keccak256(encoded);
        if hash == pre_state_root {
            root_node = Some(node.clone());
        }
        node_by_hash.insert(hash, node.clone());
        node_map.insert(node.reference(), node);
    }

    // Step 2: Use root_node or fallback to Digest
    let root = root_node.unwrap_or_else(|| MptNodeData::Digest(pre_state_root).into());

    // Build state trie.
    let mut storage_tries_detected = vec![];
    let state_trie =
        resolve_state_nodes(&root, &node_map, &mut storage_tries_detected, Nibbles::default());

    // Step 3: Build storage tries per account efficiently
    let mut storage_tries: alloy_primitives::map::HashMap<alloy_primitives::B256, MptNode> =
        alloy_primitives::map::HashMap::default();

    for (hashed_address, storage_root) in storage_tries_detected {
        let root_node = match node_by_hash.get(&storage_root).cloned() {
            Some(node) => node,
            None => {
                // An execution witness can include an account leaf (with non-empty storageRoot), but omit
                // its entire storage trie when that account's storage was NOT touched during the block.
                continue;
            }
        };
        let storage_trie = resolve_nodes(&root_node, &node_map);

        if storage_trie.is_digest() {
            panic!("Could not resolve storage trie for {storage_root}");
        }

        // Insert resolved storage trie.
        storage_tries.insert(hashed_address, storage_trie);
    }

    // Step 3a: Verify that state_trie was built correctly - confirm tree hash with pre state root.
    validate_state_trie(&state_trie, pre_state_root);

    // Step 3b: Verify that each storage trie matches the declared storage_root in the state trie.
    validate_storage_tries(&state_trie, &storage_tries)?;

    Ok((state_trie, storage_tries))
}

/// Computes the state root (over state trie).
pub fn eth_state_compute_state_root(eth_state: &EthereumState) -> alloy_primitives::B256 {
    eth_state.state_trie.hash()
}

/// Mutates state based on diffs provided in [`HashedPostState`].
pub fn eth_state_update(eth_state: &mut EthereumState, post_state: &reth_trie::HashedPostState) {
    // Apply *all* storage-slot updates first and remember new roots.
    let mut new_storage_roots: alloy_primitives::map::HashMap<Vec<u8>, alloy_primitives::B256> =
        alloy_primitives::map::HashMap::default(); // TODO: Use `with_capacity(post_state.storages.len())`.
    for (hashed_addr, storage) in post_state.storages.iter() {
        // Take existing storage trie or create an empty one.
        let storage_trie = eth_state.storage_tries.entry(*hashed_addr).or_default();

        // Wipe the trie if requested.
        if storage.wiped {
            storage_trie.clear();
        }

        // Apply slot-level changes.
        for (slot, value) in storage.storage.iter() {
            let key = slot.as_slice();
            if value.is_zero() {
                storage_trie.delete(key).unwrap();
            } else {
                storage_trie.insert_rlp(key, *value).unwrap();
            }
        }

        // Memorise the freshly-computed root.
        new_storage_roots.insert(hashed_addr.to_vec(), storage_trie.hash());
    }

    // Walk the accounts, using the roots computed above.
    for (hashed_addr, maybe_acct) in post_state.accounts.iter() {
        let addr = hashed_addr.as_slice();

        match maybe_acct {
            // Handle account update / creation.
            Some(acct) => {
                // Which storage root should we encode?
                let storage_root = new_storage_roots
                    .get(addr)
                    .copied() // root from step 1
                    .or_else(|| eth_state.storage_tries.get(addr).map(|t| t.hash()))
                    .unwrap_or(EMPTY_ROOT);

                // If both the account and its storage are empty we simply delete.
                if acct.is_empty() && storage_root == EMPTY_ROOT {
                    eth_state.state_trie.delete(addr).unwrap();
                    eth_state.storage_tries.remove(addr); // keep maps in sync
                    continue;
                }

                // Encode and insert the account leaf.
                let trie_acct = reth_trie::TrieAccount {
                    nonce: acct.nonce,
                    balance: acct.balance,
                    storage_root,
                    code_hash: acct.get_bytecode_hash(),
                };
                eth_state.state_trie.insert_rlp(addr, trie_acct).unwrap();
            }

            // Handle account deletion.
            None => {
                eth_state.state_trie.delete(addr).unwrap();
                eth_state.storage_tries.remove(addr); // NOTE: Could be skipped in zkVM.
            }
        }
    }
}

// NOTE: It provides 1-to-1 mapping with `StatelessTrie::account`.
pub fn eth_state_account(
    eth_state: &EthereumState,
    address: alloy_primitives::Address,
) -> Option<reth_trie::TrieAccount> {
    let hashed_address = alloy_primitives::keccak256(address);
    let hashed_address = hashed_address.as_slice();

    let account_in_trie =
        eth_state.state_trie.get_rlp::<reth_trie::TrieAccount>(hashed_address).unwrap();

    account_in_trie
}

// NOTE: It provides 1-to-1 mapping with `StatelessTrie::storage`.
pub fn eth_state_storage(
    eth_state: &EthereumState,
    address: alloy_primitives::Address,
    index: alloy_primitives::U256,
) -> alloy_primitives::U256 {
    let hashed_address = alloy_primitives::keccak256(address);
    let hashed_address = hashed_address.as_slice();

    // Usual case, where given storage slot is present.
    if let Some(storage_trie) = eth_state.storage_tries.get(hashed_address) {
        return storage_trie
            .get_rlp::<alloy_primitives::U256>(
                alloy_primitives::keccak256(index.to_be_bytes::<32>()).as_slice(),
            )
            .expect("Can get storage from MPT")
            .unwrap_or_default();
    }

    // Storage slot value is not present in the trie, validate that the witness is complete.
    // TODO: Implement witness checks like in reth - https://github.com/paradigmxyz/reth/blob/127595e23079de2c494048d0821ea1f1107eb624/crates/stateless/src/trie.rs#L68C9-L87.
    let account = eth_state
        .state_trie
        .get_rlp::<reth_trie::TrieAccount>(hashed_address)
        .expect("Can get account from MPT");
    match account {
        Some(account) => {
            if account.storage_root != EMPTY_ROOT {
                todo!("Validate that storage witness is valid");
            }
        }
        None => {
            todo!("Validate that account witness is valid");
        }
    }

    // Account doesn't exist or has empty storage root.
    alloy_primitives::U256::ZERO
}

fn eth_state_new(
    witness: &ExecutionWitness,
    pre_state_root: alloy_primitives::B256,
) -> (EthereumState, alloy_primitives::map::B256Map<revm::state::Bytecode>) {
    let ethereum_state = eth_state_from_execution_witness(witness, pre_state_root);

    let mut bytecodes: alloy_primitives::map::B256Map<revm::state::Bytecode> =
        alloy_primitives::map::B256Map::default();
    for encoded in &witness.codes {
        let hash = alloy_primitives::keccak256(encoded);
        bytecodes.insert(hash, revm::state::Bytecode::new_raw(encoded.clone()));
    }

    (ethereum_state, bytecodes)
}

fn eth_state_calculate_state_root(
    eth_state: &mut EthereumState,
    post_state: reth_trie::HashedPostState,
) -> alloy_primitives::B256 {
    eth_state_update(eth_state, &post_state);
    eth_state_compute_state_root(eth_state)
}

// Validate that state_trie was built correctly - confirm tree hash with pre state root.
pub fn validate_state_trie(state_trie: &MptNode, pre_state_root: alloy_primitives::FixedBytes<32>) {
    if state_trie.hash() != pre_state_root {
        panic!("Computed state root does not match pre_state_root");
    }
}

// Validates that each storage trie matches the declared storage_root in the state trie.
pub fn validate_storage_tries(
    state_trie: &MptNode,
    storage_tries: &alloy_primitives::map::hash_map::HashMap<
        alloy_primitives::FixedBytes<32>,
        MptNode,
        alloy_primitives::map::foldhash::fast::RandomState,
    >,
) -> Result<(), String> {
    for (hashed_address, storage_trie) in storage_tries.iter() {
        let account = state_trie
            .get_rlp::<reth_trie::TrieAccount>(hashed_address.as_slice())
            .map_err(|_| "Failed to decode account from state trie")?
            .ok_or("Account not found in state trie")?;

        let storage_root = account.storage_root;
        let actual_hash = storage_trie.hash();

        if storage_root != actual_hash {
            return Err(format!(
                "Mismatched storage root for address hash {:?}: expected {:?}, got {:?}",
                hashed_address, storage_root, actual_hash
            )
            .into());
        }
    }

    Ok(())
}
