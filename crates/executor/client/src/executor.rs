use std::sync::Arc;

use alloy_consensus::{BlockHeader, Header};
use alloy_eip7928::{compute_block_access_list_hash, total_bal_items, ITEM_COST};
use alloy_primitives::{keccak256, Address, B256};
use evm2::{
    evm::{DbRef, DynDatabase},
    install_crypto,
};
use reth_chainspec::ChainSpec;
use reth_errors::BlockExecutionError;
use reth_evm::{
    execute::{BlockExecutionOutput, BlockExecutor, BlockExecutorFactory},
    BundleState, ConfigureEvm,
};
use reth_evm_ethereum::EthEvmConfig;
use reth_primitives_traits::{Block, NodePrimitives, RecoveredBlock};
use reth_trie_common::{HashedPostState, KeccakKeyHasher};

use crate::{
    custom::CustomCrypto,
    error::ClientError,
    into_primitives::FromInput,
    io::{ClientExecutorInput, CommittedHeader, WitnessInput},
    BlockValidator,
};

pub const DESERIALZE_INPUTS: &str = "deserialize inputs";
pub const INIT_WITNESS_DB: &str = "initialize witness db";
pub const RECOVER_SENDERS: &str = "recover senders";
pub const BLOCK_EXECUTION: &str = "block execution";
pub const VALIDATE_HEADER: &str = "validate header";
pub const VALIDATE_EXECUTION: &str = "validate block post-execution";
pub const COMPUTE_STATE_ROOT: &str = "compute state root";

pub type EthClientExecutor = ClientExecutor<EthEvmConfig<ChainSpec>, ChainSpec>;

/// Gives out the block executor factory of an EVM configuration.
///
/// `ConfigureEvm::block_executor_factory` does the same but exists only with reth-evm's `std`
/// feature, and that feature also selects the C secp256k1 library, which a zkVM guest cannot link.
pub trait ExecutorFactoryAccess: ConfigureEvm {
    fn executor_factory(&self) -> &Self::BlockExecutorFactory;
}

impl ExecutorFactoryAccess for EthEvmConfig<ChainSpec> {
    fn executor_factory(&self) -> &Self::BlockExecutorFactory {
        &self.executor_factory
    }
}

/// Executes `block` on `database` and returns its output together with the hash of the block
/// access list that the execution produced (EIP-7928). The hash is `None` for blocks before
/// Amsterdam, whose header does not commit to an access list.
///
/// This is the sequence of reth's `BasicBlockExecutor`, written out because that type sits behind
/// reth-evm's `std` feature (see [`ExecutorFactoryAccess`]).
pub fn execute_block<C>(
    evm_config: &C,
    block: &RecoveredBlock<<C::Primitives as NodePrimitives>::Block>,
    database: impl DynDatabase,
) -> Result<
    (BlockExecutionOutput<<C::Primitives as NodePrimitives>::Receipt>, Option<B256>),
    ClientError,
>
where
    C: ExecutorFactoryAccess,
{
    let factory = evm_config.executor_factory();
    let evm_env = evm_config.evm_env(block.header()).map_err(BlockExecutionError::other)?;
    let evm = factory.evm_with_env(database, evm_env);
    let ctx =
        evm_config.context_for_block(block.sealed_block()).map_err(BlockExecutionError::other)?;
    let mut executor = factory.create_executor(evm, ctx);
    if block.header().block_access_list_hash().is_some() {
        executor.enable_block_access_list_builder();
    }

    executor.apply_pre_execution_changes()?;
    for transaction in block.transactions_recovered() {
        executor.execute_transaction(transaction)?;
    }
    let (output, block_access_list) = executor.finish_with_block_access_list()?;

    let block_access_list_hash = match block_access_list {
        Some(bal) => {
            // EIP-7928: every item in the list is paid for out of the block gas limit.
            let items = total_bal_items(&bal);
            let limit = block.header().gas_limit() / ITEM_COST as u64;
            if items > limit {
                return Err(ClientError::BlockAccessListTooLarge(items, limit));
            }
            Some(compute_block_access_list_hash(&bal))
        }
        None => None,
    };

    Ok((output, block_access_list_hash))
}

/// Hashes the state changes of a block for the trie update, and lists the accounts whose storage
/// was wiped because the account was destroyed in the block.
///
/// reth's [`HashedPostState`] no longer marks wiped storages, so the tries have to be told
/// separately: call `clear_storage` for every returned address before `update`. Each of them
/// gets an entry in `storages` here, empty if need be, so that `update` recomputes its root.
pub fn hashed_post_state(bundle: &BundleState) -> (HashedPostState, Vec<B256>) {
    let mut post_state = HashedPostState::from_bundle_state::<KeccakKeyHasher>(bundle.state.iter());
    let wiped: Vec<B256> = bundle
        .state
        .iter()
        .filter(|(_, account)| account.was_destroyed())
        .map(|(address, _)| keccak256(address))
        .collect();
    for hashed_address in &wiped {
        post_state.storages.entry(*hashed_address).or_default();
    }
    (post_state, wiped)
}

/// An executor that executes a block inside a zkVM.
#[derive(Debug, Clone)]
pub struct ClientExecutor<C: ConfigureEvm, CS> {
    evm_config: C,
    chain_spec: Arc<CS>,
}

impl<C, CS> ClientExecutor<C, CS>
where
    C: ExecutorFactoryAccess,
    C::Primitives: FromInput + BlockValidator<CS>,
{
    // Under the `arena` backend `input` is only ever read (the tries live in a separate
    // bump-scoped value), so the `mut` binding needed by the legacy in-place update is unused.
    #[cfg_attr(any(feature = "arena", feature = "flat"), allow(unused_mut))]
    pub fn execute(
        &self,
        mut input: ClientExecutorInput<C::Primitives>,
    ) -> Result<Header, ClientError> {
        // Opcode tracking was an inspector on revm's interpreter and has not been ported; fail
        // rather than hand back a report with no opcode figures in it.
        if input.opcode_tracking {
            return Err(ClientError::OpcodeTrackingUnsupported);
        }

        let sealed_headers = input.sealed_headers().collect::<Vec<_>>();

        // Decode the arena witness into bump-scoped tries (zero-copy, hash-verifying). The bump
        // and the decoded tries live for the whole function so the witness DB can borrow them;
        // `tries` is mutated by the post-execution state update.
        #[cfg(feature = "arena")]
        let bump = bumpalo::Bump::new();
        #[cfg(feature = "arena")]
        let mut tries = input.tries(&bump).unwrap();
        // The flat witness is read in place: `flat_state` borrows the node blobs out of the input
        // and the tries verify them once, here.
        #[cfg(feature = "flat")]
        let flat_state = input.flat_state().unwrap();
        #[cfg(feature = "flat")]
        let mut tries = rsp_mpt::FlatTries::new(&flat_state).unwrap();

        let block = profile_report!(RECOVER_SENDERS, {
            C::Primitives::from_input_block(input.current_block.clone())
                .try_into_recovered()
                .map_err(|_| ClientError::SignatureRecoveryFailed)
        })?;

        // Validate the block, and its header against its parent's.
        //
        // The ancestors older than the parent are in the input for the hashes `BLOCKHASH`
        // reads, and are not validated. `build_trie_db` holds every ancestor to the hash its
        // child names, which makes them the ancestors of this block; whether one of them was
        // valid against its own parent is for the proof of that block. reth's stateless
        // validation asks the same of the headers it is given.
        profile_report!(VALIDATE_HEADER, {
            C::Primitives::validate_block(&block, self.chain_spec.clone())
                .expect("The block is invalid");

            let [header, parent, ..] = sealed_headers.as_slice() else {
                panic!("The parent header is missing");
            };
            C::Primitives::validate_header_against_parent(header, parent, self.chain_spec.clone())
                .expect("The header is invalid against its parent");
        });

        // The witnessed database borrows the tries, so execution is scoped to let the state
        // update below take them mutably again.
        let (execution_output, block_access_list_hash) = {
            // Initialize the witnessed database with verified storage proofs.
            let db = profile_report!(INIT_WITNESS_DB, {
                #[cfg(not(any(feature = "arena", feature = "flat")))]
                let trie_db = input.witness_db(&sealed_headers).unwrap();
                #[cfg(any(feature = "arena", feature = "flat"))]
                let trie_db = input.witness_db(&tries, &sealed_headers).unwrap();
                DbRef::new(trie_db)
            });

            profile_report!(BLOCK_EXECUTION, { execute_block(&self.evm_config, &block, db) })?
        };

        // Validate the block post execution. From Amsterdam on this includes the hash of the
        // block access list against the one in the header.
        profile_report!(VALIDATE_EXECUTION, {
            C::Primitives::validate_block_post_execution(
                &block,
                self.chain_spec.clone(),
                &execution_output,
                block_access_list_hash,
            )
        })?;

        // Verify the state root.
        //
        // SOUNDNESS: this final state-root check is load-bearing for every host-controlled value
        // that flows into the tries — not just the trie nodes (whose hashes are verified during
        // decode), but also, on the arena backend, the untrusted `pre_state_storage_roots` cache
        // consumed by `ArenaTries::update` (see the `SOUNDNESS:` docs on
        // `EthereumState::to_arena_witness` / `ArenaTries`). A tampered witness input surfaces
        // here as `MismatchedStateRoot` and no proof is produced.
        let state_root = profile_report!(COMPUTE_STATE_ROOT, {
            let (post_state, wiped_storages) = hashed_post_state(&execution_output.state);
            #[cfg(not(any(feature = "arena", feature = "flat")))]
            {
                for hashed_address in wiped_storages {
                    input.parent_state.clear_storage(hashed_address);
                }
                input.parent_state.update(&post_state);
                input.parent_state.state_root()
            }
            #[cfg(any(feature = "arena", feature = "flat"))]
            {
                for hashed_address in wiped_storages {
                    tries.clear_storage(hashed_address);
                }
                tries.update(&post_state);
                tries.state_root()
            }
        });

        if state_root != input.current_block.header().state_root() {
            return Err(ClientError::MismatchedStateRoot);
        }

        // Derive the block header.
        // Note: the receipts root, the gas used and (from Amsterdam on) the block access list
        // hash are verified by `validate_block_post_execution`.
        let header = Header {
            parent_hash: input.current_block.header().parent_hash(),
            ommers_hash: input.current_block.header().ommers_hash(),
            beneficiary: input.current_block.header().beneficiary(),
            state_root,
            transactions_root: input.current_block.header().transactions_root(),
            receipts_root: input.current_block.header().receipts_root(),
            logs_bloom: input.current_block.logs_bloom,
            difficulty: input.current_block.header().difficulty(),
            number: input.current_block.header().number(),
            gas_limit: input.current_block.header().gas_limit(),
            gas_used: input.current_block.header().gas_used(),
            timestamp: input.current_block.header().timestamp(),
            extra_data: input.current_block.header().extra_data().clone(),
            mix_hash: input.current_block.header().mix_hash().unwrap(),
            nonce: input.current_block.header().nonce().unwrap(),
            base_fee_per_gas: input.current_block.header().base_fee_per_gas(),
            withdrawals_root: input.current_block.header().withdrawals_root(),
            blob_gas_used: input.current_block.header().blob_gas_used(),
            excess_blob_gas: input.current_block.header().excess_blob_gas(),
            parent_beacon_block_root: input.current_block.header().parent_beacon_block_root(),
            requests_hash: input.current_block.header().requests_hash(),
            block_access_list_hash: input.current_block.header().block_access_list_hash(),
            slot_number: input.current_block.header().slot_number(),
        };

        Ok(header)
    }
}

impl EthClientExecutor {
    /// `custom_beneficiary` is for Clique chains, which this build does not support: it runs
    /// reth's stock Ethereum EVM configuration.
    pub fn eth(chain_spec: Arc<ChainSpec>, custom_beneficiary: Option<Address>) -> Self {
        assert!(custom_beneficiary.is_none(), "a custom beneficiary is not supported on evm2");
        install_crypto(CustomCrypto::default());

        Self { evm_config: EthEvmConfig::new(chain_spec.clone()), chain_spec }
    }

    /// Executes `input` under the configuration it names, and returns what a guest commits: the
    /// header it derives, with the digest of that configuration.
    ///
    /// The executor is built here, from the input's own `genesis` and `custom_beneficiary`, so
    /// the digest covers exactly what configured the run, and a guest that goes through this
    /// cannot leave the digest off. `genesis`, `custom_beneficiary` and `opcode_tracking` change
    /// execution and appear nowhere in the header; see [`CommittedHeader`].
    pub fn execute_committed(
        input: crate::io::EthClientExecutorInput,
    ) -> Result<CommittedHeader, ClientError> {
        let config_digest = input.config_digest()?;
        let chain_spec: ChainSpec = (&input.genesis).try_into()?;
        let executor = Self::eth(Arc::new(chain_spec), input.custom_beneficiary);
        let header = executor.execute(input)?;
        Ok(CommittedHeader::new(header, config_digest))
    }
}

/// What the client asks of the headers in its input, on blocks that need no state: empty blocks
/// of the Paris period over the empty state, which the legacy backend holds without a witness.
#[cfg(all(test, not(any(feature = "arena", feature = "flat"))))]
mod header_tests {
    use alloy_consensus::{constants::EMPTY_OMMER_ROOT_HASH, BlockBody};
    use alloy_primitives::Bytes;
    use reth_chainspec::EthChainSpec;
    use reth_errors::ConsensusError;
    use reth_ethereum_primitives::EthPrimitives;
    use reth_primitives_traits::SealedHeader;
    use reth_trie_common::EMPTY_ROOT_HASH;
    use rsp_mpt::EthereumState;
    use rsp_primitives::genesis::Genesis;

    use super::*;
    use crate::io::EthClientExecutorInput;

    fn mainnet() -> Arc<ChainSpec> {
        Arc::new((&Genesis::Mainnet).try_into().unwrap())
    }

    /// The header of an empty block over the empty state.
    fn first_header() -> Header {
        Header {
            ommers_hash: EMPTY_OMMER_ROOT_HASH,
            state_root: EMPTY_ROOT_HASH,
            transactions_root: EMPTY_ROOT_HASH,
            receipts_root: EMPTY_ROOT_HASH,
            number: 16_000_000,
            gas_limit: 30_000_000,
            // The Merge (2022-09-15) is behind it and Shanghai (2023-04-12) ahead: no block
            // reward, no withdrawals and no system call, so the state stays empty.
            timestamp: 1_670_000_000,
            base_fee_per_gas: Some(1_000_000_000),
            ..Default::default()
        }
    }

    /// The header of the empty block that follows `parent`.
    fn child_of(parent: &Header) -> Header {
        let timestamp = parent.timestamp + 12;
        Header {
            parent_hash: parent.hash_slow(),
            number: parent.number + 1,
            timestamp,
            base_fee_per_gas: mainnet().next_block_base_fee(parent, timestamp),
            ..parent.clone()
        }
    }

    /// Four headers, each the child of the one before it.
    fn chain() -> [Header; 4] {
        let first = first_header();
        let second = child_of(&first);
        let third = child_of(&second);
        let fourth = child_of(&third);
        [first, second, third, fourth]
    }

    /// Executes the empty block of `header` with `ancestors` in the input, the parent first.
    fn execute(header: Header, ancestors: Vec<Header>) -> Result<Header, ClientError> {
        let input = EthClientExecutorInput {
            current_block: alloy_consensus::Block {
                header,
                body: BlockBody { transactions: Vec::new(), ommers: Vec::new(), withdrawals: None },
            },
            ancestor_headers: ancestors,
            parent_state: EthereumState {
                state_trie: Default::default(),
                storage_tries: Default::default(),
            },
            bytecodes: Vec::new(),
            genesis: Genesis::Mainnet,
            custom_beneficiary: None,
            opcode_tracking: false,
        };
        EthClientExecutor::eth(mainnet(), None).execute(input)
    }

    #[test]
    fn a_block_runs_on_its_ancestors() {
        let [great_grandparent, grandparent, parent, block] = chain();

        let derived = execute(block.clone(), vec![parent, grandparent, great_grandparent]).unwrap();
        assert_eq!(derived.hash_slow(), block.hash_slow());
    }

    /// The block's header is held to its parent's.
    #[test]
    #[should_panic(expected = "The header is invalid against its parent")]
    fn a_block_that_does_not_follow_from_its_parent_is_refused() {
        let parent = first_header();
        let block = Header { base_fee_per_gas: Some(7), ..child_of(&parent) };

        let _ = execute(block, vec![parent]);
    }

    /// An ancestor older than the block is held to the hash its child names, and to nothing
    /// else. The parent here has a base fee that does not follow from its own parent's, which
    /// the validation of a header against its parent refuses, and the block runs on it.
    #[test]
    fn an_ancestor_is_not_validated_against_its_own_parent() {
        let grandparent = first_header();
        let parent = Header { base_fee_per_gas: Some(7), ..child_of(&grandparent) };
        let block = child_of(&parent);

        let refused = EthPrimitives::validate_header_against_parent(
            &SealedHeader::seal_slow(parent.clone()),
            &SealedHeader::seal_slow(grandparent.clone()),
            mainnet(),
        );
        assert!(matches!(refused, Err(ConsensusError::BaseFeeDiff(_))), "{refused:?}");

        let derived = execute(block.clone(), vec![parent, grandparent]).unwrap();
        assert_eq!(derived.hash_slow(), block.hash_slow());
    }

    /// What holds the older ancestors: each is the header whose hash its child names.
    #[test]
    #[should_panic(expected = "InvalidHeaderParentHash")]
    fn an_ancestor_its_child_does_not_name_is_refused() {
        let [_, grandparent, parent, block] = chain();
        let another = Header { extra_data: Bytes::from_static(b"another"), ..grandparent };

        let _ = execute(block, vec![parent, another]);
    }

    #[test]
    #[should_panic(expected = "InvalidHeaderBlockNumber")]
    fn an_ancestor_out_of_sequence_is_refused() {
        let [great_grandparent, _, parent, block] = chain();

        let _ = execute(block, vec![parent, great_grandparent]);
    }

    #[test]
    #[should_panic(expected = "The parent header is missing")]
    fn a_block_without_its_parent_is_refused() {
        let [.., block] = chain();

        let _ = execute(block, Vec::new());
    }
}
