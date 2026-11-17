use std::sync::Arc;

use alloy_consensus::{BlockHeader, Header};
use alloy_eip7928::{compute_block_access_list_hash, total_bal_items, ITEM_COST};
use alloy_primitives::{keccak256, Address, B256};
use evm2::{
    evm::{DbRef, DynDatabase},
    install_crypto,
};
use itertools::Itertools;
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
    io::{ClientExecutorInput, WitnessInput},
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

        // Validate the blocks.
        profile_report!(VALIDATE_HEADER, {
            C::Primitives::validate_block(&block, self.chain_spec.clone())
                .expect("The block is invalid");

            for (header, parent) in sealed_headers.iter().tuple_windows() {
                C::Primitives::validate_header(parent, self.chain_spec.clone())
                    .expect("A parent header is invalid");

                C::Primitives::validate_header_against_parent(
                    header,
                    parent,
                    self.chain_spec.clone(),
                )
                .expect("The header is invalid against its parent");
            }
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
}
