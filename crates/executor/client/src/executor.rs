use crate::{
    custom::{CustomCrypto, CustomEvmFactory},
    error::ClientError,
    into_primitives::FromInput,
    io::{ClientExecutorInput, CommittedHeader, TrieDB, WitnessInput},
    tracking::OpCodesTrackingBlockExecutor,
    BlockValidator,
};
use alloy_consensus::{BlockHeader, Header};
use itertools::Itertools;
use reth_chainspec::ChainSpec;
use reth_errors::BlockExecutionError;
use reth_evm::{
    execute::{BasicBlockExecutor, Executor},
    ConfigureEvm, OnStateHook,
};
use reth_evm_ethereum::EthEvmConfig;
use reth_execution_types::ExecutionOutcome;
use reth_primitives_traits::Block;
use reth_trie::KeccakKeyHasher;
use revm::{database::WrapDatabaseRef, install_crypto};
use revm_primitives::Address;
use rsp_primitives::genesis::Genesis;
use std::sync::Arc;

pub const DESERIALZE_INPUTS: &str = "deserialize inputs";
pub const INIT_WITNESS_DB: &str = "initialize witness db";
pub const RECOVER_SENDERS: &str = "recover senders";
pub const BLOCK_EXECUTION: &str = "block execution";
pub const VALIDATE_HEADER: &str = "validate header";
pub const VALIDATE_EXECUTION: &str = "validate block post-execution";
pub const COMPUTE_STATE_ROOT: &str = "compute state root";

pub type EthClientExecutor = ClientExecutor<EthEvmConfig<ChainSpec, CustomEvmFactory>, ChainSpec>;

#[cfg(feature = "optimism")]
pub type OpClientExecutor =
    ClientExecutor<reth_optimism_evm::OpEvmConfig, reth_optimism_chainspec::OpChainSpec>;

/// An executor that executes a block inside a zkVM.
///
/// `genesis` and `custom_beneficiary` are kept here, not read back off the input, because they
/// are what *built* `evm_config` and `chain_spec`. The committed digest has to name the
/// configuration that ran; taking it from the input instead would let the two disagree, and a
/// digest attesting to a configuration the block did not execute under is worse than none.
#[derive(Debug, Clone)]
pub struct ClientExecutor<C: ConfigureEvm, CS> {
    evm_config: C,
    chain_spec: Arc<CS>,
    genesis: Genesis,
    custom_beneficiary: Option<Address>,
}

impl<C, CS> ClientExecutor<C, CS>
where
    C: ConfigureEvm,
    C::Primitives: FromInput + BlockValidator<CS>,
{
    /// Executes the block and returns the value the guest commits.
    ///
    /// Returns a [`CommittedHeader`] rather than a bare `Header` so a caller cannot leave the
    /// configuration digest off: `genesis`, `custom_beneficiary` and `opcode_tracking` change
    /// execution and appear nowhere in the header.
    pub fn execute(
        &self,
        input: ClientExecutorInput<'_, C::Primitives>,
    ) -> Result<CommittedHeader, ClientError> {
        // Digest what configured *this executor*, and refuse an input that asks for anything
        // else. In the guest both come from the same `ClientExecutorInput`, so this can only
        // fire on a caller that built the executor from one configuration and handed it a
        // witness naming another -- which the type system otherwise permits.
        if input.genesis != self.genesis || input.custom_beneficiary != self.custom_beneficiary {
            return Err(ClientError::MismatchedConfig);
        }
        let config_digest =
            crate::io::config_digest(&self.genesis, &self.custom_beneficiary, input.opcode_tracking)?;
        let sealed_headers = input.sealed_headers().collect::<Vec<_>>();

        // Every fallible step from here on propagates rather than panicking. `verified_views`
        // returns `Err(MismatchedStateRoot)` when the witness does not hash to the parent
        // header's root -- the anchor the whole trust chain hangs from -- and an abort there
        // fails closed but says nothing. `?` outside `profile_report!` so the zkvm arm still
        // prints its end marker on the error path.
        let (views, accounts, block_hashes, bytecodes_by_hash) =
            profile_report!(INIT_WITNESS_DB, {
                let (views, accounts) = input.verified_views()?;
                let (block_hashes, bytecodes_by_hash) = input.witness_aux(&sealed_headers)?;
                Ok::<_, ClientError>((views, accounts, block_hashes, bytecodes_by_hash))
            })?;
        let db = WrapDatabaseRef(TrieDB::new(&views, accounts, block_hashes, bytecodes_by_hash));

        let block_executor = BlockExecutor::new(self.evm_config.clone(), db, input.opcode_tracking);

        let block = profile_report!(RECOVER_SENDERS, {
            C::Primitives::from_input_block(input.current_block.clone())
                .try_into_recovered()
                .map_err(|_| ClientError::SignatureRecoveryFailed)
        })?;

        // Consensus rejections are the expected answer for a block the prover made up, so they
        // travel out as `ClientError::PostExecutionError`, as `validate_block_post_execution`
        // below already did.
        profile_report!(VALIDATE_HEADER, {
            C::Primitives::validate_block(&block, self.chain_spec.clone())?;

            for (header, parent) in sealed_headers.iter().tuple_windows() {
                C::Primitives::validate_header(parent, self.chain_spec.clone())?;

                C::Primitives::validate_header_against_parent(
                    header,
                    parent,
                    self.chain_spec.clone(),
                )?;
            }
            Ok::<_, ClientError>(())
        })?;

        let execution_output =
            profile_report!(BLOCK_EXECUTION, { block_executor.execute(&block) })?;

        // Validate the block post execution.
        profile_report!(VALIDATE_EXECUTION, {
            C::Primitives::validate_block_post_execution(
                &block,
                self.chain_spec.clone(),
                &execution_output,
            )
        })?;

        // Convert the output to an execution outcome.
        let executor_outcome = ExecutionOutcome::new(
            execution_output.state,
            vec![execution_output.result.receipts],
            input.current_block.header().number(),
            vec![execution_output.result.requests],
        );

        // One batched bottom-up delta pass over the verified blobs. Fallible on
        // attacker-controlled input: `post_state_root` rejects a witness that omits a modified
        // account's storage trie.
        let state_root = profile_report!(COMPUTE_STATE_ROOT, {
            let hashed_state = executor_outcome.hash_state_slow::<KeccakKeyHasher>();
            views.post_state_root(&hashed_state)
        })?;

        if state_root != input.current_block.header().state_root() {
            return Err(ClientError::MismatchedStateRoot);
        }

        // Derive the block header.
        // Note: the receipts root and gas used are verified by `validate_block_post_execution`.
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
        };

        Ok(CommittedHeader::new(header, config_digest))
    }
}

impl EthClientExecutor {
    /// Builds the executor from the `Genesis` the witness carries, deriving the `ChainSpec`
    /// here rather than taking one.
    ///
    /// Taking a prebuilt spec let a caller pass one that disagreed with the `genesis` the
    /// committed digest names. Deriving it is what makes "the digest describes the run" true
    /// by construction rather than by convention.
    pub fn eth(
        genesis: &Genesis,
        custom_beneficiary: Option<Address>,
    ) -> Result<Self, ClientError> {
        install_crypto(CustomCrypto::default());

        let chain_spec: Arc<ChainSpec> = Arc::new(genesis.try_into()?);

        Ok(Self {
            evm_config: EthEvmConfig::new_with_evm_factory(
                chain_spec.clone(),
                CustomEvmFactory::new(custom_beneficiary),
            ),
            chain_spec,
            genesis: genesis.clone(),
            custom_beneficiary,
        })
    }
}

#[cfg(feature = "optimism")]
impl OpClientExecutor {
    /// As [`EthClientExecutor::eth`]: the spec is derived here so it cannot disagree with the
    /// `genesis` the committed digest names. OP has no `custom_beneficiary`.
    pub fn optimism(genesis: &Genesis) -> Result<Self, ClientError> {
        install_crypto(CustomCrypto::default());

        let chain_spec: Arc<reth_optimism_chainspec::OpChainSpec> = Arc::new(genesis.try_into()?);

        Ok(Self {
            evm_config: reth_optimism_evm::OpEvmConfig::optimism(chain_spec.clone()),
            chain_spec,
            genesis: genesis.clone(),
            custom_beneficiary: None,
        })
    }
}

enum BlockExecutor<'a, C> {
    Basic(BasicBlockExecutor<C, WrapDatabaseRef<TrieDB<'a>>>),
    OpcodeTracking(OpCodesTrackingBlockExecutor<C, WrapDatabaseRef<TrieDB<'a>>>),
}

impl<'a, C: ConfigureEvm> BlockExecutor<'a, C> {
    fn new(strategy_factory: C, db: WrapDatabaseRef<TrieDB<'a>>, opcode_tracking: bool) -> Self {
        if opcode_tracking {
            Self::OpcodeTracking(OpCodesTrackingBlockExecutor::new(strategy_factory, db))
        } else {
            Self::Basic(BasicBlockExecutor::new(strategy_factory, db))
        }
    }
}

impl<'a, C> Executor<WrapDatabaseRef<TrieDB<'a>>> for BlockExecutor<'a, C>
where
    C: ConfigureEvm,
{
    type Primitives = C::Primitives;
    type Error = BlockExecutionError;

    fn execute_one(
        &mut self,
        block: &reth_primitives_traits::RecoveredBlock<
            <Self::Primitives as reth_primitives_traits::NodePrimitives>::Block,
        >,
    ) -> Result<
        reth_execution_types::BlockExecutionResult<
            <Self::Primitives as reth_primitives_traits::NodePrimitives>::Receipt,
        >,
        Self::Error,
    > {
        match self {
            BlockExecutor::Basic(basic_block_executor) => basic_block_executor.execute_one(block),
            BlockExecutor::OpcodeTracking(op_codes_tracking_block_executor) => {
                op_codes_tracking_block_executor.execute_one(block)
            }
        }
    }

    fn execute_one_with_state_hook<H>(
        &mut self,
        block: &reth_primitives_traits::RecoveredBlock<
            <Self::Primitives as reth_primitives_traits::NodePrimitives>::Block,
        >,
        state_hook: H,
    ) -> Result<
        reth_execution_types::BlockExecutionResult<
            <Self::Primitives as reth_primitives_traits::NodePrimitives>::Receipt,
        >,
        Self::Error,
    >
    where
        H: OnStateHook + 'static,
    {
        match self {
            BlockExecutor::Basic(basic_block_executor) => {
                basic_block_executor.execute_one_with_state_hook(block, state_hook)
            }
            BlockExecutor::OpcodeTracking(op_codes_tracking_block_executor) => {
                op_codes_tracking_block_executor.execute_one_with_state_hook(block, state_hook)
            }
        }
    }

    fn into_state(self) -> revm::database::State<WrapDatabaseRef<TrieDB<'a>>> {
        match self {
            BlockExecutor::Basic(basic_block_executor) => basic_block_executor.into_state(),
            BlockExecutor::OpcodeTracking(op_codes_tracking_block_executor) => {
                op_codes_tracking_block_executor.into_state()
            }
        }
    }

    fn size_hint(&self) -> usize {
        match self {
            BlockExecutor::Basic(basic_block_executor) => basic_block_executor.size_hint(),
            BlockExecutor::OpcodeTracking(op_codes_tracking_block_executor) => {
                op_codes_tracking_block_executor.size_hint()
            }
        }
    }
}
