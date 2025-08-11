use std::{sync::Arc, time::Instant};

use alloy_consensus::{BlockHeader, Header, TxReceipt};
use alloy_evm::EthEvmFactory;
use alloy_primitives::{Bloom, Sealable};
use alloy_provider::{ext::DebugApi, Network, Provider};
use alloy_rlp::Decodable;
use alloy_rpc_types::BlockNumberOrTag;
use reth_chainspec::ChainSpec;
use reth_evm::{
    execute::{BasicBlockExecutor, Executor},
    ConfigureEvm,
};
use reth_evm_ethereum::EthEvmConfig;
use reth_optimism_chainspec::OpChainSpec;
use reth_optimism_evm::OpEvmConfig;
use reth_primitives_traits::{Block, BlockBody};
use reth_trie::HashedPostState;
use reth_trie::KeccakKeyHasher;
use revm::database::CacheDB;
use revm_primitives::keccak256;
use revm_primitives::Address;
use revm_state::Bytecode;
use rsp_client_executor::{
    custom::CustomEvmFactory, io::ClientExecutorInput, BlockValidator, IntoInput, IntoPrimitives,
};
use rsp_primitives::genesis::Genesis;
use rsp_rpc_db::{ExecutionWitnessRpcDb, RpcDb};

use crate::execution_witness::eth_state_from_execution_witness;
use crate::HostError;

pub type EthHostExecutor = HostExecutor<EthEvmConfig<CustomEvmFactory<EthEvmFactory>>, ChainSpec>;

pub type OpHostExecutor = HostExecutor<OpEvmConfig, OpChainSpec>;

/// An executor that fetches data from a [Provider] to execute blocks in the [ClientExecutor].
#[derive(Debug, Clone)]
pub struct HostExecutor<C: ConfigureEvm, CS> {
    evm_config: C,
    chain_spec: Arc<CS>,
}

impl EthHostExecutor {
    pub fn eth(chain_spec: Arc<ChainSpec>, custom_beneficiary: Option<Address>) -> Self {
        Self {
            evm_config: EthEvmConfig::new_with_evm_factory(
                chain_spec.clone(),
                CustomEvmFactory::<EthEvmFactory>::new(custom_beneficiary),
            ),
            chain_spec,
        }
    }
}

impl OpHostExecutor {
    pub fn optimism(chain_spec: Arc<OpChainSpec>) -> Self {
        Self { evm_config: OpEvmConfig::optimism(chain_spec.clone()), chain_spec }
    }
}

impl<C: ConfigureEvm, CS> HostExecutor<C, CS> {
    /// Creates a new [HostExecutor].
    pub fn new(evm_config: C, chain_spec: Arc<CS>) -> Self {
        Self { evm_config, chain_spec }
    }

    /// Executes the block with the given block number.
    pub async fn execute<P, N>(
        &self,
        block_number: u64,
        provider: &P,
        genesis: Genesis,
        custom_beneficiary: Option<Address>,
        opcode_tracking: bool,
    ) -> Result<ClientExecutorInput<C::Primitives>, HostError>
    where
        C::Primitives: IntoPrimitives<N> + IntoInput + BlockValidator<CS>,
        P: DebugApi<N> + Provider<N> + Clone,
        N: Network,
    {
        let fetch_start = Instant::now();
        // Fetch the current block and the previous block from the provider.
        tracing::info!("fetching the current block and the previous block");
        let current_block = provider
            .get_block_by_number(block_number.into())
            .full()
            .await?
            .ok_or(HostError::ExpectedBlock(block_number))
            .map(C::Primitives::into_primitive_block)?;

        let previous_block = provider
            .get_block_by_number((block_number - 1).into())
            .full()
            .await?
            .ok_or(HostError::ExpectedBlock(block_number))
            .map(C::Primitives::into_primitive_block)?;

        // Setup the database for the block executor.
        tracing::info!("setting up the database for the block executor");
        let execution_witness =
            provider.debug_execution_witness(BlockNumberOrTag::Number(block_number)).await?;
        let state = eth_state_from_execution_witness(
            &execution_witness,
            previous_block.header().state_root(),
        );
        let codes = execution_witness
            .codes
            .iter()
            .map(|encoded| (keccak256(encoded), Bytecode::new_raw(encoded.clone())))
            .collect();
        let ancestor_headers = execution_witness
            .headers
            .iter()
            .map(|encoded| Header::decode(&mut encoded.as_ref()).unwrap())
            .map(|h| (h.number, h))
            .collect();
        let rpc_db = ExecutionWitnessRpcDb { state, codes, ancestor_headers };
        let cache_db = CacheDB::new(&rpc_db);

        let block_executor = BasicBlockExecutor::new(self.evm_config.clone(), cache_db);

        // Execute the block and fetch all the necessary data along the way.
        tracing::info!(
            "executing the block with rpc db: block_number={}, transaction_count={}",
            block_number,
            current_block.body().transactions().len()
        );

        let block = current_block
            .clone()
            .try_into_recovered()
            .map_err(|_| HostError::FailedToRecoverSenders)
            .unwrap();

        // Validate the block header.
        C::Primitives::validate_header(
            block.sealed_block().sealed_header(),
            self.chain_spec.clone(),
        )?;

        let execution_output = block_executor.execute(&block)?;

        // Validate the block post execution.
        tracing::info!("validating the block post execution");
        C::Primitives::validate_block_post_execution(
            &block,
            self.chain_spec.clone(),
            &execution_output,
        )?;

        // Accumulate the logs bloom.
        tracing::info!("accumulating the logs bloom");
        let mut logs_bloom = Bloom::default();
        execution_output.result.receipts.iter().for_each(|r| {
            logs_bloom.accrue_bloom(&r.bloom());
        });

        let state = rpc_db.state(&execution_output.state).unwrap();

        // Verify the state root.
        tracing::info!("verifying the state root");
        let state_root = {
            let mut mutated_state = state.clone();
            mutated_state.update(&HashedPostState::from_bundle_state::<KeccakKeyHasher>(
                &execution_output.state.state,
            ));
            mutated_state.state_root()
        };
        if state_root != current_block.header().state_root() {
            return Err(HostError::StateRootMismatch(
                state_root,
                current_block.header().state_root(),
            ));
        }

        // Derive the block header.
        //
        // Note: the receipts root and gas used are verified by `validate_block_post_execution`.
        let header = Header {
            parent_hash: current_block.header().parent_hash(),
            ommers_hash: current_block.header().ommers_hash(),
            beneficiary: current_block.header().beneficiary(),
            state_root,
            transactions_root: current_block.header().transactions_root(),
            receipts_root: current_block.header().receipts_root(),
            logs_bloom,
            difficulty: current_block.header().difficulty(),
            number: current_block.header().number(),
            gas_limit: current_block.header().gas_limit(),
            gas_used: current_block.header().gas_used(),
            timestamp: current_block.header().timestamp(),
            extra_data: current_block.header().extra_data().clone(),
            mix_hash: current_block.header().mix_hash().unwrap(),
            nonce: current_block.header().nonce().unwrap(),
            base_fee_per_gas: current_block.header().base_fee_per_gas(),
            withdrawals_root: current_block.header().withdrawals_root(),
            blob_gas_used: current_block.header().blob_gas_used(),
            excess_blob_gas: current_block.header().excess_blob_gas(),
            parent_beacon_block_root: current_block.header().parent_beacon_block_root(),
            requests_hash: current_block.header().requests_hash(),
        };

        let ancestor_headers = rpc_db.ancestor_headers().unwrap();

        // Assert the derived header is correct.
        let constructed_header_hash = header.hash_slow();
        let target_hash = current_block.header().hash_slow();
        if constructed_header_hash != target_hash {
            return Err(HostError::HeaderMismatch(constructed_header_hash, target_hash));
        }

        // Log the result.
        tracing::info!(
            "successfully executed block: block_number={}, block_hash={}, state_root={}",
            current_block.header().number(),
            constructed_header_hash,
            state_root
        );

        // Create the client input.
        let client_input = ClientExecutorInput {
            current_block: C::Primitives::into_input_block(current_block),
            ancestor_headers,
            parent_state: state,
            bytecodes: rpc_db.bytecodes(),
            genesis,
            custom_beneficiary,
            opcode_tracking,
        };
        tracing::info!("successfully generated client input");
        let fetch_duration = fetch_start.elapsed();
        tracing::info!("fetch client input cost: {:?}", fetch_duration);
        Ok(client_input)
    }
}
