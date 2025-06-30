use alloy_consensus::{BlockHeader, Header, TxReceipt};
use alloy_evm::EthEvmFactory;
use alloy_primitives::{Bloom, Sealable};
use alloy_provider::{Network, Provider};
use alloy_rpc_types::EIP1186AccountProofResponse;
use futures::{stream, StreamExt};
use reth_chainspec::ChainSpec;
use reth_evm::{
    execute::{BasicBlockExecutor, Executor},
    ConfigureEvm,
};
use reth_evm_ethereum::EthEvmConfig;
use reth_execution_types::ExecutionOutcome;
use reth_optimism_chainspec::OpChainSpec;
use reth_optimism_evm::OpEvmConfig;
use reth_primitives_traits::{Block, BlockBody};
use reth_trie::KeccakKeyHasher;
use revm::database::CacheDB;
use revm_primitives::{Address, B256};
use rsp_client_executor::{
    custom::CustomEvmFactory, io::ClientExecutorInput, BlockValidator, IntoInput, IntoPrimitives,
};
use rsp_mpt::EthereumState;
use rsp_primitives::{account_proof::eip1186_proof_to_account_proof, genesis::Genesis};
use rsp_rpc_db::RpcDb;
use std::collections::HashSet;
use std::{collections::BTreeSet, sync::Arc, time::Instant};
use tokio::try_join;

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
        rpc_db: &RpcDb<P, N>,
        provider: &P,
        genesis: Genesis,
        custom_beneficiary: Option<Address>,
        opcode_tracking: bool,
    ) -> Result<ClientExecutorInput<C::Primitives>, HostError>
    where
        C::Primitives: IntoPrimitives<N> + IntoInput + BlockValidator<CS>,
        P: Provider<N> + Clone,
        N: Network,
    {
        let fetch_start = Instant::now();
        // Fetch the current block and the previous block from the provider.
        tracing::info!("fetching the current block and the previous block");

        let t_fetch_blocks = Instant::now();
        let (current_block, previous_block) = try_join!(
            async {
                provider
                    .get_block_by_number(block_number.into())
                    .full()
                    .await?
                    .ok_or(HostError::ExpectedBlock(block_number))
                    .map(C::Primitives::into_primitive_block)
            },
            async {
                provider
                    .get_block_by_number((block_number - 1).into())
                    .full()
                    .await?
                    .ok_or(HostError::ExpectedBlock(block_number - 1))
                    .map(C::Primitives::into_primitive_block)
            }
        )?;
        tracing::info!("fetch_blocks: {:?}", t_fetch_blocks.elapsed());

        let t_db_setup = Instant::now();
        // Setup the database for the block executor.
        tracing::info!("setting up the database for the block executor");
        let cache_db = CacheDB::new(rpc_db);

        let block_executor = BasicBlockExecutor::new(self.evm_config.clone(), cache_db);
        tracing::info!("db_setup: {:?}", t_db_setup.elapsed());

        let t_try_recover = Instant::now();
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
        tracing::info!("try_recover: {:?}", t_try_recover.elapsed());


        let t_validate_header = Instant::now();
        // Validate the block header.
        C::Primitives::validate_header(
            block.sealed_block().sealed_header(),
            self.chain_spec.clone(),
        )?;
        tracing::info!("validate_header: {:?}", t_validate_header.elapsed());


        let t_execute = Instant::now();
        let execution_output = block_executor.execute(&block)?;
        tracing::info!("evm_execute: {:?}", t_execute.elapsed());


        let t_validate_post = Instant::now();
        // Validate the block post execution.
        tracing::info!("validating the block post execution");
        C::Primitives::validate_block_post_execution(
            &block,
            self.chain_spec.clone(),
            &execution_output,
        )?;
        tracing::info!("validate_post: {:?}", t_validate_post.elapsed());


        let t_accumulate_bloom = Instant::now();
        // Accumulate the logs bloom.
        tracing::info!("accumulating the logs bloom");
        let mut logs_bloom = Bloom::default();
        execution_output.result.receipts.iter().for_each(|r| {
            logs_bloom.accrue_bloom(&r.bloom());
        });
        tracing::info!("accumulate_bloom: {:?}", t_accumulate_bloom.elapsed());


        let t_fetch_proofs = Instant::now();
        // Convert the output to an execution outcome.
        let executor_outcome = ExecutionOutcome::new(
            execution_output.state,
            vec![execution_output.result.receipts],
            current_block.header().number(),
            vec![execution_output.result.requests],
        );

        let state_requests = rpc_db.get_state_requests();

        // For every account we touched, fetch the storage proofs for all the slots we touched.
        tracing::info!("fetching storage proofs");
        let mut before_storage_proofs = Vec::new();
        let mut after_storage_proofs = Vec::new();
        // TODO: unordered?
        {
            let proof_stream = stream::iter(state_requests.iter())
                .map(|(address, used_keys)| {
                    let modified_keys: Vec<B256> = executor_outcome
                        .state()
                        .state
                        .get(address)
                        .map(|acct| acct.storage.keys().map(|k| B256::from(*k)).collect::<Vec<_>>())
                        .unwrap_or_default();

                    let used_hs: HashSet<B256> = used_keys.iter().map(|k| B256::from(*k)).collect();

                    let provider_cloned = provider.clone();
                    let addr = *address;
                    let bn = block_number;

                    async move {
                        fetch_account_proofs(provider_cloned, addr, &used_hs, modified_keys, bn)
                            .await
                    }
                })
                .buffer_unordered(32);

            futures::pin_mut!(proof_stream);

            while let Some(res) = proof_stream.next().await {
                let (before, after_opt) = res?;
                before_storage_proofs.push(before.clone());
                match after_opt {
                    Some(after) => after_storage_proofs.push(after),
                    None => after_storage_proofs.push(before),
                }
            }
        }

        // for (address, used_keys) in state_requests.iter() {
        //     let modified_keys = executor_outcome
        //         .state()
        //         .state
        //         .get(address)
        //         .map(|account| {
        //             account.storage.keys().map(|key| B256::from(*key)).collect::<BTreeSet<_>>()
        //         })
        //         .unwrap_or_default()
        //         .into_iter()
        //         .collect::<Vec<_>>();
        //
        //     let keys = used_keys
        //         .iter()
        //         .map(|key| B256::from(*key))
        //         .chain(modified_keys.clone().into_iter())
        //         .collect::<BTreeSet<_>>()
        //         .into_iter()
        //         .collect::<Vec<_>>();
        //
        //     let storage_proof = provider
        //         .get_proof(*address, keys.clone())
        //         .block_id((block_number - 1).into())
        //         .await?;
        //     before_storage_proofs.push(eip1186_proof_to_account_proof(storage_proof));
        //
        //     let storage_proof =
        //         provider.get_proof(*address, modified_keys).block_id((block_number).into()).await?;
        //     after_storage_proofs.push(eip1186_proof_to_account_proof(storage_proof));
        // }

        tracing::info!("fetch_proofs: {:?}", t_fetch_proofs.elapsed());

        let t_build_state = Instant::now();
        let state = EthereumState::from_transition_proofs(
            previous_block.header().state_root(),
            &before_storage_proofs.iter().map(|item| (item.address, item.clone())).collect(),
            &after_storage_proofs.iter().map(|item| (item.address, item.clone())).collect(),
        )?;

        // Verify the state root.
        tracing::info!("verifying the state root");
        let state_root = {
            let mut mutated_state = state.clone();
            mutated_state.update(&executor_outcome.hash_state_slow::<KeccakKeyHasher>());
            mutated_state.state_root()
        };
        if state_root != current_block.header().state_root() {
            return Err(HostError::StateRootMismatch(
                state_root,
                current_block.header().state_root(),
            ));
        }
        tracing::info!("build_state: {:?}", t_build_state.elapsed());

        let t_build_header = Instant::now();
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
        tracing::info!("build_header: {:?}", t_build_header.elapsed());

        let t_fetch_ancestor = Instant::now();
        // Fetch the parent headers needed to constrain the BLOCKHASH opcode.
        let oldest_ancestor = *rpc_db.oldest_ancestor.read().unwrap();
        let mut ancestor_headers = vec![];
        tracing::info!("fetching {} ancestor headers", block_number - oldest_ancestor);
        for height in (oldest_ancestor..=(block_number - 1)).rev() {
            let block = provider
                .get_block_by_number(height.into())
                .await?
                .ok_or(HostError::ExpectedBlock(height))?;

            ancestor_headers.push(C::Primitives::into_primitive_header(block))
        }
        tracing::info!("fetch_ancestor_headers: {:?}", t_fetch_ancestor.elapsed());


        let t_assemble_input = Instant::now();
        // Create the client input.
        let client_input = ClientExecutorInput {
            current_block: C::Primitives::into_input_block(current_block),
            ancestor_headers,
            parent_state: state,
            state_requests,
            bytecodes: rpc_db.get_bytecodes(),
            genesis,
            custom_beneficiary,
            opcode_tracking,
        };
        tracing::info!("successfully generated client input");
        tracing::info!("assemble_input: {:?}", t_assemble_input.elapsed());

        let fetch_duration = fetch_start.elapsed();
        tracing::info!("fetch client input cost: {:?}", fetch_duration);
        Ok(client_input)
    }
}

type AccountProof = rsp_primitives::account_proof::AccountProof;

async fn fetch_account_proofs<P, N>(
    provider: P,
    address: Address,
    used_keys: &HashSet<B256>,
    modified_keys: Vec<B256>,
    block_number: u64,
) -> Result<(AccountProof, Option<AccountProof>), HostError>
where
    P: Provider<N> + Clone + Send + Sync,
    N: Network,
{
    // --------- 1. BEFORE  (height-1) ---------
    let before_keys: Vec<_> = used_keys
        .iter()
        .cloned()
        .chain(modified_keys.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let before_fut = provider.get_proof(address, before_keys).block_id((block_number - 1).into());

    // --------- 2. AFTER  (height) -------------
    if modified_keys.is_empty() {
        let before = before_fut.await.map_err(|e| HostError::Provider(e))?;
        return Ok((eip1186_proof_to_account_proof(before), None));
    }

    let after_fut =
        provider.get_proof(address, modified_keys.clone()).block_id(block_number.into());

    let (before, after): (EIP1186AccountProofResponse, EIP1186AccountProofResponse) =
        try_join!(before_fut, after_fut).map_err(|e| HostError::Provider(e))?;

    Ok((eip1186_proof_to_account_proof(before), Some(eip1186_proof_to_account_proof(after))))
}
