use std::sync::Arc;

use alloy_consensus::{Block, Header, TxEnvelope};
use alloy_network::{Ethereum, Network};
use alloy_primitives::B256;
use reth_chainspec::{ChainSpec, EthChainSpec, EthereumHardforks, NamedChain};
use reth_consensus::HeaderValidator;
use reth_consensus_common::validation::validate_block_pre_execution;
use reth_errors::ConsensusError;
use reth_ethereum_consensus::EthBeaconConsensus;
use reth_ethereum_primitives::EthPrimitives;
use reth_execution_types::BlockExecutionOutput;
use reth_primitives_traits::{NodePrimitives, RecoveredBlock, SealedHeader};

use crate::receipts::receipts_root_and_logs_bloom;

pub trait IntoPrimitives<N: Network>: NodePrimitives {
    fn into_primitive_block(block: N::BlockResponse) -> Self::Block;

    fn into_consensus_header(header: N::HeaderResponse) -> Header;
}

pub trait FromInput: NodePrimitives {
    fn from_input_block(block: Block<Self::SignedTx>) -> Self::Block;
}

pub trait IntoInput: NodePrimitives {
    fn into_input_block(block: Self::Block) -> Block<Self::SignedTx>;
}

pub trait BlockValidator<CS>: NodePrimitives {
    fn validate_header(header: &SealedHeader, chain_spec: Arc<CS>) -> Result<(), ConsensusError>;

    fn validate_block(
        block: &RecoveredBlock<Self::Block>,
        chain_spec: Arc<CS>,
    ) -> Result<(), ConsensusError>;

    fn validate_header_against_parent(
        header: &SealedHeader,
        parent: &SealedHeader,
        chain_spec: Arc<CS>,
    ) -> Result<(), ConsensusError>;

    /// `block_access_list_hash` is the hash of the access list that executing the block
    /// produced (EIP-7928); `None` before Amsterdam.
    fn validate_block_post_execution(
        block: &RecoveredBlock<Self::Block>,
        chain_spec: Arc<CS>,
        execution_output: &BlockExecutionOutput<Self::Receipt>,
        block_access_list_hash: Option<B256>,
    ) -> Result<(), ConsensusError>;
}

impl IntoPrimitives<Ethereum> for EthPrimitives {
    fn into_primitive_block(block: alloy_rpc_types::Block) -> Self::Block {
        let block = block.map_transactions(|tx| TxEnvelope::from(tx).into());
        block.into_consensus()
    }

    fn into_consensus_header(header: alloy_rpc_types::Header) -> Header {
        header.into()
    }
}

impl FromInput for EthPrimitives {
    fn from_input_block(block: Block<Self::SignedTx>) -> Self::Block {
        block
    }
}

impl IntoInput for EthPrimitives {
    fn into_input_block(block: Self::Block) -> Block<Self::SignedTx> {
        block
    }
}

impl BlockValidator<ChainSpec> for EthPrimitives {
    fn validate_header(
        header: &SealedHeader,
        chain_spec: Arc<ChainSpec>,
    ) -> Result<(), ConsensusError> {
        let validator = EthBeaconConsensus::new(chain_spec.clone());

        handle_custom_chains(validator.validate_header(header), chain_spec)
    }

    fn validate_block(
        recovered: &RecoveredBlock<Self::Block>,
        chain_spec: Arc<ChainSpec>,
    ) -> Result<(), ConsensusError> {
        Self::validate_header(recovered.sealed_header(), chain_spec.clone())?;

        // What a block has to satisfy before it is executed, as reth checks it when it imports
        // one. Comparing the body with the header's roots (`validate_body_against_header`) is
        // only part of that: this also holds the header's `blob_gas_used` to the blob gas of
        // the transactions (Cancun) and the block to the size limit (Osaka). Neither is
        // looked at again later -- the header the client commits takes `blob_gas_used` from
        // the input -- so without it a header that disagrees with its body is proved as it is.
        validate_block_pre_execution(recovered.sealed_block(), chain_spec.as_ref())?;

        Ok(())
    }

    fn validate_header_against_parent(
        header: &SealedHeader,
        parent: &SealedHeader,
        chain_spec: Arc<ChainSpec>,
    ) -> Result<(), ConsensusError> {
        let validator = EthBeaconConsensus::new(chain_spec);

        validator.validate_header_against_parent(header, parent)
    }

    fn validate_block_post_execution(
        block: &RecoveredBlock<Self::Block>,
        chain_spec: Arc<ChainSpec>,
        execution_output: &BlockExecutionOutput<Self::Receipt>,
        block_access_list_hash: Option<B256>,
    ) -> Result<(), ConsensusError> {
        // The receipts root and the logs bloom by `crate::receipts`. reth compares them with
        // the header's, where it would otherwise compute its own through alloy's encoder.
        // Before Byzantium it looks at neither (EIP-658).
        let receipt_root_bloom = chain_spec
            .is_byzantium_active_at_block(block.header().number)
            .then(|| receipts_root_and_logs_bloom(&execution_output.result.receipts));
        reth_ethereum_consensus::validate_block_post_execution(
            block,
            &chain_spec,
            &execution_output.result,
            receipt_root_bloom,
            block_access_list_hash,
        )
    }
}

fn handle_custom_chains(
    result: Result<(), ConsensusError>,
    chain_spec: Arc<ChainSpec>,
) -> Result<(), ConsensusError> {
    let err = if let Err(err) = result { err } else { return Ok(()) };

    let chain = if let Ok(chain) = NamedChain::try_from(chain_spec.chain_id()) {
        chain
    } else {
        return Err(err);
    };

    match chain {
        NamedChain::Linea | NamedChain::LineaSepolia | NamedChain::LineaGoerli => {
            // Skip extra data and Merge difficulty checks for Linea chains
            if matches!(
                err,
                ConsensusError::ExtraDataExceedsMax { .. }
                    | ConsensusError::TheMergeDifficultyIsNotZero
            ) {
                Ok(())
            } else {
                Err(err)
            }
        }
        _ => Err(err),
    }
}

#[cfg(test)]
mod tests {
    use alloy_consensus::{BlockBody, Header};
    use reth_trie_common::EMPTY_ROOT_HASH;
    use rsp_primitives::genesis::Genesis;

    use super::*;

    /// A mainnet block of the Cancun period with no transactions, whose header claims
    /// `blob_gas_used`.
    fn cancun_block(blob_gas_used: u64) -> RecoveredBlock<reth_ethereum_primitives::Block> {
        let header = Header {
            number: 20_000_000,
            // Cancun (2024-03-13) is active, Prague (2025-05-07) is not.
            timestamp: 1_720_000_000,
            gas_limit: 30_000_000,
            base_fee_per_gas: Some(1_000_000_000),
            withdrawals_root: Some(EMPTY_ROOT_HASH),
            blob_gas_used: Some(blob_gas_used),
            excess_blob_gas: Some(0),
            parent_beacon_block_root: Some(B256::ZERO),
            ..Default::default()
        };
        let body = BlockBody {
            transactions: Vec::new(),
            ommers: Vec::new(),
            withdrawals: Some(Default::default()),
        };
        RecoveredBlock::new_unhashed(Block { header, body }, Vec::new())
    }

    /// The header's `blob_gas_used` is held to what the transactions use. The client copies the
    /// field into the header it commits, and nothing after this point compares the two.
    #[test]
    fn a_header_whose_blob_gas_is_not_its_bodys_is_refused() {
        let chain_spec: Arc<ChainSpec> = Arc::new((&Genesis::Mainnet).try_into().unwrap());

        // No blob transaction, and a header that says so.
        EthPrimitives::validate_block(&cancun_block(0), chain_spec.clone()).unwrap();

        // No blob transaction, and a header that claims one blob's gas.
        let refused = EthPrimitives::validate_block(&cancun_block(131_072), chain_spec);
        assert!(matches!(refused, Err(ConsensusError::BlobGasUsedDiff(_))), "{refused:?}");
    }
}
