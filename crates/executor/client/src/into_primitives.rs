use std::sync::Arc;

use alloy_consensus::{
    proofs::calculate_receipt_root, Block, BlockHeader, Header, ReceiptWithBloom, TxEnvelope,
    TxReceipt,
};
use alloy_network::{Ethereum, Network};
use alloy_primitives::Bloom;
use reth_chainspec::{ChainSpec, EthChainSpec, EthereumHardforks, NamedChain};
use reth_consensus::HeaderValidator;
use reth_consensus_common::validation::validate_body_against_header;
use reth_errors::ConsensusError;
use reth_ethereum_consensus::EthBeaconConsensus;
use reth_ethereum_primitives::EthPrimitives;
use reth_execution_types::BlockExecutionOutput;
use reth_primitives_traits::{
    receipt::gas_spent_by_transactions, GotExpected, NodePrimitives, RecoveredBlock, SealedHeader,
};

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

    fn validate_block_post_execution(
        block: &RecoveredBlock<Self::Block>,
        chain_spec: Arc<CS>,
        execution_output: &BlockExecutionOutput<Self::Receipt>,
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

        validate_body_against_header(recovered.body(), recovered.header())?;

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

    /// `reth_ethereum_consensus::validate_block_post_execution`, with the per-receipt logs
    /// bloom computed by [`rsp_mpt::logs_bloom`] instead of `alloy_primitives::Bloom`.
    ///
    /// Every check reth makes is made here, in the same order and against the same header
    /// fields: cumulative gas used, the receipts root and the header logs bloom (post
    /// Byzantium), and the requests hash (post Prague). What changes is only *how* each
    /// receipt's bloom is computed — 25,847 of the guest's 70,722 keccak hashes on mainnet
    /// block 24006677 go through it, and alloy's version spends 238.8 retired instructions
    /// per log on copying a byte-aligned `Address` by value and materialising a `B256`
    /// digest whose first six bytes are all it reads. See the [`rsp_mpt::bloom`] module docs.
    ///
    /// Note the two comparisons are self-checking in the direction that matters: the same
    /// bloom feeds the receipts trie and the header comparison, so a bloom this code got
    /// wrong makes both `receipts_root` and `logs_bloom` differ from the header's and the
    /// block is rejected. It cannot make a wrong block pass.
    fn validate_block_post_execution(
        block: &RecoveredBlock<Self::Block>,
        chain_spec: Arc<ChainSpec>,
        execution_output: &BlockExecutionOutput<Self::Receipt>,
    ) -> Result<(), ConsensusError> {
        let receipts = &execution_output.result.receipts;

        // Gas used must match the header.
        let cumulative_gas_used =
            receipts.last().map(|receipt| receipt.cumulative_gas_used()).unwrap_or(0);
        if block.header().gas_used() != cumulative_gas_used {
            return Err(ConsensusError::BlockGasUsed {
                gas: GotExpected {
                    got: cumulative_gas_used,
                    expected: block.header().gas_used(),
                },
                gas_spent_by_tx: gas_spent_by_transactions(receipts),
            });
        }

        // Before Byzantium a receipt carried a state root rather than a status flag, and
        // this crate has never had to encode one; reth gates the receipts check the same
        // way (EIP-658).
        if chain_spec.is_byzantium_active_at_block(block.header().number()) {
            // One bloom per receipt, computed once and used both for the receipts trie —
            // the bloom is part of a receipt's RLP encoding — and for the header's bloom.
            let mut logs_bloom = Bloom::ZERO;
            let receipts_with_bloom = receipts
                .iter()
                .map(|receipt| {
                    let bloom = rsp_mpt::logs_bloom(receipt.logs());
                    logs_bloom |= bloom;
                    ReceiptWithBloom::new(receipt, bloom)
                })
                .collect::<Vec<_>>();

            let receipts_root = calculate_receipt_root(&receipts_with_bloom);
            if receipts_root != block.header().receipts_root() {
                return Err(ConsensusError::BodyReceiptRootDiff(
                    GotExpected {
                        got: receipts_root,
                        expected: block.header().receipts_root(),
                    }
                    .into(),
                ));
            }
            if logs_bloom != block.header().logs_bloom() {
                return Err(ConsensusError::BodyBloomLogDiff(
                    GotExpected { got: logs_bloom, expected: block.header().logs_bloom() }
                        .into(),
                ));
            }
        }

        // The requests hash must match the header once Prague is active.
        if chain_spec.is_prague_active_at_timestamp(block.header().timestamp()) {
            let Some(header_requests_hash) = block.header().requests_hash() else {
                return Err(ConsensusError::RequestsHashMissing);
            };
            let requests_hash = execution_output.result.requests.requests_hash();
            if requests_hash != header_requests_hash {
                return Err(ConsensusError::BodyRequestsHashDiff(
                    GotExpected::new(requests_hash, header_requests_hash).into(),
                ));
            }
        }

        Ok(())
    }
}

#[cfg(feature = "optimism")]
impl IntoPrimitives<op_alloy_network::Optimism> for reth_optimism_primitives::OpPrimitives {
    fn into_primitive_block(
        block: alloy_rpc_types::Block<op_alloy_rpc_types::Transaction>,
    ) -> Self::Block {
        let block = block.map_transactions(|tx| tx.inner.inner.into_inner());
        block.into_consensus()
    }

    fn into_consensus_header(header: alloy_rpc_types::Header) -> Header {
        header.into()
    }
}

#[cfg(feature = "optimism")]
impl FromInput for reth_optimism_primitives::OpPrimitives {
    fn from_input_block(block: Block<Self::SignedTx>) -> Self::Block {
        block
    }
}

#[cfg(feature = "optimism")]
impl IntoInput for reth_optimism_primitives::OpPrimitives {
    fn into_input_block(block: Self::Block) -> Block<Self::SignedTx> {
        block
    }
}

#[cfg(feature = "optimism")]
impl BlockValidator<reth_optimism_chainspec::OpChainSpec>
    for reth_optimism_primitives::OpPrimitives
{
    fn validate_header(
        header: &SealedHeader,
        chain_spec: Arc<reth_optimism_chainspec::OpChainSpec>,
    ) -> Result<(), ConsensusError> {
        let validator = reth_optimism_consensus::OpBeaconConsensus::new(chain_spec);

        validator.validate_header(header)
    }

    fn validate_block(
        recovered: &RecoveredBlock<Self::Block>,
        chain_spec: Arc<reth_optimism_chainspec::OpChainSpec>,
    ) -> Result<(), ConsensusError> {
        Self::validate_header(recovered.sealed_header(), chain_spec.clone())?;

        reth_optimism_consensus::validation::validate_body_against_header_op(
            chain_spec,
            recovered.body(),
            recovered.header(),
        )?;

        Ok(())
    }

    fn validate_header_against_parent(
        header: &SealedHeader,
        parent: &SealedHeader,
        chain_spec: Arc<reth_optimism_chainspec::OpChainSpec>,
    ) -> Result<(), ConsensusError> {
        let validator = reth_optimism_consensus::OpBeaconConsensus::new(chain_spec);

        validator.validate_header_against_parent(header, parent)
    }

    fn validate_block_post_execution(
        block: &RecoveredBlock<Self::Block>,
        chain_spec: Arc<reth_optimism_chainspec::OpChainSpec>,
        execution_output: &BlockExecutionOutput<Self::Receipt>,
    ) -> Result<(), ConsensusError> {
        reth_optimism_consensus::validate_block_post_execution(
            block.header(),
            &chain_spec,
            &execution_output.result,
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
                ConsensusError::ExtraDataExceedsMax { .. } |
                    ConsensusError::TheMergeDifficultyIsNotZero
            ) {
                Ok(())
            } else {
                Err(err)
            }
        }
        _ => Err(err),
    }
}
