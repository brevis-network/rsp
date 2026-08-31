use std::sync::Arc;

use alloy_consensus::{Block, BlockHeader, Header, ReceiptWithBloom, TxEnvelope, TxReceipt};
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

            let receipts_root = fast_receipts::receipts_root(&receipts_with_bloom);
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

/// The receipts root, computed with a purpose-built EIP-2718 receipt encoder.
///
/// # Why this exists
///
/// `alloy_consensus::proofs::calculate_receipt_root` costs 10.04 M retired guest
/// instructions on mainnet block 24006677 -- 2.9 % of the whole guest -- of which 7.84 M is
/// the RLP encoding alone, 79 K per receipt over 99 receipts. That is not the output buffer:
/// `ordered_trie_root_with_encoder` reuses one `Vec` and `clear()`s it between receipts, so
/// its growth is already amortised (the whole guest's `RawVec::grow` is 76 K, and presizing
/// the buffer measured under 0.1 M). The cost is the *shape* of the generic encoder:
/// `Encodable::length()` recomputes every nested length two or three times
/// (`length_of_length` alone is 1.05 M), every byte goes through `&mut dyn BufMut`, and the
/// payload arrives as ~41,515 small misaligned `memcpy` calls.
///
/// This module computes each receipt's length once, reserves exactly that, and writes the
/// bytes through a raw cursor. Block 24006677: -1,187,892.
///
/// # Why it is safe to hand-roll a consensus-critical encoding here
///
/// The output is not trusted -- it is hashed into a trie root and compared against the
/// header's `receipts_root`, so any encoding error makes the block *fail*, and cannot make a
/// bad block pass. `encode_2718` additionally asserts that the bytes written equal the
/// length computed, so a mismatch between the two halves is a panic rather than a short
/// buffer. Verified by mutation: corrupting one byte of this encoder's output makes the
/// guest reject block 24006677.
///
/// `adjust_index_for_rlp` is `alloy_consensus::proofs::ordered_trie_root_with_encoder`'s
/// index ordering, kept identical on purpose.
mod fast_receipts {
    use alloy_consensus::{ReceiptWithBloom, TxType};
    use alloy_primitives::{Bloom, Log, B256};
    use reth_ethereum_primitives::Receipt;
    use reth_trie::{HashBuilder, Nibbles};
    use std::vec::Vec;

    #[inline(always)]
    fn header_len(payload: usize) -> usize {
        if payload < 56 {
            1
        } else {
            1 + (8 - (payload.leading_zeros() as usize / 8))
        }
    }

    #[inline(always)]
    fn u64_len(v: u64) -> usize {
        if v < 0x80 {
            1
        } else {
            1 + (8 - (v.leading_zeros() as usize / 8))
        }
    }

    #[inline(always)]
    fn bytes_len(d: &[u8]) -> usize {
        if d.len() == 1 && d[0] < 0x80 {
            1
        } else {
            d.len() + header_len(d.len())
        }
    }

    #[inline(always)]
    fn log_payload_len(l: &Log) -> usize {
        let tp = l.topics().len() * 33;
        21 + header_len(tp) + tp + bytes_len(&l.data.data)
    }

    #[inline(always)]
    unsafe fn pb(c: &mut *mut u8, v: u8) {
        **c = v;
        *c = (*c).add(1);
    }

    #[inline(always)]
    unsafe fn pcp(c: &mut *mut u8, s: &[u8]) {
        core::ptr::copy_nonoverlapping(s.as_ptr(), *c, s.len());
        *c = (*c).add(s.len());
    }

    #[inline(always)]
    unsafe fn phdr(c: &mut *mut u8, list: bool, payload: usize) {
        let base: u8 = if list { 0xc0 } else { 0x80 };
        if payload < 56 {
            pb(c, base + payload as u8);
        } else {
            let skip = (payload.leading_zeros() / 8) as usize;
            let be = payload.to_be_bytes();
            pb(c, base + 55 + (8 - skip) as u8);
            pcp(c, &be[skip..]);
        }
    }

    #[inline(always)]
    unsafe fn pu64(c: &mut *mut u8, v: u64) {
        if v == 0 {
            pb(c, 0x80);
        } else if v < 0x80 {
            pb(c, v as u8);
        } else {
            let skip = (v.leading_zeros() / 8) as usize;
            let be = v.to_be_bytes();
            pb(c, 0x80 + (8 - skip) as u8);
            pcp(c, &be[skip..]);
        }
    }

    #[inline(always)]
    unsafe fn pbytes(c: &mut *mut u8, d: &[u8]) {
        if d.len() == 1 && d[0] < 0x80 {
            pb(c, d[0]);
        } else {
            phdr(c, false, d.len());
            pcp(c, d);
        }
    }

    fn encode_2718(r: &Receipt, bloom: &Bloom, out: &mut Vec<u8>, lens: &mut Vec<usize>) {
        lens.clear();
        let mut logs_payload = 0usize;
        for l in &r.logs {
            let p = log_payload_len(l);
            lens.push(p);
            logs_payload += header_len(p) + p;
        }
        let payload =
            1 + u64_len(r.cumulative_gas_used) + 259 + header_len(logs_payload) + logs_payload;
        let legacy = matches!(r.tx_type, TxType::Legacy);
        let total = (!legacy) as usize + header_len(payload) + payload;

        out.clear();
        out.reserve(total);
        unsafe {
            let base = out.as_mut_ptr();
            let mut c = base;
            if !legacy {
                pb(&mut c, r.tx_type as u8);
            }
            phdr(&mut c, true, payload);
            pb(&mut c, if r.success { 0x01 } else { 0x80 });
            pu64(&mut c, r.cumulative_gas_used);
            pb(&mut c, 0xb9);
            pb(&mut c, 0x01);
            pb(&mut c, 0x00);
            pcp(&mut c, bloom.as_slice());
            phdr(&mut c, true, logs_payload);
            for (l, &p) in r.logs.iter().zip(lens.iter()) {
                phdr(&mut c, true, p);
                pb(&mut c, 0x94);
                pcp(&mut c, l.address.as_slice());
                let topics = l.topics();
                phdr(&mut c, true, topics.len() * 33);
                for t in topics {
                    pb(&mut c, 0xa0);
                    pcp(&mut c, t.as_slice());
                }
                pbytes(&mut c, &l.data.data);
            }
            let written = c.offset_from(base) as usize;
            assert_eq!(written, total, "receipt rlp length mismatch");
            out.set_len(total);
        }
    }

    #[inline]
    const fn adjust_index_for_rlp(i: usize, len: usize) -> usize {
        if i > 0x7f {
            i
        } else if i == 0x7f || i + 1 == len {
            0
        } else {
            i + 1
        }
    }

    pub fn receipts_root(items: &[ReceiptWithBloom<&Receipt>]) -> B256 {
        if items.is_empty() {
            return alloy_consensus::constants::EMPTY_ROOT_HASH;
        }
        let mut hb = HashBuilder::default();
        let mut buf: Vec<u8> = Vec::new();
        let mut lens: Vec<usize> = Vec::new();
        let n = items.len();
        for i in 0..n {
            let index = adjust_index_for_rlp(i, n);
            let index_buffer = alloy_rlp::encode_fixed_size(&index);
            encode_2718(items[index].receipt, &items[index].logs_bloom, &mut buf, &mut lens);
            hb.add_leaf(Nibbles::unpack(&index_buffer), &buf);
        }
        hb.root()
    }
}
