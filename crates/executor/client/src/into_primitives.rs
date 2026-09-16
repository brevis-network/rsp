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
                gas: GotExpected { got: cumulative_gas_used, expected: block.header().gas_used() },
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
                    GotExpected { got: receipts_root, expected: block.header().receipts_root() }
                        .into(),
                ));
            }
            if logs_bloom != block.header().logs_bloom() {
                return Err(ConsensusError::BodyBloomLogDiff(
                    GotExpected { got: logs_bloom, expected: block.header().logs_bloom() }.into(),
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
/// # What the root comparison does and does not cover
///
/// The output is hashed into a trie root and compared against the header's `receipts_root`,
/// so an encoding error makes the block fail. Verified by mutation: corrupting one byte of
/// this encoder's output makes the guest reject block 24006677.
///
/// **That is not the same as "it cannot make a bad block pass", which is what this paragraph
/// used to say.** `receipts_root` is a *prover-supplied wire field*, like `state_root` at
/// `executor.rs:118` and the logs bloom at `:131`: the comparison says the input is
/// self-consistent, not that it is a canonical block. A prover who can predict a divergence
/// writes the matching wrong root themselves. So the reason it is safe to hand-roll this is
/// the *differential* below -- two independent oracles over 43,006 receipts and 66 blocks --
/// and not the root check. Do not use the root check as a warrant for skipping a test.
///
/// That argument covers wrong *bytes*, not a wrong *length*. What keeps the length halves
/// honest is that each writer has exactly one length twin -- `header_len`/`phdr`,
/// `u64_len`/`pu64`, `bytes_len`/`pbytes`, `log_payload_len`/the per-log writer -- and
/// `fast_receipts_parity` sweeps both together. `encode_2718`'s `assert_eq!(written, total)`
/// is a real runtime assert, not a `debug_assert!`, so it survives into the guest -- but it
/// runs after the cursor has moved, which decides what it can and cannot do. If the writers
/// ran *past* `total` they have already written outside the reservation, since they go
/// through a raw cursor into `out.as_mut_ptr()` behind nothing but `out.reserve(total)`; the
/// assert reports that, it cannot prevent it. If they stopped *short* of `total` the assert
/// precedes `set_len`, so those uninitialised bytes never become part of the `Vec`. That
/// distinction is about memory safety; acceptance is covered by the root comparison above.
///
/// `adjust_index_for_rlp` is `alloy_consensus::proofs::ordered_trie_root_with_encoder`'s
/// index ordering, kept identical on purpose.
mod fast_receipts {
    use alloy_consensus::{ReceiptWithBloom, TxType, Typed2718};
    use alloy_primitives::{Bloom, Log, B256};
    use reth_ethereum_primitives::Receipt;
    use reth_trie::{HashBuilder, Nibbles};
    use std::vec::Vec;

    /// The number of significant big-endian bytes of a non-zero `v`, i.e. `8 -
    /// v.leading_zeros() / 8`.
    ///
    /// Written as a ladder rather than with `leading_zeros` because RV64IM has no
    /// count-leading-zeros instruction: LLVM expands `u64::leading_zeros` into a ~23
    /// instruction sequence, and this encoder asks for it about 40,000 times on mainnet
    /// block 24006677 — 922 K retired instructions, 27 % of everything
    /// `validate_block_post_execution` did. Every length it actually sees is a receipt or
    /// log payload of at most a few tens of kilobytes, so the small cases are tested first
    /// and the answer costs one or two compares.
    ///
    /// `v == 0` never reaches here: both callers below take their short-string/short-list
    /// branch for anything under 56 (respectively 0x80).
    #[inline(always)]
    pub(super) fn be_len(v: u64) -> usize {
        debug_assert!(v != 0);
        if v < 0x100 {
            1
        } else if v < 0x1_0000 {
            2
        } else if v < 0x100_0000 {
            3
        } else if v < 0x1_0000_0000 {
            4
        } else if v < 0x100_0000_0000 {
            5
        } else if v < 0x1_0000_0000_0000 {
            6
        } else if v < 0x100_0000_0000_0000 {
            7
        } else {
            8
        }
    }

    #[inline(always)]
    fn header_len(payload: usize) -> usize {
        if payload < 56 {
            1
        } else {
            1 + be_len(payload as u64)
        }
    }

    #[inline(always)]
    fn u64_len(v: u64) -> usize {
        if v < 0x80 {
            1
        } else {
            1 + be_len(v)
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

    /// Append the `n` low-order bytes of `v`, most significant first.
    ///
    /// This replaces `let be = v.to_be_bytes(); pcp(c, &be[skip..])`, which on RV64IM is a
    /// `swap_bytes` expansion (no byte-reverse instruction either), eight stores of the
    /// result into a stack slot, and then a `memcpy` of the two or three bytes that are
    /// actually wanted. `n` is 1..=8 and comes from [`be_len`], so this loop is short.
    ///
    /// # Safety
    ///
    /// `*c` must have room for `n` more bytes, and `n` must be at most 8.
    #[inline(always)]
    unsafe fn pbe(c: &mut *mut u8, v: u64, n: usize) {
        debug_assert!(n >= 1 && n <= 8);
        let mut i = n;
        while i > 0 {
            i -= 1;
            // SAFETY: forwarded from the caller.
            unsafe { pb(c, (v >> (8 * i)) as u8) };
        }
    }

    #[inline(always)]
    unsafe fn phdr(c: &mut *mut u8, list: bool, payload: usize) {
        let base: u8 = if list { 0xc0 } else { 0x80 };
        if payload < 56 {
            pb(c, base + payload as u8);
        } else {
            let n = be_len(payload as u64);
            pb(c, base + 55 + n as u8);
            pbe(c, payload as u64, n);
        }
    }

    #[inline(always)]
    unsafe fn pu64(c: &mut *mut u8, v: u64) {
        if v == 0 {
            pb(c, 0x80);
        } else if v < 0x80 {
            pb(c, v as u8);
        } else {
            let n = be_len(v);
            pb(c, 0x80 + n as u8);
            pbe(c, v, n);
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

    /// The transaction-type tripwire, in the *shipped* build.
    ///
    /// `encode_2718` takes its legacy-or-not decision from `matches!(r.tx_type, TxType::Legacy)`,
    /// and that is silent about a `TxType` that did not exist when it was written: a new variant
    /// gets the non-legacy shape, which is right only by luck.
    ///
    /// The exhaustiveness check for that lived in `#[cfg(test)]`, on `ALL_TX_TYPES` -- which is a
    /// fixed-size array literal and not a tripwire, whatever its comment said -- and the natural
    /// repair if alloy ever marks `TxType` `#[non_exhaustive]` is a `_ =>` arm, which removes the
    /// check permanently with nothing signalling the loss. Here it is a `const`, so adding a
    /// variant is a compile error in the encoder module, next to the `matches!` that needs
    /// revisiting.
    ///
    /// # What this deliberately does *not* assert
    ///
    /// The discriminants. An earlier version of this block pinned `TxType::Eip2930 as u8 == 1`
    /// and so on, which was the right guard while the type byte was written as `r.tx_type as u8`.
    /// #22 changed that to `Typed2718::ty()` -- the wire id from `#[envelope(ty = N)]`, which is
    /// what EIP-2718 actually asks for and is not tied to the discriminant. With `ty()` the
    /// discriminant is irrelevant to the encoding, so asserting it would *fail the build* on a
    /// divergence that is not a defect. The exhaustiveness half is the part that is still load-
    /// bearing, and `every_tx_type_matches_alloy` covers the wire ids against alloy.
    const _: () = {
        const fn is_legacy(ty: TxType) -> bool {
            match ty {
                TxType::Legacy => true,
                TxType::Eip2930 | TxType::Eip1559 | TxType::Eip4844 | TxType::Eip7702 => false,
            }
        }
        // `Legacy` is the one variant that carries no 2718 prefix at all, which is the whole
        // reason `encode_2718` branches on it.
        assert!(is_legacy(TxType::Legacy));
        assert!(!is_legacy(TxType::Eip2930));
        assert!(!is_legacy(TxType::Eip1559));
        assert!(!is_legacy(TxType::Eip4844));
        assert!(!is_legacy(TxType::Eip7702));
    };

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
                // `Typed2718::ty()`, not `as u8`. They agree for all five of today's
                // variants (`every_tx_type_matches_alloy` checks that), but `TxType` is
                // generated by the `TransactionEnvelope` derive and its wire id comes from
                // `#[envelope(ty = N)]`, which is not tied to the discriminant. The wire id
                // is what EIP-2718 asks for here.
                pb(&mut c, r.tx_type.ty());
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

    pub(super) fn receipts_root(items: &[ReceiptWithBloom<&Receipt>]) -> B256 {
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

/// `be_len`, the length ladder that replaced `8 - v.leading_zeros() / 8`.
#[cfg(test)]
mod fast_receipts_be_len {
    /// `be_len` must agree with `8 - v.leading_zeros() / 8` for every non-zero `v`, since
    /// that is literally the expression it replaced. Swept over both every power of two and
    /// its neighbours (the ladder's boundaries) and a pseudo-random spread.
    #[test]
    fn be_len_matches_leading_zeros() {
        fn reference(v: u64) -> usize {
            8 - (v.leading_zeros() as usize / 8)
        }
        let mut boundaries = 0usize;
        let mut spread = 0usize;
        for bit in 0..64u32 {
            for v in [1u64 << bit, (1u64 << bit) - 1, (1u64 << bit) + 1] {
                if v == 0 {
                    continue;
                }
                assert_eq!(super::fast_receipts::be_len(v), reference(v), "v={v:#x}");
                boundaries += 1;
            }
        }
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..1000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            for shift in 0..8 {
                let v = x >> (shift * 8);
                if v == 0 {
                    continue;
                }
                assert_eq!(super::fast_receipts::be_len(v), reference(v), "v={v:#x}");
                spread += 1;
            }
        }
        // 64 bits x 3 neighbours, minus the one `(1 << 0) - 1 == 0` that is skipped: every
        // ladder boundary is hit from both sides. The spread only has to be large; a few of
        // its high shifts land on zero.
        assert_eq!(boundaries, 191, "the boundary sweep did not run in full");
        assert!(spread > 7900, "the random spread did not run in full: {spread}");
    }
}

/// Parity between [`fast_receipts`] and the alloy encoder it replaced.
///
/// # Why this test exists
///
/// `fast_receipts` is a *fork* of consensus logic: `calculate_receipt_root` and the
/// `Encodable2718` impl for `ReceiptWithBloom` used to come from alloy, and now a copy of
/// that encoding lives here. Everything else in this crate that could go wrong fails
/// loudly the first time it runs. This does not: it stays correct right up until upstream
/// changes the encoding -- a new transaction type, a field added to a receipt -- at which
/// point alloy gets updated and this copy silently does not. The failure then is a
/// receipts-root mismatch on every block, and nothing points at this file.
///
/// So the test's job is not to check that the encoder is right today (the nine benchmark
/// blocks and the header comparison do that). Its job is to **go red when alloy moves**.
/// If you are here because it failed after a dependency bump, the fix is to bring
/// `fast_receipts` back in line with `alloy_consensus`, not to relax the test.
#[cfg(test)]
mod fast_receipts_parity {
    use alloy_consensus::{proofs::calculate_receipt_root, ReceiptWithBloom, TxReceipt, TxType};
    use alloy_network::eip2718::Encodable2718;
    use alloy_primitives::{Address, Bytes, Log, LogData, B256};
    use reth_ethereum_primitives::Receipt;

    /// A tiny deterministic PRNG, so a failure is reproducible from the seed alone.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn bytes(&mut self, len: usize) -> Vec<u8> {
            (0..len).map(|_| self.next() as u8).collect()
        }
    }

    /// Every `TxType` there is. Listed rather than matched on `i % 5` so that the round trip
    /// below is exhaustive: a variant added upstream fails to compile here, which is the
    /// tripwire `every_tx_type_matches_alloy` is supposed to be. A `_ =>` arm would have let
    /// a new type slip through as a wrong root on mainnet instead. Being `#[cfg(test)]`, it
    /// fires on `cargo test`, not on a plain build of the guest.
    const ALL_TX_TYPES: [TxType; 5] =
        [TxType::Legacy, TxType::Eip2930, TxType::Eip1559, TxType::Eip4844, TxType::Eip7702];

    fn tx_type(i: usize) -> TxType {
        let ty = ALL_TX_TYPES[i % ALL_TX_TYPES.len()];
        match ty {
            TxType::Legacy |
            TxType::Eip2930 |
            TxType::Eip1559 |
            TxType::Eip4844 |
            TxType::Eip7702 => ty,
        }
    }

    /// One receipt, drawn to hit the places where RLP changes shape rather than uniformly:
    /// the single-byte-string case, the empty string, the 55/56-byte header boundary, and
    /// the u64 lengths either side of 0x80.
    fn receipt(rng: &mut Rng, i: usize) -> Receipt {
        let n_logs = rng.below(4);
        let logs = (0..n_logs)
            .map(|_| {
                let n_topics = rng.below(5);
                let topics: Vec<B256> =
                    (0..n_topics).map(|_| B256::from_slice(&rng.bytes(32))).collect();
                // Lengths that straddle every RLP boundary that matters here.
                let data_len = match rng.below(6) {
                    0 => 0,
                    1 => 1,
                    2 => 55,
                    3 => 56,
                    4 => 256,
                    _ => rng.below(300),
                };
                let mut data = rng.bytes(data_len);
                // The single-byte-below-0x80 case is its own RLP form; make sure it occurs.
                if data_len == 1 && i % 2 == 0 {
                    data[0] = 0x7f;
                }
                Log {
                    address: Address::from_slice(&rng.bytes(20)),
                    data: LogData::new_unchecked(topics, Bytes::from(data)),
                }
            })
            .collect();

        let cumulative_gas_used = match rng.below(6) {
            0 => 0,
            1 => 1,
            2 => 0x7f,
            3 => 0x80,
            4 => u64::MAX,
            _ => rng.next(),
        };

        Receipt { tx_type: tx_type(i), success: i % 3 != 0, cumulative_gas_used, logs }
    }

    /// The root over a block's worth of receipts must equal alloy's, receipt for receipt.
    ///
    /// The guards at the bottom tally what the generator actually produced -- logs at all, a
    /// receipt with none (RLP's empty list), every `TxType`, both extremes of the topic list, and
    /// RLP's single-byte and long-form string forms. A count of iterations would prove nothing,
    /// the bounds being constants. Not guarded: the two-byte length header, and the five
    /// `cumulative_gas_used` shapes.
    #[test]
    fn receipts_root_matches_alloy() {
        let mut rng = Rng(0x5DEE_CE66_D000_0001);
        let mut types_seen = [0usize; ALL_TX_TYPES.len()];
        let mut topics_seen = [0usize; 5];
        let mut short_data = 0usize; // a one-byte log payload below 0x80: its own RLP form
        let mut long_data = 0usize; // >= 56 bytes: the long-form string header
        let mut n_logs = 0usize; // drawn per receipt, so this is not a loop-bound count
        let mut empty_log_lists = 0usize;

        // Block sizes chosen around `adjust_index_for_rlp`'s two edges: the `i == 0x7f`
        // case and the `i + 1 == len` case only differ once the block is long enough.
        for &n in &[1usize, 2, 3, 16, 127, 128, 129, 200] {
            for round in 0..4 {
                let receipts: Vec<Receipt> = (0..n).map(|i| receipt(&mut rng, i + round)).collect();
                let with_bloom: Vec<ReceiptWithBloom<&Receipt>> = receipts
                    .iter()
                    .map(|r| ReceiptWithBloom::new(r, TxReceipt::bloom(r)))
                    .collect();

                let ours = super::fast_receipts::receipts_root(&with_bloom);
                let theirs = calculate_receipt_root(&with_bloom);
                assert_eq!(
                    ours, theirs,
                    "receipts root diverged from alloy at n={n} round={round}"
                );

                for r in &receipts {
                    types_seen[r.tx_type as usize] += 1;
                    n_logs += r.logs.len();
                    if r.logs.is_empty() {
                        empty_log_lists += 1;
                    }
                    for log in &r.logs {
                        topics_seen[log.topics().len()] += 1;
                        let d = log.data.data.as_ref();
                        if d.len() == 1 && d[0] < 0x80 {
                            short_data += 1;
                        }
                        if d.len() >= 56 {
                            long_data += 1;
                        }
                    }
                }
            }
        }

        assert!(n_logs > 1000, "only {n_logs} logs were generated");
        assert!(empty_log_lists > 0, "no receipt with an empty log list was generated");
        for (ty, &n) in types_seen.iter().enumerate() {
            assert!(n > 0, "no receipt of tx type {ty} was generated");
        }
        assert!(topics_seen[0] > 0, "no log with zero topics was generated");
        assert!(topics_seen[4] > 0, "no log with four topics was generated");
        assert!(short_data > 0, "no single-byte log payload below 0x80 was generated");
        assert!(long_data > 0, "no log payload in RLP's long-string form was generated");
    }

    /// The two ladders the block corpus never climbs: RLP's **long-form length header above
    /// two bytes**, and **three-byte `rlp(index)` trie keys**.
    ///
    /// Both are mainnet-reachable and neither was covered: the generator above tops out at a
    /// 300-byte log payload (`max_logs_list_payload = 1199`, measured from both sides) and a
    /// 200-receipt block, short of the 65,536 threshold where `header_len` goes to four bytes and
    /// the 256 receipts where the trie key goes to three. A 64 KB log is one transaction, and
    /// blocks past 256 transactions are routine. Two one-token mutants of the ladder change six
    /// receipts roots and leave the rest of the suite green.
    ///
    /// The guards read the encoder's own output, not the loop bounds -- `max(&[.., 65_536])` is a
    /// constant, and asserting on it holds for any behaviour of the code under test.
    /// `max_len_bytes` is the RLP list-header width that came out of the encoded receipt, and
    /// `max_key_len` the trie-key width `receipts_root` builds with `encode_fixed_size`.
    #[test]
    fn receipts_root_over_long_payloads_and_large_blocks() {
        /// Length bytes in the RLP list header of an encoded receipt -- 0 for the short-list
        /// form, 1..=8 for `0xf8..=0xff`. A typed receipt is `ty || rlp_list`; the five wire ids
        /// are all <= 0x04, well below any list tag.
        fn list_length_bytes(value: &[u8]) -> usize {
            let b0 = if value[0] <= 0x04 { value[1] } else { value[0] };
            if b0 >= 0xf8 {
                (b0 - 0xf7) as usize
            } else {
                0
            }
        }

        let mut rng = Rng(0x0BAD_C0DE_1234_5678);
        let mut max_len_bytes = 0usize;
        let mut max_key_len = 0usize;

        // (a) One log per receipt, with a data length sitting on each rung of `header_len`'s
        // ladder and on both sides of it. 55/56 is the short-to-long edge, 255/256 is where
        // the length needs two bytes, 65,535/65,536 is where it needs three -- the rung the
        // corpus never reaches.
        for &len in &[
            0usize, 1, 55, 56, 57, 254, 255, 256, 257, 1_000, 65_534, 65_535, 65_536, 65_537,
            70_000,
        ] {
            let topics: Vec<B256> = (0..3).map(|_| B256::from_slice(&rng.bytes(32))).collect();
            let log = Log {
                address: Address::from_slice(&rng.bytes(20)),
                data: LogData::new_unchecked(topics, Bytes::from(rng.bytes(len))),
            };
            for (i, &ty) in ALL_TX_TYPES.iter().enumerate() {
                let r = Receipt {
                    tx_type: ty,
                    success: i % 2 == 0,
                    cumulative_gas_used: 21_000 * (i as u64 + 1),
                    logs: std::vec![log.clone()],
                };
                let with_bloom = std::vec![ReceiptWithBloom::new(&r, TxReceipt::bloom(&r))];
                max_len_bytes = max_len_bytes.max(list_length_bytes(&with_bloom[0].encoded_2718()));
                assert_eq!(
                    super::fast_receipts::receipts_root(&with_bloom),
                    calculate_receipt_root(&with_bloom),
                    "receipts root diverged at log data length {len}, tx type {ty:?}"
                );
            }
        }

        // ... and the same lengths in a block, so the *receipt* payload -- not just the log's
        // -- crosses the rung too.
        for &len in &[56usize, 300, 65_536] {
            let receipts: Vec<Receipt> = (0..4usize)
                .map(|i| {
                    let topics: Vec<B256> =
                        (0..(i % 5)).map(|_| B256::from_slice(&rng.bytes(32))).collect();
                    Receipt {
                        tx_type: tx_type(i),
                        success: true,
                        cumulative_gas_used: 1_000_000 * (i as u64 + 1),
                        logs: std::vec![Log {
                            address: Address::from_slice(&rng.bytes(20)),
                            data: LogData::new_unchecked(topics, Bytes::from(rng.bytes(len))),
                        }],
                    }
                })
                .collect();
            let with_bloom: Vec<ReceiptWithBloom<&Receipt>> =
                receipts.iter().map(|r| ReceiptWithBloom::new(r, TxReceipt::bloom(r))).collect();
            assert_eq!(
                super::fast_receipts::receipts_root(&with_bloom),
                calculate_receipt_root(&with_bloom),
                "receipts root diverged for a block of 64 KB logs at length {len}"
            );
        }

        // (b) Block sizes across the trie-key ladder. `rlp(index)` is one byte below 0x80,
        // two up to 0xff, and **three** from 0x100 -- which needs 256 receipts.
        for &n in &[127usize, 128, 129, 254, 255, 256, 257, 300, 512] {
            let receipts: Vec<Receipt> = (0..n).map(|i| receipt(&mut rng, i)).collect();
            let with_bloom: Vec<ReceiptWithBloom<&Receipt>> =
                receipts.iter().map(|r| ReceiptWithBloom::new(r, TxReceipt::bloom(r))).collect();
            // The trie keys `receipts_root` will build. `adjust_index_for_rlp` permutes
            // `0..n`, so the widest key in the block is the widest over that whole range.
            for i in 0..n {
                max_key_len = max_key_len.max(alloy_rlp::encode_fixed_size(&i).len());
            }
            assert_eq!(
                super::fast_receipts::receipts_root(&with_bloom),
                calculate_receipt_root(&with_bloom),
                "receipts root diverged for a block of {n} receipts"
            );
        }

        assert_eq!(
            max_len_bytes, 3,
            "the encoder never emitted an RLP list header with three length bytes, so the rung \
             above 65,536 was not exercised (widest header produced: {max_len_bytes} length \
             bytes)"
        );
        assert_eq!(
            max_key_len, 3,
            "no three-byte `rlp(index)` trie key was produced, so the 256-receipt rung was not \
             exercised (widest key produced: {max_key_len} bytes)"
        );
    }

    /// The empty case, which `receipts_root` short-circuits.
    #[test]
    fn empty_receipts_root_matches_alloy() {
        let empty: Vec<ReceiptWithBloom<&Receipt>> = Vec::new();
        assert_eq!(super::fast_receipts::receipts_root(&empty), calculate_receipt_root(&empty));
    }

    /// Every transaction type on its own, so a new variant upstream shows up here as a
    /// compile error on `tx_type` or a mismatch, rather than as a wrong root on one block.
    #[test]
    fn every_tx_type_matches_alloy() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for i in 0..5 {
            let r = Receipt { tx_type: tx_type(i), ..receipt(&mut rng, i) };
            let with_bloom = vec![ReceiptWithBloom::new(&r, TxReceipt::bloom(&r))];
            assert_eq!(
                super::fast_receipts::receipts_root(&with_bloom),
                calculate_receipt_root(&with_bloom),
                "tx type {:?} diverged from alloy",
                tx_type(i)
            );
        }
    }
}
