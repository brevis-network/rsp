//! The logs bloom of a receipt, without alloy's `Bloom::accrue_log`.
//!
//! Computing the receipts root and the header's logs bloom is the guest's second-largest
//! consumer of keccak: 25,847 of the 70,722 hashes on mainnet block 24006677, one per log
//! address and one per log topic. (They are *not* computed twice —
//! `reth_ethereum_consensus::verify_receipts` memoises each receipt's bloom into a
//! `ReceiptWithBloom` and the `logs_bloom` fold then reads `bloom_ref()`. 7,373 logs, 2.51
//! topics each, is exactly 25,847.)
//!
//! What alloy's version costs on top of those hashes is 238.8 retired instructions per log,
//! and none of it is the bloom arithmetic:
//!
//! * `Bloom::accrue_raw_log(address: Address, ..)` takes the address **by value**. `Address`
//!   is `[u8; 20]` with alignment 1, so the copy into the callee's slot is 20 `lbu` plus 20
//!   `sb`. Hashing it straight out of the `Log` needs no copy at all.
//! * `m3_2048` goes through `keccak256(bytes) -> B256`. `B256` is alignment 1 too, so the
//!   digest is written to the caller's slot with a 28-`srli`/32-`sb` scatter — and then only
//!   its first six bytes are ever read.
//!
//! This module reads those six bytes as one aligned `u64` out of a slot it owns, and the bit
//! arithmetic is byte-for-byte alloy's (see [`m3_2048`]). `bloom_parity` checks that against
//! alloy over random logs.

use alloy_primitives::{Bloom, Log};

/// Size of the bloom filter in bytes; `alloy_primitives::BLOOM_SIZE_BYTES`.
const BLOOM_SIZE_BYTES: usize = 256;

/// The logs bloom of `logs`, identical to `logs.iter().fold(Bloom::ZERO, accrue_log)`.
pub fn logs_bloom(logs: &[Log]) -> Bloom {
    let mut bloom = Bloom::ZERO;
    {
        let data = bloom.data_mut();
        for log in logs {
            m3_2048(data, log.address.as_slice());
            for topic in log.topics() {
                m3_2048(data, topic.as_slice());
            }
        }
    }
    bloom
}

/// Set the three bloom bits of `keccak256(bytes)` in `data`.
///
/// This is `alloy_primitives::Bloom::m3_2048`: for `i` in 0, 2, 4 take the 16-bit
/// big-endian value at `hash[i..i + 2]`, mask it to 11 bits, and set that bit counting
/// bytes from the *end* of the filter — `data[255 - bit / 8] |= 1 << (bit % 8)`.
#[inline]
fn m3_2048(data: &mut [u8; BLOOM_SIZE_BYTES], bytes: &[u8]) {
    let w = digest_prefix(bytes);
    // `hash[j]` is byte `j` of the digest, so `w >> (8 * j)` in the little-endian word.
    for i in [0u32, 2, 4] {
        let hi = (w >> (8 * i)) & 0xff;
        let lo = (w >> (8 * (i + 1))) & 0xff;
        let bit = ((hi << 8) | lo) & 0x7FF;
        // `bit <= 0x7FF`, so `bit / 8 <= 255` and the index is in `0..256`.
        data[BLOOM_SIZE_BYTES - 1 - (bit >> 3) as usize] |= 1u8 << (bit & 7);
    }
}

/// The first eight bytes of `keccak256(bytes)`, little-endian.
#[cfg(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64"))]
#[inline]
fn digest_prefix(bytes: &[u8]) -> u64 {
    /// A digest slot the compiler knows is 8-aligned, so reading the first word out of it is
    /// one `ld` rather than eight `lbu` and a shift/or tree.
    #[repr(align(8))]
    struct Digest([u8; 32]);
    // Uninitialized rather than `[0u8; 32]`: the sponge writes all four words, but it takes a
    // raw pointer, so LLVM cannot see that and would keep four `sd` of zero.
    let mut d = core::mem::MaybeUninit::<Digest>::uninit();
    let p = d.as_mut_ptr().cast::<u8>();
    // SAFETY: `p` is 32 writable bytes, which is `keccak_into`'s whole contract -- it handles
    // an unaligned destination itself. The 8-alignment is for the `ld` below, not for the
    // callee. `keccak_into` writes all 32 bytes, so that read is of initialized memory.
    unsafe {
        crate::mpt::keccak_into(bytes, p);
        p.cast::<u64>().read()
    }
}

/// The first eight bytes of `keccak256(bytes)`, little-endian.
#[cfg(not(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64")))]
#[inline]
fn digest_prefix(bytes: &[u8]) -> u64 {
    let h = alloy_primitives::keccak256(bytes);
    u64::from_le_bytes(h[..8].try_into().expect("keccak256 is 32 bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{Address, LogData, B256};

    /// A deterministic xorshift, so the sweep below is reproducible without a dev-dependency.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.next() as u8).collect()
        }
    }

    /// The bloom must match alloy's for every topic count a log can have (0..=4), for logs
    /// whose addresses and topics are unrelated to each other, and for the empty log list.
    ///
    /// The comparison is against `Bloom::accrue_log`, i.e. the exact function this module
    /// replaces, not against a recorded constant: a constant would only pin one input.
    #[test]
    fn bloom_parity() {
        let mut rng = Rng(0x243f_6a88_85a3_08d3);
        let mut checked = 0usize;
        for n_logs in [0usize, 1, 2, 7] {
            for n_topics in 0..=4usize {
                let logs: Vec<Log> = (0..n_logs)
                    .map(|_| {
                        let address = Address::from_slice(&rng.bytes(20));
                        let topics: Vec<B256> =
                            (0..n_topics).map(|_| B256::from_slice(&rng.bytes(32))).collect();
                        Log {
                            address,
                            data: LogData::new_unchecked(topics, rng.bytes(13).into()),
                        }
                    })
                    .collect();
                let mut expected = Bloom::ZERO;
                for log in &logs {
                    expected.accrue_log(log);
                }
                assert_eq!(logs_bloom(&logs), expected, "n_logs={n_logs} n_topics={n_topics}");
                checked += 1;
            }
        }
        // Guard against the sweep collapsing to nothing: 4 log counts x 5 topic counts.
        assert_eq!(checked, 20);
        // And one hand-checked case, so a bug that broke both this module *and* the
        // comparison would still be caught: the canonical ERC-20 Transfer topic on its own.
        let mut expected = Bloom::ZERO;
        expected.m3_2048(
            alloy_primitives::b256!(
                "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
            )
            .as_slice(),
        );
        let mut got = Bloom::ZERO;
        m3_2048(
            got.data_mut(),
            alloy_primitives::b256!(
                "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
            )
            .as_slice(),
        );
        assert_eq!(got, expected);
    }
}
