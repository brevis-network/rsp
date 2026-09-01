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
//!
//! On top of that the guest memoises the digests: see [`MemoTable`].

use alloy_primitives::{Bloom, Log};

/// Size of the bloom filter in bytes; `alloy_primitives::BLOOM_SIZE_BYTES`.
const BLOOM_SIZE_BYTES: usize = 256;

/// The logs bloom of `logs`, identical to `logs.iter().fold(Bloom::ZERO, accrue_log)`.
pub fn logs_bloom(logs: &[Log]) -> Bloom {
    let mut bloom = Bloom::ZERO;
    {
        let data = bloom.data_mut();
        for log in logs {
            m3_2048::<20>(data, &log.address.0 .0);
            for topic in log.topics() {
                m3_2048::<32>(data, &topic.0);
            }
        }
    }
    bloom
}

/// Set the three bloom bits of `keccak256(bytes)` in `data`.
#[inline]
fn m3_2048<const N: usize>(data: &mut [u8; BLOOM_SIZE_BYTES], bytes: &[u8; N]) {
    apply_ops(data, bloom_ops(bytes));
}

/// The three `(byte index, bit mask)` pairs that `keccak256(bytes)` sets, packed one pair per
/// 16 bits: index in the low byte, mask in the high byte.
///
/// This is `alloy_primitives::Bloom::m3_2048`'s arithmetic: for `i` in 0, 2, 4 take the
/// 16-bit big-endian value at `hash[i..i + 2]`, mask it to 11 bits, and set that bit counting
/// bytes from the *end* of the filter — `data[255 - bit / 8] |= 1 << (bit % 8)`.
///
/// Deriving it is what the memo remembers, not the digest: the derivation is another ~15
/// retired instructions on top of the eight-byte digest prefix, and it is a pure function of
/// the same input, so a repeat should not pay for it either.
#[inline]
fn ops_from_digest(w: u64) -> u64 {
    let mut ops = 0u64;
    // `hash[j]` is byte `j` of the digest, so `w >> (8 * j)` in the little-endian word.
    let mut k = 0u32;
    while k < 3 {
        let i = 2 * k;
        let hi = (w >> (8 * i)) & 0xff;
        let lo = (w >> (8 * (i + 1))) & 0xff;
        let bit = ((hi << 8) | lo) & 0x7FF;
        // `bit <= 0x7FF`, so `bit / 8 <= 255` and the index is in `0..256`.
        let idx = (BLOOM_SIZE_BYTES as u64 - 1) - (bit >> 3);
        let mask = 1u64 << (bit & 7);
        ops |= (idx | (mask << 8)) << (16 * k);
        k += 1;
    }
    ops
}

/// Apply the packed pairs from [`ops_from_digest`].
#[inline]
fn apply_ops(data: &mut [u8; BLOOM_SIZE_BYTES], ops: u64) {
    let mut k = 0u32;
    while k < 3 {
        // Masked to a byte, so the index is in `0..256` and needs no bounds check.
        let idx = ((ops >> (16 * k)) & 0xff) as usize;
        let mask = ((ops >> (16 * k + 8)) & 0xff) as u8;
        data[idx] |= mask;
        k += 1;
    }
}

// --- digest memo ------------------------------------------------------------------------

/// Number of memo slots. 16,384 x 64 B is 1 MiB, and because the table is all zeros it
/// lives in `.bss`: the zkVM's memory image provides it at no retired-instruction cost.
#[allow(dead_code)] // host builds do not memoise; the table is still exercised by the tests
const MEMO_LEN: usize = 1 << 14;

#[repr(C, align(8))]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct MemoEntry {
    /// The remembered input as little-endian `u64`, zero-padded past `len`.
    key: [u64; 4],
    /// The packed bloom ops of `key`, i.e. exactly what [`bloom_ops`] returns.
    val: u64,
    /// The input's length in bytes; `0` marks an empty slot. It is part of the match test,
    /// so a 20-byte address can never be answered with the digest of a 32-byte topic whose
    /// last twelve bytes happen to be zero.
    len: u64,
    /// Pads the entry to 64 bytes so that addressing slot `i` is one `slli` and one `add`
    /// rather than a multiply. Costs nothing: the table is `.bss`.
    _pad: [u64; 2],
}

/// A direct-mapped, **exact** memo for [`bloom_ops`].
///
/// A mainnet block's log addresses and topics repeat hard: one event signature is topic 0 of
/// thousands of logs, and a handful of contracts emit most of them. Only the first eight
/// bytes of each digest are ever read, so one word per distinct input is all that has to be
/// remembered.
///
/// A hit requires the stored key to equal the probe *byte for byte* (`len` included), so a
/// slot collision costs a rehash and can never produce a wrong digest. That is what keeps
/// this sound: the bloom feeds both the receipts root and the header comparison, so a wrong
/// digest would not silently pass — but it would break correct blocks, and no hash-only
/// fingerprint is used here.
#[repr(C)]
#[allow(dead_code)]
struct MemoTable {
    slots: [MemoEntry; MEMO_LEN],
}

#[allow(dead_code)]
impl MemoTable {
    /// The `N` bytes at `p` as four little-endian words, zero-padded.
    ///
    /// Returned as a tuple, not `[u64; 4]`: an array round-trips through a stack slot here
    /// (measured at 8 retired instructions for what should be four `ld`), and comparing two
    /// `[u64; 4]` is a `bcmp` libcall on this target, which is what made the first version of
    /// this memo a net loss of 258 K.
    ///
    /// # Safety
    ///
    /// `p` must be 8-aligned and point at `N` readable bytes; `N` must be 20 or 32.
    #[inline(always)]
    unsafe fn load_words<const N: usize>(p: *const u8) -> (u64, u64, u64, u64) {
        const {
            assert!(N == 20 || N == 32);
        }
        let q = p.cast::<u64>();
        if N == 32 {
            // SAFETY: `p` is 8-aligned with 32 readable bytes.
            unsafe { (q.read(), q.add(1).read(), q.add(2).read(), q.add(3).read()) }
        } else {
            // The 20-byte address shape: two whole words and a 4-byte tail. `p + 16` is
            // 8-aligned, so the `u32` read is aligned; four `lbu` and a shift/or chain
            // become one `lwu`. Nothing past byte 20 is touched.
            // SAFETY: `p` is 8-aligned with 20 readable bytes.
            unsafe {
                (q.read(), q.add(1).read(), u64::from(p.add(16).cast::<u32>().read()), 0)
            }
        }
    }

    /// `compute(bytes)`, from the table when it is already there.
    ///
    /// `compute` must be a pure function of the bytes — it is called at most once per
    /// distinct input, and the table answers for it afterwards.
    #[inline]
    fn lookup<const N: usize>(
        &mut self,
        bytes: &[u8; N],
        compute: impl Fn(&[u8; N]) -> u64,
    ) -> u64 {
        let p = bytes.as_ptr();
        if !(p as usize).is_multiple_of(8) {
            // The word loads below need an 8-aligned probe. Both call sites are 8-aligned in
            // practice (`Log` has alignment 8 and its `address` is at offset 0; topics are a
            // 32-byte stride inside a `Vec`), so this is a safety net, not a path that runs.
            return compute(bytes);
        }
        // SAFETY: `p` is 8-aligned and `bytes` is `N` readable bytes, `N` in {20, 32} by the
        // `const` assertion inside `load_words`.
        let (w0, w1, w2, w3) = unsafe { Self::load_words::<N>(p) };
        let idx = (w0 as usize) & (MEMO_LEN - 1);
        // SAFETY: `idx < MEMO_LEN` by the mask above.
        let e = unsafe { self.slots.get_unchecked_mut(idx) };
        // One or-reduced xor rather than `e.key == [w0, w1, w2, w3]`: array equality lowers
        // to a `bcmp` libcall on `riscv64im-pico-zkvm-elf` (no unaligned scalar access, so
        // LLVM never expands it inline), and that libcall cost more than the hash it saved.
        let diff = (e.key[0] ^ w0)
            | (e.key[1] ^ w1)
            | (e.key[2] ^ w2)
            | (e.key[3] ^ w3)
            | (e.len ^ N as u64);
        if diff == 0 {
            return e.val;
        }
        let v = compute(bytes);
        e.key = [w0, w1, w2, w3];
        e.val = v;
        e.len = N as u64;
        v
    }
}

#[cfg(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64"))]
struct Memo(core::cell::UnsafeCell<MemoTable>);

// SAFETY: the pico guest is single-threaded, so the `&mut` handed out below is never aliased.
// This impl is gated to that target for exactly that reason.
#[cfg(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64"))]
unsafe impl Sync for Memo {}

#[cfg(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64"))]
static MEMO: Memo = Memo(core::cell::UnsafeCell::new(MemoTable {
    slots: [MemoEntry { key: [0; 4], val: 0, len: 0, _pad: [0; 2] }; MEMO_LEN],
}));

/// The first eight bytes of `keccak256(bytes)`, little-endian, straight from the sponge.
#[cfg(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64"))]
#[inline]
fn hash_prefix<const N: usize>(bytes: &[u8; N]) -> u64 {
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

/// The packed bloom ops of `keccak256(bytes)`, from the memo when it is already there.
#[cfg(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64"))]
#[inline]
fn bloom_ops<const N: usize>(bytes: &[u8; N]) -> u64 {
    // SAFETY: single-threaded guest (see the `Sync` impl), and the borrow does not escape.
    let table = unsafe { &mut *MEMO.0.get() };
    table.lookup(bytes, |b| ops_from_digest(hash_prefix::<N>(b)))
}

/// The packed bloom ops of `keccak256(bytes)`.
///
/// The host build does not memoise: the table's soundness argument rests on the guest being
/// single-threaded, and the host runs this from rayon workers.
#[cfg(not(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64")))]
#[inline]
fn bloom_ops<const N: usize>(bytes: &[u8; N]) -> u64 {
    let h = alloy_primitives::keccak256(bytes);
    ops_from_digest(u64::from_le_bytes(h[..8].try_into().expect("keccak256 is 32 bytes")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{Address, LogData, B256};
    use std::cell::Cell;

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
        m3_2048::<32>(
            got.data_mut(),
            &alloy_primitives::b256!(
                "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
            )
            .0,
        );
        assert_eq!(got, expected);
    }

    /// An 8-aligned `[u8; 32]`, so the memo's alignment guard does not silently turn the
    /// tests below into "everything missed".
    #[repr(align(8))]
    #[derive(Clone, Copy)]
    struct Key32([u8; 32]);

    #[repr(align(8))]
    #[derive(Clone, Copy)]
    struct Key20([u8; 20]);

    /// An all-zero table on the heap. `Box::new(MemoTable { .. })` would materialise the
    /// whole 1 MiB on the stack first and blow it in a debug build.
    fn empty_table() -> Box<MemoTable> {
        let v: Vec<MemoEntry> =
            vec![MemoEntry { key: [0; 4], val: 0, len: 0, _pad: [0; 2] }; MEMO_LEN];
        let b: Box<[MemoEntry]> = v.into_boxed_slice();
        assert_eq!(b.len(), MEMO_LEN);
        // SAFETY: `MemoTable` is `#[repr(C)]` around exactly `[MemoEntry; MEMO_LEN]`, and `b`
        // holds exactly `MEMO_LEN` elements, so the data pointer is a valid `*mut MemoTable`
        // over an allocation of the right size and alignment.
        unsafe { Box::from_raw(Box::into_raw(b) as *mut MemoTable) }
    }

    fn reference(bytes: &[u8]) -> u64 {
        let h = alloy_primitives::keccak256(bytes);
        u64::from_le_bytes(h[..8].try_into().unwrap())
    }

    /// The memo must answer exactly what a fresh hash would, and it must actually memoise:
    /// the underlying hash is called once per *distinct* key, never twice for a repeat that
    /// did not get evicted.
    #[test]
    fn memo_is_exact_and_actually_caches() {
        let mut table = empty_table();
        let calls = Cell::new(0usize);
        let mut rng = Rng(0x1234_5678_9abc_def0);

        // 400 distinct 32-byte keys and 200 distinct 20-byte ones, each probed 5 times in an
        // interleaved order. 600 keys into 16,384 slots: no eviction is expected, so the hash
        // count pins the behaviour exactly.
        let k32: Vec<Key32> = (0..400)
            .map(|_| {
                let mut k = [0u8; 32];
                k.copy_from_slice(&rng.bytes(32));
                Key32(k)
            })
            .collect();
        let k20: Vec<Key20> = (0..200)
            .map(|_| {
                let mut k = [0u8; 20];
                k.copy_from_slice(&rng.bytes(20));
                Key20(k)
            })
            .collect();
        assert!(k32.iter().all(|k| (k.0.as_ptr() as usize) % 8 == 0));
        assert!(k20.iter().all(|k| (k.0.as_ptr() as usize) % 8 == 0));

        let mut probes = 0usize;
        for round in 0..5 {
            for (i, k) in k32.iter().enumerate() {
                let got = table.lookup(&k.0, |b| {
                    calls.set(calls.get() + 1);
                    reference(b)
                });
                assert_eq!(got, reference(&k.0), "round={round} i={i} (32)");
                probes += 1;
            }
            for (i, k) in k20.iter().enumerate() {
                let got = table.lookup(&k.0, |b| {
                    calls.set(calls.get() + 1);
                    reference(b)
                });
                assert_eq!(got, reference(&k.0), "round={round} i={i} (20)");
                probes += 1;
            }
        }
        assert_eq!(probes, 5 * 600, "the probe sweep did not run in full");
        // Every key is hashed on its first probe. Distinct keys can still collide on the
        // 14-bit index, and two that do evict each other on every round, so allow more than
        // 600 -- but the memo must be doing real work, not re-hashing everything.
        assert!(calls.get() >= 600, "fewer hashes than distinct keys: {}", calls.get());
        assert!(
            calls.get() < 900,
            "the memo barely cached anything: {} hashes for 600 distinct keys",
            calls.get()
        );
    }

    /// Two different keys that land in the same slot must not answer for each other, and a
    /// 20-byte key must not be answered by a 32-byte key with the same leading bytes.
    #[test]
    fn memo_rejects_collisions_and_length_aliases() {
        let mut table = empty_table();
        let calls = Cell::new(0usize);
        let hash = |b: &[u8]| {
            calls.set(calls.get() + 1);
            reference(b)
        };

        // Same low 14 bits of word 0 (identical first two bytes), different elsewhere.
        let mut a = Key32([0u8; 32]);
        let mut b = Key32([0u8; 32]);
        a.0[0] = 0xAB;
        b.0[0] = 0xAB;
        a.0[1] = 0xCD;
        b.0[1] = 0xCD;
        a.0[31] = 1;
        b.0[31] = 2;
        assert_eq!(
            (u64::from_le_bytes(a.0[..8].try_into().unwrap()) as usize) & (MEMO_LEN - 1),
            (u64::from_le_bytes(b.0[..8].try_into().unwrap()) as usize) & (MEMO_LEN - 1),
            "the two keys must share a slot for this test to mean anything"
        );
        assert_eq!(table.lookup(&a.0, |x| hash(x)), reference(&a.0));
        assert_eq!(table.lookup(&b.0, |x| hash(x)), reference(&b.0));
        assert_eq!(table.lookup(&a.0, |x| hash(x)), reference(&a.0));
        assert_eq!(calls.get(), 3, "a collision must force a rehash, not return the neighbour");

        // A 20-byte key whose bytes are the first 20 of a 32-byte key that is zero after
        // byte 20: both canonicalize to the same four words, so only `len` separates them.
        let mut long = Key32([0u8; 32]);
        for i in 0..20 {
            long.0[i] = (i as u8).wrapping_mul(37).wrapping_add(3);
        }
        let mut short = Key20([0u8; 20]);
        short.0.copy_from_slice(&long.0[..20]);
        let before = calls.get();
        let hl = table.lookup(&long.0, |x| hash(x));
        let hs = table.lookup(&short.0, |x| hash(x));
        assert_eq!(hl, reference(&long.0));
        assert_eq!(hs, reference(&short.0));
        assert_ne!(hl, hs, "the 20- and 32-byte digests differ, so the memo must too");
        assert_eq!(calls.get() - before, 2, "the length must be part of the match test");

        // And the empty-slot marker must not answer an all-zero key.
        let zero32 = Key32([0u8; 32]);
        let before = calls.get();
        assert_eq!(table.lookup(&zero32.0, |x| hash(x)), reference(&zero32.0));
        assert_eq!(calls.get() - before, 1, "an empty slot must not match an all-zero key");
    }


    /// `ops_from_digest` + `apply_ops` must set exactly the bits `Bloom::m3_2048` sets, for
    /// every digest shape. Swept over random digests plus the boundary values of the 11-bit
    /// index (0 and 0x7FF in each of the three positions).
    #[test]
    fn ops_match_alloy_m3_2048() {
        let mut rng = Rng(0x0f0e_0d0c_0b0a_0908);
        let mut checked = 0usize;
        let case = |bytes: &[u8]| {
            let mut expected = Bloom::ZERO;
            expected.m3_2048(bytes);
            let mut got = Bloom::ZERO;
            let h = alloy_primitives::keccak256(bytes);
            let w = u64::from_le_bytes(h[..8].try_into().unwrap());
            apply_ops(got.data_mut(), ops_from_digest(w));
            assert_eq!(got, expected, "bytes={bytes:?}");
        };
        for n in [20usize, 32] {
            for _ in 0..200 {
                case(&rng.bytes(n));
                checked += 1;
            }
        }
        assert_eq!(checked, 400, "the ops sweep did not run in full");
        // And directly on the packing: a digest whose first six bytes are all zero sets bit 0
        // three times (byte 255, mask 1); all-ones sets bit 0x7FF three times (byte 0, 0x80).
        assert_eq!(ops_from_digest(0), (255) | (1 << 8) | (255 << 16) | (1 << 24) | (255 << 32) | (1 << 40));
        assert_eq!(
            ops_from_digest(u64::MAX),
            (0x80 << 8) | (0x80 << 24) | (0x80 << 40)
        );
    }
    /// `load_words` must be the identity on the bytes it packs, for both shapes the guest
    /// uses, so that "equal words" really is "equal bytes".
    #[test]
    fn load_words_round_trips() {
        let mut rng = Rng(0xdead_beef_0bad_f00d);
        let mut cases = 0usize;
        let unpack = |w: (u64, u64, u64, u64)| {
            let mut back = [0u8; 32];
            back[0..8].copy_from_slice(&w.0.to_le_bytes());
            back[8..16].copy_from_slice(&w.1.to_le_bytes());
            back[16..24].copy_from_slice(&w.2.to_le_bytes());
            back[24..32].copy_from_slice(&w.3.to_le_bytes());
            back
        };
        for _ in 0..64 {
            let mut k = Key32([0u8; 32]);
            k.0.copy_from_slice(&rng.bytes(32));
            assert_eq!(unpack(unsafe { MemoTable::load_words::<32>(k.0.as_ptr()) }), k.0);
            cases += 1;

            let mut s = Key20([0u8; 20]);
            s.0.copy_from_slice(&rng.bytes(20));
            let back = unpack(unsafe { MemoTable::load_words::<20>(s.0.as_ptr()) });
            assert_eq!(back[..20], s.0[..]);
            assert_eq!(back[20..], [0u8; 12], "the tail must be zero-padded");
            cases += 1;
        }
        assert_eq!(cases, 128, "the round-trip sweep did not run in full");
    }
}
