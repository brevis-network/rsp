// This code is modified from the original implementation of Zeth.
//
// Reference: https://github.com/risc0/zeth
//
// Copyright 2023 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![allow(unreachable_pub)]
#![allow(dead_code)]
// target_vendor = "pico" is defined by the custom riscv64im-pico-zkvm-elf target
#![allow(unexpected_cfgs)]

// `keccak256_sponge` assembles each `u64` from little-endian byte order, both when reading
// unaligned input and when writing the squeezed state.
const _: () = assert!(cfg!(target_endian = "little"));

use alloc::boxed::Box;
use alloy_primitives::{b256, map::HashMap, B256};
use alloy_rlp::Encodable;
use core::{
    cmp,
    fmt::{Debug, Write},
    iter, mem,
};
use reth_trie::{AccountProof, Nibbles};
use std::sync::Mutex;

use rlp::{Decodable, DecoderError, Prototype, Rlp};
use serde::{Deserialize, Serialize};
use thiserror::Error as ThisError;

use alloy_primitives::Address;

use super::{EthereumState, FromProofError};

pub trait RlpBytes {
    /// Returns the RLP-encoding.
    fn to_rlp(&self) -> Vec<u8>;
}

impl<T> RlpBytes for T
where
    T: alloy_rlp::Encodable,
{
    #[inline]
    fn to_rlp(&self) -> Vec<u8> {
        let rlp_length = self.length();
        let mut out = Vec::with_capacity(rlp_length);
        self.encode(&mut out);
        debug_assert_eq!(out.len(), rlp_length);
        out
    }
}

/// Root hash of an empty trie.
pub const EMPTY_ROOT: B256 =
    b256!("56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421");

extern crate alloc;

/// Represents the Keccak-256 hash of an empty byte slice.
///
/// This is a constant value and can be used as a default or placeholder
/// in various cryptographic operations.
pub const KECCAK_EMPTY: B256 =
    b256!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");

/// Computes the Keccak-256 hash of the provided data.
///
/// This function is a thin wrapper around the Keccak256 hashing algorithm
/// and is optimized for performance.
///
/// # TODO
/// - Consider switching the return type to `B256` for consistency with other parts of the codebase.
#[inline]
#[cfg(not(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64")))]
pub fn keccak(data: impl AsRef<[u8]>) -> [u8; 32] {
    *alloy_primitives::utils::keccak256(data)
}

/// On the Pico zkVM target, hash with a direct sponge over the keccak permute syscall. This
/// skips the generic digest machinery (block buffering, trait dispatch) that costs ~1-2K cycles
/// per call on small inputs; the permute syscall is the same one the patched sha3 crate uses.
#[inline]
#[cfg(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64"))]
pub fn keccak(data: impl AsRef<[u8]>) -> [u8; 32] {
    #[repr(align(8))]
    struct Out([u8; 32]);
    let mut out = Out([0u8; 32]);
    // SAFETY: `out.0` is 32 writable, 8-aligned bytes.
    unsafe { keccak256_into(data.as_ref(), permute_syscall, out.0.as_mut_ptr()) };
    out.0
}

/// The keccak permutation, as a *named* function.
///
/// `keccak256_sponge_into` takes the permutation as an `impl Fn`, so it is monomorphized once
/// per closure type it is handed. `keccak` and `keccak_into` used to write out `|state|
/// syscall_keccak_permute(state)` each -- two closures, two distinct types, and therefore two
/// copies of the `#[inline(never)]` sponge in the ELF (they show up in the profile as two
/// symbols, 12.21 M and 3.01 M retired instructions on mainnet block 24006677). A `fn` item
/// has one type, so passing this collapses them into one.
#[cfg(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64"))]
#[inline(always)]
fn permute_syscall(state: &mut [u64; 25]) {
    extern "C" {
        fn syscall_keccak_permute(state: *mut [u64; 25]);
    }
    // SAFETY: `state` is a live, aligned `[u64; 25]`, which is what the syscall expects.
    unsafe { syscall_keccak_permute(state) }
}

/// Keccak-256 straight into `out`, for the callers that already own the 32-byte destination.
///
/// Returning `[u8; 32]` costs the sponge 52 retired instructions per call: the type's
/// alignment is 1, so LLVM writes the caller's slot with 28 `srli` + 32 `sb`, and keeping the
/// four state words live across that is what makes the function save and restore twelve
/// callee-saved registers. Handing the destination down instead lets the common case be four
/// `sd`.
///
/// # Safety
///
/// `out` must point at 32 writable bytes.
#[cfg(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64"))]
#[inline]
pub unsafe fn keccak_into(data: &[u8], out: *mut u8) {
    // SAFETY: forwarded from the caller.
    unsafe { keccak256_into(data, permute_syscall, out) }
}

/// The sponge's rate in bytes. An input shorter than this is one block and one permutation.
pub(crate) const RATE: usize = 136;

/// Keccak-256 into `out`, over a permute function.
///
/// The body is the single-block case, `data.len() < RATE`; longer inputs go to
/// [`keccak256_sponge_into`]. 66,283 of the 70,722 hashes on mainnet block 24006677 are
/// single-block: every EVM `KECCAK256` of a 64-byte mapping key, every hashed account
/// address and storage slot, every logs-bloom address and topic. The general sponge serves
/// them through machinery they never use, and the measured cost of that is mostly not the
/// branches:
///
/// * It saves and restores twelve callee-saved registers, 26 instructions per call, because
///   the multi-block absorb keeps seventeen state words live across the permute.
/// * The final block's padding word is read-modify-written (`ld`, `xor`, `sd`) and the
///   `0x80` terminator likewise, because either may land on a word an earlier block already
///   filled. On a single block both words are known to be zero, so both are plain stores.
/// * The block count, the 136-byte reciprocal and the multi-block loop's induction
///   variables are all materialised before the length is tested.
///
/// The length test lives here rather than in the inlined wrappers on purpose. Split at the
/// call sites it costs the two instructions of the test at each of the 66 of them plus the
/// register pressure of a second live call target: measured at +0.65 M retired instructions
/// against +0.14 M for one test inside one function.
///
/// # Safety
///
/// `out` must point at 32 writable bytes.
#[inline(never)]
#[allow(dead_code)]
pub(crate) unsafe fn keccak256_into(
    data: &[u8],
    permute: impl Fn(&mut [u64; 25]),
    out: *mut u8,
) {
    if data.len() >= RATE {
        // SAFETY: forwarded from the caller.
        return unsafe { keccak256_sponge_into(data, permute, out) };
    }

    let n = data.len();
    let p = data.as_ptr();
    let full = n / 8; // 0..=16, because `n <= RATE - 1`
    let t = n & 7;

    let mut state = core::mem::MaybeUninit::<[u64; 25]>::uninit();
    let q = state.as_mut_ptr().cast::<u64>();

    // SAFETY: every arm below writes all 25 words of `state` exactly once, so the reference
    // taken afterwards is to initialized memory. `p` points at `n` readable bytes and
    // `full * 8 + t == n`, with `full <= 16`, so no state index exceeds 16.
    unsafe {
        if p as usize & 7 == 0 && (t == 0 || n == 20) {
            // The two shapes that a mainnet block hashes over and over: a whole number of
            // words (a 32-byte topic or storage key, a 64-byte mapping slot, an RLP node
            // whose length happens to divide) and a 20-byte address. Both fill the state
            // from a constant-offset, fully unrolled `sd` run.
            if t == 0 {
                match full {
                    0 => fill_block::<0, 0>(q, p),
                    1 => fill_block::<1, 0>(q, p),
                    2 => fill_block::<2, 0>(q, p),
                    3 => fill_block::<3, 0>(q, p),
                    4 => fill_block::<4, 0>(q, p),
                    5 => fill_block::<5, 0>(q, p),
                    6 => fill_block::<6, 0>(q, p),
                    7 => fill_block::<7, 0>(q, p),
                    8 => fill_block::<8, 0>(q, p),
                    9 => fill_block::<9, 0>(q, p),
                    10 => fill_block::<10, 0>(q, p),
                    11 => fill_block::<11, 0>(q, p),
                    12 => fill_block::<12, 0>(q, p),
                    13 => fill_block::<13, 0>(q, p),
                    14 => fill_block::<14, 0>(q, p),
                    15 => fill_block::<15, 0>(q, p),
                    // `n <= RATE - 1 == 135`, so `full <= 16`.
                    _ => fill_block::<16, 0>(q, p),
                }
            } else {
                fill_block::<2, 4>(q, p);
            }
        } else {
            fill_block_dyn(q, p, full, t);
        }
        let state = &mut *state.as_mut_ptr();
        permute(state);
        // SAFETY: the caller guarantees 32 writable bytes at `out`. See
        // `keccak256_sponge_into` for why the alignment is a run-time test.
        squeeze_into(state, out);
    }
}

/// The `0x80` that terminates Keccak padding, in the last word of the rate.
const PAD_HI: u64 = 0x80u64 << 56;
/// The rate in `u64` words.
const RATE_WORDS: usize = RATE / 8; // 17

/// Fill all 25 words of the state for a single block of exactly `8 * W + T` bytes read from
/// the 8-aligned `p`.
///
/// With `W` and `T` constant, the absorbed words, the padding word, the zeros between it and
/// the `0x80` terminator and the whole capacity are one straight run of `sd` at constant
/// offsets: `25 + W` instructions. [`fill_block_dyn`] instead zeroes 25 words and then
/// re-writes the first `full` of them through a five-instruction loop body (`ld`, `sd`, two
/// `addi`, `bne`), which is `4 * W + 3` instructions more.
///
/// # Safety
///
/// `q` must point at 25 writable, 8-aligned `u64`; `p` must be 8-aligned and point at
/// `8 * W + T` readable bytes.
#[inline(always)]
#[allow(dead_code)]
unsafe fn fill_block<const W: usize, const T: usize>(q: *mut u64, p: *const u8) {
    // `W * 8 + T` is the input length, which must be below the rate. `W == RATE_WORDS - 1`
    // is allowed: the padding word and the terminator then share word 16.
    const {
        assert!(W < RATE_WORDS && T < 8 && W * 8 + T < RATE);
    }
    // Volatile throughout: a run of plain stores of this shape is LLVM's `memcpy`/`memset`
    // idiom and becomes a libcall, which measured +2.5 M retired instructions on mainnet
    // block 24006677 when the absorb was written that way.
    let s = p.cast::<u64>();
    let mut i = 0;
    while i < W {
        // SAFETY: `p` is 8-aligned with `8 * W` readable bytes, so `s[i]` is an aligned read
        // inside the input; `i < W < 25`.
        unsafe { q.add(i).write_volatile(s.add(i).read()) };
        i += 1;
    }
    // The padding byte 0x01 sits just past the input, preceded in its word by the `T`
    // trailing bytes.
    // SAFETY: the `T` bytes at `p + 8 * W` are the tail of the input.
    let pad = unsafe { tail_word::<T>(p.add(8 * W)) } | (1u64 << (8 * T));
    let mut i = W;
    while i < 25 {
        let v = if i == W && W == RATE_WORDS - 1 {
            pad | PAD_HI
        } else if i == W {
            pad
        } else if i == RATE_WORDS - 1 {
            PAD_HI
        } else {
            0
        };
        // SAFETY: `i < 25`.
        unsafe { q.add(i).write_volatile(v) };
        i += 1;
    }
}

/// The `T` bytes at `p`, little-endian, in the low `8 * T` bits.
///
/// # Safety
///
/// `p` must point at `T` readable bytes, and at 4 `4`-aligned bytes when `T == 4`.
#[inline(always)]
#[allow(dead_code)]
unsafe fn tail_word<const T: usize>(p: *const u8) -> u64 {
    // `T == 4` is the 20-byte address shape; the callers only reach it with `p` a multiple of
    // 8 bytes past an 8-aligned base, so the `u32` read is aligned. Four `lbu` plus six
    // shift/or become one `lwu`.
    if T == 4 {
        // SAFETY: 4 readable, 4-aligned bytes, per the contract above.
        return u64::from(unsafe { p.cast::<u32>().read() });
    }
    let mut v = 0u64;
    let mut j = 0;
    while j < T {
        // SAFETY: `j < T` and `p` has `T` readable bytes.
        v |= u64::from(unsafe { *p.add(j) }) << (8 * j);
        j += 1;
    }
    v
}

/// [`fill_block`] for a length and an input alignment that are only known at run time.
///
/// # Safety
///
/// `q` must point at 25 writable, 8-aligned `u64`; `p` must point at `full * 8 + t` readable
/// bytes with `full <= RATE_WORDS - 1` and `t < 8`.
#[inline]
#[allow(dead_code)]
unsafe fn fill_block_dyn(q: *mut u64, p: *const u8, full: usize, t: usize) {
    // `[0u64; 25]` is 200 bytes, which LLVM zeroes with a `memset` call — ~60 instructions,
    // paid once per hash. Volatile stores keep it as 25 `sd` with constant offsets. Zeroing
    // all of it and then overwriting the first `full` words costs `full` stores more than
    // zeroing only the words the absorb does not fill, but that shorter zero run has a
    // run-time trip count, so it is a four-instruction loop body per word rather than one
    // `sd` — worse for every `full` below about 20, and `full <= 16` here.
    // SAFETY: 25 in-bounds writes.
    unsafe {
        for i in 0..25 {
            q.add(i).write_volatile(0);
        }
    }
    // SAFETY: `q` is a live, aligned, now-initialized `[u64; 25]`.
    let state = unsafe { &mut *q.cast::<[u64; 25]>() };
    // SAFETY: the `full` whole words and the `t` trailing bytes together are the input, and
    // `full < RATE_WORDS`, so every state index written below is in `0..17`.
    unsafe {
        if full != 0 {
            // Assign rather than xor: the state is still all zeros. See `absorb_words`, whose
            // `FIRST` arm this is, including the `write_volatile`.
            absorb_first_words(state, p, full);
        }
        let mut last = 1u64 << (8 * t);
        for j in 0..t {
            last |= u64::from(*p.add(full * 8 + j)) << (8 * j);
        }
        // `full == RATE_WORDS - 1` is the one case where the padding byte and the `0x80`
        // terminator share a word (an input of 128..=135 bytes).
        if full == RATE_WORDS - 1 {
            *state.get_unchecked_mut(RATE_WORDS - 1) = last | PAD_HI;
        } else {
            *state.get_unchecked_mut(full) = last;
            *state.get_unchecked_mut(RATE_WORDS - 1) = PAD_HI;
        }
    }
}

/// Assign the `k` little-endian `u64` words held in the `8 * k` bytes at `p` to `state[..k]`.
///
/// This is [`keccak256_sponge_into`]'s `absorb_words::<true>` without the xor arm; see the
/// comments there for why the loop must not be unrolled and why the stores are volatile.
///
/// # Safety
///
/// `p` must point at `8 * k` readable bytes and `k` must be in `1..=17`.
#[inline]
#[allow(dead_code)]
pub(crate) unsafe fn absorb_first_words(state: &mut [u64; 25], p: *const u8, k: usize) {
    /// # Safety
    /// `i` must be less than 25.
    #[inline(always)]
    unsafe fn put(state: &mut [u64; 25], i: usize, v: u64) {
        // SAFETY: caller guarantees `i < 25`.
        let slot = unsafe { state.get_unchecked_mut(i) };
        // SAFETY: `slot` is a live, aligned `u64`.
        unsafe { core::ptr::write_volatile(slot, v) };
    }
    let off = p as usize & 7;
    if off == 0 {
        let q = p.cast::<u64>();
        for i in 0..k {
            put(state, i, q.add(i).read());
        }
        return;
    }
    let r = 8 - off;
    let sl = (r * 8) as u32;
    let sr = (off * 8) as u32;
    let mut head = 0u64;
    for j in 0..r {
        head |= (*p.add(j) as u64) << (8 * j);
    }
    let mut tail = 0u64;
    for j in 0..off {
        tail |= (*p.add(8 * k - off + j) as u64) << (8 * j);
    }
    let a = p.add(r).cast::<u64>();
    if k == 1 {
        put(state, 0, head | (tail << sl));
        return;
    }
    let mut prev = a.read();
    put(state, 0, head | (prev << sl));
    for i in 1..k - 1 {
        let cur = a.add(i).read();
        put(state, i, (prev >> sr) | (cur << sl));
        prev = cur;
    }
    put(state, k - 1, (prev >> sr) | (tail << sl));
}

/// Write the first 32 bytes of the squeezed state to `out`.
///
/// # Safety
///
/// `out` must point at 32 writable bytes.
#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn squeeze_into(state: &[u64; 25], out: *mut u8) {
    // RV64 is little-endian, so an aligned `u64` store writes the same bytes as
    // `to_le_bytes`. Digests land in a `B256`, whose alignment is 1 as far as the compiler is
    // concerned but which is in practice a stack slot the backend has aligned; check at run
    // time rather than pay 60 instructions of byte scatter for it.
    unsafe {
        if (out as usize).is_multiple_of(core::mem::align_of::<u64>()) {
            let o = out.cast::<u64>();
            o.write(state[0]);
            o.add(1).write(state[1]);
            o.add(2).write(state[2]);
            o.add(3).write(state[3]);
            return;
        }
        let mut i = 0;
        while i < 4 {
            let w = *state.get_unchecked(i);
            let b = out.add(i * 8);
            b.write(w as u8);
            b.add(1).write((w >> 8) as u8);
            b.add(2).write((w >> 16) as u8);
            b.add(3).write((w >> 24) as u8);
            b.add(4).write((w >> 32) as u8);
            b.add(5).write((w >> 40) as u8);
            b.add(6).write((w >> 48) as u8);
            b.add(7).write((w >> 56) as u8);
            i += 1;
        }
    }
}

/// Keccak-256 returned by value, through [`keccak256_into`]. Test helper.
#[inline]
#[allow(dead_code)]
pub(crate) fn keccak256_block(data: &[u8], permute: impl Fn(&mut [u64; 25])) -> [u8; 32] {
    #[repr(align(8))]
    struct Out([u8; 32]);
    let mut out = Out([0u8; 32]);
    // SAFETY: `out.0` is 32 writable, 8-aligned bytes.
    unsafe { keccak256_into(data, permute, out.0.as_mut_ptr()) };
    out.0
}

/// Keccak-256 sponge (rate 136 bytes, Keccak padding 0x01/0x80) over a permute function.
///
/// RV64IM has no misaligned scalar loads, so anything that reads the input as `u64` through a
/// byte pointer of unknown alignment is expanded by LLVM into `lbu` + shift/or chains (measured
/// at ~650 instructions for a single 136-byte block). Instead every `u64` is assembled from the
/// two *aligned* words that contain it, and the 8 bytes at the two ends of the region — the only
/// ones no fully-contained aligned word covers — are read with `lbu`. Nothing outside the slice
/// is ever touched.
///
/// The final (partial) block is absorbed in place: only the `ceil((n+1)/8)` words that can be
/// non-zero are XORed, so a short input no longer pays for zeroing and copying a 136-byte
/// staging buffer and XORing 17 words.
#[inline]
#[allow(dead_code)]
pub(crate) fn keccak256_sponge(data: &[u8], permute: impl Fn(&mut [u64; 25])) -> [u8; 32] {
    #[repr(align(8))]
    struct Out([u8; 32]);
    let mut out = Out([0u8; 32]);
    // SAFETY: `out.0` is 32 writable, 8-aligned bytes.
    unsafe { keccak256_sponge_into(data, permute, out.0.as_mut_ptr()) };
    out.0
}

/// The sponge itself; writes the 32-byte digest to `out`.
///
/// # Safety
///
/// `out` must point at 32 writable bytes.
/// Deliberately out of line: it is entered from both `keccak` and `keccak_into`, and a second
/// copy of a body this size costs more in register pressure than the call saves.
#[inline(never)]
#[allow(dead_code)]
pub(crate) unsafe fn keccak256_sponge_into(
    data: &[u8],
    permute: impl Fn(&mut [u64; 25]),
    out: *mut u8,
) {

    /// XOR the `k` little-endian `u64` words held in the `8 * k` bytes at `p` into `state[..k]`.
    ///
    /// With `FIRST`, assign instead: the state is still all zeros when the first block is
    /// absorbed, so the load and the xor of each word are dead - two instructions per word,
    /// and up to 17 words per call.
    ///
    /// # Safety
    /// `p` must point at `8 * k` readable bytes and `k` must be in `1..=RATE_WORDS`.
    #[inline]
    unsafe fn absorb_words<const FIRST: bool>(state: &mut [u64; 25], p: *const u8, k: usize) {
        /// `state[i] ^= v`, or `state[i] = v` on the first block.
        ///
        /// # Safety
        /// `i` must be less than 25.
        #[inline(always)]
        unsafe fn put<const FIRST: bool>(state: &mut [u64; 25], i: usize, v: u64) {
            // SAFETY: caller guarantees `i < 25`.
            let slot = unsafe { state.get_unchecked_mut(i) };
            if FIRST {
                // Volatile: a plain `state[i] = *src.add(i)` loop is exactly LLVM's memcpy
                // idiom, and it turns the aligned first block into a `memcpy` libcall -
                // measured at +2.5 M retired instructions, more than the assignment saves.
                // SAFETY: `slot` is a live, aligned `u64`.
                unsafe { core::ptr::write_volatile(slot, v) };
            } else {
                *slot ^= v;
            }
        }
        let off = p as usize & 7;
        if off == 0 {
            // Do NOT unroll this. The loop is `ld`, `sd`, two `addi` and a branch per word,
            // and a four-at-a-time version with a tail ladder is fewer instructions on paper.
            // Measured, it is worse by 240 K retired instructions on mainnet block 24006677:
            // the bigger body stops `absorb_words` being inlined into the sponge, and the two
            // out-of-line copies that appear then cost more than the addressing saves.
            let q = p.cast::<u64>();
            for i in 0..k {
                put::<FIRST>(state, i, q.add(i).read());
            }
            return;
        }
        let r = 8 - off;
        let sl = (r * 8) as u32;
        let sr = (off * 8) as u32;
        // The `r` bytes before the first 8-byte boundary inside the region ...
        let mut head = 0u64;
        for j in 0..r {
            head |= (*p.add(j) as u64) << (8 * j);
        }
        // ... and the `off` bytes after the last aligned word fully inside it.
        let mut tail = 0u64;
        for j in 0..off {
            tail |= (*p.add(8 * k - off + j) as u64) << (8 * j);
        }
        // `a[i]` spans bytes `[r + 8i, r + 8i + 8)`, so `a[..k - 1]` stays inside the region.
        let a = p.add(r).cast::<u64>();
        if k == 1 {
            put::<FIRST>(state, 0, head | (tail << sl));
            return;
        }
        let mut prev = a.read();
        put::<FIRST>(state, 0, head | (prev << sl));
        for i in 1..k - 1 {
            let cur = a.add(i).read();
            put::<FIRST>(state, i, (prev >> sr) | (cur << sl));
            prev = cur;
        }
        put::<FIRST>(state, k - 1, (prev >> sr) | (tail << sl));
    }

    // `[0u64; 25]` is 200 bytes, which LLVM zeroes with a `memset` call — ~60 instructions,
    // paid once per hash. Volatile stores keep it as 25 `sd`.
    let mut state = core::mem::MaybeUninit::<[u64; 25]>::uninit();
    // SAFETY: the 25 in-bounds writes initialize every word of the array.
    let state = unsafe {
        let q = state.as_mut_ptr().cast::<u64>();
        for i in 0..25 {
            q.add(i).write_volatile(0);
        }
        &mut *state.as_mut_ptr()
    };
    let p = data.as_ptr();
    // 99.5 % of the calls on a mainnet block hash fewer than `RATE` bytes -- every EVM topic,
    // address and account hash -- so the division by 136 lives inside the branch. Left
    // outside it costs those calls the four instructions that materialise the magic
    // multiplier plus the `mulhu`/`srli`/`mul` that use it, for a quotient that is zero.
    let mut absorbed = 0usize;
    if data.len() >= RATE {
        let nblocks = data.len() / RATE;
        let off = p as usize & 7;
        // `RATE` is 17 whole words, so every block of one input shares `p`'s alignment. When
        // that alignment is not 8, absorbing each block on its own makes it pay two byte
        // loops -- the `8 - off` bytes before its first fully-contained aligned word, and the
        // `off` bytes after its last -- and that measured 855 K retired instructions over the
        // 15,552 unaligned block absorbs on mainnet block 24006677 (66 % of the block
        // absorbs; the long inputs are RLP node blobs at arbitrary offsets in the witness).
        //
        // The absorbed region is contiguous, so one shifted stream over it needs the leading
        // byte loop once and no trailing one at all: the aligned word that supplies the end
        // of block `b` is the one that supplies the start of block `b + 1`.
        //
        // The stream reads the aligned word at `p + r + 8 * (17 * nblocks - 1)`, whose last
        // byte is at offset `RATE * nblocks + r - 1`; it therefore needs `r` bytes of input
        // beyond the absorbed region, and `r <= 7`. Otherwise fall back per block.
        let r = 8 - off;
        if off != 0 && data.len() - nblocks * RATE >= r {
            /// Absorb one block of 17 words from the shifted stream.
            ///
            /// `carry` holds the low `r` bytes of the next word -- the same position
            /// `cur >> sr` leaves them in -- so word 0 of the input and every later word
            /// share one formula.
            ///
            /// # Safety
            ///
            /// `a` must point at 17 readable, 8-aligned `u64`, and `sl + sr` must be 64 with
            /// both below 64.
            #[inline(always)]
            unsafe fn absorb_shifted<const FIRST: bool>(
                state: &mut [u64; 25],
                a: *const u64,
                carry: &mut u64,
                sl: u32,
                sr: u32,
            ) {
                let mut c = *carry;
                // Constant trip count, so this is 17 loads and 17 stores at constant
                // offsets, the same shape the per-block version already had.
                for i in 0..RATE / 8 {
                    // SAFETY: `a` has 17 readable aligned words; `i < 17 < 25`.
                    let cur = unsafe { a.add(i).read() };
                    let w = c | (cur << sl);
                    c = cur >> sr;
                    // SAFETY: `i < 25`.
                    let slot = unsafe { state.get_unchecked_mut(i) };
                    if FIRST {
                        // Untouched state: assign. Volatile for the reason recorded on
                        // `absorb_words` -- a plain store loop here is LLVM's memcpy idiom.
                        // SAFETY: `slot` is a live, aligned `u64`.
                        unsafe { core::ptr::write_volatile(slot, w) };
                    } else {
                        *slot ^= w;
                    }
                }
                *carry = c;
            }

            let sl = (r * 8) as u32;
            let sr = (off * 8) as u32;
            // SAFETY: `p` has `RATE * nblocks + r` readable bytes by the guard above, so
            // every aligned word `p + r + 8 * j` for `j < 17 * nblocks` is readable, and
            // `p + j` for `j < r` is inside the input.
            unsafe {
                let mut carry = 0u64;
                for j in 0..r {
                    carry |= u64::from(*p.add(j)) << (8 * j);
                }
                let mut a = p.add(r).cast::<u64>();
                // `nblocks >= 1`, because `data.len() >= RATE`.
                absorb_shifted::<true>(state, a, &mut carry, sl, sr);
                permute(state);
                for _ in 1..nblocks {
                    a = a.add(RATE / 8);
                    absorb_shifted::<false>(state, a, &mut carry, sl, sr);
                    permute(state);
                }
            }
        } else {
            for b in 0..nblocks {
                // SAFETY: block `b` is the `RATE` bytes at `p + b * RATE`, inside `data`.
                unsafe {
                    if b == 0 {
                        absorb_words::<true>(state, p, RATE_WORDS);
                    } else {
                        absorb_words::<false>(state, p.add(b * RATE), RATE_WORDS);
                    }
                }
                permute(state);
            }
        }
        absorbed = nblocks * RATE;
    }

    // Final (partial) block with padding; also covers empty input and exact multiples.
    let rem = data.len() - absorbed; // 0..RATE
    let full = rem / 8;
    let t = rem & 7;
    // SAFETY: the remainder is the `rem` bytes at the end of `data`, and `full * 8 + t == rem`.
    unsafe {
        let q = p.add(absorbed);
        // Untouched state when this is also the first block: assign rather than xor, and
        // then `state[full]` is still zero so the padding word can be assigned too.
        let first = absorbed == 0;
        if full != 0 {
            if first {
                absorb_words::<true>(state, q, full);
            } else {
                absorb_words::<false>(state, q, full);
            }
        }
        let mut last = 1u64 << (8 * t);
        for j in 0..t {
            last |= (*q.add(full * 8 + j) as u64) << (8 * j);
        }
        // `full <= RATE_WORDS - 1` because `rem < RATE`.
        if first {
            *state.get_unchecked_mut(full) = last;
        } else {
            *state.get_unchecked_mut(full) ^= last;
        }
    }
    state[RATE_WORDS - 1] ^= 0x80u64 << 56;
    permute(state);

    // SAFETY: the caller guarantees 32 writable bytes at `out`.
    unsafe { squeeze_into(state, out) };
}

/// Keccak-256 of `data` written into an existing `B256`.
///
/// The trie-verification pass hashes every node blob and then either compares the digest
/// against a reference or stores it; going through a `[u8; 32]` return value costs it the
/// 28 `srli` + 32 `sb` scatter described on [`keccak_into`].
#[inline]
pub(crate) fn keccak_into_b256(data: &[u8], out: &mut B256) {
    #[cfg(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64"))]
    // SAFETY: `out` is a live `B256`, i.e. 32 writable bytes.
    unsafe {
        keccak_into(data, out.0.as_mut_ptr())
    };
    #[cfg(not(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64")))]
    {
        *out = B256::from(keccak(data));
    }
}

/// Keccak-256 for the Pico zkVM guest via the permute syscall; provided so the guest binary can
/// export it as alloy's `native-keccak` hook (routing all alloy keccak256 calls — EVM opcodes,
/// transaction hashing, receipts root — through the direct sponge).
#[cfg(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64"))]
pub fn keccak256_zkvm(data: &[u8]) -> [u8; 32] {
    keccak(data)
}

/// Keccak-256 for the Pico zkVM guest, written straight into `out`. See [`keccak_into`].
///
/// # Safety
///
/// `out` must point at 32 writable bytes.
#[cfg(all(target_os = "zkvm", target_vendor = "pico", target_arch = "riscv64"))]
pub unsafe fn keccak256_zkvm_into(data: &[u8], out: *mut u8) {
    // SAFETY: forwarded from the caller.
    unsafe { keccak_into(data, out) }
}

/// Represents the root node of a sparse Merkle Patricia Trie.
///
/// The "sparse" nature of this trie allows for truncation of certain unneeded parts,
/// representing them by their node hash. This design choice is particularly useful for
/// optimizing storage. However, operations targeting a truncated part will fail and
/// return an error. Another distinction of this implementation is that branches cannot
/// store values, aligning with the construction of MPTs in Ethereum.
#[derive(Default, Serialize, Deserialize)]
pub struct MptNode {
    /// The type and data of the node.
    data: MptNodeData,
    /// Cache for a previously computed reference of this node. This is skipped during
    /// serialization.
    #[serde(skip)]
    cached_reference: Mutex<Option<MptNodeReference>>,
}

impl Ord for MptNode {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.data.cmp(&other.data)
    }
}

impl PartialOrd for MptNode {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for MptNode {}

impl PartialEq for MptNode {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Clone for MptNode {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            cached_reference: Mutex::new(self.cached_reference.lock().unwrap().clone()),
        }
    }
}

/// Represents custom error types for the sparse Merkle Patricia Trie (MPT).
///
/// These errors cover various scenarios that can occur during trie operations, such as
/// encountering unresolved nodes, finding values in branches where they shouldn't be, and
/// issues related to RLP (Recursive Length Prefix) encoding and decoding.
#[derive(Debug, ThisError)]
pub enum Error {
    /// Triggered when an operation reaches an unresolved node. The associated `B256`
    /// value provides details about the unresolved node.
    #[error("reached an unresolved node: {0:#}")]
    NodeNotResolved(B256),
    /// Occurs when a value is unexpectedly found in a branch node.
    #[error("branch node with value")]
    ValueInBranch,
    /// Represents errors related to the RLP encoding and decoding using the `alloy_rlp`
    /// library.
    #[error("RLP error")]
    Rlp(#[from] alloy_rlp::Error),
    /// Represents errors related to the RLP encoding and decoding, specifically legacy
    /// errors.
    #[error("RLP error")]
    LegacyRlp(#[from] DecoderError),
    /// A malformed flat-RLP trie wire encoding.
    #[error("malformed flat trie: {0}")]
    FlatTrie(&'static str),
}

/// Represents the various types of data that can be stored within a node in the sparse
/// Merkle Patricia Trie (MPT).
///
/// Each node in the trie can be of one of several types, each with its own specific data
/// structure. This enum provides a clear and type-safe way to represent the data
/// associated with each node type.
#[derive(Clone, Default, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum MptNodeData {
    /// Represents an empty trie node.
    #[default]
    Null,
    /// A node that can have up to 16 children. Each child is an optional boxed [MptNode].
    Branch([Option<Box<MptNode>>; 16]),
    /// A leaf node that contains a key and a value, both represented as byte vectors.
    Leaf(Vec<u8>, Vec<u8>),
    /// A node that has exactly one child and is used to represent a shared prefix of
    /// several keys.
    Extension(Vec<u8>, Box<MptNode>),
    /// Represents a sub-trie by its hash, allowing for efficient storage of large
    /// sub-tries without storing their entire content.
    Digest(B256),
}

/// Represents the ways in which one node can reference another node inside the sparse
/// Merkle Patricia Trie (MPT).
///
/// Nodes in the MPT can reference other nodes either directly through their byte
/// representation or indirectly through a hash of their encoding. This enum provides a
/// clear and type-safe way to represent these references.
#[derive(Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum MptNodeReference {
    /// Represents a direct reference to another node using its byte encoding. Typically
    /// used for short encodings that are less than 32 bytes in length.
    Bytes(Vec<u8>),
    /// Represents an indirect reference to another node using the Keccak hash of its long
    /// encoding. Used for encodings that are not less than 32 bytes in length.
    Digest(B256),
}

/// Provides a conversion from [MptNodeData] to [MptNode].
///
/// This implementation allows for conversion from [MptNodeData] to [MptNode],
/// initializing the `data` field with the provided value and setting the
/// `cached_reference` field to `None`.
impl From<MptNodeData> for MptNode {
    fn from(value: MptNodeData) -> Self {
        Self { data: value, cached_reference: Mutex::new(None) }
    }
}

impl core::fmt::Debug for MptNodeReference {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MptNodeReference::Bytes(b) => {
                write!(f, "Ref::Bytes({})", alloy_primitives::hex::encode(b))
            }
            MptNodeReference::Digest(h) => {
                write!(f, "Ref::Digest({})", alloy_primitives::hex::encode(h.as_slice()))
            }
        }
    }
}

impl core::fmt::Debug for MptNodeData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MptNodeData::Null => write!(f, "Null"),
            MptNodeData::Leaf(k, v) => write!(
                f,
                "Leaf(key={}, value={})",
                alloy_primitives::hex::encode(k),
                alloy_primitives::hex::encode(v)
            ),
            MptNodeData::Extension(k, child) => f
                .debug_struct("Extension")
                .field("key", &alloy_primitives::hex::encode(k))
                .field("child", child)
                .finish(),
            MptNodeData::Digest(h) => write!(f, "Digest({})", alloy_primitives::hex::encode(h)),
            MptNodeData::Branch(children) => {
                let mut ds = f.debug_struct("Branch");
                for (i, child) in children.iter().enumerate() {
                    if let Some(c) = child {
                        ds.field(&format!("child_{i}"), c);
                    }
                }
                ds.finish()
            }
        }
    }
}

impl core::fmt::Debug for MptNode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut ds = f.debug_struct("MptNode");
        ds.field("data", &self.data);
        if let Ok(guard) = self.cached_reference.lock() {
            if let Some(reference) = guard.as_ref() {
                ds.field("cached_reference", reference);
            }
        }
        ds.finish()
    }
}

/// Provides encoding functionalities for the `MptNode` type.
///
/// This implementation allows for the serialization of an [MptNode] into its RLP-encoded
/// form. The encoding is done based on the type of node data ([MptNodeData]) it holds.
impl Encodable for MptNode {
    /// Encodes the node into the provided `out` buffer.
    ///
    /// The encoding is done using the Recursive Length Prefix (RLP) encoding scheme. The
    /// method handles different node data types and encodes them accordingly.
    #[inline]
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        match &self.data {
            MptNodeData::Null => {
                out.put_u8(alloy_rlp::EMPTY_STRING_CODE);
            }
            MptNodeData::Branch(nodes) => {
                alloy_rlp::Header { list: true, payload_length: self.payload_length() }.encode(out);
                nodes.iter().for_each(|child| match child {
                    Some(node) => node.reference_encode(out),
                    None => out.put_u8(alloy_rlp::EMPTY_STRING_CODE),
                });
                // in the MPT reference, branches have values so always add empty value
                out.put_u8(alloy_rlp::EMPTY_STRING_CODE);
            }
            MptNodeData::Leaf(prefix, value) => {
                alloy_rlp::Header { list: true, payload_length: self.payload_length() }.encode(out);
                prefix.as_slice().encode(out);
                value.as_slice().encode(out);
            }
            MptNodeData::Extension(prefix, node) => {
                alloy_rlp::Header { list: true, payload_length: self.payload_length() }.encode(out);
                prefix.as_slice().encode(out);
                node.reference_encode(out);
            }
            MptNodeData::Digest(digest) => {
                digest.encode(out);
            }
        }
    }

    /// Returns the length of the encoded node in bytes.
    ///
    /// This method calculates the length of the RLP-encoded node. It's useful for
    /// determining the size requirements for storage or transmission.
    #[inline]
    fn length(&self) -> usize {
        let payload_length = self.payload_length();
        payload_length + alloy_rlp::length_of_length(payload_length)
    }
}

/// Provides decoding functionalities for the [MptNode] type.
///
/// This implementation allows for the deserialization of an RLP-encoded [MptNode] back
/// into its original form. The decoding is done based on the prototype of the RLP data,
/// ensuring that the node is reconstructed accurately.
///
/// **Note**: This implementation is still using the older RLP library and needs to be
/// migrated to `alloy_rlp` in the future.
// TODO: migrate to alloy_rlp
impl Decodable for MptNode {
    /// Decodes an RLP-encoded node from the provided `rlp` buffer.
    ///
    /// The method handles different RLP prototypes and reconstructs the `MptNode` based
    /// on the encoded data. If the RLP data does not match any known prototype or if
    /// there's an error during decoding, an error is returned.
    fn decode(rlp: &Rlp<'_>) -> Result<Self, DecoderError> {
        match rlp.prototype()? {
            Prototype::Null | Prototype::Data(0) => Ok(MptNodeData::Null.into()),
            Prototype::List(2) => {
                let path: Vec<u8> = rlp.val_at(0)?;
                let prefix = path[0];
                if (prefix & (2 << 4)) == 0 {
                    let node: MptNode = Decodable::decode(&rlp.at(1)?)?;
                    Ok(MptNodeData::Extension(path, Box::new(node)).into())
                } else {
                    Ok(MptNodeData::Leaf(path, rlp.val_at(1)?).into())
                }
            }
            Prototype::List(17) => {
                let mut node_list = Vec::with_capacity(16);
                for node_rlp in rlp.iter().take(16) {
                    match node_rlp.prototype()? {
                        Prototype::Null | Prototype::Data(0) => {
                            node_list.push(None);
                        }
                        _ => node_list.push(Some(Box::new(Decodable::decode(&node_rlp)?))),
                    }
                }
                let value: Vec<u8> = rlp.val_at(16)?;
                if value.is_empty() {
                    Ok(MptNodeData::Branch(node_list.try_into().unwrap()).into())
                } else {
                    Err(DecoderError::Custom("branch node with value"))
                }
            }
            Prototype::Data(32) => {
                let bytes: Vec<u8> = rlp.as_val()?;
                Ok(MptNodeData::Digest(B256::from_slice(&bytes)).into())
            }
            _ => Err(DecoderError::RlpIncorrectListLen),
        }
    }
}

/// Represents a node in the sparse Merkle Patricia Trie (MPT).
///
/// The [MptNode] type encapsulates the data and functionalities associated with a node in
/// the MPT. It provides methods for manipulating the trie, such as inserting, deleting,
/// and retrieving values, as well as utility methods for encoding, decoding, and
/// debugging.
impl MptNode {
    /// Creates a Merkle Patricia trie from an EIP-1186 proof.
    pub fn from_account_proof(account_proof: &[impl AsRef<[u8]>]) -> Result<Self, FromProofError> {
        let nodes = parse_proof(account_proof)?;
        mpt_from_proof(&nodes)
    }

    /// Clears the trie, replacing its data with an empty node, [MptNodeData::Null].
    ///
    /// This method effectively removes all key-value pairs from the trie.
    #[inline]
    pub fn clear(&mut self) {
        self.data = MptNodeData::Null;
        self.invalidate_ref_cache();
    }

    /// Decodes an RLP-encoded [MptNode] from the provided byte slice.
    ///
    /// This method allows for the deserialization of a previously serialized [MptNode].
    #[inline]
    pub fn decode(bytes: impl AsRef<[u8]>) -> Result<MptNode, Error> {
        rlp::decode(bytes.as_ref()).map_err(Error::from)
    }

    /// Retrieves the underlying data of the node.
    ///
    /// This method provides a reference to the node's data, allowing for inspection and
    /// manipulation.
    #[inline]
    pub fn as_data(&self) -> &MptNodeData {
        &self.data
    }

    /// Retrieves the [MptNodeReference] reference of the node when it's referenced inside
    /// another node.
    ///
    /// This method provides a way to obtain a compact representation of the node for
    /// storage or transmission purposes.
    #[inline]
    pub fn reference(&self) -> MptNodeReference {
        self.cached_reference.lock().unwrap().get_or_insert_with(|| self.calc_reference()).clone()
    }

    pub fn for_each_leaves<F: FnMut(&[u8], &[u8])>(&self, mut f: F) {
        let mut stack = vec![(self, Nibbles::default())];

        while let Some((node, path)) = stack.pop() {
            match node.as_data() {
                MptNodeData::Null | MptNodeData::Digest(_) => (),
                MptNodeData::Branch(branch) => {
                    for (i, n) in
                        branch.iter().enumerate().filter_map(|(i, n)| n.as_ref().map(|n| (i, n)))
                    {
                        let mut new_path = path;
                        new_path.push(i as u8);
                        stack.push((n, new_path));
                    }
                }
                MptNodeData::Leaf(prefix, value) => {
                    let mut full_path = path;
                    full_path.extend(&Nibbles::from_nibbles(prefix_nibs(prefix)));
                    f(&full_path.pack(), value)
                }
                MptNodeData::Extension(prefix, node) => {
                    let mut new_path = path;
                    new_path.extend(&Nibbles::from_nibbles(prefix_nibs(prefix)));
                    stack.push((node, new_path));
                }
            }
        }
    }

    /// Computes and returns the 256-bit hash of the node.
    ///
    /// This method provides a unique identifier for the node based on its content.
    #[inline]
    pub fn hash(&self) -> B256 {
        match self.data {
            MptNodeData::Null => EMPTY_ROOT,
            _ => match self.reference() {
                MptNodeReference::Digest(digest) => digest,
                MptNodeReference::Bytes(bytes) => keccak(bytes).into(),
            },
        }
    }

    /// Encodes the [MptNodeReference] of this node into the `out` buffer.
    fn reference_encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        match self.reference() {
            // if the reference is an RLP-encoded byte slice, copy it directly
            MptNodeReference::Bytes(bytes) => out.put_slice(&bytes),
            // if the reference is a digest, RLP-encode it with its fixed known length
            MptNodeReference::Digest(digest) => {
                out.put_u8(alloy_rlp::EMPTY_STRING_CODE + 32);
                out.put_slice(digest.as_slice());
            }
        }
    }

    /// Returns the length of the encoded [MptNodeReference] of this node.
    fn reference_length(&self) -> usize {
        match self.reference() {
            MptNodeReference::Bytes(bytes) => bytes.len(),
            MptNodeReference::Digest(_) => 1 + 32,
        }
    }

    fn calc_reference(&self) -> MptNodeReference {
        match &self.data {
            MptNodeData::Null => MptNodeReference::Bytes(vec![alloy_rlp::EMPTY_STRING_CODE]),
            MptNodeData::Digest(digest) => MptNodeReference::Digest(*digest),
            _ => {
                let encoded = alloy_rlp::encode(self);
                if encoded.len() < 32 {
                    MptNodeReference::Bytes(encoded)
                } else {
                    MptNodeReference::Digest(keccak(encoded).into())
                }
            }
        }
    }

    /// Determines if the trie is empty.
    ///
    /// This method checks if the node represents an empty trie, i.e., it doesn't contain
    /// any key-value pairs.
    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(&self.data, MptNodeData::Null)
    }

    /// Determines if the node represents a digest.
    ///
    /// A digest is a compact representation of a sub-trie, represented by its hash.
    #[inline]
    pub fn is_digest(&self) -> bool {
        matches!(&self.data, MptNodeData::Digest(_))
    }

    /// Retrieves the nibbles corresponding to the node's prefix.
    ///
    /// Nibbles are half-bytes, and in the context of the MPT, they represent parts of
    /// keys.
    #[inline]
    pub fn nibs(&self) -> Vec<u8> {
        match &self.data {
            MptNodeData::Null | MptNodeData::Branch(_) | MptNodeData::Digest(_) => vec![],
            MptNodeData::Leaf(prefix, _) | MptNodeData::Extension(prefix, _) => prefix_nibs(prefix),
        }
    }

    /// Retrieves the value associated with a given key in the trie.
    ///
    /// If the key is not present in the trie, this method returns `None`. Otherwise, it
    /// returns a reference to the associated value. If [None] is returned, the key is
    /// provably not in the trie.
    #[inline]
    pub fn get(&self, key: &[u8]) -> Result<Option<&[u8]>, Error> {
        self.get_internal(&to_nibs(key))
    }

    /// Retrieves the RLP-decoded value corresponding to the key.
    ///
    /// If the key is not present in the trie, this method returns `None`. Otherwise, it
    /// returns the RLP-decoded value.
    #[inline]
    pub fn get_rlp<T: alloy_rlp::Decodable>(&self, key: &[u8]) -> Result<Option<T>, Error> {
        match self.get(key)? {
            Some(mut bytes) => Ok(Some(T::decode(&mut bytes)?)),
            None => Ok(None),
        }
    }

    fn get_internal(&self, key_nibs: &[u8]) -> Result<Option<&[u8]>, Error> {
        match &self.data {
            MptNodeData::Null => Ok(None),
            MptNodeData::Branch(nodes) => {
                if let Some((i, tail)) = key_nibs.split_first() {
                    match nodes[*i as usize] {
                        Some(ref node) => node.get_internal(tail),
                        None => Ok(None),
                    }
                } else {
                    Ok(None)
                }
            }
            MptNodeData::Leaf(prefix, value) => {
                if prefix_nibs(prefix) == key_nibs {
                    Ok(Some(value))
                } else {
                    Ok(None)
                }
            }
            MptNodeData::Extension(prefix, node) => {
                if let Some(tail) = key_nibs.strip_prefix(prefix_nibs(prefix).as_slice()) {
                    node.get_internal(tail)
                } else {
                    Ok(None)
                }
            }
            MptNodeData::Digest(_digest) => Ok(None),
        }
    }

    /// Removes a key from the trie.
    ///
    /// This method attempts to remove a key-value pair from the trie. If the key is
    /// present, it returns `true`. Otherwise, it returns `false`.
    #[inline]
    pub fn delete(&mut self, key: &[u8]) -> Result<bool, Error> {
        self.delete_internal(&to_nibs(key))
    }

    fn delete_internal(&mut self, key_nibs: &[u8]) -> Result<bool, Error> {
        match &mut self.data {
            MptNodeData::Null => return Ok(false),
            MptNodeData::Branch(children) => {
                if let Some((i, tail)) = key_nibs.split_first() {
                    let child = &mut children[*i as usize];
                    match child {
                        Some(node) => {
                            if !node.delete_internal(tail)? {
                                return Ok(false);
                            }
                            // if the node is now empty, remove it
                            if node.is_empty() {
                                *child = None;
                            }
                        }
                        None => return Ok(false),
                    }
                } else {
                    return Err(Error::ValueInBranch);
                }

                let mut remaining = children.iter_mut().enumerate().filter(|(_, n)| n.is_some());
                // there will always be at least one remaining node
                let (index, node) = remaining.next().unwrap();
                // if there is only exactly one node left, we need to convert the branch
                if remaining.next().is_none() {
                    let mut orphan = node.take().unwrap();
                    match &mut orphan.data {
                        // if the orphan is a leaf, prepend the corresponding nib to it
                        MptNodeData::Leaf(prefix, orphan_value) => {
                            let new_nibs: Vec<_> =
                                iter::once(index as u8).chain(prefix_nibs(prefix)).collect();
                            self.data = MptNodeData::Leaf(
                                to_encoded_path(&new_nibs, true),
                                mem::take(orphan_value),
                            );
                        }
                        // if the orphan is an extension, prepend the corresponding nib to it
                        MptNodeData::Extension(prefix, orphan_child) => {
                            let new_nibs: Vec<_> =
                                iter::once(index as u8).chain(prefix_nibs(prefix)).collect();
                            self.data = MptNodeData::Extension(
                                to_encoded_path(&new_nibs, false),
                                mem::take(orphan_child),
                            );
                        }
                        // if the orphan is a branch or digest, convert to an extension
                        MptNodeData::Branch(_) | MptNodeData::Digest(_) => {
                            self.data = MptNodeData::Extension(
                                to_encoded_path(&[index as u8], false),
                                orphan,
                            );
                        }
                        MptNodeData::Null => unreachable!(),
                    }
                }
            }
            MptNodeData::Leaf(prefix, _) => {
                if prefix_nibs(prefix) != key_nibs {
                    return Ok(false);
                }
                self.data = MptNodeData::Null;
            }
            MptNodeData::Extension(prefix, child) => {
                let mut self_nibs = prefix_nibs(prefix);
                if let Some(tail) = key_nibs.strip_prefix(self_nibs.as_slice()) {
                    if !child.delete_internal(tail)? {
                        return Ok(false);
                    }
                } else {
                    return Ok(false);
                }

                // an extension can only point to a branch or a digest; since it's sub trie was
                // modified, we need to make sure that this property still holds
                match &mut child.data {
                    // if the child is empty, remove the extension
                    MptNodeData::Null => {
                        self.data = MptNodeData::Null;
                    }
                    // for a leaf, replace the extension with the extended leaf
                    MptNodeData::Leaf(prefix, value) => {
                        self_nibs.extend(prefix_nibs(prefix));
                        self.data =
                            MptNodeData::Leaf(to_encoded_path(&self_nibs, true), mem::take(value));
                    }
                    // for an extension, replace the extension with the extended extension
                    MptNodeData::Extension(prefix, node) => {
                        self_nibs.extend(prefix_nibs(prefix));
                        self.data = MptNodeData::Extension(
                            to_encoded_path(&self_nibs, false),
                            mem::take(node),
                        );
                    }
                    // for a branch or digest, the extension is still correct
                    MptNodeData::Branch(_) | MptNodeData::Digest(_) => {}
                }
            }
            MptNodeData::Digest(digest) => return Err(Error::NodeNotResolved(*digest)),
        };

        self.invalidate_ref_cache();
        Ok(true)
    }

    /// Inserts a key-value pair into the trie.
    ///
    /// This method attempts to insert a new key-value pair into the trie. If the
    /// insertion is successful, it returns `true`. If the key already exists, it updates
    /// the value and returns `false`.
    #[inline]
    pub fn insert(&mut self, key: &[u8], value: Vec<u8>) -> Result<bool, Error> {
        if value.is_empty() {
            panic!("value must not be empty");
        }
        self.insert_internal(&to_nibs(key), value)
    }

    /// Inserts an RLP-encoded value into the trie.
    ///
    /// This method inserts a value that's been encoded using RLP into the trie.
    #[inline]
    pub fn insert_rlp(&mut self, key: &[u8], value: impl Encodable) -> Result<bool, Error> {
        self.insert_internal(&to_nibs(key), value.to_rlp())
    }

    fn insert_internal(&mut self, key_nibs: &[u8], value: Vec<u8>) -> Result<bool, Error> {
        match &mut self.data {
            MptNodeData::Null => {
                self.data = MptNodeData::Leaf(to_encoded_path(key_nibs, true), value);
            }
            MptNodeData::Branch(children) => {
                if let Some((i, tail)) = key_nibs.split_first() {
                    let child = &mut children[*i as usize];
                    match child {
                        Some(node) => {
                            if !node.insert_internal(tail, value)? {
                                return Ok(false);
                            }
                        }
                        // if the corresponding child is empty, insert a new leaf
                        None => {
                            *child = Some(Box::new(
                                MptNodeData::Leaf(to_encoded_path(tail, true), value).into(),
                            ));
                        }
                    }
                } else {
                    return Err(Error::ValueInBranch);
                }
            }
            MptNodeData::Leaf(prefix, old_value) => {
                let self_nibs = prefix_nibs(prefix);
                let common_len = lcp(&self_nibs, key_nibs);
                if common_len == self_nibs.len() && common_len == key_nibs.len() {
                    // if self_nibs == key_nibs, update the value if it is different
                    if old_value == &value {
                        return Ok(false);
                    }
                    *old_value = value;
                } else if common_len == self_nibs.len() || common_len == key_nibs.len() {
                    return Err(Error::ValueInBranch);
                } else {
                    let split_point = common_len + 1;
                    // otherwise, create a branch with two children
                    let mut children: [Option<Box<MptNode>>; 16] = Default::default();

                    children[self_nibs[common_len] as usize] = Some(Box::new(
                        MptNodeData::Leaf(
                            to_encoded_path(&self_nibs[split_point..], true),
                            mem::take(old_value),
                        )
                        .into(),
                    ));
                    children[key_nibs[common_len] as usize] = Some(Box::new(
                        MptNodeData::Leaf(to_encoded_path(&key_nibs[split_point..], true), value)
                            .into(),
                    ));

                    let branch = MptNodeData::Branch(children);
                    if common_len > 0 {
                        // create parent extension for new branch
                        self.data = MptNodeData::Extension(
                            to_encoded_path(&self_nibs[..common_len], false),
                            Box::new(branch.into()),
                        );
                    } else {
                        self.data = branch;
                    }
                }
            }
            MptNodeData::Extension(prefix, existing_child) => {
                let self_nibs = prefix_nibs(prefix);
                let common_len = lcp(&self_nibs, key_nibs);
                if common_len == self_nibs.len() {
                    // traverse down for update
                    if !existing_child.insert_internal(&key_nibs[common_len..], value)? {
                        return Ok(false);
                    }
                } else if common_len == key_nibs.len() {
                    return Err(Error::ValueInBranch);
                } else {
                    let split_point = common_len + 1;
                    // otherwise, create a branch with two children
                    let mut children: [Option<Box<MptNode>>; 16] = Default::default();

                    children[self_nibs[common_len] as usize] = if split_point < self_nibs.len() {
                        Some(Box::new(
                            MptNodeData::Extension(
                                to_encoded_path(&self_nibs[split_point..], false),
                                mem::take(existing_child),
                            )
                            .into(),
                        ))
                    } else {
                        Some(mem::take(existing_child))
                    };
                    children[key_nibs[common_len] as usize] = Some(Box::new(
                        MptNodeData::Leaf(to_encoded_path(&key_nibs[split_point..], true), value)
                            .into(),
                    ));

                    let branch = MptNodeData::Branch(children);
                    if common_len > 0 {
                        // Create parent extension for new branch
                        self.data = MptNodeData::Extension(
                            to_encoded_path(&self_nibs[..common_len], false),
                            Box::new(branch.into()),
                        );
                    } else {
                        self.data = branch;
                    }
                }
            }
            MptNodeData::Digest(digest) => return Err(Error::NodeNotResolved(*digest)),
        };

        self.invalidate_ref_cache();
        Ok(true)
    }

    fn invalidate_ref_cache(&mut self) {
        self.cached_reference.lock().unwrap().take();
    }

    /// Returns the number of traversable nodes in the trie.
    ///
    /// This method provides a count of all the nodes that can be traversed within the
    /// trie.
    pub fn size(&self) -> usize {
        match self.as_data() {
            MptNodeData::Null => 0,
            MptNodeData::Branch(children) => {
                children.iter().flatten().map(|n| n.size()).sum::<usize>() + 1
            }
            MptNodeData::Leaf(_, _) => 1,
            MptNodeData::Extension(_, child) => child.size() + 1,
            MptNodeData::Digest(_) => 0,
        }
    }

    /// Formats the trie as a string list, where each line corresponds to a trie leaf.
    ///
    /// This method is primarily used for debugging purposes, providing a visual
    /// representation of the trie's structure.
    pub fn debug_rlp<T: alloy_rlp::Decodable + Debug>(&self) -> Vec<String> {
        // convert the nibs to hex
        let nibs: String = self.nibs().iter().fold(String::new(), |mut output, n| {
            let _ = write!(output, "{n:x}");
            output
        });

        match self.as_data() {
            MptNodeData::Null => vec![format!("{:?}", MptNodeData::Null)],
            MptNodeData::Branch(children) => children
                .iter()
                .enumerate()
                .flat_map(|(i, child)| {
                    match child {
                        Some(node) => node.debug_rlp::<T>(),
                        None => vec!["None".to_string()],
                    }
                    .into_iter()
                    .map(move |s| format!("{i:x} {s}"))
                })
                .collect(),
            MptNodeData::Leaf(_, data) => {
                vec![format!("{} -> {:?}", nibs, T::decode(&mut &data[..]).unwrap())]
            }
            MptNodeData::Extension(_, node) => {
                node.debug_rlp::<T>().into_iter().map(|s| format!("{nibs} {s}")).collect()
            }
            MptNodeData::Digest(digest) => vec![format!("#{:#}", digest)],
        }
    }

    /// Returns the length of the RLP payload of the node.
    fn payload_length(&self) -> usize {
        match &self.data {
            MptNodeData::Null => 0,
            MptNodeData::Branch(nodes) => {
                1 + nodes
                    .iter()
                    .map(|child| child.as_ref().map_or(1, |node| node.reference_length()))
                    .sum::<usize>()
            }
            MptNodeData::Leaf(prefix, value) => {
                prefix.as_slice().length() + value.as_slice().length()
            }
            MptNodeData::Extension(prefix, node) => {
                prefix.as_slice().length() + node.reference_length()
            }
            MptNodeData::Digest(_) => 32,
        }
    }
}

/// Converts a byte slice into a vector of nibbles.
///
/// A nibble is 4 bits or half of an 8-bit byte. This function takes each byte from the
/// input slice, splits it into two nibbles, and appends them to the resulting vector.
pub fn to_nibs(slice: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(2 * slice.len());
    for byte in slice {
        result.push(byte >> 4);
        result.push(byte & 0xf);
    }
    result
}

/// Encodes a slice of nibbles into a vector of bytes, with an additional prefix to
/// indicate the type of node (leaf or extension).
///
/// The function starts by determining the type of node based on the `is_leaf` parameter.
/// If the node is a leaf, the prefix is set to `0x20`. If the length of the nibbles is
/// odd, the prefix is adjusted and the first nibble is incorporated into it.
///
/// The remaining nibbles are then combined into bytes, with each pair of nibbles forming
/// a single byte. The resulting vector starts with the prefix, followed by the encoded
/// bytes.
pub fn to_encoded_path(mut nibs: &[u8], is_leaf: bool) -> Vec<u8> {
    let mut prefix = (is_leaf as u8) * 0x20;
    if !nibs.len().is_multiple_of(2) {
        prefix += 0x10 + nibs[0];
        nibs = &nibs[1..];
    }
    iter::once(prefix).chain(nibs.chunks_exact(2).map(|byte| (byte[0] << 4) + byte[1])).collect()
}

/// Returns the length of the common prefix.
pub(crate) fn lcp(a: &[u8], b: &[u8]) -> usize {
    for (i, (a, b)) in iter::zip(a, b).enumerate() {
        if a != b {
            return i;
        }
    }
    cmp::min(a.len(), b.len())
}

pub(crate) fn prefix_nibs(prefix: &[u8]) -> Vec<u8> {
    let (extension, tail) = prefix.split_first().unwrap();
    // the first bit of the first nibble denotes the parity
    let is_odd = extension & (1 << 4) != 0;

    let mut result = Vec::with_capacity(2 * tail.len() + is_odd as usize);
    // for odd lengths, the second nibble contains the first element
    if is_odd {
        result.push(extension & 0xf);
    }
    for nib in tail {
        result.push(nib >> 4);
        result.push(nib & 0xf);
    }
    result
}

/// Parses proof bytes into a vector of MPT nodes.
pub fn parse_proof(proof: &[impl AsRef<[u8]>]) -> Result<Vec<MptNode>, Error> {
    proof.iter().map(MptNode::decode).collect()
}

/// Creates a Merkle Patricia trie from an EIP-1186 proof.
/// For inclusion proofs the returned trie contains exactly one leaf with the value.
pub fn mpt_from_proof(proof_nodes: &[MptNode]) -> Result<MptNode, FromProofError> {
    let mut next: Option<MptNode> = None;
    for (i, node) in proof_nodes.iter().enumerate().rev() {
        // there is nothing to replace for the last node
        let Some(replacement) = next else {
            next = Some(node.clone());
            continue;
        };

        // the next node must have a digest reference
        let MptNodeReference::Digest(ref child_ref) = replacement.reference() else {
            return Err(FromProofError::NodeNotFoundByHash(i + 1));
        };
        // find the child that references the next node
        let resolved: MptNode = match node.as_data().clone() {
            MptNodeData::Branch(mut children) => {
                if let Some(child) = children.iter_mut().flatten().find(
                    |child| matches!(child.as_data(), MptNodeData::Digest(d) if d == child_ref),
                ) {
                    *child = Box::new(replacement);
                } else {
                    return Err(FromProofError::NodeHasInvalidSuccessor(i));
                }
                MptNodeData::Branch(children).into()
            }
            MptNodeData::Extension(prefix, child) => {
                if !matches!(child.as_data(), MptNodeData::Digest(d) if d == child_ref) {
                    return Err(FromProofError::NodeHasInvalidSuccessor(i));
                }
                MptNodeData::Extension(prefix, Box::new(replacement)).into()
            }
            MptNodeData::Null | MptNodeData::Leaf(_, _) | MptNodeData::Digest(_) => {
                return Err(FromProofError::NodeCannotHaveChildren(i))
            }
        };

        next = Some(resolved);
    }

    // the last node in the proof should be the root
    Ok(next.unwrap_or_default())
}

/// Verifies that the given proof is a valid proof of exclusion for the given key.
pub fn is_not_included(key: &[u8], proof_nodes: &[MptNode]) -> Result<bool, FromProofError> {
    let proof_trie = mpt_from_proof(proof_nodes).unwrap();
    // for valid proofs, the get must not fail
    let value = proof_trie.get(key).unwrap();

    Ok(value.is_none())
}

/// Creates a new MPT trie where all the digests contained in `node_store` are resolved.
pub fn resolve_nodes(root: &MptNode, node_store: &HashMap<MptNodeReference, MptNode>) -> MptNode {
    let trie = match root.as_data() {
        MptNodeData::Null | MptNodeData::Leaf(_, _) => root.clone(),
        MptNodeData::Branch(children) => {
            let children: Vec<_> = children
                .iter()
                .map(|child| child.as_ref().map(|node| Box::new(resolve_nodes(node, node_store))))
                .collect();
            MptNodeData::Branch(children.try_into().unwrap()).into()
        }
        MptNodeData::Extension(prefix, target) => {
            MptNodeData::Extension(prefix.clone(), Box::new(resolve_nodes(target, node_store)))
                .into()
        }
        MptNodeData::Digest(digest) => {
            if let Some(node) = node_store.get(&MptNodeReference::Digest(*digest)) {
                resolve_nodes(node, node_store)
            } else {
                root.clone()
            }
        }
    };
    // the root hash must not change
    debug_assert_eq!(root.hash(), trie.hash());

    trie
}

/// Returns a list of all possible nodes that can be created by shortening the path of the
/// given node.
/// When nodes in an MPT are deleted, leaves or extensions may be extended. To still be
/// able to identify the original nodes, we create all shortened versions of the node.
pub fn shorten_node_path(node: &MptNode) -> Vec<MptNode> {
    let mut res = Vec::new();
    let nibs = node.nibs();
    match node.as_data() {
        MptNodeData::Null | MptNodeData::Branch(_) | MptNodeData::Digest(_) => {}
        MptNodeData::Leaf(_, value) => {
            for i in 0..=nibs.len() {
                res.push(MptNodeData::Leaf(to_encoded_path(&nibs[i..], true), value.clone()).into())
            }
        }
        MptNodeData::Extension(_, child) => {
            for i in 0..=nibs.len() {
                res.push(
                    MptNodeData::Extension(to_encoded_path(&nibs[i..], false), child.clone())
                        .into(),
                )
            }
        }
    };
    res
}

pub fn proofs_to_tries(
    state_root: B256,
    proofs: &HashMap<Address, AccountProof>,
) -> Result<EthereumState, FromProofError> {
    // if no addresses are provided, return the trie only consisting of the state root
    if proofs.is_empty() {
        return Ok(EthereumState {
            state_trie: node_from_digest(state_root),
            storage_tries: HashMap::with_hasher(Default::default()),
        });
    }

    let mut storage: HashMap<B256, MptNode> =
        HashMap::with_capacity_and_hasher(proofs.len(), Default::default());

    let mut state_nodes = HashMap::with_hasher(Default::default());
    let mut state_root_node = MptNode::default();
    for (address, proof) in proofs {
        let proof_nodes = parse_proof(&proof.proof).unwrap();
        mpt_from_proof(&proof_nodes).unwrap();

        // the first node in the proof is the root
        if let Some(node) = proof_nodes.first() {
            state_root_node = node.clone();
        }

        proof_nodes.into_iter().for_each(|node| {
            state_nodes.insert(node.reference(), node);
        });

        // if no slots are provided, return the trie only consisting of the storage root
        let storage_root = proof.storage_root;
        if proof.storage_proofs.is_empty() {
            let storage_root_node = node_from_digest(storage_root);
            storage.insert(B256::from(&keccak(address)), storage_root_node);
            continue;
        }

        let mut storage_nodes = HashMap::with_hasher(Default::default());
        let mut storage_root_node = MptNode::default();
        for storage_proof in &proof.storage_proofs {
            let proof_nodes = parse_proof(&storage_proof.proof).unwrap();
            mpt_from_proof(&proof_nodes).unwrap();

            // the first node in the proof is the root
            if let Some(node) = proof_nodes.first() {
                storage_root_node = node.clone();
            }

            proof_nodes.into_iter().for_each(|node| {
                storage_nodes.insert(node.reference(), node);
            });
        }

        // create the storage trie, from all the relevant nodes
        let storage_trie = resolve_nodes(&storage_root_node, &storage_nodes);
        let storage_trie_hash = storage_trie.hash();
        if storage_trie_hash != storage_root {
            return Err(FromProofError::MismatchedStorageRoot(
                *address,
                storage_trie_hash,
                storage_root,
            ));
        }

        storage.insert(B256::from(&keccak(address)), storage_trie);
    }
    let state_trie = resolve_nodes(&state_root_node, &state_nodes);
    let state_trie_hash = state_trie.hash();
    if state_trie_hash != state_root {
        return Err(FromProofError::MismatchedStateRoot(state_trie_hash, state_root));
    }

    Ok(EthereumState { state_trie, storage_tries: storage })
}

pub fn transition_proofs_to_tries(
    state_root: B256,
    parent_proofs: &HashMap<Address, AccountProof>,
    proofs: &HashMap<Address, AccountProof>,
) -> Result<EthereumState, FromProofError> {
    // if no addresses are provided, return the trie only consisting of the state root
    if parent_proofs.is_empty() {
        return Ok(EthereumState {
            state_trie: node_from_digest(state_root),
            storage_tries: HashMap::with_hasher(Default::default()),
        });
    }

    let mut storage: HashMap<B256, MptNode> =
        HashMap::with_capacity_and_hasher(parent_proofs.len(), Default::default());

    let mut state_nodes = HashMap::with_hasher(Default::default());
    let mut state_root_node = MptNode::default();
    for (address, proof) in parent_proofs {
        let proof_nodes = parse_proof(&proof.proof).unwrap();
        mpt_from_proof(&proof_nodes).unwrap();

        // the first node in the proof is the root
        if let Some(node) = proof_nodes.first() {
            state_root_node = node.clone();
        }

        proof_nodes.into_iter().for_each(|node| {
            state_nodes.insert(node.reference(), node);
        });

        let fini_proofs = proofs.get(address).unwrap();

        // assure that addresses can be deleted from the state trie
        add_orphaned_leafs(address, &fini_proofs.proof, &mut state_nodes)?;

        // if no slots are provided, return the trie only consisting of the storage root
        let storage_root = proof.storage_root;
        if proof.storage_proofs.is_empty() {
            let storage_root_node = node_from_digest(storage_root);
            storage.insert(B256::from(&keccak(address)), storage_root_node);
            continue;
        }

        let mut storage_nodes = HashMap::with_hasher(Default::default());
        let mut storage_root_node = MptNode::default();
        for storage_proof in &proof.storage_proofs {
            let proof_nodes = parse_proof(&storage_proof.proof).unwrap();
            mpt_from_proof(&proof_nodes).unwrap();

            // the first node in the proof is the root
            if let Some(node) = proof_nodes.first() {
                storage_root_node = node.clone();
            }

            proof_nodes.into_iter().for_each(|node| {
                storage_nodes.insert(node.reference(), node);
            });
        }

        // assure that slots can be deleted from the storage trie
        for storage_proof in &fini_proofs.storage_proofs {
            add_orphaned_leafs(storage_proof.key.0, &storage_proof.proof, &mut storage_nodes)?;
        }
        // create the storage trie, from all the relevant nodes
        let storage_trie = resolve_nodes(&storage_root_node, &storage_nodes);
        let storage_trie_hash = storage_trie.hash();
        if storage_trie_hash != storage_root {
            return Err(FromProofError::MismatchedStorageRoot(
                *address,
                storage_trie_hash,
                storage_root,
            ));
        }

        storage.insert(B256::from(&keccak(address)), storage_trie);
    }

    let state_trie = resolve_nodes(&state_root_node, &state_nodes);
    let state_trie_hash = state_trie.hash();
    if state_trie_hash != state_root {
        return Err(FromProofError::MismatchedStateRoot(state_trie_hash, state_root));
    }

    Ok(EthereumState { state_trie, storage_tries: storage })
}

/// Adds all the leaf nodes of non-inclusion proofs to the nodes.
fn add_orphaned_leafs(
    key: impl AsRef<[u8]>,
    proof: &[impl AsRef<[u8]>],
    nodes_by_reference: &mut HashMap<MptNodeReference, MptNode>,
) -> Result<(), FromProofError> {
    if !proof.is_empty() {
        let proof_nodes = parse_proof(proof)?;
        if is_not_included(&keccak(key), &proof_nodes)? {
            // add the leaf node to the nodes
            let leaf = proof_nodes.last().unwrap();
            shorten_node_path(leaf).into_iter().for_each(|node| {
                nodes_by_reference.insert(node.reference(), node);
            });
        }
    }

    Ok(())
}

/// Creates an MPT node with a pre-computed reference cache. The caller must guarantee that
/// `reference` is the reference of `data` as encoded.
pub(crate) fn node_with_cached_reference(data: MptNodeData, reference: MptNodeReference) -> MptNode {
    MptNode { data, cached_reference: Mutex::new(Some(reference)) }
}

/// Creates a new MPT node from a digest.
pub(crate) fn node_from_digest(digest: B256) -> MptNode {
    match digest {
        EMPTY_ROOT | B256::ZERO => MptNode::default(),
        _ => MptNodeData::Digest(digest).into(),
    }
}

#[cfg(test)]
mod tests {
    use hex_literal::hex;

    use super::*;

    #[test]
    pub fn test_keccak_sponge_matches_alloy() {
        for len in 0..300usize {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 + len) as u8).collect();
            let sponge = keccak256_sponge(&data, keccak::f1600);
            assert_eq!(sponge, *alloy_primitives::utils::keccak256(&data), "len={len}");
        }
        // multi-block sizes around the rate boundary
        for len in [135usize, 136, 137, 271, 272, 273, 1000] {
            let data: Vec<u8> = (0..len).map(|i| (i * 13) as u8).collect();
            let sponge = keccak256_sponge(&data, keccak::f1600);
            assert_eq!(sponge, *alloy_primitives::utils::keccak256(&data), "len={len}");
        }
        // every start alignment x every length: the word-assembly path reads the input
        // through aligned loads, so its head/tail handling depends on `ptr % 8`.
        let backing: Vec<u8> = (0..1200usize).map(|i| (i * 31 + 5) as u8).collect();
        let base = backing.as_ptr() as usize;
        for skew in 0..8usize {
            let start = (8 - (base & 7)) % 8 + skew; // absolute alignment `skew`
            for len in 0..420usize {
                let data = &backing[start..start + len];
                assert_eq!(data.as_ptr() as usize % 8, skew);
                let sponge = keccak256_sponge(data, keccak::f1600);
                assert_eq!(
                    sponge,
                    *alloy_primitives::utils::keccak256(data),
                    "skew={skew} len={len}"
                );
            }
        }
    }

    /// The single-block fast path must agree with alloy for every length it accepts and every
    /// input alignment, and must agree with the general sponge too — the guest picks between
    /// them on `len < RATE` alone, so a disagreement at any length below the rate would make
    /// two callers of the same hash see different digests.
    ///
    /// This is deliberately not a "hash something and compare to a constant" test: the
    /// interesting cases are `len % 8` (how many bytes share the padding word), `len / 8 ==
    /// 16` (where the padding byte and the `0x80` terminator share word 16) and `ptr % 8`
    /// (which of the two absorb arms runs). All three are swept.
    #[test]
    pub fn test_keccak_block_matches_alloy_and_sponge() {
        let backing: Vec<u8> = (0..RATE + 16).map(|i| (i * 29 + 11) as u8).collect();
        let base = backing.as_ptr() as usize;
        let mut lens_seen = 0usize;
        for skew in 0..8usize {
            let start = (8 - (base & 7)) % 8 + skew; // absolute alignment `skew`
            for len in 0..RATE {
                if start + len > backing.len() {
                    continue;
                }
                let data = &backing[start..start + len];
                assert_eq!(data.as_ptr() as usize % 8, skew % 8);
                let block = keccak256_block(data, keccak::f1600);
                assert_eq!(
                    block,
                    *alloy_primitives::utils::keccak256(data),
                    "skew={skew} len={len}"
                );
                assert_eq!(
                    block,
                    keccak256_sponge(data, keccak::f1600),
                    "block vs sponge: skew={skew} len={len}"
                );
                lens_seen += 1;
            }
        }
        // Guard against the sweep silently collapsing: 8 alignments x 136 lengths.
        assert_eq!(lens_seen, 8 * RATE, "the alignment x length sweep did not run in full");
    }

    #[test]
    pub fn test_trie_pointer_no_keccak() {
        let cases = [("do", "verb"), ("dog", "puppy"), ("doge", "coin"), ("horse", "stallion")];
        for (k, v) in cases {
            let node: MptNode =
                MptNodeData::Leaf(k.as_bytes().to_vec(), v.as_bytes().to_vec()).into();
            assert!(
                matches!(node.reference(),MptNodeReference::Bytes(bytes) if bytes == node.to_rlp().to_vec())
            );
        }
    }

    #[test]
    pub fn test_to_encoded_path() {
        // extension node with an even path length
        let nibbles = vec![0x0a, 0x0b, 0x0c, 0x0d];
        assert_eq!(to_encoded_path(&nibbles, false), vec![0x00, 0xab, 0xcd]);
        // extension node with an odd path length
        let nibbles = vec![0x0a, 0x0b, 0x0c];
        assert_eq!(to_encoded_path(&nibbles, false), vec![0x1a, 0xbc]);
        // leaf node with an even path length
        let nibbles = vec![0x0a, 0x0b, 0x0c, 0x0d];
        assert_eq!(to_encoded_path(&nibbles, true), vec![0x20, 0xab, 0xcd]);
        // leaf node with an odd path length
        let nibbles = vec![0x0a, 0x0b, 0x0c];
        assert_eq!(to_encoded_path(&nibbles, true), vec![0x3a, 0xbc]);
    }

    #[test]
    pub fn test_lcp() {
        let cases = [
            (vec![], vec![], 0),
            (vec![0xa], vec![0xa], 1),
            (vec![0xa, 0xb], vec![0xa, 0xc], 1),
            (vec![0xa, 0xb], vec![0xa, 0xb], 2),
            (vec![0xa, 0xb], vec![0xa, 0xb, 0xc], 2),
            (vec![0xa, 0xb, 0xc], vec![0xa, 0xb, 0xc], 3),
            (vec![0xa, 0xb, 0xc], vec![0xa, 0xb, 0xc, 0xd], 3),
            (vec![0xa, 0xb, 0xc, 0xd], vec![0xa, 0xb, 0xc, 0xd], 4),
        ];
        for (a, b, cpl) in cases {
            assert_eq!(lcp(&a, &b), cpl)
        }
    }

    #[test]
    pub fn test_empty() {
        let trie = MptNode::default();

        assert!(trie.is_empty());
        assert_eq!(trie.reference(), MptNodeReference::Bytes(vec![0x80]));
        let expected = hex!("56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421");
        assert_eq!(expected, trie.hash().0);

        // test RLP encoding
        let mut out = Vec::new();
        trie.encode(&mut out);
        assert_eq!(out, vec![0x80]);
        assert_eq!(trie.length(), out.len());
        let decoded = MptNode::decode(out).unwrap();
        assert_eq!(trie.hash(), decoded.hash());
    }

    #[test]
    pub fn test_empty_key() {
        let mut trie = MptNode::default();

        trie.insert(&[], b"empty".to_vec()).unwrap();
        assert_eq!(trie.get(&[]).unwrap(), Some(b"empty".as_ref()));
        assert!(trie.delete(&[]).unwrap());
    }

    #[test]
    pub fn test_clear() {
        let mut trie = MptNode::default();
        trie.insert(b"dog", b"puppy".to_vec()).unwrap();
        assert!(!trie.is_empty());
        assert_ne!(trie.hash(), EMPTY_ROOT);

        trie.clear();
        assert!(trie.is_empty());
        assert_eq!(trie.hash(), EMPTY_ROOT);
    }

    #[test]
    pub fn test_tiny() {
        // trie consisting of an extension, a branch and two leafs
        let mut trie = MptNode::default();
        trie.insert_rlp(b"a", 0u8).unwrap();
        trie.insert_rlp(b"b", 1u8).unwrap();

        assert!(!trie.is_empty());
        let exp_rlp = hex!("d816d680c3208180c220018080808080808080808080808080");
        assert_eq!(trie.reference(), MptNodeReference::Bytes(exp_rlp.to_vec()));
        let exp_hash = hex!("6fbf23d6ec055dd143ff50d558559770005ff44ae1d41276f1bd83affab6dd3b");
        assert_eq!(trie.hash().0, exp_hash);

        // test RLP encoding
        let mut out = Vec::new();
        trie.encode(&mut out);
        assert_eq!(out, exp_rlp.to_vec());
        assert_eq!(trie.length(), out.len());
        let decoded = MptNode::decode(out).unwrap();
        assert_eq!(trie.hash(), decoded.hash());
    }

    #[test]
    pub fn test_partial() {
        let mut trie = MptNode::default();
        trie.insert_rlp(b"aa", 0u8).unwrap();
        trie.insert_rlp(b"ab", 1u8).unwrap();
        trie.insert_rlp(b"ba", 2u8).unwrap();

        let exp_hash = trie.hash();

        // replace one node with its digest
        let MptNodeData::Extension(_, node) = &mut trie.data else { panic!("extension expected") };
        **node = MptNodeData::Digest(node.hash()).into();
        assert!(node.is_digest());

        let trie = MptNode::decode(trie.to_rlp()).unwrap();
        assert_eq!(trie.hash(), exp_hash);

        // lookups should fail
        trie.get(b"aa").unwrap_err();
        trie.get(b"a0").unwrap_err();
    }

    #[test]
    pub fn test_for_each_leaves() {
        let mut trie = MptNode::default();
        trie.insert(b"dog", b"puppy".to_vec()).unwrap();
        trie.insert(b"dock", b"boat".to_vec()).unwrap();

        trie.for_each_leaves(|k, v| {
            println!("key: {k:?}");
            println!("value: {v:?}");
        });
    }

    #[test]
    pub fn test_branch_value() {
        let mut trie = MptNode::default();
        trie.insert(b"do", b"verb".to_vec()).unwrap();
        // leads to a branch with value which is not supported
        trie.insert(b"dog", b"puppy".to_vec()).unwrap_err();
    }

    #[test]
    pub fn test_insert() {
        let mut trie = MptNode::default();
        let vals = vec![
            ("painting", "place"),
            ("guest", "ship"),
            ("mud", "leave"),
            ("paper", "call"),
            ("gate", "boast"),
            ("tongue", "gain"),
            ("baseball", "wait"),
            ("tale", "lie"),
            ("mood", "cope"),
            ("menu", "fear"),
        ];
        for (key, val) in &vals {
            assert!(trie.insert(key.as_bytes(), val.as_bytes().to_vec()).unwrap());
        }

        let expected = hex!("2bab6cdf91a23ebf3af683728ea02403a98346f99ed668eec572d55c70a4b08f");
        assert_eq!(expected, trie.hash().0);

        for (key, value) in &vals {
            assert_eq!(trie.get(key.as_bytes()).unwrap(), Some(value.as_bytes()));
        }

        // check inserting duplicate keys
        assert!(trie.insert(vals[0].0.as_bytes(), b"new".to_vec()).unwrap());
        assert!(!trie.insert(vals[0].0.as_bytes(), b"new".to_vec()).unwrap());

        // try RLP roundtrip
        let decoded = MptNode::decode(trie.to_rlp()).unwrap();
        assert_eq!(trie.hash(), decoded.hash());
    }

    #[test]
    pub fn test_keccak_trie() {
        const N: usize = 512;

        // insert
        let mut trie = MptNode::default();
        for i in 0..N {
            assert!(trie.insert_rlp(&keccak(i.to_be_bytes()), i).unwrap());

            // check hash against trie build in reverse
            let mut reference = MptNode::default();
            for j in (0..=i).rev() {
                reference.insert_rlp(&keccak(j.to_be_bytes()), j).unwrap();
            }
            assert_eq!(trie.hash(), reference.hash());
        }

        let expected = hex!("7310027edebdd1f7c950a7fb3413d551e85dff150d45aca4198c2f6315f9b4a7");
        assert_eq!(trie.hash().0, expected);

        // get
        for i in 0..N {
            assert_eq!(trie.get_rlp(&keccak(i.to_be_bytes())).unwrap(), Some(i));
            assert!(trie.get(&keccak((i + N).to_be_bytes())).unwrap().is_none());
        }

        // delete
        for i in 0..N {
            assert!(trie.delete(&keccak(i.to_be_bytes())).unwrap());

            let mut reference = MptNode::default();
            for j in ((i + 1)..N).rev() {
                reference.insert_rlp(&keccak(j.to_be_bytes()), j).unwrap();
            }
            assert_eq!(trie.hash(), reference.hash());
        }
        assert!(trie.is_empty());
    }

    #[test]
    pub fn test_index_trie() {
        const N: usize = 512;

        // insert
        let mut trie = MptNode::default();
        for i in 0..N {
            assert!(trie.insert_rlp(&i.to_rlp(), i).unwrap());

            // check hash against trie build in reverse
            let mut reference = MptNode::default();
            for j in (0..=i).rev() {
                reference.insert_rlp(&j.to_rlp(), j).unwrap();
            }
            assert_eq!(trie.hash(), reference.hash());

            // try RLP roundtrip
            let decoded = MptNode::decode(trie.to_rlp()).unwrap();
            assert_eq!(trie.hash(), decoded.hash());
        }

        // get
        for i in 0..N {
            assert_eq!(trie.get_rlp(&i.to_rlp()).unwrap(), Some(i));
            assert!(trie.get(&(i + N).to_rlp()).unwrap().is_none());
        }

        // delete
        for i in 0..N {
            assert!(trie.delete(&i.to_rlp()).unwrap());

            let mut reference = MptNode::default();
            for j in ((i + 1)..N).rev() {
                reference.insert_rlp(&j.to_rlp(), j).unwrap();
            }
            assert_eq!(trie.hash(), reference.hash());
        }
        assert!(trie.is_empty());
    }
}
