//! Keccak-256 as a sponge written directly over a keccak-f[1600] permutation: in the Pico guest,
//! the permute syscall.
//!
//! Carried over from our `pico-rv64` branch, where it sat in `mpt.rs`. The digest machinery a
//! hashing crate goes through (block buffering, byte-wise absorb) costs most of a small hash in
//! the guest, and small hashes are nearly all of them: storage keys, hashed addresses, mapping
//! slots, bloom topics.
//!
//! The guest binary exports [`keccak256_zkvm_into`] as alloy's `native-keccak` hook, so every
//! `keccak256` alloy computes there comes through this file. Outside the guest nothing calls it;
//! the tests run the same code over a software permutation.

// `target_vendor = "pico"` comes from the custom riscv64im-pico-zkvm-elf target.
#![allow(unexpected_cfgs)]
// Outside the guest only the tests use this file.
#![allow(dead_code)]

// The sponge assembles each `u64` from little-endian byte order, both when reading unaligned
// input and when writing the squeezed state.
const _: () = assert!(cfg!(target_endian = "little"));

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
/// [`keccak256_sponge_into`]. 66,283 of the 70,722 hashes on block 24006677 are single-block --
/// every EVM `KECCAK256` of a mapping key, every hashed address and slot, every bloom topic --
/// and the general sponge serves them through machinery they never use. The measured cost is
/// mostly not the branches:
///
/// * twelve callee-saved registers saved and restored, 26 instructions per call, because the
///   multi-block absorb keeps seventeen state words live across the permute;
/// * the padding word and the `0x80` terminator are read-modify-written, since either may land on a
///   word an earlier block filled — on a single block both are known zero, so plain stores;
/// * the block count, the 136-byte reciprocal and the loop's induction variables are all
///   materialised before the length is tested.
///
/// The length test lives here rather than in the inlined wrappers deliberately: split across the
/// 66 call sites it measured +0.65 M retired instructions, against +0.14 M for one test here.
///
/// # Safety
///
/// `out` must point at 32 writable bytes.
#[inline(never)]
#[allow(dead_code)]
pub(crate) unsafe fn keccak256_into(data: &[u8], permute: impl Fn(&mut [u64; 25]), out: *mut u8) {
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
            // Measured, it is worse on mainnet block 24006677: the bigger body stops
            // `absorb_words` being inlined into the sponge, and the two out-of-line copies
            // that appear then cost more than the addressing saves.
            //
            // **The size of the regression is not reliably recorded.** This comment said
            // 240 K; the commit that rejected the experiment (`3fbd43fa`) says +1.33 M, 5.5x
            // apart, for the same experiment. The convention in this project is that a
            // negative result is written down so it is not retried, and a 5.5x spread makes
            // the record useless for that -- so if anyone does retry it, *re-measure* rather
            // than trusting either figure, and replace this paragraph with the answer.
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

    // `[0u64; 25]` is 200 bytes, which LLVM zeroes with a `memset` call (~60 instructions per
    // hash); volatile stores keep it as 25 `sd`.
    //
    // On the multi-block path only the capacity (words 17..25) needs zeroing: `FIRST` *assigns*
    // all 17 rate words, so the zeros are overwritten before anything reads them. 17 `sd` saved
    // on each of the 4,439 multi-block hashes of block 24006677.
    //
    // The single-block entry still needs all 25 -- it absorbs `full <= 16` words then reads
    // `state[full]` and `state[RATE_WORDS - 1]`. Unreachable from the guest, where
    // `keccak256_into` handles short inputs itself, but `keccak256_sponge` enters here directly.
    let multi = data.len() >= RATE;
    let mut state = core::mem::MaybeUninit::<[u64; 25]>::uninit();
    // SAFETY: the writes below initialize words 17..25 always and words 0..17 either here or,
    // when `multi`, in the `FIRST` absorb of block 0 further down; every word of the array is
    // written before it is read.
    let state = unsafe {
        let q = state.as_mut_ptr().cast::<u64>();
        if !multi {
            for i in 0..RATE_WORDS {
                q.add(i).write_volatile(0);
            }
        }
        for i in RATE_WORDS..25 {
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
    if multi {
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

            // The guard above is a *memory-safety* guard, not a correctness one, and no test
            // can see the difference: the stream's last aligned word reaches `r - 1` bytes
            // past the absorbed region, but those bytes only ever land in `carry`, which is
            // dead after the final block. So the digest is right either way, and removing
            // the `data.len() - nblocks * RATE >= r` half of the condition leaves the whole
            // 8-alignment x 420-length sweep green while reading out of bounds. This says
            // what the guard is for, in a form the tests do check.
            debug_assert!(
                data.len() >= nblocks * RATE + r,
                "the shifted stream reads r bytes past the absorbed region"
            );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keccak_sponge_matches_alloy() {
        for len in 0..300usize {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 + len) as u8).collect();
            let sponge = keccak256_sponge(&data, ::keccak::f1600);
            assert_eq!(sponge, *alloy_primitives::utils::keccak256(&data), "len={len}");
        }
        // multi-block sizes around the rate boundary
        for len in [135usize, 136, 137, 271, 272, 273, 1000] {
            let data: Vec<u8> = (0..len).map(|i| (i * 13) as u8).collect();
            let sponge = keccak256_sponge(&data, ::keccak::f1600);
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
                let sponge = keccak256_sponge(data, ::keccak::f1600);
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
    fn test_keccak_block_matches_alloy_and_sponge() {
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
                let block = keccak256_block(data, ::keccak::f1600);
                assert_eq!(
                    block,
                    *alloy_primitives::utils::keccak256(data),
                    "skew={skew} len={len}"
                );
                assert_eq!(
                    block,
                    keccak256_sponge(data, ::keccak::f1600),
                    "block vs sponge: skew={skew} len={len}"
                );
                lens_seen += 1;
            }
        }
        // Guard against the sweep silently collapsing: 8 alignments x 136 lengths.
        assert_eq!(lens_seen, 8 * RATE, "the alignment x length sweep did not run in full");
    }

    /// `keccak256_into` is what the guest actually calls, and it hands anything at or above
    /// the rate to the general sponge. The sweep above stops one byte short of that hand-off,
    /// which leaves the single-block path's `full <= 16` premise resting on an untested
    /// comparison: were it ever to admit a 136-byte input, the `_ =>` arm of `fill_block`
    /// would absorb 128 bytes and pad there, silently. Cross the boundary in both directions.
    #[test]
    fn test_keccak_into_crosses_the_rate_boundary() {
        let backing: Vec<u8> = (0..3 * RATE + 16).map(|i| (i * 23 + 7) as u8).collect();
        let base = backing.as_ptr() as usize;
        let mut lens_seen = 0usize;
        for skew in 0..8usize {
            let start = (8 - (base & 7)) % 8 + skew; // absolute alignment `skew`
            for len in RATE - 8..2 * RATE + 8 {
                let data = &backing[start..start + len];
                assert_eq!(data.as_ptr() as usize % 8, skew);
                assert_eq!(
                    keccak256_block(data, ::keccak::f1600),
                    *alloy_primitives::utils::keccak256(data),
                    "skew={skew} len={len}"
                );
                lens_seen += 1;
            }
        }
        assert_eq!(lens_seen, 8 * (RATE + 16), "the rate-boundary sweep did not run in full");
    }

    /// Two digest destinations are align-1 as far as the compiler is concerned -- the `B256`
    /// `keccak_into_b256` writes through, and the raw out-pointer alloy hands
    /// `native_keccak256` -- so `squeeze_into` chooses between four `sd` and a byte scatter
    /// at run time. (`digest_prefix` and the guest's own `keccak` deliberately pass an
    /// `#[repr(align(8))]` slot and always take the wide arm.) Every test helper declares
    /// that alignment too, so the scatter was unreached; drive it directly.
    ///
    /// Note what a host test cannot settle: x86 and aarch64 tolerate the misaligned `u64`
    /// store that RV64 traps on, so inverting the alignment predicate keeps this green. What
    /// is pinned here is the scatter's arithmetic, not the choice between the two arms.
    #[test]
    fn test_keccak_into_unaligned_destination() {
        #[repr(align(8))]
        struct Out([u8; 40]);
        let backing: Vec<u8> = (0..600usize).map(|i| (i * 17 + 3) as u8).collect();
        let mut checked = 0usize;
        for off in 1..8usize {
            for len in [0usize, 1, 20, 32, 135, 136, 137, 271, 272, 400] {
                let data = &backing[..len];
                let mut out = Out([0u8; 40]);
                assert_ne!((out.0.as_ptr() as usize + off) % 8, 0);
                // SAFETY: `out.0[off..off + 32]` is 32 writable bytes.
                unsafe { keccak256_into(data, ::keccak::f1600, out.0.as_mut_ptr().add(off)) };
                assert_eq!(
                    &out.0[off..off + 32],
                    &*alloy_primitives::utils::keccak256(data),
                    "off={off} len={len}"
                );
                // The buffer has room either side, so a write outside the 32 bytes lands in
                // it rather than in the assertion above.
                assert!(
                    out.0[..off].iter().all(|&b| b == 0)
                        && out.0[off + 32..].iter().all(|&b| b == 0),
                    "squeeze_into wrote outside its 32 bytes: off={off} len={len}"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 7 * 10, "the unaligned-destination sweep did not run in full");
    }
}
