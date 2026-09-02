//! Word-at-a-time `memcmp`/`bcmp`/`memset` for the pico guest.
//!
//! `compiler_builtins`' `memcmp` is a byte-at-a-time loop — seven instructions per byte on
//! `riscv64im-pico-zkvm-elf`. It matters more here than it looks: because the target has no
//! misaligned scalar memory access, LLVM's RISC-V backend does not expand `memcmp` inline
//! (`RISCVTTIImpl::enableMemCmpExpansion` is gated on `enableUnalignedScalarMem`), so *every*
//! byte-array equality becomes a libcall — `U256` comparisons in the EVM interpreter
//! (`JUMPI`/`ISZERO`/`EQ`), hash and address keys in the journal's maps, RLP node references.
//! Measured on block 24006677: 136.3 M of 1018.2 M retired instructions, 13.4 % of the guest.
//!
//! The strong symbols defined here override `compiler_builtins`' weak definitions at link time.
//! `#![no_builtins]` prevents LLVM from recognizing the comparison loop as a memcmp idiom and
//! lowering it back into a `memcmp` libcall.

#![cfg_attr(not(test), no_std)]
#![no_builtins]

// The word-compare path derives the first differing byte from the least significant differing
// bits, which is only correct on little-endian targets.
const _: () = assert!(cfg!(target_endian = "little"));

const WORD: usize = core::mem::size_of::<usize>();

/// C `memcmp`'s answer for two differing words read from the same offset.
///
/// Little-endian: the lowest differing byte *address* holds the least significant differing
/// byte of the word, so the first difference is at the lowest set bit of the xor.
///
/// Note for future rounds: unrolling the word loop four wide and replacing the sub-word tail
/// with three size tests was measured at **+1.29 M** retired instructions on block 24006677.
/// That is the third independent attempt to speed this function up and the third negative
/// one; the loop below is at its floor and the win, if there is one, is in *not calling* it.
///
/// The obvious spelling of that, `((x ^ y).trailing_zeros() / 8) * 8`, costs 20 instructions on
/// RV64IM, which has no count-trailing-zeros: LLVM expands `cttz.i64` into
/// `(d & -d) * DEBRUIJN >> 58` plus a 64-entry table load. Measured on block 24006677 that tail
/// runs 67,910 times out of `memcmp`'s 105,382 calls -- 1.36 M retired instructions, 40 % of
/// `memcmp`. The exact ctz is never needed to *get there*, though: the callers compare digests,
/// `U256` limbs and address words, so once two words are known to differ the *lowest* byte
/// differs with probability 255/256. Testing that byte first and leaving the other seven to an
/// unrolled ladder brings the common case to six instructions, and the returned value is
/// bit-identical to what the shift spelling produced.
#[inline(always)]
fn word_diff(x: usize, y: usize) -> i32 {
    // Spelled out rather than looped or split out. Two other shapes were measured on block
    // 24006677 and both gave most of the win back:
    //   * bytes 1..8 in a `#[cold] #[inline(never)]` helper: -0.49 M against this shape's -1.01 M,
    //     because the call makes `memcmp` save `ra` -- two instructions in the prologue and two in
    //     every epilogue, on all 105,382 calls;
    //   * a `while i < WORD` loop: -0.03 M, because the extra counter pushed `compare_bytes` past
    //     LLVM's inlining threshold and `memcmp` became an eight-instruction thunk. `compare_bytes`
    //     carries `#[inline(always)]` now so that cannot come back.
    macro_rules! byte_step {
        ($k:expr) => {{
            let xb = (x >> ($k * 8)) as u8;
            let yb = (y >> ($k * 8)) as u8;
            if xb != yb {
                return i32::from(xb) - i32::from(yb);
            }
        }};
    }
    byte_step!(0);
    byte_step!(1);
    byte_step!(2);
    byte_step!(3);
    byte_step!(4);
    byte_step!(5);
    byte_step!(6);
    byte_step!(7);
    // Unreachable for the only caller, which has already found the two words unequal.
    // Returning 0 keeps the function total instead of adding a panic path to `memcmp`.
    0
}

/// Compares `n` bytes at `a` and `b` with C `memcmp` semantics, reading a word at a time when
/// both pointers share the same alignment.
///
/// # Safety
///
/// `a` and `b` must be valid for reads of `n` bytes.
#[inline(always)]
pub unsafe fn compare_bytes(mut a: *const u8, mut b: *const u8, mut n: usize) -> i32 {
    if (a as usize) % WORD == (b as usize) % WORD {
        while (a as usize) % WORD != 0 && n > 0 {
            let (x, y) = (*a, *b);
            if x != y {
                return i32::from(x) - i32::from(y);
            }
            a = a.add(1);
            b = b.add(1);
            n -= 1;
        }
        while n >= WORD {
            let x = *a.cast::<usize>();
            let y = *b.cast::<usize>();
            if x != y {
                return word_diff(x, y);
            }
            a = a.add(WORD);
            b = b.add(WORD);
            n -= WORD;
        }
    }
    while n > 0 {
        let (x, y) = (*a, *b);
        if x != y {
            return i32::from(x) - i32::from(y);
        }
        a = a.add(1);
        b = b.add(1);
        n -= 1;
    }
    0
}

/// Writes fewer than `WORD` copies of `c`, without a loop.
///
/// A `while i < n` byte loop costs five instructions per byte (store, pointer bump, branch,
/// bound test); three size tests instead bring it down to about one.
///
/// # Safety
///
/// `n < WORD` and `dst` is valid for `n` writes.
#[inline(always)]
unsafe fn set_sub_word(mut dst: *mut u8, c: u8, n: usize) {
    if n & 4 != 0 {
        *dst = c;
        *dst.add(1) = c;
        *dst.add(2) = c;
        *dst.add(3) = c;
        dst = dst.add(4);
    }
    if n & 2 != 0 {
        *dst = c;
        *dst.add(1) = c;
        dst = dst.add(2);
    }
    if n & 1 != 0 {
        *dst = c;
    }
}

/// Writes `n` copies of `c` at `dst`, a word at a time.
///
/// `compiler_builtins`' `memset` aligns to 4 bytes and stores with `sw`, behind ~40
/// instructions of small-size dispatch; the guest's average call is ~78 bytes, so that
/// prologue is most of the cost.
///
/// # Safety
///
/// `dst` must be valid for writes of `n` bytes. Nothing outside `dst..dst + n` is touched.
#[inline(always)]
pub unsafe fn set_bytes(dst: *mut u8, c: u8, n: usize) {
    let mut d = dst;
    let mut n = n;

    if n < WORD {
        set_sub_word(d, c, n);
        return;
    }

    let v = (c as usize).wrapping_mul(usize::MAX / 0xff);

    let head = (WORD - (d as usize % WORD)) % WORD;
    if head != 0 {
        set_sub_word(d, c, head);
        d = d.add(head);
        n -= head;
    }

    let words = n / WORD;
    let mut p = d.cast::<usize>();
    let mut k = words;
    while k >= 4 {
        p.write(v);
        p.add(1).write(v);
        p.add(2).write(v);
        p.add(3).write(v);
        p = p.add(4);
        k -= 4;
    }
    while k != 0 {
        p.write(v);
        p = p.add(1);
        k -= 1;
    }

    set_sub_word(d.add(words * WORD), c, n - words * WORD);
}

#[cfg(target_os = "zkvm")]
mod c_exports {
    #[no_mangle]
    unsafe extern "C" fn memcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
        super::compare_bytes(a, b, n)
    }

    // Equality-only variant of `memcmp`; the compiler emits calls to it for slice `==` where
    // the ordering is unused.
    #[no_mangle]
    unsafe extern "C" fn bcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
        super::compare_bytes(a, b, n)
    }

    // Overriding `memset` by defining the symbol does *not* work, even though
    // `compiler_builtins` spells its `mem*` intrinsics with `linkage = "weak"`: linking the
    // guest that way fails with `rust-lld: error: duplicate symbol: memset` (verified). The
    // difference from `memcmp` above is most likely reachability -- nothing inside
    // `compiler_builtins` calls `memcmp`, so the linker never pulls that object out of the
    // archive, whereas `memset` is referenced internally and always comes along.
    //
    // So the guest is linked with `--wrap=memset` instead (see `build-guest.sh`), which
    // redirects every call here. Note this makes the optimisation depend on the build
    // invocation: a plain `cargo pico build` silently drops it and costs ~7 M instructions,
    // with no error. The cycle regression check exists to catch exactly that.
    //
    // Only `memset` is replaced. A word-at-a-time `memcpy` was tried too and measured *worse*
    // than `compiler_builtins`' (53.4 M vs 50.3 M retired instructions on block 24006677): 42 %
    // of the guest's copies have a misaligned destination but are shorter than 32 bytes, and for
    // those the straight-line byte blocks that `compiler_builtins` falls back to beat a
    // destination-aligning prologue plus a shift loop.
    #[no_mangle]
    unsafe extern "C" fn __wrap_memset(dst: *mut u8, c: i32, n: usize) -> *mut u8 {
        super::set_bytes(dst, c as u8, n);
        dst
    }
}

#[cfg(test)]
mod tests {
    use super::{compare_bytes, set_bytes, word_diff};

    fn reference(a: &[u8], b: &[u8]) -> i32 {
        for (x, y) in core::iter::zip(a, b) {
            if x != y {
                return i32::from(*x) - i32::from(*y);
            }
        }
        0
    }

    /// `word_diff` against the definition it replaced, over every non-empty subset of the
    /// eight byte positions and both directions of the difference.
    ///
    /// The old spelling is written out here rather than referenced so the test stays a
    /// comparison against an independent implementation.
    ///
    /// The sweep covers every non-empty subset of the eight byte positions, not just one
    /// differing byte at a time. That is the difference between pinning the ladder and
    /// pinning nothing: with exactly one byte differing, "the lowest differing byte decides"
    /// and "the highest differing byte decides" agree on every case, so an earlier version of
    /// this test stayed green when the whole ladder was reversed. `discriminating` counts the
    /// cases where the two rules disagree, which only a multi-byte difference can produce, so
    /// it is a guard the generated data has to earn rather than a constant.
    #[test]
    fn word_diff_matches_the_shift_spelling() {
        fn old(x: usize, y: usize) -> i32 {
            let shift = ((x ^ y).trailing_zeros() / 8) * 8;
            i32::from((x >> shift) as u8) - i32::from((y >> shift) as u8)
        }

        // "The highest differing byte decides" -- the rule the ladder must *not* implement.
        fn highest(x: usize, y: usize) -> i32 {
            let shift = (usize::BITS - 1 - (x ^ y).leading_zeros()) / 8 * 8;
            i32::from((x >> shift) as u8) - i32::from((y >> shift) as u8)
        }

        let mut discriminating = 0usize;
        for fill in [
            0x0000_0000_0000_0000u64,
            0xffff_ffff_ffff_ffff,
            0x0f1e_2d3c_4b5a_6978,
            0x8080_8080_8080_8080,
        ] {
            for mask in 1u32..=0xff {
                for delta in [0x01u64, 0xa5] {
                    // Flip `delta` into every byte position named by `mask`, so differences
                    // of one through eight bytes are all covered.
                    // A different amount per position, so the bytes at the lowest and the
                    // highest differing positions rarely hold the same value -- otherwise the
                    // two rules coincide numerically even where the positions differ, and the
                    // sweep discriminates far less than its size suggests.
                    let mut d = 0u64;
                    for pos in 0..8usize {
                        if mask & (1 << pos) != 0 {
                            let b = (delta as u8).wrapping_mul(pos as u8 * 2 + 1) | 1;
                            d |= u64::from(b) << (pos * 8);
                        }
                    }
                    let x = fill as usize;
                    let y = (fill ^ d) as usize;
                    assert_ne!(x, y, "mask={mask:#x} delta={delta:#x}");
                    let got = word_diff(x, y);
                    assert_eq!(got, old(x, y), "mask={mask:#x} delta={delta:#x} fill={fill:#x}");
                    assert_ne!(got, 0);
                    // and the mirror, so both signs are covered
                    assert_eq!(word_diff(y, x), old(y, x));
                    if got != highest(x, y) {
                        discriminating += 1;
                    }
                }
            }
        }
        assert!(
            discriminating > 900,
            "only {discriminating} cases distinguish the lowest differing byte from the \
             highest; the sweep cannot see the ladder's order"
        );
    }

    #[test]
    fn set_bytes_matches_reference_across_alignments_and_lengths() {
        let mut buf = [0u8; 512];
        for d_off in 0..8 {
            for len in 0..200 {
                let at = 16 + d_off;
                buf.fill(0xAA);
                unsafe { set_bytes(buf.as_mut_ptr().add(at), 0x5C, len) };
                assert!(buf[at..at + len].iter().all(|&b| b == 0x5C), "len={len}");
                assert!(buf[..at].iter().all(|&b| b == 0xAA));
                assert!(buf[at + len..].iter().all(|&b| b == 0xAA));
            }
        }
    }

    #[test]
    fn matches_reference_across_alignments_lengths_and_diff_positions() {
        let mut base = [0u8; 96];
        for (i, byte) in base.iter_mut().enumerate() {
            *byte = (i * 37 % 251) as u8;
        }
        for a_off in 0..8 {
            for b_off in 0..8 {
                for len in 0..48 {
                    let a = &base[a_off..a_off + len];
                    let mut b_buf = [0u8; 96];
                    b_buf[b_off..b_off + len].copy_from_slice(a);
                    // diff_pos == len leaves the buffers equal.
                    for diff_pos in 0..=len {
                        let mut b_buf = b_buf;
                        if diff_pos < len {
                            b_buf[b_off + diff_pos] ^= 0x80;
                        }
                        let b = &b_buf[b_off..b_off + len];
                        let got = unsafe { compare_bytes(a.as_ptr(), b.as_ptr(), len) };
                        assert_eq!(
                            got,
                            reference(a, b),
                            "a_off={a_off} b_off={b_off} len={len} diff_pos={diff_pos}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod compare_bytes_tests {
    use super::compare_bytes;

    /// Every length up to 40, every pair of start offsets in a 16-byte window (so both the
    /// same-alignment and the different-alignment arms are hit), and every position of a
    /// single differing byte, against the reference the C contract asks for.
    #[test]
    fn matches_reference() {
        let mut x = vec![0u8; 128];
        let mut y = vec![0u8; 128];
        for (i, b) in x.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7).wrapping_add(3);
        }
        y.copy_from_slice(&x);
        for n in 0..=40usize {
            for oa in 0..16usize {
                for ob in 0..16usize {
                    for diff in 0..=n {
                        y.copy_from_slice(&x);
                        // The two ranges have to hold the same bytes up to the injected
                        // difference. Copying `x` wholesale leaves them differing at k = 0
                        // whenever `oa != ob`, so the comparison returns before the loop runs
                        // and the mismatched-alignment arm is never reached.
                        y[ob..ob + n].copy_from_slice(&x[oa..oa + n]);
                        if diff < n {
                            y[ob + diff] =
                                x[oa + diff].wrapping_add(if diff % 2 == 0 { 1 } else { 200 });
                        }
                        let want = {
                            let mut r = 0i32;
                            for k in 0..n {
                                let (p, q) = (x[oa + k], y[ob + k]);
                                if p != q {
                                    r = i32::from(p) - i32::from(q);
                                    break;
                                }
                            }
                            r
                        };
                        let got =
                            unsafe { compare_bytes(x.as_ptr().add(oa), y.as_ptr().add(ob), n) };
                        assert_eq!(
                            got.signum(),
                            want.signum(),
                            "n={n} oa={oa} ob={ob} diff={diff} got={got} want={want}"
                        );
                        if want == 0 {
                            assert_eq!(got, 0, "n={n} oa={oa} ob={ob}");
                        }
                    }
                }
            }
        }
    }
}
