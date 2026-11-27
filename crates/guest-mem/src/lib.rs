//! Word-at-a-time `memcmp`/`bcmp`, and `memcpy`/`memset` on Pico's MEMCPY / MEMSET ecalls, for
//! the pico guest.
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
//!
//! `memcpy` and `memset` are brevis-vm's RV64 assembly ones (`memcpy.s`, `memset.s`), linked as
//! `__wrap_memcpy` / `__wrap_memset` under `--wrap=memcpy` / `--wrap=memset`. A call of 16 bytes
//! or more is one ecall per 2048 bytes, MEMCPY (`0x136`) or MEMSET (`0x137`), which Pico proves in
//! the chunk that runs it on its `MemoryCopy` chip, a row per destination dword. Shorter calls
//! stay in the assembly, where every load and store is naturally aligned. The guest needs a Pico
//! prover that has the two ecalls.

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
/// # The **sign** is load-bearing, not just zero/non-zero
///
/// 45 of the 1,107 shipped call sites consume the result as a sign test, traced in the ELF to
/// `post_state_root -> delta_root -> ipnsort` -- the sort, and the strict-ascending check, that
/// `apply_branch`'s precondition depends on. An approximately-ordered `memcmp` would mis-sort a
/// delta batch and hand `apply_branch` a list it is not allowed to receive. **Do not simplify
/// this to an equality test**, whatever a static sweep of the call sites suggests.
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

#[cfg(target_os = "zkvm")]
mod c_exports {
    // `memcpy` and `memset` cannot be overridden by defining the symbols, even though
    // `compiler_builtins` spells its `mem*` intrinsics with `linkage = "weak"`: linking the guest
    // that way fails with `rust-lld: error: duplicate symbol: memset` (verified). Its `memset` is
    // absorbed by `lto = "fat"` and re-emitted *strong* in a regular object, so the weak linkage
    // is gone before the linker ever chooses; `memcmp` below survives only because
    // `#![no_builtins]` excludes this crate from the LTO merge. The practical consequence: **if
    // fat LTO is ever dropped, `memcmp` could start colliding the same way.** pico-sdk defines
    // `memcpy` and `memset` too.
    //
    // So the guest is linked with `--wrap=memcpy` and `--wrap=memset` (see `build-guest.sh`),
    // which redirect every call to the `__wrap_` symbols these files define. That makes the
    // optimisation depend on the build invocation: a plain `cargo pico build` silently drops it,
    // which is why `build-guest.sh` asserts the linked ELF defines both.
    core::arch::global_asm!(include_str!("memcpy.s"));
    core::arch::global_asm!(include_str!("memset.s"));

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
}

#[cfg(test)]
mod tests {
    use super::{compare_bytes, word_diff};

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
