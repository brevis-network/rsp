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

/// Compares `n` bytes at `a` and `b` with C `memcmp` semantics, reading a word at a time when
/// both pointers share the same alignment.
///
/// # Safety
///
/// `a` and `b` must be valid for reads of `n` bytes.
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
                // Little-endian: the lowest differing byte address holds the least
                // significant differing byte of the word.
                let shift = ((x ^ y).trailing_zeros() / 8) * 8;
                return i32::from((x >> shift) as u8) - i32::from((y >> shift) as u8);
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
    use super::{compare_bytes, set_bytes};

    fn reference(a: &[u8], b: &[u8]) -> i32 {
        for (x, y) in core::iter::zip(a, b) {
            if x != y {
                return i32::from(*x) - i32::from(*y);
            }
        }
        0
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
