//! MODEXP on Pico's UINT256_MUL precompile, for a base and a modulus of at most 256 bits.
//!
//! evm2's backend is aurora-engine-modexp: Montgomery multiplication in software. In the guest
//! `(x * y) % m` on 256-bit numbers is one call of the UINT256_MUL precompile, so here the
//! exponentiation is plain square-and-multiply over that call, at most two calls for a bit of
//! the exponent. Wider operands are left to evm2.
//!
//! The precompile's chip proves `x * y = q * m + r` with `q` in 256 bits, so it can prove a call
//! only when `x * y < m * 2^256`. Every call here has an operand under the modulus: the base is
//! reduced first, by a multiplication by 1, and the accumulator is under the modulus from then
//! on.

/// A 256-bit number as the precompile takes it: four little-endian limbs.
type Limbs = [u64; 4];

/// The value of `bytes`, big-endian, if it fits 256 bits.
fn limbs(bytes: &[u8]) -> Option<Limbs> {
    let first = bytes.iter().position(|&byte| byte != 0).unwrap_or(bytes.len());
    let bytes = &bytes[first..];
    if bytes.len() > 32 {
        return None;
    }
    let mut limbs = [0u64; 4];
    for (i, &byte) in bytes.iter().rev().enumerate() {
        limbs[i / 8] |= u64::from(byte) << (8 * (i % 8));
    }
    Some(limbs)
}

/// `value`, big-endian, in `len` bytes. It fits: it is below a modulus of `len` bytes.
fn bytes(value: Limbs, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    for (i, byte) in out.iter_mut().rev().take(32).enumerate() {
        *byte = (value[i / 8] >> (8 * (i % 8))) as u8;
    }
    out
}

/// `base^exp % modulus` in the modulus' length, or `None` when the base or the modulus has more
/// than 256 bits.
///
/// `mulmod(x, y_and_m)` sets `x` to `(x * y) % m`, with `y` and `m` the two halves of its second
/// argument: the precompile in the guest, a model of it in the tests.
pub fn modexp_u256(
    base: &[u8],
    exp: &[u8],
    modulus: &[u8],
    mulmod: impl Fn(&mut Limbs, &[u64; 8]),
) -> Option<Vec<u8>> {
    let (mut base, m) = (limbs(base)?, limbs(modulus)?);
    // A modulus of 0 gives 0 by definition, and everything is 0 modulo 1.
    if (m[1] | m[2] | m[3]) == 0 && m[0] <= 1 {
        return Some(vec![0; modulus.len()]);
    }
    let times = |x: &mut Limbs, y: Limbs| {
        mulmod(x, &[y[0], y[1], y[2], y[3], m[0], m[1], m[2], m[3]]);
    };
    // The base under the modulus.
    times(&mut base, [1, 0, 0, 0]);
    // Square and multiply, from the exponent's top bit. There is nothing to square before its
    // first set bit.
    let mut acc: Option<Limbs> = None;
    for &byte in exp {
        for bit in (0..8).rev() {
            if let Some(acc) = &mut acc {
                let square = *acc;
                times(acc, square);
            }
            if (byte >> bit) & 1 == 1 {
                if let Some(acc) = &mut acc {
                    times(acc, base);
                } else {
                    acc = Some(base);
                }
            }
        }
    }
    // An exponent of 0 gives 1, which is under a modulus of 2 or more.
    Some(bytes(acc.unwrap_or([1, 0, 0, 0]), modulus.len()))
}

/// [`modexp_u256`] on the precompile.
#[cfg(target_os = "zkvm")]
pub(crate) fn on_uint256_mul(base: &[u8], exp: &[u8], modulus: &[u8]) -> Option<Vec<u8>> {
    extern "C" {
        /// pico-sdk's `riscv_ecalls/uint256_mul.rs`: `x` becomes `(x * y) % m`, where `y` points
        /// at `y` and then `m`.
        fn syscall_uint256_mulmod(x: *mut [u64; 4], y: *const [u64; 4]);
    }
    // SAFETY: four limbs at `x` and eight at `y_and_m`, aligned to eight bytes, which is what
    // the precompile reads and writes.
    modexp_u256(base, exp, modulus, |x, y_and_m| unsafe {
        syscall_uint256_mulmod(x, y_and_m.as_ptr().cast())
    })
}
