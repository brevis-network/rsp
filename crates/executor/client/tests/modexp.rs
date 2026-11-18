//! [`modexp_u256`], the guest's MODEXP for operands of at most 256 bits, against evm2's backend.
//!
//! In the guest every multiplication is a call of Pico's UINT256_MUL precompile, which a host
//! cannot make. Here it is a model of the precompile: `(x * y) % m` by ruint, refusing what the
//! precompile's chip could not prove, a call whose quotient has more than 256 bits.

use alloy_primitives::{hex, U256, U512};
use evm2::precompiles::{Crypto, DefaultCrypto};
use rsp_client_executor::modexp::modexp_u256;
use std::cell::Cell;

fn model(x: &mut [u64; 4], y_and_m: &[u64; 8]) {
    let a = U256::from_limbs(*x);
    let b = U256::from_limbs(y_and_m[..4].try_into().unwrap());
    let m = U256::from_limbs(y_and_m[4..].try_into().unwrap());
    assert!(!m.is_zero(), "a modulus of 0 means 2^256 to the precompile, and MODEXP passes none");
    let (quotient, remainder) = a.widening_mul::<256, 4, 512, 8>(b).div_rem(U512::from(m));
    assert!(quotient.bit_len() <= 256, "UINT256_MUL cannot prove ({a} * {b}) % {m}");
    *x = remainder.as_limbs()[..4].try_into().unwrap();
}

fn significant(bytes: &[u8]) -> &[u8] {
    &bytes[bytes.iter().position(|&byte| byte != 0).unwrap_or(bytes.len())..]
}

/// Ours has the modulus' length and the value evm2's backend computes.
fn check(base: &[u8], exp: &[u8], modulus: &[u8]) {
    let ours = modexp_u256(base, exp, modulus, model).expect("operands of at most 256 bits");
    let evm2 = DefaultCrypto::new().modexp(base, exp, modulus).unwrap();
    let case = format!("{} ^ {} % {}", hex::encode(base), hex::encode(exp), hex::encode(modulus));
    assert_eq!(ours.len(), modulus.len(), "{case}");
    assert_eq!(significant(&ours), significant(&evm2), "{case}");
}

/// xorshift64: deterministic, so a failure reproduces.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }

    /// At most 32 significant bytes, under some leading zeros, with the values a reduction and a
    /// comparison turn on mixed in.
    fn operand(&mut self) -> Vec<u8> {
        let mut out = vec![0u8; self.below(4) as usize];
        if self.below(40) == 0 {
            out.extend(vec![0u8; 40]);
        }
        let len = self.below(33) as usize;
        match self.below(12) {
            0 => {}
            1 => out.extend(vec![0u8; len]),
            2 => out.push(1),
            3 => out.push(2),
            4 => out.extend(vec![0xffu8; len]),
            5 if len > 0 => {
                out.push(1 << self.below(8));
                out.extend(vec![0u8; len - 1]);
            }
            _ => out.extend(self.bytes(len)),
        }
        out
    }

    fn exponent(&mut self) -> Vec<u8> {
        let len = self.below(41) as usize;
        match self.below(10) {
            0 => vec![],
            1 => vec![0u8; len],
            2 => vec![1],
            3 => vec![0xffu8; len],
            4 => {
                // One set bit somewhere under leading zeros.
                let mut out = vec![0u8; len + 1];
                let at = self.below(len as u64 + 1) as usize;
                out[at] = 1 << self.below(8);
                out
            }
            _ => self.bytes(len),
        }
    }
}

#[test]
fn matches_evm2_on_random_operands() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    for _ in 0..20_000 {
        let (base, exp, modulus) = (rng.operand(), rng.exponent(), rng.operand());
        check(&base, &exp, &modulus);
    }
}

/// A base equal to the modulus, one to either side of it, and one to either side of twice it:
/// where the first reduction's quotient goes from 0 to 1 to 2.
#[test]
fn matches_evm2_around_the_modulus() {
    let mut rng = Rng(0x0fed_cba9_8765_4321);
    for _ in 0..5_000 {
        let m = U256::from_be_slice(significant(&rng.operand()));
        let step = U256::from(rng.below(3));
        let base = match rng.below(5) {
            0 => m.wrapping_sub(step),
            1 => m.wrapping_add(step),
            2 => m.wrapping_add(m).wrapping_sub(step),
            3 => m.wrapping_add(m).wrapping_add(step),
            _ => U256::MAX.wrapping_sub(step),
        };
        check(&base.to_be_bytes::<32>(), &rng.exponent(), &m.to_be_bytes::<32>());
    }
}

#[test]
fn the_values_it_turns_on() {
    let max = [0xffu8; 32];
    let top_bit = hex!("8000000000000000000000000000000000000000000000000000000000000000");
    let operands: [&[u8]; 9] =
        [&[], &[0], &[1], &[2], &[3], &[0, 0, 7], &top_bit, &max[..31], &max];
    let exponents: [&[u8]; 8] = [&[], &[0], &[1], &[2], &[0, 0, 1], &[0x80], &top_bit, &max];
    for base in operands {
        for exp in exponents {
            for modulus in operands {
                check(base, exp, modulus);
            }
        }
    }
    // A modulus of 0 and a modulus of 1 give 0, in the modulus' length.
    assert_eq!(modexp_u256(&[5], &[3], &[0, 0, 0], model), Some(vec![0, 0, 0]));
    assert_eq!(modexp_u256(&[5], &[3], &[0, 1], model), Some(vec![0, 0]));
    assert_eq!(modexp_u256(&[5], &[3], &[], model), Some(vec![]));
    // 0^0 is 1.
    assert_eq!(modexp_u256(&[], &[], &[9], model), Some(vec![1]));
}

#[test]
fn wider_operands_are_left_to_evm2() {
    assert_eq!(modexp_u256(&[1; 33], &[1], &[7], model), None);
    assert_eq!(modexp_u256(&[1], &[1], &[1; 33], model), None);
    // Leading zeros do not make an operand wide: 5^3 % 7 is 6.
    let mut base = vec![0u8; 100];
    base.push(5);
    let mut modulus = vec![0u8; 40];
    modulus.push(7);
    let mut six = vec![0u8; 40];
    six.push(6);
    assert_eq!(modexp_u256(&base, &[3], &modulus, model), Some(six));
}

/// EIP-198's two examples over secp256k1's field prime: `3^(p - 1) % p` is 1 by Fermat, and
/// `0^(p - 1) % p` is 0.
#[test]
fn the_examples_of_eip_198() {
    let p = hex!("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f");
    let p_minus_1 = hex!("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2e");
    let mut one = vec![0u8; 32];
    one[31] = 1;
    assert_eq!(modexp_u256(&[3], &p_minus_1, &p, model), Some(one));
    assert_eq!(modexp_u256(&[], &p_minus_1, &p, model), Some(vec![0u8; 32]));
}

/// One call to reduce the base, one to square for every bit under the exponent's top bit, one
/// to multiply for every set bit under it.
#[test]
fn a_bit_of_the_exponent_costs_at_most_two_calls() {
    let mut rng = Rng(0x5555_3333_0f0f_00ff);
    for _ in 0..2_000 {
        let (base, exp) = (rng.operand(), rng.exponent());
        let calls = Cell::new(0u32);
        let counting = |x: &mut [u64; 4], y_and_m: &[u64; 8]| {
            calls.set(calls.get() + 1);
            model(x, y_and_m);
        };
        modexp_u256(&base, &exp, &[0xfe, 0xdc, 0xba, 0x98], counting).unwrap();
        let e = significant(&exp);
        let (bits, ones) = match e.first() {
            None => (0, 0),
            Some(top) => (
                8 * e.len() as u32 - top.leading_zeros(),
                e.iter().map(|byte| byte.count_ones()).sum::<u32>(),
            ),
        };
        let expected = if bits == 0 { 1 } else { 1 + (bits - 1) + (ones - 1) };
        assert_eq!(calls.get(), expected, "exponent {}", hex::encode(&exp));
    }
}
