//! The guest's bn254 precompile code (`rsp_client_executor::bn254`) against evm2's default
//! backend.
//!
//! Ours is substrate-bn with affine G1 points (the path that is syscalls in the guest); evm2's
//! default is arkworks, so the two sides share no code. On random inputs,
//! valid and broken in each way the checks tell apart, both must reject the same inputs and
//! return the same bytes for the rest.
//!
//! These run the plain-Rust path. In the guest the same code does its arithmetic in Pico
//! syscalls, which a host test cannot reach.

use alloy_primitives::hex;
use bn::{AffineG1, AffineG2, Fr, Group, G1, G2};
use evm2::precompiles::{Crypto, DefaultCrypto, PrecompileHalt};
use rsp_client_executor::bn254::{g1_add, g1_mul, pairing_check};
use std::fmt::Debug;

/// p, the modulus of the base field.
const P: [u8; 32] = hex!("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47");
/// r, the order of G1 and G2.
const R: [u8; 32] = hex!("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

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

    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        for chunk in out.chunks_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes()[..chunk.len()]);
        }
        out
    }
}

/// `n + d` for a big-endian `n`, wrapping.
fn be_add(mut n: [u8; 32], d: i64) -> [u8; 32] {
    let mut carry = d as i128;
    for byte in n.iter_mut().rev() {
        let sum = *byte as i128 + carry;
        *byte = sum.rem_euclid(256) as u8;
        carry = sum.div_euclid(256);
    }
    n
}

/// `a - b` for big-endian integers with `a >= b`.
fn be_sub(a: [u8; 32], b: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut borrow = 0i16;
    for i in (0..32).rev() {
        let diff = a[i] as i16 - b[i] as i16 - borrow;
        out[i] = diff.rem_euclid(256) as u8;
        borrow = (diff < 0) as i16;
    }
    out
}

fn random_scalar(rng: &mut Rng) -> Fr {
    Fr::from_slice(&rng.bytes::<32>()).unwrap()
}

/// `x || y` of a point on the curve, never the point at infinity.
fn valid_g1(rng: &mut Rng) -> [u8; 64] {
    loop {
        if let Some(p) = AffineG1::from_jacobian(G1::one() * random_scalar(rng)) {
            let mut out = [0u8; 64];
            p.x().to_big_endian(&mut out[..32]).unwrap();
            p.y().to_big_endian(&mut out[32..]).unwrap();
            return out;
        }
    }
}

/// `x.c1 || x.c0 || y.c1 || y.c0` of a point of G2, never the point at infinity.
fn valid_g2(rng: &mut Rng) -> [u8; 128] {
    loop {
        if let Some(q) = AffineG2::from_jacobian(G2::one() * random_scalar(rng)) {
            let mut out = [0u8; 128];
            q.x().imaginary().to_big_endian(&mut out[..32]).unwrap();
            q.x().real().to_big_endian(&mut out[32..64]).unwrap();
            q.y().imaginary().to_big_endian(&mut out[64..96]).unwrap();
            q.y().real().to_big_endian(&mut out[96..]).unwrap();
            return out;
        }
    }
}

/// -P: the same x, p - y.
fn negated_g1(point: [u8; 64]) -> [u8; 64] {
    let mut out = point;
    out[32..].copy_from_slice(&be_sub(P, &point[32..]));
    out
}

/// Something that is not a field element.
fn bad_fq(rng: &mut Rng) -> [u8; 32] {
    match rng.below(3) {
        0 => P,
        1 => be_add(P, 1 + rng.below(1 << 20) as i64),
        _ => [0xff; 32],
    }
}

fn g1_input(rng: &mut Rng) -> [u8; 64] {
    match rng.below(9) {
        0..=4 => valid_g1(rng),
        5 => [0; 64],
        // all but surely off the curve
        6 => {
            let mut out = valid_g1(rng);
            out[63] ^= 1;
            out
        }
        7 => {
            let mut out = valid_g1(rng);
            out[..32].copy_from_slice(&bad_fq(rng));
            out
        }
        _ => {
            let mut out = valid_g1(rng);
            out[32..].copy_from_slice(&bad_fq(rng));
            out
        }
    }
}

fn g2_input(rng: &mut Rng) -> [u8; 128] {
    match rng.below(8) {
        0..=4 => valid_g2(rng),
        5 => [0; 128],
        // all but surely off the curve
        6 => {
            let mut out = valid_g2(rng);
            out[127] ^= 1;
            out
        }
        _ => {
            let mut out = valid_g2(rng);
            let at = 32 * rng.below(4) as usize;
            out[at..at + 32].copy_from_slice(&bad_fq(rng));
            out
        }
    }
}

fn scalar_input(rng: &mut Rng) -> [u8; 32] {
    match rng.below(9) {
        0 => [0; 32],
        1 => be_add([0; 32], 1),
        2 => be_add(R, -1),
        3 => R,
        4 => be_add(R, 1),
        5 => [0xff; 32],
        _ => rng.bytes(),
    }
}

/// Counts both outcomes, so that inputs which never reach one side of the checks fail the test.
#[derive(Default)]
struct Tally {
    accepted: usize,
    rejected: usize,
}

impl Tally {
    fn check<T: PartialEq + Debug>(
        &mut self,
        what: &str,
        i: usize,
        ours: Result<T, PrecompileHalt>,
        evm2s: Result<T, PrecompileHalt>,
    ) {
        match (&ours, &evm2s) {
            (Ok(a), Ok(b)) if a == b => self.accepted += 1,
            (Err(_), Err(_)) => self.rejected += 1,
            _ => panic!("{what} #{i}: ours {ours:?}, evm2's {evm2s:?}"),
        }
    }

    fn assert_both_sides(&self, what: &str) {
        let Self { accepted, rejected } = *self;
        assert!(accepted > 0 && rejected > 0, "{what}: {accepted} accepted, {rejected} rejected");
    }
}

#[test]
fn g1_add_matches_evm2() {
    let (mut rng, mut tally) = (Rng(0x9e37_79b9_7f4a_7c15), Tally::default());
    let evm2s = DefaultCrypto::new();
    for i in 0..2000 {
        let a = g1_input(&mut rng);
        let b = match rng.below(6) {
            // doubling, and the sum that is the point at infinity
            0 => a,
            1 => negated_g1(a),
            _ => g1_input(&mut rng),
        };
        tally.check("g1_add", i, g1_add(&a, &b), evm2s.bn254_g1_add(&a, &b));
    }
    tally.assert_both_sides("g1_add");
}

#[test]
fn g1_mul_matches_evm2() {
    let (mut rng, mut tally) = (Rng(0x2545_f491_4f6c_dd1d), Tally::default());
    let evm2s = DefaultCrypto::new();
    for i in 0..600 {
        let point = g1_input(&mut rng);
        let scalar = scalar_input(&mut rng);
        tally.check("g1_mul", i, g1_mul(&point, &scalar), evm2s.bn254_g1_mul(&point, &scalar));
    }
    tally.assert_both_sides("g1_mul");
}

#[test]
fn pairing_check_matches_evm2() {
    let (mut rng, mut tally) = (Rng(0xdead_beef_cafe_f00d), Tally::default());
    let evm2s = DefaultCrypto::new();
    for i in 0..120 {
        let product_is_one = rng.below(3) == 0;
        let pairs: Vec<([u8; 64], [u8; 128])> = if product_is_one {
            // e(P, Q) * e(-P, Q), at times with a pair that has the point at infinity on one side
            let (p, q) = (valid_g1(&mut rng), valid_g2(&mut rng));
            let mut pairs = vec![(p, q), (negated_g1(p), q)];
            match rng.below(3) {
                0 => pairs.push(([0; 64], valid_g2(&mut rng))),
                1 => pairs.push((valid_g1(&mut rng), [0; 128])),
                _ => {}
            }
            pairs
        } else {
            (0..rng.below(4)).map(|_| (g1_input(&mut rng), g2_input(&mut rng))).collect()
        };
        let borrowed: Vec<(&[u8], &[u8])> = pairs.iter().map(|(p, q)| (&p[..], &q[..])).collect();
        let a = pairing_check(&borrowed);
        if product_is_one {
            assert_eq!(a, Ok(true), "pairing_check #{i}");
        }
        tally.check("pairing_check", i, a, evm2s.bn254_pairing_check(&borrowed));
    }
    tally.assert_both_sides("pairing_check");
}
