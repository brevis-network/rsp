//! The bn254 precompiles (EIP-196, EIP-197) on brevis's `substrate-bn`, the way our revm branch
//! runs them.
//!
//! G1 points stay affine: in the Pico guest `AffineG1 + AffineG1` is one BN254_ADD or
//! BN254_DOUBLE syscall and `AffineG1 * Fr` a double-and-add over them, where evm2's own
//! substrate-bn backend goes through Jacobian coordinates and pays for every field operation of
//! every group operation. The pairing check is the same as evm2's.
//!
//! The checks are those of evm2's backends: a coordinate is 32 big-endian bytes below p; (0, 0)
//! is the point at infinity and any other point must be on the curve (and, in G2, in the
//! subgroup); a scalar is any 256-bit integer.
//!
//! Only the guest's [`CustomCrypto`](crate::custom::CustomCrypto) calls these. Outside the guest
//! the patched crate does its field arithmetic through a slow emulation of the syscalls (about
//! 10 ms for one scalar multiplication), so there evm2's default backend stays in place and
//! these functions are reached by their tests only.

use bn::{AffineG1, AffineG2, Fq, Fq2, Fr, Group, Gt, G1, G2};
use evm2::precompiles::PrecompileHalt;

const FQ_LEN: usize = 32;
const FQ2_LEN: usize = 2 * FQ_LEN;
const G1_LEN: usize = 2 * FQ_LEN;
const SCALAR_LEN: usize = 32;

fn read_fq(input: &[u8]) -> Result<Fq, PrecompileHalt> {
    Fq::from_slice(&input[..FQ_LEN]).map_err(|_| PrecompileHalt::Bn254FieldPointNotAMember)
}

/// An Fq2 element; the imaginary part comes first in the input.
fn read_fq2(input: &[u8]) -> Result<Fq2, PrecompileHalt> {
    let y = read_fq(&input[..FQ_LEN])?;
    let x = read_fq(&input[FQ_LEN..2 * FQ_LEN])?;
    Ok(Fq2::new(x, y))
}

/// A G1 point, `None` for the point at infinity.
fn read_g1(input: &[u8]) -> Result<Option<AffineG1>, PrecompileHalt> {
    let px = read_fq(&input[..FQ_LEN])?;
    let py = read_fq(&input[FQ_LEN..2 * FQ_LEN])?;
    if px == Fq::zero() && py == Fq::zero() {
        Ok(None)
    } else {
        AffineG1::new(px, py).map(Some).map_err(|_| PrecompileHalt::Bn254AffineGFailedToCreate)
    }
}

fn encode_g1(point: Option<AffineG1>) -> [u8; G1_LEN] {
    let mut output = [0u8; G1_LEN];
    if let Some(p) = point {
        p.x().to_big_endian(&mut output[..FQ_LEN]).unwrap();
        p.y().to_big_endian(&mut output[FQ_LEN..]).unwrap();
    }
    output
}

fn read_g2(input: &[u8]) -> Result<G2, PrecompileHalt> {
    let x = read_fq2(&input[..FQ2_LEN])?;
    let y = read_fq2(&input[FQ2_LEN..2 * FQ2_LEN])?;
    if x.is_zero() && y.is_zero() {
        Ok(G2::zero())
    } else {
        Ok(G2::from(AffineG2::new(x, y).map_err(|_| PrecompileHalt::Bn254AffineGFailedToCreate)?))
    }
}

pub fn g1_add(p1: &[u8], p2: &[u8]) -> Result<[u8; G1_LEN], PrecompileHalt> {
    let sum = match (read_g1(p1)?, read_g1(p2)?) {
        (None, None) => None,
        (Some(p), None) | (None, Some(p)) => Some(p),
        // P + (-P): the affine addition has no result to give for it.
        (Some(a), Some(b)) if a.x() == b.x() && a.y() != b.y() => None,
        (Some(a), Some(b)) => Some(a + b),
    };
    Ok(encode_g1(sum))
}

pub fn g1_mul(point: &[u8], scalar: &[u8]) -> Result<[u8; G1_LEN], PrecompileHalt> {
    assert_eq!(scalar.len(), SCALAR_LEN, "unexpected scalar length");
    // Any 256-bit integer is a scalar; `from_slice` reduces it.
    let scalar = Fr::from_slice(scalar).unwrap();
    let product = match read_g1(point)? {
        // `AffineG1 * Fr` does not take a zero scalar.
        Some(p) if scalar != Fr::zero() => Some(p * scalar),
        _ => None,
    };
    Ok(encode_g1(product))
}

pub fn pairing_check(pairs: &[(&[u8], &[u8])]) -> Result<bool, PrecompileHalt> {
    let mut parsed = Vec::with_capacity(pairs.len());
    for (g1, g2) in pairs {
        let g1 = read_g1(g1)?;
        let g2 = read_g2(g2)?;
        // A pair with the point at infinity on either side contributes 1, once both are valid.
        if let Some(g1) = g1 {
            if !g2.is_zero() {
                // The crate converts through `Into` only.
                let g1: G1 = g1.into();
                parsed.push((g1, g2));
            }
        }
    }
    if parsed.is_empty() {
        return Ok(true);
    }
    Ok(bn::pairing_batch(&parsed) == Gt::one())
}
