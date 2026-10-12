//! EIP-2537 (BLS12-381) precompiles on brevis's `bls12_381`, whose field and G1 arithmetic run as
//! Pico syscalls in the guest (and as plain Rust elsewhere).
//!
//! revm's default backend is arkworks in software, which made these precompiles the most
//! expensive code a mainnet block runs: one BLS signature check (two `map_fp2_to_g2`, one
//! `g2_add`, one pairing check) retired ~36 M instructions, and its 64-bit multiplications filled
//! the Mul chip of several core chunks past every proving shape.
//!
//! Each check mirrors revm's arkworks backend (`revm_precompile::bls12_381::arkworks`), so an
//! input one backend rejects the other rejects too, and an accepted input gives the same bytes:
//! - an Fp is 48 big-endian bytes, below p;
//! - (0, 0) is the point at infinity, and any other point must be on the curve;
//! - addition does no subgroup check; MSM and the pairing check do;
//! - the pairing check skips a pair holding the point at infinity, after validating its other
//!   point;
//! - an MSM scalar is any 256-bit integer, taken mod r; a pair with a zero scalar is skipped after
//!   its point is validated.

use bls12_381::{
    fp::Fp, fp2::Fp2, hash_to_curve::MapToCurve, multi_miller_loop, G1Affine, G1Projective,
    G2Affine, G2Prepared, G2Projective, Gt, Scalar,
};
use revm::precompile::{
    bls12_381::{G1Point, G1PointScalar, G2Point, G2PointScalar},
    PrecompileError,
};

const FP_LENGTH: usize = 48;
const G1_LENGTH: usize = 2 * FP_LENGTH;
const G2_LENGTH: usize = 4 * FP_LENGTH;

fn read_fp(be: &[u8; FP_LENGTH]) -> Result<Fp, PrecompileError> {
    Option::from(Fp::from_bytes(be)).ok_or(PrecompileError::NonCanonicalFp)
}

fn is_zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

/// A G1 point from its coordinates, with no subgroup check.
///
/// zkcrypto reads the top three bits of an encoding as flags (compression, infinity, sort), so
/// the coordinates are range-checked on the raw bytes first: below p those bits are zero, and the
/// encoding is plain `x || y`. Decoding first would, for one, take `x = 0x40 00..00` (2^382, not a
/// field element) together with `y = 0` as the point at infinity.
fn read_g1_no_subgroup_check((x, y): &G1Point) -> Result<G1Affine, PrecompileError> {
    let (px, py) = (read_fp(x)?, read_fp(y)?);
    if bool::from(px.is_zero() & py.is_zero()) {
        return Ok(G1Affine::identity());
    }
    let mut bytes = [0u8; G1_LENGTH];
    bytes[..FP_LENGTH].copy_from_slice(x);
    bytes[FP_LENGTH..].copy_from_slice(y);
    let point: G1Affine = Option::from(G1Affine::from_uncompressed_unchecked(&bytes))
        .ok_or(PrecompileError::NonCanonicalFp)?;
    if !bool::from(point.is_on_curve()) {
        return Err(PrecompileError::Bls12381G1NotOnCurve);
    }
    Ok(point)
}

fn read_g1(p: &G1Point) -> Result<G1Affine, PrecompileError> {
    let point = read_g1_no_subgroup_check(p)?;
    if !bool::from(point.is_torsion_free()) {
        return Err(PrecompileError::Bls12381G1NotInSubgroup);
    }
    Ok(point)
}

fn encode_g1(point: &G1Affine) -> [u8; G1_LENGTH] {
    if bool::from(point.is_identity()) {
        return [0u8; G1_LENGTH];
    }
    // A finite point's uncompressed encoding carries no flag bits: it is `x || y`.
    point.to_uncompressed()
}

/// A G2 point from its coordinates, with no subgroup check. EIP-2537 orders each Fp2 `c0 || c1`,
/// zkcrypto `c1 || c0`; the flags sit in the top bits of `x.c1`. See [`read_g1_no_subgroup_check`].
fn read_g2_no_subgroup_check((x0, x1, y0, y1): &G2Point) -> Result<G2Affine, PrecompileError> {
    let fps = [read_fp(x0)?, read_fp(x1)?, read_fp(y0)?, read_fp(y1)?];
    if fps.iter().all(|f| bool::from(f.is_zero())) {
        return Ok(G2Affine::identity());
    }
    let mut bytes = [0u8; G2_LENGTH];
    for (i, c) in [x1, x0, y1, y0].into_iter().enumerate() {
        bytes[i * FP_LENGTH..(i + 1) * FP_LENGTH].copy_from_slice(c);
    }
    let point: G2Affine = Option::from(G2Affine::from_uncompressed_unchecked(&bytes))
        .ok_or(PrecompileError::NonCanonicalFp)?;
    if !bool::from(point.is_on_curve()) {
        return Err(PrecompileError::Bls12381G2NotOnCurve);
    }
    Ok(point)
}

fn read_g2(p: &G2Point) -> Result<G2Affine, PrecompileError> {
    let point = read_g2_no_subgroup_check(p)?;
    if !bool::from(point.is_torsion_free()) {
        // The arkworks backend reports a G2 subgroup failure with the G1 variant; kept identical.
        return Err(PrecompileError::Bls12381G1NotInSubgroup);
    }
    Ok(point)
}

fn encode_g2(point: &G2Affine) -> [u8; G2_LENGTH] {
    let mut out = [0u8; G2_LENGTH];
    if bool::from(point.is_identity()) {
        return out;
    }
    // zkcrypto: x.c1 || x.c0 || y.c1 || y.c0.
    let zk = point.to_uncompressed();
    for (dst, src) in [1usize, 0, 3, 2].into_iter().enumerate() {
        out[dst * FP_LENGTH..(dst + 1) * FP_LENGTH]
            .copy_from_slice(&zk[src * FP_LENGTH..(src + 1) * FP_LENGTH]);
    }
    out
}

/// Any 256-bit big-endian integer, reduced mod r, as arkworks' `from_be_bytes_mod_order` does.
fn read_scalar(be: &[u8; 32]) -> Scalar {
    let mut wide = [0u8; 64];
    for (dst, src) in wide.iter_mut().zip(be.iter().rev()) {
        *dst = *src;
    }
    Scalar::from_bytes_wide(&wide)
}

pub(crate) fn g1_add(a: G1Point, b: G1Point) -> Result<[u8; G1_LENGTH], PrecompileError> {
    let p = read_g1_no_subgroup_check(&a)?;
    let q = read_g1_no_subgroup_check(&b)?;
    // In the guest: the identity cases in software, P == Q on the double syscall, P == -Q to the
    // identity, everything else on the add syscall.
    Ok(encode_g1(&p.add_affine(&q)))
}

pub(crate) fn g2_add(a: G2Point, b: G2Point) -> Result<[u8; G2_LENGTH], PrecompileError> {
    let p = read_g2_no_subgroup_check(&a)?;
    let q = read_g2_no_subgroup_check(&b)?;
    Ok(encode_g2(&G2Affine::from(G2Projective::from(p) + q)))
}

pub(crate) fn g1_msm(
    pairs: &mut dyn Iterator<Item = Result<G1PointScalar, PrecompileError>>,
) -> Result<[u8; G1_LENGTH], PrecompileError> {
    let mut acc = G1Projective::identity();
    for pair in pairs {
        let (point, scalar) = pair?;
        let point = read_g1(&point)?;
        if is_zero(&scalar) {
            continue;
        }
        acc += point * read_scalar(&scalar);
    }
    Ok(encode_g1(&G1Affine::from(acc)))
}

pub(crate) fn g2_msm(
    pairs: &mut dyn Iterator<Item = Result<G2PointScalar, PrecompileError>>,
) -> Result<[u8; G2_LENGTH], PrecompileError> {
    let mut acc = G2Projective::identity();
    for pair in pairs {
        let (point, scalar) = pair?;
        let point = read_g2(&point)?;
        if is_zero(&scalar) {
            continue;
        }
        acc += point * read_scalar(&scalar);
    }
    Ok(encode_g2(&G2Affine::from(acc)))
}

pub(crate) fn pairing_check(pairs: &[(G1Point, G2Point)]) -> Result<bool, PrecompileError> {
    let mut g1 = Vec::with_capacity(pairs.len());
    let mut g2 = Vec::with_capacity(pairs.len());
    for (p, q) in pairs {
        let p_zero = is_zero(&p.0) && is_zero(&p.1);
        let q_zero = is_zero(&q.0) && is_zero(&q.1) && is_zero(&q.2) && is_zero(&q.3);
        if p_zero || q_zero {
            // e(O, Q) = e(P, O) = 1, but the other point must still be valid.
            if !p_zero {
                read_g1(p)?;
            }
            if !q_zero {
                read_g2(q)?;
            }
            continue;
        }
        g1.push(read_g1(p)?);
        g2.push(G2Prepared::from(read_g2(q)?));
    }
    if g1.is_empty() {
        return Ok(true);
    }
    let terms: Vec<(&G1Affine, &G2Prepared)> = g1.iter().zip(g2.iter()).collect();
    Ok(multi_miller_loop(&terms).final_exponentiation() == Gt::identity())
}

/// `map_to_curve` (simplified SWU onto the 11-isogenous curve, then the isogeny) and cofactor
/// clearing by `h_eff`, as EIP-2537 specifies (not the full `hash_to_curve`).
pub(crate) fn map_fp_to_g1(fp: &[u8; FP_LENGTH]) -> Result<[u8; G1_LENGTH], PrecompileError> {
    let u = read_fp(fp)?;
    let point = <G1Projective as MapToCurve>::map_to_curve(&u).clear_h();
    Ok(encode_g1(&G1Affine::from(point)))
}

/// As [`map_fp_to_g1`], over Fp2 with the 3-isogeny. The input is `c0 || c1`.
pub(crate) fn map_fp2_to_g2(
    (c0, c1): ([u8; FP_LENGTH], [u8; FP_LENGTH]),
) -> Result<[u8; G2_LENGTH], PrecompileError> {
    let u = Fp2 { c0: read_fp(&c0)?, c1: read_fp(&c1)? };
    let point = <G2Projective as MapToCurve>::map_to_curve(&u).clear_h();
    Ok(encode_g2(&G2Affine::from(point)))
}
