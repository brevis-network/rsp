//! The EIP-2537 (BLS12-381) precompiles on [`CustomCrypto`], against EIP-2537's own vectors and
//! against revm's default backend.
//!
//! An integration test so that it has a process of its own: `install_crypto` sets a process-wide
//! provider, once, and the vector test has to be the caller that sets it.
//!
//! These run the backend's plain-Rust path. In the guest the same code does its field and G1
//! arithmetic in Pico syscalls, which a host test cannot reach.

use alloy_primitives::hex;
use bls12_381::{
    fp::Fp, fp2::Fp2, hash_to_curve::MapToCurve, G1Affine, G1Projective, G2Affine, G2Projective,
    Scalar,
};
use revm::precompile::{
    bls12_381::{
        g1_add::g1_add, g1_msm::g1_msm, g2_add::g2_add, g2_msm::g2_msm,
        map_fp2_to_g2::map_fp2_to_g2, map_fp_to_g1::map_fp_to_g1, pairing::pairing, G1Point,
        G1PointScalar, G2Point, G2PointScalar,
    },
    install_crypto, Crypto, DefaultCrypto, PrecompileError, PrecompileFn,
};
use rsp_client_executor::custom::CustomCrypto;
use std::{fmt::Debug, fs, path::Path};

/// EIP-2537's vectors (see `testdata/eip2537/README.md`) through revm's precompile entry points,
/// input decoding and gas included, with [`CustomCrypto`] installed as the executor installs it.
#[test]
fn eip2537_vectors() {
    assert!(install_crypto(CustomCrypto::default()), "a crypto provider was already installed");
    let files: [(&str, PrecompileFn); 12] = [
        ("add_G1_bls", g1_add),
        ("fail-add_G1_bls", g1_add),
        ("add_G2_bls", g2_add),
        ("fail-add_G2_bls", g2_add),
        ("fail-msm_G1_bls", g1_msm),
        ("fail-msm_G2_bls", g2_msm),
        ("pairing_check_bls", pairing),
        ("fail-pairing_check_bls", pairing),
        ("map_fp_to_G1_bls", map_fp_to_g1),
        ("fail-map_fp_to_G1_bls", map_fp_to_g1),
        ("map_fp2_to_G2_bls", map_fp2_to_g2),
        ("fail-map_fp2_to_G2_bls", map_fp2_to_g2),
    ];
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/eip2537");
    let mut failures = Vec::new();
    for (file, precompile) in files {
        let text = fs::read_to_string(dir.join(format!("{file}.json"))).unwrap();
        let vectors: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
        assert!(!vectors.is_empty(), "{file}: no vectors");
        for v in vectors {
            let name = format!("{file} {}", v["Name"].as_str().unwrap());
            let input = hex::decode(v["Input"].as_str().unwrap()).unwrap();
            // A vector of valid input carries `Expected` and `Gas`, a `fail-` one `ExpectedError`.
            match (precompile(&input, u64::MAX), v["Expected"].as_str()) {
                (Ok(out), Some(expected)) => {
                    if hex::encode(&out.bytes) != expected {
                        failures.push(format!("{name}: returned {}", hex::encode(&out.bytes)));
                    }
                    if v["Gas"].as_u64() != Some(out.gas_used) {
                        failures.push(format!("{name}: used {} gas", out.gas_used));
                    }
                }
                (Err(e), Some(_)) => failures.push(format!("{name}: rejected with {e:?}")),
                (Ok(_), None) => failures.push(format!("{name}: accepted")),
                (Err(_), None) => {}
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// The rest compares `CustomCrypto` with revm's `DefaultCrypto` (blst or arkworks, whichever this
// build enables) on random inputs. Besides valid points they draw each kind of input the checks
// tell apart: off the curve, outside the subgroup, coordinates at or past p (including the
// encodings zkcrypto reads as flags), the point at infinity, P == Q and P == -Q, scalars at and
// past r. Both sides must reject the same inputs -- which error does not matter, any error fails
// the call -- and return the same bytes for the rest.

/// p, the modulus of the base field.
const P: [u8; 48] = hex!(
    "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf"
    "6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"
);
/// r, the order of G1 and G2.
const R: [u8; 32] = hex!("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

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
fn be_add<const N: usize>(mut n: [u8; N], d: i64) -> [u8; N] {
    let mut carry = d as i128;
    for byte in n.iter_mut().rev() {
        let sum = *byte as i128 + carry;
        *byte = sum.rem_euclid(256) as u8;
        carry = sum.div_euclid(256);
    }
    n
}

/// A canonical field element.
fn fp_bytes(rng: &mut Rng) -> [u8; 48] {
    loop {
        let mut bytes: [u8; 48] = rng.bytes();
        bytes[0] &= 0x1f;
        if bool::from(Fp::from_bytes(&bytes).is_some()) {
            return bytes;
        }
    }
}

fn fp(rng: &mut Rng) -> Fp {
    Fp::from_bytes(&fp_bytes(rng)).unwrap()
}

/// zkcrypto's encoding of the point at infinity: the infinity flag, every other bit clear. As a
/// coordinate it is 2^382, past p.
fn infinity_flag() -> [u8; 48] {
    let mut bytes = [0u8; 48];
    bytes[0] = 0x40;
    bytes
}

/// Something that is not a canonical field element, in place of `good`.
fn bad_fp(rng: &mut Rng, good: [u8; 48]) -> [u8; 48] {
    match rng.below(6) {
        0 => P,
        1 => be_add(P, 1 + rng.below(1 << 20) as i64),
        2 => infinity_flag(),
        3 => {
            // one of zkcrypto's flag bits set on a coordinate that is otherwise fine
            let mut bytes = good;
            bytes[0] |= [0x20, 0x40, 0x80][rng.below(3) as usize];
            bytes
        }
        4 => {
            // 2^381 - 1, the largest value with the flag bits clear
            let mut bytes = [0xff; 48];
            bytes[0] = 0x1f;
            bytes
        }
        _ => [0xff; 48],
    }
}

fn g1_enc(p: &G1Affine) -> G1Point {
    if bool::from(p.is_identity()) {
        return ([0; 48], [0; 48]);
    }
    let bytes = p.to_uncompressed();
    (bytes[..48].try_into().unwrap(), bytes[48..].try_into().unwrap())
}

fn g2_enc(p: &G2Affine) -> G2Point {
    if bool::from(p.is_identity()) {
        return ([0; 48], [0; 48], [0; 48], [0; 48]);
    }
    // zkcrypto writes x.c1 || x.c0 || y.c1 || y.c0; EIP-2537 puts c0 first.
    let bytes = p.to_uncompressed();
    let c = |i: usize| -> [u8; 48] { bytes[48 * i..48 * (i + 1)].try_into().unwrap() };
    (c(1), c(0), c(3), c(2))
}

/// -P for whatever point the encoding holds, on the curve or not.
fn g1_negated((x, y): G1Point) -> Option<G1Point> {
    let p = G1Affine::from_uncompressed_unchecked(&[x, y].concat().try_into().unwrap());
    Option::<G1Affine>::from(p).map(|p| g1_enc(&-p))
}

fn g2_negated((x0, x1, y0, y1): G2Point) -> Option<G2Point> {
    let p = G2Affine::from_uncompressed_unchecked(&[x1, x0, y1, y0].concat().try_into().unwrap());
    Option::<G2Affine>::from(p).map(|p| g2_enc(&-p))
}

fn valid_g1(rng: &mut Rng) -> G1Affine {
    G1Affine::from(G1Projective::generator() * Scalar::from(rng.next()))
}

fn valid_g2(rng: &mut Rng) -> G2Affine {
    G2Affine::from(G2Projective::generator() * Scalar::from(rng.next()))
}

/// On the curve but, all but surely, outside the subgroup: a map-to-curve output whose cofactor
/// has not been cleared.
fn off_subgroup_g1(rng: &mut Rng) -> G1Affine {
    G1Affine::from(<G1Projective as MapToCurve>::map_to_curve(&fp(rng)))
}

fn off_subgroup_g2(rng: &mut Rng) -> G2Affine {
    let u = Fp2 { c0: fp(rng), c1: fp(rng) };
    G2Affine::from(<G2Projective as MapToCurve>::map_to_curve(&u))
}

fn g1_input(rng: &mut Rng) -> G1Point {
    match rng.below(13) {
        0..=4 => g1_enc(&valid_g1(rng)),
        5 => g1_enc(&off_subgroup_g1(rng)),
        // all but surely off the curve
        6 => (fp_bytes(rng), fp_bytes(rng)),
        7 => ([0; 48], [0; 48]),
        8 => {
            let (x, y) = g1_enc(&valid_g1(rng));
            if rng.below(2) == 0 {
                (bad_fp(rng, x), y)
            } else {
                (x, bad_fp(rng, y))
            }
        }
        // (0, 2) and (0, -2) are on y^2 = x^3 + 4, of order 3
        9 => ([0; 48], if rng.below(2) == 0 { be_add([0; 48], 2) } else { be_add(P, -2) }),
        10 => ([0; 48], fp_bytes(rng)),
        11 => (fp_bytes(rng), [0; 48]),
        _ => (infinity_flag(), [0; 48]),
    }
}

fn g2_input(rng: &mut Rng) -> G2Point {
    match rng.below(11) {
        0..=4 => g2_enc(&valid_g2(rng)),
        5 => g2_enc(&off_subgroup_g2(rng)),
        6 => (fp_bytes(rng), fp_bytes(rng), fp_bytes(rng), fp_bytes(rng)),
        7 => ([0; 48], [0; 48], [0; 48], [0; 48]),
        8 => {
            let (mut x0, mut x1, mut y0, mut y1) = g2_enc(&valid_g2(rng));
            match rng.below(4) {
                0 => x0 = bad_fp(rng, x0),
                1 => x1 = bad_fp(rng, x1),
                2 => y0 = bad_fp(rng, y0),
                _ => y1 = bad_fp(rng, y1),
            }
            (x0, x1, y0, y1)
        }
        9 => {
            let (x0, x1, _, _) = g2_enc(&valid_g2(rng));
            (x0, x1, [0; 48], [0; 48])
        }
        // the flags sit in x.c1, which zkcrypto writes first
        _ => ([0; 48], infinity_flag(), [0; 48], [0; 48]),
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

/// The pairs as the MSM precompiles hand them to the backend, all decoded without error. (A pair
/// that fails to decode is an `Err` item, which the vector test's `fail-msm` files cover.)
fn decoded<T: Copy>(pairs: &[T]) -> impl Iterator<Item = Result<T, PrecompileError>> + '_ {
    pairs.iter().map(|&pair| Ok(pair))
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
        ours: Result<T, PrecompileError>,
        revms: Result<T, PrecompileError>,
    ) {
        match (&ours, &revms) {
            (Ok(a), Ok(b)) if a == b => self.accepted += 1,
            (Err(_), Err(_)) => self.rejected += 1,
            _ => panic!("{what} #{i}: ours {ours:?}, revm's {revms:?}"),
        }
    }

    fn assert_both_sides(&self, what: &str) {
        let Self { accepted, rejected } = *self;
        assert!(accepted > 0 && rejected > 0, "{what}: {accepted} accepted, {rejected} rejected");
    }
}

#[test]
fn g1_add_matches_revm() {
    let (mut rng, mut tally) = (Rng(0x9e37_79b9_7f4a_7c15), Tally::default());
    let crypto = CustomCrypto::default();
    for i in 0..2000 {
        let a = g1_input(&mut rng);
        let b = match rng.below(6) {
            0 => a,
            1 => g1_negated(a).unwrap_or(a),
            _ => g1_input(&mut rng),
        };
        let ours = crypto.bls12_381_g1_add(a, b);
        tally.check("g1_add", i, ours, DefaultCrypto.bls12_381_g1_add(a, b));
    }
    tally.assert_both_sides("g1_add");
}

#[test]
fn g2_add_matches_revm() {
    let (mut rng, mut tally) = (Rng(0x2545_f491_4f6c_dd1d), Tally::default());
    let crypto = CustomCrypto::default();
    for i in 0..1000 {
        let a = g2_input(&mut rng);
        let b = match rng.below(6) {
            0 => a,
            1 => g2_negated(a).unwrap_or(a),
            _ => g2_input(&mut rng),
        };
        let ours = crypto.bls12_381_g2_add(a, b);
        tally.check("g2_add", i, ours, DefaultCrypto.bls12_381_g2_add(a, b));
    }
    tally.assert_both_sides("g2_add");
}

#[test]
fn g1_msm_matches_revm() {
    let (mut rng, mut tally) = (Rng(0x1234_5678_9abc_def1), Tally::default());
    let crypto = CustomCrypto::default();
    for i in 0..300 {
        let pairs: Vec<G1PointScalar> =
            (0..1 + rng.below(4)).map(|_| (g1_input(&mut rng), scalar_input(&mut rng))).collect();
        let ours = crypto.bls12_381_g1_msm(&mut decoded(&pairs));
        tally.check("g1_msm", i, ours, DefaultCrypto.bls12_381_g1_msm(&mut decoded(&pairs)));
    }
    tally.assert_both_sides("g1_msm");
}

#[test]
fn g2_msm_matches_revm() {
    let (mut rng, mut tally) = (Rng(0x0fed_cba9_8765_4321), Tally::default());
    let crypto = CustomCrypto::default();
    for i in 0..100 {
        let pairs: Vec<G2PointScalar> =
            (0..1 + rng.below(3)).map(|_| (g2_input(&mut rng), scalar_input(&mut rng))).collect();
        let ours = crypto.bls12_381_g2_msm(&mut decoded(&pairs));
        tally.check("g2_msm", i, ours, DefaultCrypto.bls12_381_g2_msm(&mut decoded(&pairs)));
    }
    tally.assert_both_sides("g2_msm");
}

#[test]
fn pairing_check_matches_revm() {
    let (mut rng, mut tally) = (Rng(0xdead_beef_cafe_f00d), Tally::default());
    let crypto = CustomCrypto::default();
    for i in 0..200 {
        let product_is_one = rng.below(3) == 0;
        let pairs: Vec<(G1Point, G2Point)> = if product_is_one {
            // e(P, Q) * e(-P, Q), at times with a pair that has the point at infinity on one side
            let (p, q) = (valid_g1(&mut rng), valid_g2(&mut rng));
            let mut pairs = vec![(g1_enc(&p), g2_enc(&q)), (g1_enc(&-p), g2_enc(&q))];
            match rng.below(3) {
                0 => pairs.push((g1_enc(&G1Affine::identity()), g2_enc(&valid_g2(&mut rng)))),
                1 => pairs.push((g1_enc(&valid_g1(&mut rng)), g2_enc(&G2Affine::identity()))),
                _ => {}
            }
            pairs
        } else {
            (0..rng.below(4)).map(|_| (g1_input(&mut rng), g2_input(&mut rng))).collect()
        };
        let ours = crypto.bls12_381_pairing_check(&pairs);
        if product_is_one {
            assert_eq!(ours, Ok(true), "pairing_check #{i}");
        }
        tally.check("pairing_check", i, ours, DefaultCrypto.bls12_381_pairing_check(&pairs));
    }
    tally.assert_both_sides("pairing_check");
}

#[test]
fn map_fp_to_g1_matches_revm() {
    let (mut rng, mut tally) = (Rng(0x5555_aaaa_3333_cccc), Tally::default());
    let crypto = CustomCrypto::default();
    for i in 0..500 {
        let good = fp_bytes(&mut rng);
        let u = match rng.below(10) {
            0..=1 => bad_fp(&mut rng, good),
            2 => [0; 48],
            _ => good,
        };
        let ours = crypto.bls12_381_fp_to_g1(&u);
        tally.check("map_fp_to_g1", i, ours, DefaultCrypto.bls12_381_fp_to_g1(&u));
    }
    tally.assert_both_sides("map_fp_to_g1");
}

#[test]
fn map_fp2_to_g2_matches_revm() {
    let (mut rng, mut tally) = (Rng(0x7777_1111_9999_2222), Tally::default());
    let crypto = CustomCrypto::default();
    for i in 0..300 {
        let (mut c0, mut c1) = (fp_bytes(&mut rng), fp_bytes(&mut rng));
        match rng.below(8) {
            0 => c0 = bad_fp(&mut rng, c0),
            1 => c1 = bad_fp(&mut rng, c1),
            2 => c0 = [0; 48],
            3 => c1 = [0; 48],
            _ => {}
        }
        let ours = crypto.bls12_381_fp2_to_g2((c0, c1));
        tally.check("map_fp2_to_g2", i, ours, DefaultCrypto.bls12_381_fp2_to_g2((c0, c1)));
    }
    tally.assert_both_sides("map_fp2_to_g2");
}
