//! [`CustomCrypto`]'s SHA-256, ECRECOVER and P256VERIFY against evm2's default backend.
//!
//! `CustomCrypto` runs these on older major versions of the RustCrypto crates than evm2 does (the
//! ones Pico accelerates), so the two sides share no code. On random inputs, valid and broken in
//! each way the checks tell apart, both must reject the same inputs and return the same bytes for
//! the rest.
//!
//! These run the plain-Rust path. In the guest the same code does its arithmetic in Pico
//! syscalls, which a host test cannot reach.

use alloy_primitives::hex;
use evm2::precompiles::{Crypto, DefaultCrypto};
use rsp_client_executor::custom::CustomCrypto;

/// n, the order of secp256k1.
const K256_N: [u8; 32] = hex!("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141");
/// n, the order of P-256.
const P256_N: [u8; 32] = hex!("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551");

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
        self.fill(&mut out);
        out
    }

    fn fill(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes()[..chunk.len()]);
        }
    }
}

/// `a - b` for big-endian 32-byte integers with `a >= b`.
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

/// Counts both outcomes, so that inputs which never reach one side of the checks fail the test.
#[derive(Default)]
struct Tally {
    accepted: usize,
    rejected: usize,
}

impl Tally {
    fn assert_both_sides(&self, what: &str) {
        let Self { accepted, rejected } = *self;
        assert!(accepted > 0 && rejected > 0, "{what}: {accepted} accepted, {rejected} rejected");
    }
}

#[test]
fn sha256_matches_evm2() {
    let mut rng = Rng(0x243f_6a88_85a3_08d3);
    let (ours, evm2s) = (CustomCrypto::default(), DefaultCrypto::new());
    // Around the padding boundaries of one and two blocks, and a few long inputs.
    for len in [0, 1, 31, 32, 54, 55, 56, 57, 63, 64, 65, 118, 119, 120, 127, 128, 1000, 4096] {
        let mut input = vec![0u8; len];
        rng.fill(&mut input);
        assert_eq!(ours.sha256(&input), evm2s.sha256(&input), "sha256 of {len} bytes");
    }
    assert_eq!(
        ours.sha256(b"abc"),
        hex!("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    );
}

#[test]
fn ecrecover_matches_evm2() {
    use k256::ecdsa::SigningKey;

    let (mut rng, mut tally) = (Rng(0x1319_8a2e_0370_7344), Tally::default());
    let (ours, evm2s) = (CustomCrypto::default(), DefaultCrypto::new());
    for i in 0..600 {
        let key = loop {
            if let Ok(key) = SigningKey::from_slice(&rng.bytes::<32>()) {
                break key;
            }
        };
        let msg: [u8; 32] = rng.bytes();
        let (signature, recovery_id) = key.sign_prehash_recoverable(&msg).unwrap();
        let mut sig: [u8; 64] = signature.to_bytes().into();
        // The precompile passes `v - 27`, so only the parity bit.
        let mut recid = recovery_id.to_byte() & 1;
        let kind = rng.below(10);
        match kind {
            // as signed: low s
            0..=2 => {}
            // the same signature written with the high s
            3 => {
                let s = be_sub(K256_N, &sig[32..]);
                sig[32..].copy_from_slice(&s);
                recid ^= 1;
            }
            // the other recovery id: some other key, or none
            4 => recid ^= 1,
            5 => sig[..32].fill(0),
            6 => sig[32..].fill(0),
            // r and s have to be below n
            7 => sig[..32].copy_from_slice(&K256_N),
            8 => sig[32..].copy_from_slice(&[0xff; 32]),
            _ => sig = rng.bytes(),
        }

        let a = ours.secp256k1_ecrecover(&sig, recid, &msg);
        let b = evm2s.secp256k1_ecrecover(&sig, recid, &msg);
        match (&a, &b) {
            (Ok(x), Ok(y)) if x == y => tally.accepted += 1,
            (Err(_), Err(_)) => tally.rejected += 1,
            _ => panic!("ecrecover #{i} (kind {kind}): ours {a:?}, evm2's {b:?}"),
        }
        if kind <= 3 {
            // An untouched signature, in either form, recovers the signer.
            let public_key = key.verifying_key().to_encoded_point(false);
            let mut address = alloy_primitives::keccak256(&public_key.as_bytes()[1..]);
            address[..12].fill(0);
            assert_eq!(a, Ok(address.0), "ecrecover #{i} (kind {kind})");
        }
    }
    tally.assert_both_sides("ecrecover");
}

#[test]
fn p256_verify_matches_evm2() {
    use p256::ecdsa::{signature::hazmat::PrehashSigner, Signature, SigningKey};

    let (mut rng, mut tally) = (Rng(0xa409_3822_299f_31d0), Tally::default());
    let (ours, evm2s) = (CustomCrypto::default(), DefaultCrypto::new());
    for i in 0..400 {
        let key = loop {
            if let Ok(key) = SigningKey::from_slice(&rng.bytes::<32>()) {
                break key;
            }
        };
        let mut msg: [u8; 32] = rng.bytes();
        let signature: Signature = key.sign_prehash(&msg).unwrap();
        let mut sig: [u8; 64] = signature.to_bytes().into();
        let public_key = key.verifying_key().to_encoded_point(false);
        let mut pk: [u8; 64] = public_key.as_bytes()[1..].try_into().unwrap();
        let kind = rng.below(12);
        match kind {
            0..=2 => {}
            // (r, n - s) verifies too: the precompile has no low-s rule
            3 => {
                let s = be_sub(P256_N, &sig[32..]);
                sig[32..].copy_from_slice(&s);
            }
            4 => msg[rng.below(32) as usize] ^= 1,
            5 => sig[..32].fill(0),
            6 => sig[32..].fill(0),
            // r and s have to be below n
            7 => sig[..32].copy_from_slice(&P256_N),
            8 => sig[32..].copy_from_slice(&[0xff; 32]),
            // all but surely not a point on the curve
            9 => pk = rng.bytes(),
            // (0, 0) is not on the curve either
            10 => pk = [0; 64],
            // a valid key, but not the signer's
            _ => {
                let other = loop {
                    if let Ok(key) = SigningKey::from_slice(&rng.bytes::<32>()) {
                        break key;
                    }
                };
                let point = other.verifying_key().to_encoded_point(false);
                pk.copy_from_slice(&point.as_bytes()[1..]);
            }
        }

        let a = ours.secp256r1_verify_signature(&msg, &sig, &pk);
        let b = evm2s.secp256r1_verify_signature(&msg, &sig, &pk);
        assert_eq!(a, b, "p256 verify #{i} (kind {kind})");
        if kind <= 3 {
            assert!(a, "p256 verify #{i} (kind {kind}): a valid signature was rejected");
        }
        if a {
            tally.accepted += 1;
        } else {
            tally.rejected += 1;
        }
    }
    tally.assert_both_sides("p256 verify");
}
