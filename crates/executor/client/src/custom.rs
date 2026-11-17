//! The cryptography behind the EVM's precompiles, on the libraries Pico accelerates.
//!
//! evm2's own backends are k256 0.14, p256 0.14 and sha2 0.11 from RustCrypto, and arkworks for
//! the pairing curves. Pico's patched crates, whose field and curve arithmetic run as syscalls in
//! the guest, are k256 0.13, p256 0.13, sha2 0.10, substrate-bn and bls12_381, so
//! [`CustomCrypto`] sends each of those precompiles there; left to evm2 they would run in
//! software. Outside the guest the same code runs as plain Rust.
//!
//! The KZG point evaluation is on kzg-rs.

use alloy_primitives::keccak256;
use evm2::precompiles::{Crypto, PrecompileHalt};
use k256::ecdsa::{RecoveryId, Signature, VerifyingKey};
use kzg_rs::{Bytes32, Bytes48, KzgProof, KzgSettings};
use sha2::{Digest, Sha256};

use crate::eip2537::{self, G1Point, G1PointScalar, G2Point, G2PointScalar};

#[derive(Debug)]
pub struct CustomCrypto {
    kzg_settings: KzgSettings,
}

impl Default for CustomCrypto {
    fn default() -> Self {
        Self { kzg_settings: KzgSettings::load_trusted_setup_file().unwrap() }
    }
}

impl Crypto for CustomCrypto {
    fn sha256(&self, input: &[u8]) -> [u8; 32] {
        Sha256::digest(input).into()
    }

    fn secp256k1_ecrecover(
        &self,
        sig: &[u8; 64],
        recid: u8,
        msg: &[u8; 32],
    ) -> Result<[u8; 32], PrecompileHalt> {
        ecrecover(sig, recid, msg).ok_or(PrecompileHalt::Secp256k1RecoverFailed)
    }

    fn secp256r1_verify_signature(&self, msg: &[u8; 32], sig: &[u8; 64], pk: &[u8; 64]) -> bool {
        p256_verify(msg, sig, pk).is_some()
    }

    fn verify_kzg_proof(
        &self,
        z: &[u8; 32],
        y: &[u8; 32],
        commitment: &[u8; 48],
        proof: &[u8; 48],
    ) -> Result<(), PrecompileHalt> {
        match KzgProof::verify_kzg_proof(
            &Bytes48(*commitment),
            &Bytes32(*z),
            &Bytes32(*y),
            &Bytes48(*proof),
            &self.kzg_settings,
        ) {
            Ok(true) => Ok(()),
            // An input that does not decode fails the precompile the same way a bad proof does.
            Ok(false) | Err(_) => Err(PrecompileHalt::BlobVerifyKzgProofFailed),
        }
    }

    // EIP-196 and EIP-197, in the guest only; see `crate::bn254` for why.

    #[cfg(target_os = "zkvm")]
    fn bn254_g1_add(&self, p1: &[u8], p2: &[u8]) -> Result<[u8; 64], PrecompileHalt> {
        crate::bn254::g1_add(p1, p2)
    }

    #[cfg(target_os = "zkvm")]
    fn bn254_g1_mul(&self, point: &[u8], scalar: &[u8]) -> Result<[u8; 64], PrecompileHalt> {
        crate::bn254::g1_mul(point, scalar)
    }

    #[cfg(target_os = "zkvm")]
    fn bn254_pairing_check(&self, pairs: &[(&[u8], &[u8])]) -> Result<bool, PrecompileHalt> {
        crate::bn254::pairing_check(pairs)
    }

    // EIP-2537; see `crate::eip2537`.

    fn bls12_381_g1_add(&self, a: G1Point, b: G1Point) -> Result<[u8; 96], PrecompileHalt> {
        eip2537::g1_add(a, b)
    }

    fn bls12_381_g1_msm(
        &self,
        pairs: &mut dyn Iterator<Item = Result<G1PointScalar, PrecompileHalt>>,
    ) -> Result<[u8; 96], PrecompileHalt> {
        eip2537::g1_msm(pairs)
    }

    fn bls12_381_g2_add(&self, a: G2Point, b: G2Point) -> Result<[u8; 192], PrecompileHalt> {
        eip2537::g2_add(a, b)
    }

    fn bls12_381_g2_msm(
        &self,
        pairs: &mut dyn Iterator<Item = Result<G2PointScalar, PrecompileHalt>>,
    ) -> Result<[u8; 192], PrecompileHalt> {
        eip2537::g2_msm(pairs)
    }

    fn bls12_381_pairing_check(
        &self,
        pairs: &[(G1Point, G2Point)],
    ) -> Result<bool, PrecompileHalt> {
        eip2537::pairing_check(pairs)
    }

    fn bls12_381_fp_to_g1(&self, fp: &[u8; 48]) -> Result<[u8; 96], PrecompileHalt> {
        eip2537::map_fp_to_g1(fp)
    }

    fn bls12_381_fp2_to_g2(&self, fp2: ([u8; 48], [u8; 48])) -> Result<[u8; 192], PrecompileHalt> {
        eip2537::map_fp2_to_g2(fp2)
    }
}

/// ECRECOVER: the signer's address, left-padded to 32 bytes, or `None` when the signature
/// recovers no key. The steps of evm2's k256 backend, on k256 0.13.
fn ecrecover(sig: &[u8; 64], mut recid: u8, msg: &[u8; 32]) -> Option<[u8; 32]> {
    let mut sig = Signature::from_slice(sig).ok()?;
    // k256 recovers from a low-s signature only; (r, n - s) with the other recovery id is the
    // same signature.
    if let Some(normalized) = sig.normalize_s() {
        sig = normalized;
        recid ^= 1;
    }
    let recid = RecoveryId::from_byte(recid)?;
    let key = VerifyingKey::recover_from_prehash(msg, &sig, recid).ok()?;
    let mut hash = keccak256(&key.to_encoded_point(false).as_bytes()[1..]);
    hash[..12].fill(0);
    Some(hash.0)
}

/// P256VERIFY: `Some` when `sig` (`r || s`) is a signature of the prehash `msg` under the public
/// key `pk` (`x || y`). The steps of evm2's p256 backend, on p256 0.13.
fn p256_verify(msg: &[u8; 32], sig: &[u8; 64], pk: &[u8; 64]) -> Option<()> {
    use p256::{
        ecdsa::{signature::hazmat::PrehashVerifier, Signature, VerifyingKey},
        EncodedPoint,
    };

    let signature = Signature::from_slice(sig).ok()?;
    let point = EncodedPoint::from_untagged_bytes(&(*pk).into());
    let key = VerifyingKey::from_encoded_point(&point).ok()?;
    key.verify_prehash(msg, &signature).ok()
}
