//! KZG point evaluation for the EVM's precompile, on kzg-rs.

use evm2::precompiles::{Crypto, PrecompileHalt};
use kzg_rs::{Bytes32, Bytes48, KzgProof, KzgSettings};

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
}
