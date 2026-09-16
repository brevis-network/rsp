//! Host-side databases that fetch state over RPC and record what a block touched.
//!
//! Executing a block against one of these yields the witness the guest is given: the accounts,
//! slots, bytecodes and ancestor headers actually read, plus the proofs binding them to the
//! parent state root. [`BasicRpcDb`] builds that from `eth_getProof`; `ExecutionWitnessRpcDb`
//! (feature `execution-witness`) from `debug_executionWitness`.

#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use alloy_consensus::Header;
use alloy_provider::Network;
use async_trait::async_trait;
use revm_database::{BundleState, DatabaseRef};
use revm_state::Bytecode;
use rsp_mpt::EthereumState;

mod basic;
pub use basic::BasicRpcDb;

#[cfg(feature = "execution-witness")]
mod execution_witness;
#[cfg(feature = "execution-witness")]
pub use execution_witness::ExecutionWitnessRpcDb;

mod error;
pub use error::RpcDbError;

#[async_trait]
pub trait RpcDb<N: Network>: DatabaseRef {
    async fn state(&self, bundle_state: &BundleState) -> Result<EthereumState, RpcDbError>;

    /// Gets all account bytecodes.
    fn bytecodes(&self) -> Vec<Bytecode>;

    // Fetches the parent headers needed to constrain the BLOCKHASH opcode.
    async fn ancestor_headers(&self) -> Result<Vec<Header>, RpcDbError>;
}
