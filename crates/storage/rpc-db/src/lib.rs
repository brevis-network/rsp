#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use alloy_consensus::Header;
use revm_database::{BundleState, DatabaseRef};
use revm_state::Bytecode;
use rsp_mpt::EthereumState;

mod execution_witness;
pub use execution_witness::ExecutionWitnessRpcDb;

mod error;
pub use error::RpcDbError;

pub trait RpcDb: DatabaseRef {
    fn state(&self, bundle_state: &BundleState) -> Result<EthereumState, RpcDbError>;

    /// Gets all account bytecodes.
    fn bytecodes(&self) -> Vec<Bytecode>;

    // Fetches the parent headers needed to constrain the BLOCKHASH opcode.
    fn ancestor_headers(&self) -> Result<Vec<Header>, RpcDbError>;
}
