#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use alloy_primitives::{map::HashMap, Address, B256, U256};
use evm2::{
    bytecode::Bytecode,
    evm::{AccountInfo, DatabaseRef},
};
use reth_storage_errors::provider::ProviderError;
use serde::{Deserialize, Serialize};

/// A database used to witness state inside the zkVM.
#[derive(Debug, Serialize, Deserialize)]
pub struct WitnessDb {
    /// The accounts.
    pub accounts: HashMap<Address, AccountInfo>,
    /// The storage values, indexed by account address and slot.
    pub storage: HashMap<Address, HashMap<U256, U256>>,
    /// The block hashes, indexed by block number.
    pub block_hashes: HashMap<u64, B256>,
}

impl DatabaseRef for WitnessDb {
    type Error = ProviderError;

    fn get_account(&self, address: &Address) -> Result<Option<AccountInfo>, Self::Error> {
        // Even absent accounts are loaded as `None`, so if an entry is missing from `HashMap` we
        // need to panic. Otherwise the EVM would take it for an uninitialized account.
        Ok(Some(self.accounts.get(address).cloned().unwrap()))
    }

    fn get_code_by_hash(&self, _code_hash: &B256) -> Result<Bytecode, Self::Error> {
        unimplemented!()
    }

    fn get_storage(&self, address: &Address, index: &U256) -> Result<U256, Self::Error> {
        // Absence of storage trie or slot must be treated as an error here. Otherwise it's possible
        // to trick the EVM into believing a slot is `0` when it's not.
        Ok(*self.storage.get(address).unwrap().get(index).unwrap())
    }

    fn get_block_hash(&self, number: &U256) -> Result<B256, Self::Error> {
        Ok(*self.block_hashes.get(&number.to::<u64>()).unwrap())
    }
}
