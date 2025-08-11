use crate::{RpcDb, RpcDbError};
use alloy_consensus::Header;
use alloy_primitives::{map::HashMap, Address, B256};
use alloy_rlp::Decodable;
use alloy_trie::TrieAccount;
use reth_storage_errors::ProviderError;
use revm_database::{BundleState, DatabaseRef};
use revm_primitives::{keccak256, ruint::aliases::U256};
use revm_state::{AccountInfo, Bytecode};
use rsp_mpt::EthereumState;

#[derive(Debug)]
pub struct ExecutionWitnessRpcDb {
    pub state: EthereumState,
    pub codes: HashMap<B256, Bytecode>,
    pub ancestor_headers: HashMap<u64, Header>,
}

impl DatabaseRef for ExecutionWitnessRpcDb {
    type Error = ProviderError;

    fn basic_ref(&self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        let hash = keccak256(address);
        if let Some(mut bytes) = self
            .state
            .state_trie
            .get(hash.as_ref())
            .map_err(|err| ProviderError::TrieWitnessError(err.to_string()))?
        {
            let account = TrieAccount::decode(&mut bytes)?;
            let account_info = AccountInfo {
                balance: account.balance,
                nonce: account.nonce,
                code_hash: account.code_hash,
                code: None,
            };

            Ok(Some(account_info))
        } else {
            Ok(None)
        }
    }

    fn code_by_hash_ref(&self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        self.codes
            .get(&code_hash)
            .ok_or_else(|| {
                ProviderError::TrieWitnessError(format!("Code not found for {code_hash}"))
            })
            .cloned()
    }

    fn storage_ref(&self, address: Address, index: U256) -> Result<U256, Self::Error> {
        let slot = B256::from(index);
        let hashed_address = keccak256(address);
        let hashed_slot = keccak256(slot);
        if let Some(mut value) = self
            .state
            .storage_tries
            .get(&hashed_address)
            .and_then(|storage_trie| storage_trie.get(hashed_slot.as_slice()).unwrap())
        {
            Ok(U256::decode(&mut value)?)
        } else {
            Ok(U256::ZERO)
        }
    }

    fn block_hash_ref(&self, number: u64) -> Result<B256, Self::Error> {
        let header = self.ancestor_headers.get(&number).ok_or_else(|| {
            ProviderError::TrieWitnessError(format!("Header {number} not found in the ancestors"))
        })?;

        Ok(header.hash_slow())
    }
}

impl RpcDb for ExecutionWitnessRpcDb {
    fn state(&self, _bundle_state: &BundleState) -> Result<EthereumState, RpcDbError> {
        Ok(self.state.clone())
    }

    fn bytecodes(&self) -> Vec<Bytecode> {
        self.codes.values().cloned().collect()
    }

    fn ancestor_headers(&self) -> Result<Vec<Header>, RpcDbError> {
        let mut ancestor_headers: Vec<Header> = self.ancestor_headers.values().cloned().collect();
        ancestor_headers.sort_by(|a, b| b.number.cmp(&a.number));
        Ok(ancestor_headers)
    }
}
