#![no_main]
sp1_zkvm::entrypoint!(main);

use rsp_client_executor::{
    executor::{EthClientExecutor, DESERIALZE_INPUTS},
    io::{CommittedHeader, EthClientExecutorInput},
    utils::profile_report,
};

pub fn main() {
    // Read the input.
    //
    // Legacy backend: a single bincode item carrying the whole `EthClientExecutorInput`
    // (including the pointer-based `EthereumState` in `parent_state`).
    //
    // Arena backend: the witness blob is a *separate* stdin item, so bincode only handles the
    // small header (`parent_state` is `#[serde(skip)]`); the blob is then read straight from the
    // SP1 input region into `input.parent_state` and decoded zero-copy in the executor.
    let input = profile_report!(DESERIALZE_INPUTS, {
        #[cfg(not(feature = "arena"))]
        {
            let input = sp1_zkvm::io::read_vec();
            bincode::deserialize::<EthClientExecutorInput>(&input).unwrap()
        }
        #[cfg(feature = "arena")]
        {
            let header_bytes = sp1_zkvm::io::read_vec();
            let mut input: EthClientExecutorInput = bincode::deserialize(&header_bytes).unwrap();
            input.parent_state = sp1_zkvm::io::read_vec().into();
            input
        }
    });

    // Execute the block under the chain configuration the input names.
    let committed = EthClientExecutor::execute_committed(input).expect("failed to execute client");

    // Commit the derived header together with the digest of that configuration.
    sp1_zkvm::io::commit::<CommittedHeader>(&committed);
}
