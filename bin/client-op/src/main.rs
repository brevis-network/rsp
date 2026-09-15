#![no_main]
sp1_zkvm::entrypoint!(main);

use rsp_client_executor::{
    executor::{OpClientExecutor, DESERIALZE_INPUTS},
    io::{CommittedHeader, OpClientExecutorInput},
    utils::profile_report,
};
use std::sync::Arc;

pub fn main() {
    // Read the input.
    let input = profile_report!(DESERIALZE_INPUTS, {
        let input = sp1_zkvm::io::read_vec();
        bincode::deserialize::<OpClientExecutorInput>(&input).unwrap()
    });

    // Execute the block.
    let executor = OpClientExecutor::optimism(Arc::new((&input.genesis).try_into().unwrap()));
    let committed = executor.execute(input).expect("failed to execute client");

    // Commit the derived header together with the digest of the configuration it ran under;
    // see `CommittedHeader`.
    sp1_zkvm::io::commit::<CommittedHeader>(&committed);
}
