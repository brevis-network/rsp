#![no_main]
pico_sdk::entrypoint!(main);

use rsp_client_executor::{
    executor::{EthClientExecutor, DESERIALZE_INPUTS},
    io::{CommittedHeader, EthClientExecutorInput},
    utils::profile_report,
};
use std::sync::Arc;

pub fn main() {
    // Read the input. The deserialized input borrows the flat trie blobs zero-copy from `raw`,
    // so the buffer must outlive it.
    let raw = pico_sdk::io::read_vec();
    let input = profile_report!(DESERIALZE_INPUTS, {
        bincode::deserialize::<EthClientExecutorInput>(&raw).unwrap()
    });

    // Execute the block.
    let executor = EthClientExecutor::eth(
        Arc::new((&input.genesis).try_into().unwrap()),
        input.custom_beneficiary,
    );
    let header = executor.execute(input).expect("failed to execute client");

    // Commit the block header.
    pico_sdk::io::commit::<CommittedHeader>(&header.into());
}
