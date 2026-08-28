#![no_main]
pico_sdk::entrypoint!(main);

use rsp_client_executor::{
    executor::{EthClientExecutor, DESERIALZE_INPUTS},
    io::{CommittedHeader, EthClientExecutorInput},
    utils::profile_report,
};
use std::sync::Arc;

// Linked for its `memcmp`/`bcmp` symbols, which override compiler-builtins'
// byte-at-a-time versions. Nothing calls it directly.
use rsp_guest_mem as _;

/// alloy's `native-keccak` hook: routes every alloy `keccak256` call in the guest (EVM
/// opcodes, transaction hashing, receipts root, bytecode hashing) through the direct
/// keccak-permute-syscall sponge.
///
/// # Safety
/// Called by alloy with a valid input range and a 32-byte output buffer.
#[no_mangle]
pub unsafe extern "C" fn native_keccak256(bytes: *const u8, len: usize, output: *mut u8) {
    let data = core::slice::from_raw_parts(bytes, len);
    // Straight into alloy's buffer: going through a `[u8; 32]` return value cost a 32-byte
    // byte-wise copy out of the sponge plus this one.
    rsp_mpt::keccak256_zkvm_into(data, output);
}

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
