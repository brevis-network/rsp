#![no_main]
pico_sdk::entrypoint!(main);

use rsp_client_executor::{
    executor::{EthClientExecutor, DESERIALZE_INPUTS},
    io::{CommittedHeader, EthClientExecutorInput},
    utils::profile_report,
};
use std::sync::Arc;

// Linked for its `memcmp`/`bcmp` symbols, which take the place of compiler-builtins'
// byte-at-a-time ones, and for `__wrap_memset`. Nothing calls it directly.
use rsp_guest_mem as _;

/// alloy's `native-keccak` hook: every `keccak256` alloy computes in the guest (EVM opcodes,
/// transaction hashing, trie nodes, the receipts root, bytecode hashing) comes here, to the
/// sponge written directly over Pico's keccak-permute syscall.
///
/// # Safety
/// Called by alloy with a valid input range and a 32-byte output buffer.
#[no_mangle]
pub unsafe extern "C" fn native_keccak256(bytes: *const u8, len: usize, output: *mut u8) {
    let data = core::slice::from_raw_parts(bytes, len);
    // Straight into alloy's buffer rather than through a `[u8; 32]` return value.
    rsp_mpt::keccak256_zkvm_into(data, output);
}

pub fn main() {
    // Read the input: one bincode item for the pointer-based state, and for the arena backend
    // a second item with the witness blob (see the SP1 client for the same layout).
    let input = profile_report!(DESERIALZE_INPUTS, {
        #[cfg(not(feature = "arena"))]
        {
            let input = pico_sdk::io::read_vec();
            bincode::deserialize::<EthClientExecutorInput>(&input).unwrap()
        }
        #[cfg(feature = "arena")]
        {
            let header_bytes = pico_sdk::io::read_vec();
            let mut input: EthClientExecutorInput = bincode::deserialize(&header_bytes).unwrap();
            input.parent_state = pico_sdk::io::read_vec();
            input
        }
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
