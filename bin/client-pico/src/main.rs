#![no_main]
pico_sdk::entrypoint!(main);

use rsp_client_executor::{
    executor::{EthClientExecutor, DESERIALZE_INPUTS},
    io::{CommittedHeader, EthClientExecutorInput},
    stdin,
    utils::profile_report,
};

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
    // Read the input. brevis-vm's provers hand the guest one buffer for a block: the witness
    // blob (of the arena and flat backends), the bincode-serialized input, and the blob's length
    // (`rsp_client_executor::stdin`). The blob stays where it was read; nothing copies it.
    let input = profile_report!(DESERIALZE_INPUTS, {
        let (input, witness) = stdin::split(pico_sdk::io::read_vec(), |bytes| {
            bincode::deserialize::<EthClientExecutorInput>(bytes).unwrap()
        })
        .expect("the input is not a joined buffer");
        #[cfg(any(feature = "arena", feature = "flat"))]
        let input = EthClientExecutorInput { parent_state: witness, ..input };
        // A blob here means the host was built with another witness backend than this guest.
        #[cfg(not(any(feature = "arena", feature = "flat")))]
        assert!(witness.is_empty(), "the default backend takes no witness blob");
        input
    });

    // Execute the block under the chain configuration the input names.
    let committed = EthClientExecutor::execute_committed(input).expect("failed to execute client");

    // Commit the derived header together with the digest of that configuration; see
    // `CommittedHeader` for why the header alone does not identify the statement.
    pico_sdk::io::commit::<CommittedHeader>(&committed);
}
