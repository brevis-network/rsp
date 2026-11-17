//! Runs the client natively on saved inputs (the files a host wrote for the zkVM guest) and checks
//! that the header it derives hashes to the hash of the block inside the input.
//!
//! usage: exec_input <input.bin>...
//! Each file is the one buffer a Pico guest reads from stdin (`rsp_client_executor::stdin`), and
//! must come from a host built with the same witness backend.

use std::{sync::Arc, time::Instant};

use rsp_client_executor::{executor::EthClientExecutor, io::EthClientExecutorInput, stdin};

fn main() {
    let (mut passed, mut failed) = (0u32, 0u32);
    for path in std::env::args().skip(1) {
        let buffer = std::fs::read(&path).expect("read input");
        let (input, witness) = stdin::split(buffer, |bytes| {
            bincode::deserialize::<EthClientExecutorInput>(bytes).expect("decode input")
        })
        .expect("the file is not a joined input buffer");
        #[cfg(any(feature = "arena", feature = "flat"))]
        let input = EthClientExecutorInput { parent_state: witness, ..input };
        #[cfg(not(any(feature = "arena", feature = "flat")))]
        assert!(witness.is_empty(), "the default backend takes no witness blob");
        let expected = input.current_block.header.hash_slow();
        let number = input.current_block.header.number;
        let gas_used = input.current_block.header.gas_used;
        let transactions = input.current_block.body.transactions.len();

        let executor = EthClientExecutor::eth(
            Arc::new((&input.genesis).try_into().unwrap()),
            input.custom_beneficiary,
        );
        let started = Instant::now();
        let outcome = executor.execute(input);
        let seconds = started.elapsed().as_secs_f64();
        match outcome {
            Ok(header) if header.hash_slow() == expected => {
                passed += 1;
                println!(
                    "{number} ok hash={expected} gas={:.1}M txs={transactions} client={seconds:.2}s",
                    gas_used as f64 / 1e6
                );
            }
            Ok(header) => {
                failed += 1;
                println!(
                    "{number} FAILED header hashes to {}, the block is {expected}",
                    header.hash_slow()
                );
            }
            Err(err) => {
                failed += 1;
                println!("{number} FAILED {err}");
            }
        }
    }
    println!("passed {passed}, failed {failed}");
    std::process::exit(if failed == 0 { 0 } else { 1 });
}
