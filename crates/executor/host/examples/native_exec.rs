//! Runs the host and then the client natively, block by block, against an RPC endpoint.
//!
//! For each block: fetch the block and its execution witness and execute it to build the client
//! input (the host checks the state root and the block hash on the way); then run the client
//! executor on that input, which is the code the zkVM guest runs, and compare the hash of the
//! header it returns with the block hash the node reports.
//!
//! usage: native_exec <rpc-url> <mainnet|sepolia> <block | from..to>...
//! Set DUMP_DIR to also write each block's client input to `<DUMP_DIR>/<block>.bin`.
//! Set STATE_BACKEND=proofs to fetch state with `eth_getProof` instead of
//! `debug_executionWitness`.

use std::{sync::Arc, time::Instant};

use alloy_provider::{network::Ethereum, Provider, RootProvider};
use reth_chainspec::ChainSpec;
use rsp_client_executor::executor::EthClientExecutor;
use rsp_host_executor::{EthHostExecutor, StateBackend};
use rsp_primitives::genesis::Genesis;
use url::Url;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: native_exec <rpc-url> <mainnet|sepolia> <block | from..to>...");
        std::process::exit(2);
    }
    let rpc_url = Url::parse(&args[0]).expect("invalid rpc url");
    let genesis = match args[1].as_str() {
        "mainnet" => Genesis::Mainnet,
        "sepolia" => Genesis::Sepolia,
        other => panic!("unknown chain `{other}`"),
    };
    let mut blocks = Vec::new();
    for arg in &args[2..] {
        match arg.split_once("..") {
            Some((from, to)) => {
                blocks.extend(from.parse::<u64>().unwrap()..=to.parse::<u64>().unwrap())
            }
            None => blocks.push(arg.parse::<u64>().unwrap()),
        }
    }

    let state_backend: StateBackend = std::env::var("STATE_BACKEND")
        .map(|name| name.parse().expect("invalid STATE_BACKEND"))
        .unwrap_or(StateBackend::ExecutionWitness);

    let chain_spec: Arc<ChainSpec> = Arc::new((&genesis).try_into().unwrap());
    let host_executor = EthHostExecutor::eth(chain_spec.clone(), None);
    let client_executor = EthClientExecutor::eth(chain_spec, None);
    let provider = RootProvider::<Ethereum>::new_http(rpc_url);
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();

    let (mut passed, mut failed) = (0u32, 0u32);
    for block_number in blocks {
        let fetched =
            runtime.block_on(async { provider.get_block_by_number(block_number.into()).await });
        let expected = match fetched {
            Ok(Some(block)) => block.header.hash,
            other => {
                println!("{block_number} FAILED cannot fetch the block: {other:?}");
                failed += 1;
                continue;
            }
        };

        let started = Instant::now();
        let input = match runtime.block_on(host_executor.execute(
            block_number,
            &provider,
            genesis.clone(),
            None,
            false,
            state_backend,
        )) {
            Ok(input) => input,
            Err(err) => {
                println!("{block_number} FAILED in the host: {err}");
                failed += 1;
                continue;
            }
        };
        let host_seconds = started.elapsed().as_secs_f64();

        let serialized = bincode::serialize(&input).unwrap();
        // With DUMP_DIR set, keep the input: it is exactly what the zkVM guest reads from stdin.
        if let Ok(dir) = std::env::var("DUMP_DIR") {
            std::fs::write(format!("{dir}/{block_number}.bin"), &serialized).unwrap();
            // The arena backend passes the witness as a second stdin item, outside bincode.
            #[cfg(feature = "arena")]
            std::fs::write(format!("{dir}/{block_number}.blob"), &input.parent_state).unwrap();
        }
        let input_bytes = serialized.len();
        let gas_used = input.current_block.header.gas_used;
        let transactions = input.current_block.body.transactions.len();
        let access_list_hash = input.current_block.header.block_access_list_hash;

        let started = Instant::now();
        let outcome = client_executor.execute(input);
        let client_seconds = started.elapsed().as_secs_f64();

        match outcome {
            Ok(header) if header.hash_slow() == expected => {
                passed += 1;
                println!(
                    "{block_number} ok hash={expected} gas={:.1}M txs={transactions} bal={} input={:.2}MB host={host_seconds:.2}s client={client_seconds:.2}s",
                    gas_used as f64 / 1e6,
                    if access_list_hash.is_some() { "yes" } else { "no" },
                    input_bytes as f64 / 1e6,
                );
            }
            Ok(header) => {
                failed += 1;
                println!("{block_number} FAILED the client's header hashes to {}, the node says {expected}", header.hash_slow());
            }
            Err(err) => {
                failed += 1;
                println!("{block_number} FAILED in the client: {err}");
            }
        }
    }

    println!("passed {passed}, failed {failed}");
    std::process::exit(if failed == 0 { 0 } else { 1 });
}
