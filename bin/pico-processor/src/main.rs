#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod cli;

use alloy_provider::{Provider, ProviderBuilder, WsConnect};
use clap::Parser;
use cli::Args;
use dotenvy::dotenv;
use futures::StreamExt;
use pico_sdk::client::DefaultProverClient;
use pico_vm::{configs::stark_config::KoalaBearPoseidon2, emulator::stdin::EmulatorStdinBuilder};
use rsp_client_executor::io::EthClientExecutorInput;
use rsp_host_executor::{
    create_eth_block_execution_strategy_factory, BlockExecutor, EthExecutorComponents, FullExecutor,
};
use rsp_provider::create_provider;
use std::{env, fs, path::Path};
use tracing::info;
use tracing_subscriber::{
    filter::EnvFilter, fmt, prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt,
};

// reth elf file path
// in `bin/client` run `cargo pico build` to generate this elf file
const RETH_ELF_PATH: &str = "../client/elf/riscv32im-pico-zkvm-elf";

#[tokio::main]
async fn main() {
    // init environment
    dotenv().ok();
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info");
    }
    tracing_subscriber::registry().with(fmt::layer()).with(EnvFilter::from_default_env()).init();
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("pico-processor: failed to install rustls crypto provider");

    // parse arguments
    let args = Args::parse();
    let config =
        args.as_config().await.expect("pico-processor: failed to convert cli arguments to config");

    // get the current latest block number
    let http_provider = create_provider(args.provider.rpc_http_url);
    let current_block_number = http_provider
        .get_block_number()
        .await
        .expect("pico-processor: failed to get current latest block number");
    info!("pico-processor: current latest block number is {current_block_number}");

    // initialize a websocket rpc connection for receiving latest blocks
    let ws_conn = WsConnect::new(args.provider.rpc_ws_url);
    let ws_provider = ProviderBuilder::new()
        .connect_ws(ws_conn)
        .await
        .expect("pico-processor: failed to connect to rpc websocket URL");
    let subscription = ws_provider
        .subscribe_blocks()
        .await
        .expect("pico-processor: failed to subscribe the latest blocks");
    let mut latest_block_receiver = subscription.into_stream();

    // initialize block executor
    let block_execution_strategy_factory =
        create_eth_block_execution_strategy_factory(&config.genesis, config.custom_beneficiary);
    let executor = FullExecutor::<EthExecutorComponents<_>, _>::try_new(
        http_provider,
        block_execution_strategy_factory,
        (),
        config,
    )
    .await
    .unwrap_or_else(|e| panic!("pico-processor: failed to build executor {e:?}"));

    // monitor and execute the latest blocks
    while let Some(header) = latest_block_receiver.next().await {
        let block_number = header.number;
        info!("pico-processor: waiting for block {block_number}");
        executor
            .wait_for_block(block_number)
            .await
            .expect("pico-processor: failed to wait for block {block_number}");

        info!("pico-processor: fetching block {block_number}");
        let input = executor
            .execute(header.number, None)
            .await
            .expect("pico-processor: failed to execute block {block_number}");

        if args.is_input_emulated {
            info!("pico-processor: start to emulate block {block_number}");

            // check if reth elf has been build
            let elf_path = Path::new(RETH_ELF_PATH);
            if !elf_path.exists() {
                panic!("pico-processor: run `cargo pico build` in `bin/client` first");
            }

            // generate stdin builder
            let mut stdin_builder = EmulatorStdinBuilder::<Vec<u8>, KoalaBearPoseidon2>::default();
            stdin_builder.write::<EthClientExecutorInput<'_>>(&input);

            // emulate reth with block input
            let elf = fs::read(elf_path).expect("pico-processor: failed to read reth ELF file");
            let prover_client = DefaultProverClient::new(&elf);
            prover_client.emulate(stdin_builder);

            info!("pico-processor: finish emulating block {block_number}");
        }
    }
}
