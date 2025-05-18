use std::{sync::Arc, time::Duration};

use alloy_network::Ethereum;
use alloy_provider::{Provider, ProviderBuilder, RootProvider, WsConnect};
use clap::Parser;
use cli::Args;
use eth_proofs::EthProofsClient;
use futures::{future::ready, StreamExt};
use rsp_host_executor::{
    alerting::AlertingClient, create_eth_block_execution_strategy_factory, BlockExecutor,
    EthExecutorComponents, FullExecutor,
};
use rsp_provider::create_provider;
use sp1_sdk::ProverClient;
use tokio::time::sleep;
use tracing::{debug, error, info};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod cli;

mod eth_proofs;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // Initialize the environment variables.
    dotenv::dotenv().ok();

    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }

    // Initialize the logger.
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::from_default_env()
                .add_directive("sp1_core_machine=warn".parse().unwrap())
                .add_directive("sp1_core_executor=warn".parse().unwrap())
                .add_directive("sp1_prover=warn".parse().unwrap()),
        )
        .init();

    // Parse the command line arguments.
    let args = Args::parse();
    let config = args.as_config().await?;

    let elf = include_bytes!("../elf/rsp-client-elf").to_vec();

    let ws = WsConnect::new(args.ws_rpc_url);
    let ws_provider = ProviderBuilder::new().on_ws(ws).await?;
    let http_provider = create_provider::<Ethereum>(args.http_rpc_url);

    // Subscribe to block headers.
    let subscription = ws_provider.subscribe_blocks().await?;
    let mut stream =
        subscription.into_stream().filter(|h| ready(h.number % args.block_interval == 0));

    let mut builder = ProverClient::builder().cuda();
    if let Some(endpoint) = &args.moongate_endpoint {
        builder = builder.with_moongate_endpoint(endpoint)
    }

    // let client = Arc::new(builder.build());

    // let executor = FullExecutor::<EthExecutorComponents<_, _>, _>::try_new(
    //     http_provider.clone(),
    //     elf,
    //     block_execution_strategy_factory,
    //     client,
    //     eth_proofs_client,
    //     config,
    // )
    // .await?;

    info!("Latest block number: {}", http_provider.get_block_number().await?);

    while let Some(header) = stream.next().await {
        // Wait for the block to be avaliable in the HTTP provider
        // executor.wait_for_block(header.number).await?;
        wait_for_block::<Ethereum>(http_provider.clone(), header.number).await?;

        debug!("Received Block number: {}", header.number);

        // TODO: send the input buffer and elf to the pico proprocessor gPRC server.
        break;
    }

    Ok(())
}

async fn wait_for_block<N>(provider: RootProvider, block_number: u64) -> eyre::Result<()> {
    let block_number = block_number.into();

    while provider.get_block_by_number(block_number).await?.is_none() {
        sleep(Duration::from_millis(100)).await;
    }
    Ok(())
}