use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use alloy_provider::{Provider, ProviderBuilder, RootProvider, WsConnect};
use clap::Parser;
use cli::Args;
use eth_proofs::EthProofsClient;
use futures::{future::ready, StreamExt};
use prover::{prove_block, ProverLink, ProverTimings};
use rsp_host_executor::{
    create_eth_block_execution_strategy_factory, BlockExecutor, EthExecutorComponents, FullExecutor,
};
use rsp_provider::create_provider;
use tracing::{debug, error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod cli;

mod eth_proofs;

mod prover;

const TEST_BLOCK_NUMBER: u64 = 23371900;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // Initialize the environment variables.
    dotenv::dotenv().ok();

    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }

    // Initialize the logger.
    tracing_subscriber::registry().with(fmt::layer()).with(EnvFilter::from_default_env()).init();

    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    // Parse the command line arguments.
    let args = Args::parse();
    let config = args.as_config().await?;

    let block_execution_strategy_factory =
        create_eth_block_execution_strategy_factory(&config.genesis, None);

    let eth_proofs_client = EthProofsClient::new(
        args.eth_proofs_cluster_id,
        args.eth_proofs_endpoint,
        args.eth_proofs_api_token,
    );

    let ws = WsConnect::new(args.ws_rpc_url);
    let ws_provider = ProviderBuilder::new().connect_ws(ws).await?;
    let http_provider = create_provider(args.http_rpc_url);

    // Subscribe to block headers.
    let subscription = ws_provider.subscribe_blocks().await?;
    let mut stream =
        subscription.into_stream().filter(|h| ready(h.number % args.block_interval == 0));

    let executor =
        FullExecutor::<EthExecutorComponents<_, sp1_sdk::CudaProver>, RootProvider>::try_new(
            http_provider.clone(),
            block_execution_strategy_factory,
            eth_proofs_client.clone(),
            config,
        )
        .await?;

    info!("Latest block number: {}", http_provider.get_block_number().await?);

    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel::<(u64, Vec<u8>)>();

    if args.test_e2e {
        info!("Start to test block : {}", TEST_BLOCK_NUMBER);
        if let Err(err) = executor.execute(TEST_BLOCK_NUMBER, None).await {
            let error_message: String =
                format!("Error handling block {}: {err}", TEST_BLOCK_NUMBER);
            error!(error_message);
        }
        // sleep 5s
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    } else {
        let hooks = eth_proofs_client.clone();
        let timings = ProverTimings {
            poll: Duration::from_secs(args.prover_poll_secs.max(1)),
            resubmit_after: Duration::from_secs(args.prover_resubmit_secs),
            give_up_after: Duration::from_secs(args.prover_give_up_secs),
        };
        let mut link = ProverLink::new(args.witness_getaway_endpoint.clone());
        // Blocks handed to the prover task and not finished yet, the one being proved included.
        let queued = Arc::new(AtomicUsize::new(0));
        let task_queued = queued.clone();

        // One block at a time, since the prover takes one block in flight. `prove_block` waits
        // out a prover restart and resubmits, so a crash costs the blocks nothing but time.
        let prover_task = tokio::task::spawn(async move {
            while let Some((block_num, client_input)) = receiver.recv().await {
                info!(
                    "receiver client input, block_number: {}, input size: {}",
                    block_num,
                    client_input.len()
                );
                prove_block(&mut link, &hooks, block_num, &client_input, timings).await;
                task_queued.fetch_sub(1, Ordering::SeqCst);
            }
        });

        let mut last_block = 0;

        while let Some(header) = stream.next().await {
            // skip if not greater than last processed block
            if header.number <= last_block {
                warn!("Skipping duplicate/old block: {}", header.number);
                continue;
            }
            last_block = header.number;

            // The prover task only ends if something in it panicked. Nothing would reach the
            // prover after that, so exit and let systemd restart us.
            if prover_task.is_finished() {
                eyre::bail!("the prover task has stopped; exiting");
            }
            let waiting = queued.load(Ordering::SeqCst);
            if waiting >= args.max_queued_blocks {
                warn!(
                    "Skipping block {}: {waiting} block(s) are already waiting for the prover",
                    header.number
                );
                continue;
            }

            // Wait for the block to be avaliable in the HTTP provider
            executor.wait_for_block(header.number).await?;
            queued.fetch_add(1, Ordering::SeqCst);
            if let Err(err) = executor.execute(header.number, Some(&sender)).await {
                // `execute` hands the block over as its last fallible step, so an error means
                // the prover task never got it.
                queued.fetch_sub(1, Ordering::SeqCst);
                let error_message = format!("Error handling block {}: {err}", header.number);
                error!(error_message);
            }
            debug!("Received Block number: {}", header.number);
        }
    }
    Ok(())
}
