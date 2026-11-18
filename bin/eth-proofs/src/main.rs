//! Follows a chain and has its blocks proved by brevis-vm's PicoProver service.
//!
//! For every sampled block: fetch it and its execution witness from the node, execute it to build
//! the client input (which checks the state root and the header on the way), and hand the input,
//! as the one buffer the Pico guest reads, to the prover over gRPC. A second task sees one block
//! at a time through the prover and reports its proof to ethproofs, when reporting is configured.

use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use alloy_network::Ethereum;
use alloy_provider::{Provider, ProviderBuilder, RootProvider, WsConnect};
use clap::Parser;
use cli::Args;
use eth_proofs::EthProofsClient;
use futures::{future::ready, StreamExt};
use prover::{prove_block, ProverLink, ProverTimings};
use reth_chainspec::{ChainSpec, EthChainSpec};
use rsp_host_executor::{single_buffer_stdin, EthHostExecutor, StateBackend};
use rsp_primitives::genesis::Genesis;
use rsp_provider::create_provider;
use tracing::{debug, error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod cli;

mod eth_proofs;

mod prover;

/// The PicoProver service, as brevis-vm's provers expose it.
pub mod pb {
    tonic::include_proto!("prover_server");
}

/// The witness the host emits, fixed when this binary is built. The guest ELF in the prover has
/// to be built with the same one.
const WITNESS_BACKEND: &str = if cfg!(feature = "flat") {
    "flat"
} else if cfg!(feature = "arena") {
    "arena"
} else {
    "default"
};

/// How long to wait for a block the node announced to appear on its HTTP endpoint.
const WAIT_FOR_BLOCK_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// The most time one block's fetch and host execution may take. The provider has no request
/// timeout of its own, so without this a connection that hangs would hold up every later block.
const FETCH_TIMEOUT: Duration = Duration::from_secs(15 * 60);

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
    let genesis = args.chain.genesis();
    let chain_spec: Arc<ChainSpec> = Arc::new((&genesis).try_into()?);
    let chain_id = chain_spec.chain().id();
    let host_executor = EthHostExecutor::eth(chain_spec, None);

    info!(
        "chain: {:?} ({chain_id}); witness backend: {WITNESS_BACKEND}; state backend: {}",
        args.chain, args.state_backend
    );

    let eth_proofs_client =
        EthProofsClient::new(args.eth_proofs_cluster_id, args.eth_proofs_credentials());
    if !eth_proofs_client.is_enabled() {
        warn!(
            "nothing is reported to ethproofs (ETH_PROOFS_ENDPOINT and ETH_PROOFS_API_TOKEN are \
             not both set); proofs are logged"
        );
    }

    for dir in [&args.stdin_dir, &args.proof_dir].into_iter().flatten() {
        std::fs::create_dir_all(dir)?;
    }

    let ws = WsConnect::new(args.ws_rpc_url.clone());
    let ws_provider = ProviderBuilder::new().connect_ws(ws).await?;
    let http_provider: RootProvider<Ethereum> = create_provider(args.http_rpc_url.clone());

    // A host for one chain on a node of another fails every block with a state root that does
    // not match, which does not say why.
    let node_chain_id = http_provider.get_chain_id().await?;
    eyre::ensure!(
        node_chain_id == chain_id,
        "the node is on chain {node_chain_id}, and --chain names chain {chain_id}"
    );

    // Subscribe to block headers.
    let subscription = ws_provider.subscribe_blocks().await?;
    let block_interval = args.block_interval;
    let mut stream =
        subscription.into_stream().filter(move |h| ready(h.number % block_interval == 0));

    info!("Latest block number: {}", http_provider.get_block_number().await?);

    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel::<(u64, Vec<u8>)>();

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
    // Blocks proved, and whether that reached `--max-blocks`.
    let proved = Arc::new(AtomicU64::new(0));
    let task_proved = proved.clone();
    let limit_reached = Arc::new(AtomicBool::new(false));
    let task_limit_reached = limit_reached.clone();
    let wake = Arc::new(tokio::sync::Notify::new());
    let task_wake = wake.clone();
    let max_blocks = args.max_blocks;
    let proof_dir = args.proof_dir.clone();

    // One block at a time, since the prover takes one block in flight. `prove_block` waits
    // out a prover restart and resubmits, so a crash costs the blocks nothing but time.
    let prover_task = tokio::task::spawn(async move {
        while let Some((block_num, client_input)) = receiver.recv().await {
            info!(
                "receiver client input, block_number: {}, input size: {}",
                block_num,
                client_input.len()
            );
            let was_proved = prove_block(
                &mut link,
                &hooks,
                block_num,
                &client_input,
                timings,
                proof_dir.as_deref(),
            )
            .await;
            task_queued.fetch_sub(1, Ordering::SeqCst);
            if was_proved {
                let count = task_proved.fetch_add(1, Ordering::SeqCst) + 1;
                if max_blocks.is_some_and(|max| count >= max) {
                    task_limit_reached.store(true, Ordering::SeqCst);
                    task_wake.notify_one();
                    return;
                }
            }
        }
    });

    let mut last_block = 0;

    loop {
        let header = tokio::select! {
            header = stream.next() => header,
            _ = wake.notified() => None,
        };
        if limit_reached.load(Ordering::SeqCst) {
            info!("Proved {} block(s), the limit given; exiting", proved.load(Ordering::SeqCst));
            return Ok(());
        }
        // The subscription only ends when the connection to the node is gone for good. Exit
        // with an error, so that whatever supervises the service starts it again.
        let Some(header) = header else {
            eyre::bail!("the block subscription ended; exiting");
        };

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

        queued.fetch_add(1, Ordering::SeqCst);
        eth_proofs_client.queued(header.number);
        let fetched = tokio::time::timeout(
            FETCH_TIMEOUT,
            fetch_input(
                &host_executor,
                &http_provider,
                &genesis,
                args.state_backend,
                header.number,
                args.stdin_dir.as_deref(),
            ),
        )
        .await
        .unwrap_or_else(|_| Err(eyre::eyre!("no input after {FETCH_TIMEOUT:?}")));
        match fetched {
            Ok(buffer) => {
                info!(
                    "Sending client input to the sender, block_number: {}, input size: {}",
                    header.number,
                    buffer.len()
                );
                // The receiver is gone only if the prover task ended; the check above reports it.
                if sender.send((header.number, buffer)).is_err() {
                    queued.fetch_sub(1, Ordering::SeqCst);
                }
            }
            Err(err) => {
                // Nothing was handed over, so the prover task never got the block.
                queued.fetch_sub(1, Ordering::SeqCst);
                let error_message = format!("Error handling block {}: {err}", header.number);
                error!(error_message);
            }
        }
        debug!("Received Block number: {}", header.number);
    }
}

/// Fetches `block_number` and what executing it needs, executes it, and returns the client input
/// as the one buffer the Pico guest reads from stdin.
async fn fetch_input(
    host_executor: &EthHostExecutor,
    provider: &RootProvider<Ethereum>,
    genesis: &Genesis,
    state_backend: StateBackend,
    block_number: u64,
    stdin_dir: Option<&Path>,
) -> eyre::Result<Vec<u8>> {
    // Wait for the block to be avaliable in the HTTP provider
    let waiting_since = Instant::now();
    while provider.get_block_by_number(block_number.into()).await?.is_none() {
        eyre::ensure!(
            waiting_since.elapsed() < WAIT_FOR_BLOCK_TIMEOUT,
            "the node announced the block but did not serve it within {WAIT_FOR_BLOCK_TIMEOUT:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let fetch_data_start = Instant::now();
    info!("Loading client input from RPC");
    let client_input = host_executor
        .execute(block_number, provider, genesis.clone(), None, false, state_backend)
        .await?;
    let buffer = single_buffer_stdin(&client_input)?;
    info!("client input loaded, size: {}", buffer.len());
    info!("Fetch data took: {:?}", fetch_data_start.elapsed());

    if let Some(dir) = stdin_dir {
        std::fs::write(dir.join(format!("{block_number}.bin")), &buffer)?;
    }
    Ok(buffer)
}
