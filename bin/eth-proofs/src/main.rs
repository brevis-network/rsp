use alloy_provider::{Provider, ProviderBuilder, RootProvider, WsConnect};
use clap::Parser;
use cli::Args;
use eth_proofs::EthProofsClient;
use futures::{future::ready, StreamExt};
use rsp_host_executor::{
    create_eth_block_execution_strategy_factory, fetch_proving_status,
    pico_prover_client::PicoProverClient, process_client, BlockExecutor, EthExecutorComponents,
    FullExecutor,
};
use rsp_provider::create_provider;
use tonic::codec::CompressionEncoding;
use tracing::{debug, error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod cli;

mod eth_proofs;

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

    // The WS connection + block subscription are (re)established inside the reconnect loop
    // below, so a dropped WS auto-reconnects instead of terminating the service. The HTTP
    // provider is created once here and reused (it has an internal server-error retry layer;
    // transient connection errors are handled by wait_for_block's retry).
    let http_provider = create_provider(args.http_rpc_url);

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
        // Capture what the reconnect loop needs before the proving task's `async move`
        // takes ownership of parts of `args`.
        let ws_rpc_url = args.ws_rpc_url.clone();
        let block_interval = args.block_interval;

        let hooks = eth_proofs_client.clone();
        tokio::task::spawn(async move {
            let mut client = PicoProverClient::connect(args.witness_getaway_endpoint.clone())
                .await
                .unwrap()
                .max_encoding_message_size(600 * 1024 * 1024)
                .max_decoding_message_size(600 * 1024 * 1024)
                .accept_compressed(CompressionEncoding::Zstd)
                .send_compressed(CompressionEncoding::Zstd);

            while let Some((block_num, client_input)) = receiver.recv().await {
                info!(
                    "receiver client input, block_number: {}, input size: {}",
                    block_num,
                    client_input.len()
                );
                process_client::<EthExecutorComponents<_, sp1_sdk::CudaProver>>(
                    &hooks,
                    block_num,
                    client_input,
                    args.witness_getaway_endpoint.clone(),
                )
                .await
                .unwrap();
                // loop until the current block proof is ready
                let res = fetch_proving_status::<EthExecutorComponents<_, sp1_sdk::CudaProver>>(
                    block_num,
                    &hooks,
                    &mut client,
                )
                .await;
                if res.is_err() {
                    error!("Error fetching proving status: {:?}", res);
                }
            }
        });

        let mut last_block = 0u64;

        // Reconnect loop: if the WS subscription drops (stream ends), reconnect and
        // re-subscribe instead of exiting. Combined with wait_for_block's internal retry
        // and the skip-on-error below, a transient RPC/node blip no longer kills the service.
        loop {
            let ws_provider = match ProviderBuilder::new()
                .connect_ws(WsConnect::new(ws_rpc_url.clone()))
                .await
            {
                Ok(p) => p,
                Err(err) => {
                    error!("WS connect failed: {err}; retrying in 5s");
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                    continue;
                }
            };
            let subscription = match ws_provider.subscribe_blocks().await {
                Ok(s) => s,
                Err(err) => {
                    error!("subscribe_blocks failed: {err}; retrying in 5s");
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                    continue;
                }
            };
            let mut stream = subscription
                .into_stream()
                .filter(move |h| ready(h.number % block_interval == 0));
            info!("Subscribed to new blocks (last_block={last_block})");

            while let Some(header) = stream.next().await {
                // skip if not greater than last processed block
                if header.number <= last_block {
                    warn!("Skipping duplicate/old block: {}", header.number);
                    continue;
                }
                last_block = header.number;

                // Wait for the block to be available in the HTTP provider. wait_for_block
                // retries transient RPC errors internally; if it still fails (node down for
                // a long time) skip this block instead of terminating the service.
                if let Err(err) = executor.wait_for_block(header.number).await {
                    error!("Error waiting for block {}: {err}; skipping", header.number);
                    continue;
                }
                if let Err(err) = executor.execute(header.number, Some(&sender)).await {
                    let error_message = format!("Error handling block {}: {err}", header.number);
                    error!(error_message);
                }
                debug!("Received Block number: {}", header.number);
            }

            warn!("Block stream ended (WS dropped?); reconnecting in 5s");
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    }
    Ok(())
}
