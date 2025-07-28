use alloy_network::Ethereum;
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
use tracing::{debug, error, info};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod cli;

mod eth_proofs;

const TEST_BLOCK_NUMBER: u64 = 22515566;

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
            EnvFilter::from_default_env(), // .add_directive("sp1_core_machine=warn".parse().unwrap())
                                           // .add_directive("sp1_core_executor=warn".parse().unwrap())
                                           // .add_directive("sp1_prover=warn".parse().unwrap()),
        )
        .init();

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
    let ws_provider = ProviderBuilder::new().on_ws(ws).await?;
    let http_provider = create_provider::<Ethereum>(args.http_rpc_url);

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

        while let Some(header) = stream.next().await {
            // Wait for the block to be avaliable in the HTTP provider
            executor.wait_for_block(header.number).await?;
            if let Err(err) = executor.execute(header.number, Some(&sender)).await {
                let error_message = format!("Error handling block {}: {err}", header.number);
                error!(error_message);
            }
            debug!("Received Block number: {}", header.number);
        }
    }
    Ok(())
}
