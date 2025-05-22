use std::{
    fmt::{Debug, Formatter},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    pico_prover_client::PicoProverClient, GetProveResultRequest, ProvingRequest, ProvingType,
};
use alloy_provider::Provider;
use either::Either;
use eyre::bail;
use reth_primitives_traits::NodePrimitives;
use rsp_client_executor::io::ClientExecutorInput;
use rsp_rpc_db::RpcDb;
use serde::de::DeserializeOwned;
use sp1_prover::components::CpuProverComponents;
use sp1_sdk::{ExecutionReport, Prover, SP1ProvingKey, SP1PublicValues, SP1Stdin};
use tokio::{
    sync::mpsc::UnboundedSender,
    task,
    time::sleep,
};
use tonic::{codec::CompressionEncoding, transport::Channel};
use tracing::{info, info_span, warn, error};

use crate::{Config, ExecutionHooks, ExecutorComponents, HostExecutor};

pub type EitherExecutor<C, P> = Either<FullExecutor<C, P>, CachedExecutor<C>>;

pub async fn build_executor<C, P>(
    provider: Option<P>,
    evm_config: C::EvmConfig,
    hooks: C::Hooks,
    config: Config,
) -> eyre::Result<EitherExecutor<C, P>>
where
    C: ExecutorComponents,
    P: Provider<C::Network> + Clone,
{
    if let Some(provider) = provider {
        return Ok(Either::Left(FullExecutor::try_new(provider, evm_config, hooks, config).await?));
    }

    if let Some(cache_dir) = &config.cache_dir {
        return Ok(Either::Right(CachedExecutor::try_new(hooks, cache_dir.clone(), config).await?));
    }

    bail!("Either a RPC URL or a cache dir must be provided")
}

pub trait BlockExecutor<C: ExecutorComponents> {
    #[allow(async_fn_in_trait)]
    async fn execute(
        &self,
        block_number: u64,
        sender: Option<&UnboundedSender<(u64, Vec<u8>)>>,
    ) -> eyre::Result<()>;

    fn config(&self) -> &Config;
}

pub async fn process_client<C: ExecutorComponents>(
    hooks: &C::Hooks,
    block_number: u64,
    buffer: Vec<u8>,
    grpc_endpoint: String,
) -> eyre::Result<()> {
    info!("Starting proof generation");

    hooks.on_proving_start(block_number).await?;

    // TODO:START REMOTE PROVING
    let mut client = PicoProverClient::connect(grpc_endpoint)
        .await?
        .max_encoding_message_size(600 * 1024 * 1024)
        .max_decoding_message_size(600 * 1024 * 1024)
        .accept_compressed(CompressionEncoding::Zstd)
        .send_compressed(CompressionEncoding::Zstd);

    client
        .request_prover(ProvingRequest {
            block_number,
            input_buffer: Some(buffer),
            proving_type: ProvingType::Gpu as i32,
        })
        .await?;
    info!("gRPC client post to prover, blk number: {}", block_number);

    Ok(())
}

pub async fn fetch_proving_status<C: ExecutorComponents>(
    block_number: u64,
    start_time: Instant,
    hooks: &C::Hooks,
    grpc_client: &mut PicoProverClient<Channel>,
) -> eyre::Result<()> {
    // start to loop result at interval of 2s
    let mut interval = tokio::time::interval(Duration::from_secs(2));

    loop {
        interval.tick().await;
        let result = grpc_client.get_prove_result(GetProveResultRequest { block_number }).await?;
        let response = result.get_ref();

        if let Some(prover_err) = response.clone().err {
            let error_message = format!("Prove failed: {:?}", prover_err);
            error!(error_message);
            return Err(eyre::eyre!(error_message));
        }
        
        if let Some(proof_info) = response.clone().proof_info {
            if proof_info.proof_with_publics.len() > 0 {
                let prove_end = start_time.elapsed();

                // report the result to the ethproofs
                hooks
                    .on_proving_end(
                        block_number,
                        &proof_info.proof_with_publics,
                        &proof_info.verifier_id,
                        Some(proof_info.proving_cycles),
                        prove_end,
                    )
                    .await?;
                info!("Proof {:?} successfully generated!, proving time: {:?}", block_number, prove_end.clone());
                break;
            } else {
                info!("Waiting for proof {:?} generation...", block_number);
            }
        } else {
            info!("Waiting for proof {:?} generation...", block_number);
        }
    }
    Ok(())
}

impl<C, P> BlockExecutor<C> for EitherExecutor<C, P>
where
    C: ExecutorComponents,
    P: Provider<C::Network> + Clone,
{
    async fn execute(
        &self,
        block_number: u64,
        sender: Option<&UnboundedSender<(u64, Vec<u8>)>>,
    ) -> eyre::Result<()> {
        match self {
            Either::Left(ref executor) => executor.execute(block_number, sender).await,
            Either::Right(ref executor) => executor.execute(block_number, sender).await,
        }
    }

    fn config(&self) -> &Config {
        match self {
            Either::Left(executor) => executor.config(),
            Either::Right(executor) => executor.config(),
        }
    }
}

pub struct FullExecutor<C, P>
where
    C: ExecutorComponents,
    P: Provider<C::Network> + Clone,
{
    provider: P,
    host_executor: HostExecutor<C::EvmConfig, C::ChainSpec>,
    hooks: C::Hooks,
    config: Config,
}

impl<C, P> FullExecutor<C, P>
where
    C: ExecutorComponents,
    P: Provider<C::Network> + Clone,
{
    pub async fn try_new(
        provider: P,
        evm_config: C::EvmConfig,
        hooks: C::Hooks,
        config: Config,
    ) -> eyre::Result<Self> {
        Ok(Self {
            provider,
            host_executor: HostExecutor::new(
                evm_config,
                Arc::new(C::try_into_chain_spec(&config.genesis)?),
            ),
            hooks,
            config,
        })
    }

    pub async fn wait_for_block(&self, block_number: u64) -> eyre::Result<()> {
        let block_number = block_number.into();

        while self.provider.get_block_by_number(block_number).await?.is_none() {
            sleep(Duration::from_millis(100)).await;
        }
        Ok(())
    }
}

impl<C, P> BlockExecutor<C> for FullExecutor<C, P>
where
    C: ExecutorComponents,
    P: Provider<C::Network> + Clone,
{
    async fn execute(
        &self,
        block_number: u64,
        sender: Option<&UnboundedSender<(u64, Vec<u8>)>>,
    ) -> eyre::Result<()> {
        let fetch_data_start = Instant::now();

        self.hooks.on_execution_start(block_number).await?;

        let client_input_from_cache = self.config.cache_dir.as_ref().and_then(|cache_dir| {
            info!("try to load input from cache: {:}", cache_dir.display());
            match try_load_input_from_cache::<C::Primitives>(
                cache_dir,
                self.config.chain.id(),
                block_number,
            ) {
                Ok(client_input) => {
                    info!("Loaded input from cache");
                    client_input
                }
                Err(e) => {
                    warn!("Failed to load input from cache: {}", e);
                    None
                }
            }
        });

        let client_input = match client_input_from_cache {
            Some(mut client_input_from_cache) => {
                // Override opcode tracking from cache by the setting provided by the user
                client_input_from_cache.opcode_tracking = self.config.opcode_tracking;
                client_input_from_cache
            }
            None => {
                info!("client_input is None, Loading client input from RPC");
                let rpc_db = RpcDb::new(self.provider.clone(), block_number - 1);

                // Execute the host.
                let client_input = self
                    .host_executor
                    .execute(
                        block_number,
                        &rpc_db,
                        &self.provider,
                        self.config.genesis.clone(),
                        self.config.custom_beneficiary,
                        self.config.opcode_tracking,
                    )
                    .await?;

                if let Some(ref cache_dir) = self.config.cache_dir {
                    let input_folder = cache_dir.join(format!("input/{}", self.config.chain.id()));
                    if !input_folder.exists() {
                        std::fs::create_dir_all(&input_folder)?;
                    }

                    let input_path = input_folder.join(format!("{}.bin", block_number));
                    let mut cache_file = std::fs::File::create(input_path)?;

                    bincode::serialize_into(&mut cache_file, &client_input)?;
                }

                client_input
            }
        };
        let buffer: Vec<u8> = bincode::serialize(&client_input).unwrap();
        info!("client input loaded, size: {}", buffer.len());
        let fetch_data_duration = fetch_data_start.elapsed();
        info!("Fetch data took: {:?}", fetch_data_duration);
        
        // Notification to the sender the prover started
        if let Some(sender) = sender {
            info!("Sending client input to the sender, block_number: {}, input size: {}", block_number, buffer.len());

            sender.send((block_number, buffer))?;
        }
        Ok(())
    }

    fn config(&self) -> &Config {
        &self.config
    }
}

impl<C, P> Debug for FullExecutor<C, P>
where
    C: ExecutorComponents,
    P: Provider<C::Network> + Clone,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullExecutor").field("config", &self.config).finish()
    }
}

pub struct CachedExecutor<C>
where
    C: ExecutorComponents,
{
    cache_dir: PathBuf,
    hooks: C::Hooks,
    config: Config,
}

impl<C> CachedExecutor<C>
where
    C: ExecutorComponents,
{
    pub async fn try_new(
        hooks: C::Hooks,
        cache_dir: PathBuf,
        config: Config,
    ) -> eyre::Result<Self> {
        Ok(Self { cache_dir, hooks, config })
    }
}

impl<C> BlockExecutor<C> for CachedExecutor<C>
where
    C: ExecutorComponents,
{
    async fn execute(
        &self,
        block_number: u64,
        sender: Option<&UnboundedSender<(u64, Vec<u8>)>>,
    ) -> eyre::Result<()> {
        let client_input = try_load_input_from_cache::<C::Primitives>(
            &self.cache_dir,
            self.config.chain.id(),
            block_number,
        )?
        .ok_or(eyre::eyre!("No cached input found"))?;
        let buffer: Vec<u8> = bincode::serialize(&client_input).unwrap();
        process_client::<C>(&self.hooks, block_number, buffer, "".to_string()).await
    }

    fn config(&self) -> &Config {
        &self.config
    }
}

impl<C> Debug for CachedExecutor<C>
where
    C: ExecutorComponents,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedExecutor").field("cache_dir", &self.cache_dir).finish()
    }
}

// Block execution in SP1 is a long-running, blocking task, so run it in a separate thread.
async fn execute_client<P: Prover<CpuProverComponents> + 'static>(
    number: u64,
    client: Arc<P>,
    pk: Arc<SP1ProvingKey>,
    stdin: Arc<SP1Stdin>,
) -> eyre::Result<eyre::Result<(SP1PublicValues, ExecutionReport)>> {
    task::spawn_blocking(move || {
        info_span!("execute_client", number).in_scope(|| {
            let result = client.execute(&pk.elf, &stdin);
            result.map_err(|err| eyre::eyre!("{err}"))
        })
    })
    .await
    .map_err(|err| eyre::eyre!("{err}"))
}

fn try_load_input_from_cache<P: NodePrimitives + DeserializeOwned>(
    cache_dir: &Path,
    chain_id: u64,
    block_number: u64,
) -> eyre::Result<Option<ClientExecutorInput<P>>> {
    let cache_path = cache_dir.join(format!("input/{}/{}.bin", chain_id, block_number));

    if cache_path.exists() {
        // TODO: prune the cache if invalid instead
        let mut cache_file = std::fs::File::open(cache_path)?;
        let client_input = bincode::deserialize_from(&mut cache_file)?;

        Ok(Some(client_input))
    } else {
        Ok(None)
    }
}
