use alloy_chains::Chain;
use alloy_provider::{network::AnyNetwork, Provider, RootProvider};
use anyhow::Result;
use clap::Parser;
use rsp_host_executor::Config;
use std::path::PathBuf;
use url::Url;

#[derive(Debug, Clone, Parser)]
pub struct ProviderArgs {
    #[clap(long, env = "RPC_URL")]
    pub rpc_http_url: Url,

    #[clap(long, env = "RPC_WS_URL")]
    pub rpc_ws_url: Url,
}

#[derive(Debug, Clone, Parser)]
pub struct Args {
    #[clap(long, default_value = "false")]
    pub is_input_emulated: bool,

    #[clap(long)]
    pub cache_dir: Option<PathBuf>,

    #[clap(flatten)]
    pub provider: ProviderArgs,
}

impl Args {
    pub async fn as_config(&self) -> Result<Config> {
        // get the chain ID
        let provider = RootProvider::<AnyNetwork>::new_http(self.provider.rpc_http_url.clone());
        let chain_id = provider.get_chain_id().await?;

        // build chain and genesis
        let chain = Chain::from_id(chain_id);
        let genesis = chain_id.try_into()?;

        Ok(Config {
            chain,
            genesis,
            rpc_url: Some(self.provider.rpc_http_url.clone()),
            cache_dir: self.cache_dir.clone(),
            custom_beneficiary: None,
            prove_mode: None,
            skip_client_execution: false,
            opcode_tracking: false,
        })
    }
}
