use std::path::PathBuf;

use alloy_chains::Chain;
use clap::Parser;
use rsp_host_executor::Config;
use rsp_primitives::genesis::Genesis;
use sp1_sdk::SP1ProofMode;
use url::Url;

/// The arguments for the cli.
#[derive(Debug, Clone, Parser)]
pub struct Args {
    /// The HTTP rpc url used to fetch data about the block.
    #[clap(long, env)]
    pub http_rpc_url: Url,

    /// The WS rpc url used to fetch data about the block.
    #[clap(long, env)]
    pub ws_rpc_url: Url,

    /// Whether to generate a proof or just execute the block.
    #[clap(long)]
    pub execute_only: bool,

    /// The interval at which to execute blocks.
    #[clap(long, default_value_t = 100)]
    pub block_interval: u64,

    /// ETH proofs endpoint.
    #[clap(long, env)]
    pub eth_proofs_endpoint: String,

    /// ETH proofs API token.
    #[clap(long, env)]
    pub eth_proofs_api_token: String,

    /// Optional ETH proofs cluster ID.
    #[clap(long, default_value_t = 1)]
    pub eth_proofs_cluster_id: u64,

    /// PagerDuty integration key.
    #[clap(long, env)]
    pub pager_duty_integration_key: Option<String>,

    /// Moongate server endpoint.
    #[clap(long, env)]
    pub moongate_endpoint: Option<String>,

    /// Optional path to the directory containing cached client input. A new cache file will be
    /// created from RPC data if it doesn't already exist.    #[clap(long, env)]
    #[clap(long)]
    pub cache_dir: Option<PathBuf>,

    /// execute a specific block number to test e2e
    #[clap(long)]
    pub test_e2e: bool,

    /// The gRPC endopoint to execute proving.
    #[clap(long, env)]
    pub witness_getaway_endpoint: String,

    /// How often to ask the prover whether a block's proof is ready, in seconds.
    #[clap(long, default_value_t = 2)]
    pub prover_poll_secs: u64,

    /// Resubmit a block that still has no proof this many seconds after it went in. The prover
    /// refuses a block while another is in flight, so this never starts a second proof of a block
    /// it is still working on; it restarts one the prover lost, as a restarted prover does.
    #[clap(long, default_value_t = 120)]
    pub prover_resubmit_secs: u64,

    /// Give a block up after this many seconds without a proof, so one block cannot hold up all
    /// the blocks behind it.
    #[clap(long, default_value_t = 1800)]
    pub prover_give_up_secs: u64,

    /// The most blocks that may wait for the prover, the one being proved included. Past it, new
    /// blocks are skipped rather than queued: each queued block holds its client input in memory,
    /// and a longer backlog would keep proofs behind the chain long after the prover is back.
    #[clap(long, default_value_t = 128)]
    pub max_queued_blocks: usize,
}

impl Args {
    pub async fn as_config(&self) -> eyre::Result<Config> {
        let config = Config {
            chain: Chain::mainnet(),
            genesis: Genesis::Mainnet,
            rpc_url: Some(self.http_rpc_url.clone()),
            cache_dir: self.cache_dir.clone(),
            custom_beneficiary: None,
            prove_mode: (!self.execute_only).then_some(SP1ProofMode::Compressed),
            skip_client_execution: true,
            opcode_tracking: false,
        };

        Ok(config)
    }
}
