use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use rsp_host_executor::StateBackend;
use rsp_primitives::genesis::Genesis;
use url::Url;

/// The chain the node follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Chain {
    Mainnet,
    Sepolia,
}

impl Chain {
    pub fn genesis(self) -> Genesis {
        match self {
            Self::Mainnet => Genesis::Mainnet,
            Self::Sepolia => Genesis::Sepolia,
        }
    }
}

/// The arguments for the cli.
#[derive(Debug, Clone, Parser)]
pub struct Args {
    /// The HTTP rpc url used to fetch data about the block.
    #[clap(long, env)]
    pub http_rpc_url: Url,

    /// The WS rpc url used to fetch data about the block.
    #[clap(long, env)]
    pub ws_rpc_url: Url,

    /// The chain the node at those urls follows.
    #[clap(long, env, value_enum, default_value_t = Chain::Mainnet)]
    pub chain: Chain,

    /// The interval at which to prove blocks: a block is taken when its number is a multiple of
    /// this.
    #[clap(long, default_value_t = 100, value_parser = clap::value_parser!(u64).range(1..))]
    pub block_interval: u64,

    /// ETH proofs endpoint. Proofs are reported only when this and the API token are both set;
    /// without them the service proves and logs, which is how a chain ethproofs does not list
    /// (Sepolia) is run.
    #[clap(long, env)]
    pub eth_proofs_endpoint: Option<String>,

    /// ETH proofs API token.
    #[clap(long, env)]
    pub eth_proofs_api_token: Option<String>,

    /// Optional ETH proofs cluster ID.
    #[clap(long, default_value_t = 1)]
    pub eth_proofs_cluster_id: u64,

    /// The gRPC endopoint to execute proving.
    #[clap(long, env)]
    pub witness_getaway_endpoint: String,

    /// How to fetch the state a block needs: `execution-witness` is one `debug_executionWitness`
    /// call, which needs the node's `debug` namespace; `proofs` is `eth_getProof` per account.
    #[clap(long, default_value_t = StateBackend::ExecutionWitness)]
    pub state_backend: StateBackend,

    /// Write every block's client input, as the prover is sent it, to `<DIR>/<block>.bin`.
    #[clap(long, value_name = "DIR")]
    pub stdin_dir: Option<PathBuf>,

    /// Write every proof the prover returns to `<DIR>/<block>.proof`.
    #[clap(long, value_name = "DIR")]
    pub proof_dir: Option<PathBuf>,

    /// Exit once this many blocks have been proved, for a bounded run.
    #[clap(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub max_blocks: Option<u64>,

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
    /// The ethproofs endpoint and token when both are set. An empty value, as an untouched
    /// `.env` leaves, counts as unset.
    pub fn eth_proofs_credentials(&self) -> Option<(String, String)> {
        let set = |value: &Option<String>| value.clone().filter(|v| !v.is_empty());
        Some((set(&self.eth_proofs_endpoint)?, set(&self.eth_proofs_api_token)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> Args {
        let base = [
            "eth-proofs",
            "--http-rpc-url",
            "http://localhost:8545",
            "--ws-rpc-url",
            "ws://localhost:8546",
            "--witness-getaway-endpoint",
            "http://localhost:50051",
        ];
        Args::try_parse_from(base.iter().chain(extra).copied()).unwrap()
    }

    #[test]
    fn the_defaults_are_mainnet_with_the_execution_witness() {
        let args = parse(&[]);
        assert_eq!(args.chain, Chain::Mainnet);
        assert_eq!(args.chain.genesis(), Genesis::Mainnet);
        assert_eq!(args.state_backend, StateBackend::ExecutionWitness);
        assert_eq!(args.block_interval, 100);
    }

    #[test]
    fn sepolia_is_selected_by_name() {
        let args = parse(&["--chain", "sepolia", "--block-interval", "1"]);
        assert_eq!(args.chain.genesis(), Genesis::Sepolia);
        assert_eq!(args.block_interval, 1);
    }

    #[test]
    fn reporting_needs_both_the_endpoint_and_the_token() {
        assert!(parse(&[]).eth_proofs_credentials().is_none());
        assert!(parse(&["--eth-proofs-endpoint", "https://x/api/v0"])
            .eth_proofs_credentials()
            .is_none());
        assert!(parse(&["--eth-proofs-endpoint", "", "--eth-proofs-api-token", "t"])
            .eth_proofs_credentials()
            .is_none());
        assert_eq!(
            parse(&["--eth-proofs-endpoint", "https://x/api/v0", "--eth-proofs-api-token", "t"])
                .eth_proofs_credentials(),
            Some(("https://x/api/v0".to_string(), "t".to_string()))
        );
    }
}
