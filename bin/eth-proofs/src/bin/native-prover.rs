//! A stand-in for brevis-vm's PicoProver that runs the client natively instead of proving it.
//!
//! It serves the same gRPC service, so `eth-proofs` talks to it unchanged. For each block it is
//! sent it reads the input exactly as the Pico guest does, executes the block with the client
//! executor (the code the guest runs), and checks that the header it derives is the block's.
//! Nothing is proved: what it returns as the proof is the bincode of what the guest would commit
//! (the header and the config digest), and its verifier id is `native-client`.
//!
//! It is for running the whole fetching side against a node without a GPU: every block that
//! passes here is an input the guest accepts. Build it with the same witness backend as the
//! `eth-proofs` it serves.

use std::{
    collections::BTreeMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Instant,
};

use clap::Parser;
use pb::{
    pico_prover_server::{PicoProver, PicoProverServer},
    ErrCode, ErrMsg, GetProveResultRequest, GetProveResultResponse, ProofInfo, ProofType,
    ProvingRequest, ProvingResponse,
};
use rsp_client_executor::{executor::EthClientExecutor, io::EthClientExecutorInput, stdin};
use tonic::{codec::CompressionEncoding, transport::Server, Request, Response, Status};
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// The PicoProver service, as brevis-vm's provers expose it.
pub mod pb {
    tonic::include_proto!("prover_server");
}

/// gRPC message limit, both ways, as `eth-proofs` sets it.
const MAX_MESSAGE_SIZE: usize = 600 * 1024 * 1024;

/// Results kept for `GetProveResult`; the oldest blocks go first.
const RESULT_CAPACITY: usize = 64;

#[derive(Debug, Parser)]
#[clap(about = "A stand-in PicoProver that executes the client natively and proves nothing")]
struct Args {
    /// Where `eth-proofs` connects (its WITNESS_GETAWAY_ENDPOINT).
    #[clap(long, env = "GRPC_ADDR", default_value = "127.0.0.1:50051")]
    grpc_addr: SocketAddr,
}

#[derive(Default)]
struct State {
    /// The block being executed. One at a time, as the real prover takes them.
    in_flight: Option<u64>,
    results: BTreeMap<u64, Result<ProofInfo, String>>,
}

struct Service {
    state: Arc<Mutex<State>>,
}

/// What the guest does with its input, up to the header it commits.
fn execute(block: u64, buffer: Vec<u8>) -> Result<Vec<u8>, String> {
    let (input, witness) =
        stdin::split(buffer, |bytes| bincode::deserialize::<EthClientExecutorInput>(bytes))
            .ok_or("the input is not a joined buffer")?;
    let input = input.map_err(|err| format!("the input does not decode: {err}"))?;
    #[cfg(any(feature = "arena", feature = "flat"))]
    let input = EthClientExecutorInput { parent_state: witness, ..input };
    #[cfg(not(any(feature = "arena", feature = "flat")))]
    {
        if !witness.is_empty() {
            return Err("the input carries a witness blob, which the default backend does not \
                        take; was the host built with another backend?"
                .into());
        }
    }

    let number = input.current_block.header.number;
    if number != block {
        return Err(format!("the request names block {block} and its input is block {number}"));
    }
    let expected = input.current_block.header.hash_slow();
    let committed = EthClientExecutor::execute_committed(input)
        .map_err(|err| format!("the client failed: {err}"))?;
    let derived = committed.header.hash_slow();
    if derived != expected {
        return Err(format!("the client's header hashes to {derived}, the block is {expected}"));
    }
    bincode::serialize(&committed).map_err(|err| err.to_string())
}

fn refusal(code: ErrCode, msg: String) -> Option<ErrMsg> {
    Some(ErrMsg { code: code as i32, msg })
}

#[tonic::async_trait]
impl PicoProver for Service {
    async fn check_health(&self, _request: Request<()>) -> Result<Response<()>, Status> {
        Ok(Response::new(()))
    }

    async fn request_prover(
        &self,
        request: Request<ProvingRequest>,
    ) -> Result<Response<ProvingResponse>, Status> {
        let request = request.into_inner();
        let block = request.block_number;
        let received_at = Instant::now();

        {
            let mut state = self.state.lock().unwrap();
            if let Some(busy) = state.in_flight {
                warn!("rejecting block {block}: block {busy} is still in flight");
                return Ok(Response::new(ProvingResponse {
                    err: refusal(
                        ErrCode::Internal,
                        format!("another block is in flight; rejected block {block}"),
                    ),
                }));
            }
            state.in_flight = Some(block);
            state.results.remove(&block);
        }

        let input = request.input_buffer.unwrap_or_default();
        let input_len = input.len();
        let state = self.state.clone();
        tokio::spawn(async move {
            // On a blocking thread, and joined: a panic in the client is a failed block, as a
            // guest panic is to the prover, not the end of this process.
            let outcome = tokio::task::spawn_blocking(move || execute(block, input))
                .await
                .unwrap_or_else(|err| Err(format!("the client panicked: {err}")));
            let elapsed = received_at.elapsed();
            let result = match outcome {
                Ok(committed) => {
                    info!("block {block}: executed in {elapsed:.2?}, input {input_len} bytes");
                    Ok(ProofInfo {
                        proof_with_publics: committed,
                        block_number: block,
                        proving_cycles: 0,
                        verifier_id: "native-client".to_string(),
                        proof_type: ProofType::Combine as i32,
                        proving_duration: Some(prost_types::Duration {
                            seconds: elapsed.as_secs() as i64,
                            nanos: elapsed.subsec_nanos() as i32,
                        }),
                    })
                }
                Err(reason) => {
                    error!("block {block}: FAILED after {elapsed:.2?}: {reason}");
                    Err(reason)
                }
            };
            let mut state = state.lock().unwrap();
            state.results.insert(block, result);
            while state.results.len() > RESULT_CAPACITY {
                state.results.pop_first();
            }
            state.in_flight = None;
        });

        Ok(Response::new(ProvingResponse { err: None }))
    }

    async fn get_prove_result(
        &self,
        request: Request<GetProveResultRequest>,
    ) -> Result<Response<GetProveResultResponse>, Status> {
        let block = request.into_inner().block_number;
        let state = self.state.lock().unwrap();
        let response = match state.results.get(&block) {
            Some(Ok(proof)) => {
                GetProveResultResponse { err: None, proof_info: Some(proof.clone()) }
            }
            Some(Err(reason)) => GetProveResultResponse {
                err: refusal(ErrCode::ProveFailed, reason.clone()),
                proof_info: None,
            },
            // Not ready, or never sent: the caller polls, and resubmits after a while.
            None => GetProveResultResponse { err: None, proof_info: None },
        };
        Ok(Response::new(response))
    }
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    tracing_subscriber::registry().with(fmt::layer()).with(EnvFilter::from_default_env()).init();

    let args = Args::parse();
    let service = PicoProverServer::new(Service { state: Arc::default() })
        .max_decoding_message_size(MAX_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_MESSAGE_SIZE)
        .accept_compressed(CompressionEncoding::Zstd)
        .send_compressed(CompressionEncoding::Zstd);
    info!("native-prover (executes the client, proves nothing) listening on {}", args.grpc_addr);
    Server::builder().add_service(service).serve(args.grpc_addr).await?;
    Ok(())
}
