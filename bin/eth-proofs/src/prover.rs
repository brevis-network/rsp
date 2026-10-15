//! The link to the PicoProver, and the loop that sees one block through it.
//!
//! The prover can die mid-block (a GPU allocation failure aborts its process), and it is down for
//! about a minute and a half while systemd restarts it and it prewarms. A restarted prover has
//! forgotten the block it held. So nothing here gives a block up to a transport error: the client
//! is rebuilt once the prover answers again, and the block is submitted again.

use std::time::{Duration, Instant};

use rsp_host_executor::{
    pico_prover_client::PicoProverClient, ExecutionHooks, GetProveResultRequest, ProofInfo,
    ProvingRequest, ProvingType,
};
use tonic::{codec::CompressionEncoding, transport::Channel};
use tracing::{error, info, warn};

use crate::eth_proofs::EthProofsClient;

/// gRPC message limit, both ways: a client input is a few MB, a proof under one.
const MAX_MESSAGE_SIZE: usize = 600 * 1024 * 1024;

/// The longest wait between two attempts to reach a prover that is down.
const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy)]
pub struct ProverTimings {
    /// How often to ask for a block's proof.
    pub poll: Duration,
    /// Resubmit a block with no proof this long after it went in.
    pub resubmit_after: Duration,
    /// Give a block up after this long without a proof.
    pub give_up_after: Duration,
}

/// One gRPC client to the prover, rebuilt after a transport error.
pub struct ProverLink {
    endpoint: String,
    client: Option<PicoProverClient<Channel>>,
}

impl ProverLink {
    pub fn new(endpoint: String) -> Self {
        Self { endpoint, client: None }
    }

    /// The client, connecting first if there is none -- and waiting, however long it takes, for a
    /// prover that is down to come back.
    async fn client(&mut self) -> &mut PicoProverClient<Channel> {
        if self.client.is_none() {
            let since = Instant::now();
            let mut delay = Duration::from_secs(1);
            let mut attempts: u32 = 0;
            let client = loop {
                attempts += 1;
                match PicoProverClient::connect(self.endpoint.clone()).await {
                    Ok(client) => break client,
                    Err(err) => {
                        // The first failure, then every tenth: a restart that takes a minute and a
                        // half logs a handful of lines rather than one per attempt.
                        if attempts % 10 == 1 {
                            warn!("prover at {} unreachable ({err}); retrying", self.endpoint);
                        }
                        tokio::time::sleep(delay).await;
                        delay = (delay * 2).min(MAX_RECONNECT_DELAY);
                    }
                }
            };
            if attempts > 1 {
                info!("prover reachable again after {:.0?} ({attempts} attempts)", since.elapsed());
            }
            self.client = Some(
                client
                    .max_encoding_message_size(MAX_MESSAGE_SIZE)
                    .max_decoding_message_size(MAX_MESSAGE_SIZE)
                    .accept_compressed(CompressionEncoding::Zstd)
                    .send_compressed(CompressionEncoding::Zstd),
            );
        }
        self.client.as_mut().expect("connected above")
    }

    /// Forget the client after a transport error, so the next call connects afresh.
    fn reset(&mut self) {
        self.client = None;
    }
}

/// See `block` through the prover and report its proof to ethproofs.
///
/// Returns once the proof is reported, once the prover reports the block failed, or after
/// `give_up_after`. Everything else is retried:
/// - a transport error (the prover went away) rebuilds the client and resubmits, since a restarted
///   prover has lost the block;
/// - a refusal (another block is in flight) is polled, then resubmitted after `resubmit_after`;
/// - so is a block that has no proof `resubmit_after` after it went in.
///
/// A resubmission cannot start a second proof of a block the prover is still working on: the
/// prover refuses any block while one is in flight.
pub async fn prove_block(
    link: &mut ProverLink,
    hooks: &EthProofsClient,
    block: u64,
    input: &[u8],
    timings: ProverTimings,
) {
    info!("Starting proof generation");
    if let Err(err) = hooks.on_proving_start(block).await {
        warn!("block {block}: proving status not reported: {err}");
    }
    let started = Instant::now();
    // When the prover last took (or refused) this block; `None` means it has to go in again.
    let mut submitted: Option<Instant> = None;
    loop {
        if started.elapsed() > timings.give_up_after {
            error!("block {block}: no proof after {:.0?}; giving it up", timings.give_up_after);
            return;
        }
        if submitted.is_none() {
            let request = ProvingRequest {
                block_number: block,
                input_buffer: Some(input.to_vec()),
                proving_type: ProvingType::Gpu as i32,
            };
            match link.client().await.request_prover(request).await {
                Ok(response) => {
                    match response.into_inner().err {
                        None => info!("gRPC client post to prover, blk number: {block}"),
                        // Another block is in flight -- possibly this one, from a submission whose
                        // answer was lost. Polling finds out; `resubmit_after` bounds the wait.
                        Some(err) => warn!("block {block}: prover refused it for now: {}", err.msg),
                    }
                    submitted = Some(Instant::now());
                }
                Err(status) => {
                    warn!("block {block}: prover unreachable on submit ({status}); will resubmit");
                    link.reset();
                    continue;
                }
            }
        }
        tokio::time::sleep(timings.poll).await;
        let request = GetProveResultRequest { block_number: block };
        match link.client().await.get_prove_result(request).await {
            Ok(response) => {
                let response = response.into_inner();
                if let Some(err) = response.err {
                    error!("block {block}: prover reported a failure, giving it up: {err:?}");
                    return;
                }
                match response.proof_info {
                    Some(proof) if !proof.proof_with_publics.is_empty() => {
                        report(hooks, block, &proof).await;
                        return;
                    }
                    _ => {
                        let waited = submitted.map(|at| at.elapsed()).unwrap_or_default();
                        if waited > timings.resubmit_after {
                            warn!("block {block}: no proof {waited:.0?} after it went in; resubmitting");
                            submitted = None;
                        } else {
                            info!("Waiting for proof {block:?} generation...");
                        }
                    }
                }
            }
            Err(status) => {
                // A restarted prover has forgotten the block, so it goes in again once the prover
                // is back.
                warn!("block {block}: prover unreachable while waiting ({status}); will resubmit");
                link.reset();
                submitted = None;
            }
        }
    }
}

async fn report(hooks: &EthProofsClient, block: u64, proof: &ProofInfo) {
    let proving_time = proof
        .proving_duration
        .as_ref()
        .map(|d| Duration::new(d.seconds as u64, d.nanos as u32))
        .unwrap_or_default();
    if let Err(err) = hooks
        .on_proving_end(
            block,
            &proof.proof_with_publics,
            &proof.verifier_id,
            Some(proof.proving_cycles),
            proving_time,
        )
        .await
    {
        error!("block {block}: proof not reported: {err}");
        return;
    }
    info!("Proof {block:?} successfully generated!, proving time: {proving_time:?}");
}
