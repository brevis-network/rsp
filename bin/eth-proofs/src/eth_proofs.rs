//! Reports to the ethproofs API: a block queued, a block being proved, a block proved.

use std::time::Duration;

use base64::{engine::general_purpose::STANDARD, Engine};
use eyre::eyre;
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};
use tracing::{debug, error, info};

#[derive(Debug, Clone)]
pub struct EthProofsClient {
    cluster_id: u64,
    /// `None` reports nothing: the service then proves and logs, which is how a chain ethproofs
    /// does not list is run.
    submit: Option<Submit>,
}

#[derive(Debug, Clone)]
struct Submit {
    endpoint: String,
    api_token: String,
    client: ClientWithMiddleware,
}

impl EthProofsClient {
    /// `credentials` is the endpoint and the API token; without them nothing is reported.
    pub fn new(cluster_id: u64, credentials: Option<(String, String)>) -> Self {
        let submit = credentials.map(|(endpoint, api_token)| {
            let retry_policy = ExponentialBackoff::builder().build_with_max_retries(10);
            let client = ClientBuilder::new(reqwest::Client::new())
                .with(RetryTransientMiddleware::new_with_policy(retry_policy))
                .build();
            Submit { endpoint, api_token, client }
        });
        Self { cluster_id, submit }
    }

    /// Whether anything is reported.
    pub fn is_enabled(&self) -> bool {
        self.submit.is_some()
    }

    /// Posts `json` to `<endpoint>/<path>` on a task of its own, so that its retries hold up
    /// neither the block being fetched nor the one being proved. `on_success` is logged at info
    /// when the API takes it.
    fn post(&self, path: &'static str, json: serde_json::Value, on_success: Option<&'static str>) {
        let Some(submit) = self.submit.clone() else { return };
        tokio::spawn(async move {
            let response = submit
                .client
                .post(format!("{}/{path}", submit.endpoint))
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", submit.api_token))
                .json(&json)
                .send()
                .await
                .map_err(|e| eyre!(e))
                .and_then(|r| r.error_for_status().map_err(|e| eyre!(e)));

            match (response, on_success) {
                (Ok(_), Some(line)) => info!("{line}"),
                (Ok(_), None) => {}
                (Err(err), _) => error!("Failed to report {path}: {err}"),
            }
        });
    }

    pub fn queued(&self, block_number: u64) {
        let json = serde_json::json!({
            "block_number": block_number,
            "cluster_id": self.cluster_id,
        });
        self.post("proofs/queued", json, None);
    }

    pub fn proving(&self, block_number: u64) {
        let json = serde_json::json!({
            "block_number": block_number,
            "cluster_id": self.cluster_id,
        });
        self.post("proofs/proving", json, None);
    }

    pub fn proved(
        &self,
        proof_bytes: &[u8],
        block_number: u64,
        cycle_count: u64,
        elapsed: Duration,
        vk: &str,
    ) {
        let json = serde_json::json!({
            "proof": STANDARD.encode(proof_bytes),
            "block_number": block_number,
            "proving_cycles": cycle_count,
            "proving_time": (elapsed.as_secs_f32() * 1000.0) as u64,
            "verifier_id": vk,
            "cluster_id": self.cluster_id,
        });
        debug!("proved json report: {}", json);
        // Logged only on success: this line is what monitoring counts submissions by.
        self.post("proofs/proved", json, Some(" Submit proof to the API Successfully"));
    }
}
