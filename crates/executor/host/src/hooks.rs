use std::{future::Future, time::Duration};

use alloy_consensus::Block;
use reth_primitives_traits::NodePrimitives;
use sp1_sdk::{ExecutionReport, HashableKey, SP1VerifyingKey};

pub trait ExecutionHooks: Send {
    fn on_execution_start(
        &self,
        _block_number: u64,
    ) -> impl Future<Output = eyre::Result<()>> + Send {
        async { Ok(()) }
    }

    fn on_execution_end<P: NodePrimitives>(
        &self,
        _executed_block: &Block<P::SignedTx>,
        _execution_report: &ExecutionReport,
    ) -> impl Future<Output = eyre::Result<()>> {
        async { Ok(()) }
    }

    fn on_proving_start(&self, _block_number: u64) -> impl Future<Output = eyre::Result<()>> {
        async { Ok(()) }
    }

    fn on_proving_end(
        &self,
        block_number: u64,
        proof_bytes: &[u8],
        vk: &SP1VerifyingKey,
        cycles_count: Option<u64>,
        proving_duration: Duration,
    ) -> impl Future<Output = eyre::Result<()>> {
        async move {
            println!("\n===== Proving Completed =====");
            println!("on_proving_end triggered!");
            println!("block_number = {}", block_number);
            println!("proof_bytes length = {} bytes", proof_bytes.len());
            println!("verifier_id = {:?}", vk.bytes32());
            println!("proving_cycles = {:?} cycles", cycles_count);
            // println!("sp1 gas = {:?} gas", execution_report.gas);
            println!(
                "proving_duration = {:?} ms",
                (proving_duration.as_secs_f32() * 1000.0) as u64
            );
            println!("==============================\n");
            Ok(())
        }
    }
}

impl ExecutionHooks for () {}
