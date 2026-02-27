use clap::Parser;
use rsp_client_executor::{executor::EthClientExecutor, io::EthClientExecutorInput};
use std::sync::Arc;
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the input file containing bincode-serialized EthClientExecutorInput
    #[arg(long)]
    input: std::path::PathBuf,
}

fn main() {
    // Initialize the environment variables.
    dotenv::dotenv().ok();

    // Initialize the logger.
    tracing_subscriber::fmt::init();

    // Parse the command line arguments.
    let args = Args::parse();
    info!("args = {args:?}");

    // Read and deserialize input
    let input_data = std::fs::read(&args.input).expect("failed to read input file");
    let input: EthClientExecutorInput =
        bincode::deserialize(&input_data).expect("failed to deserialize input");
    info!("input block-{:?}", input.current_block.header.number);

    // Execute the block
    info!("init eth executor");
    let executor = EthClientExecutor::eth(
        Arc::new((&input.genesis).try_into().unwrap()),
        input.custom_beneficiary,
    );
    let header = executor.execute(input).expect("failed to execute client");
    info!("execution success gas_used = {}", header.gas_used);
}
