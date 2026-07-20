use clap::Parser;
use rsp_client_executor::{
    executor::EthClientExecutor,
    io::{EthClientExecutorInput, LegacyEthClientExecutorInput},
};
use std::sync::Arc;
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the input file containing bincode-serialized EthClientExecutorInput
    #[arg(long)]
    input: std::path::PathBuf,

    /// Treat the input file as the legacy (MptNode-graph) format and convert it to the flat
    /// format, writing the converted input to this path before executing it.
    #[arg(long)]
    convert_legacy_to: Option<std::path::PathBuf>,
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
    let converted;
    let input: EthClientExecutorInput<'_> = if let Some(out) = &args.convert_legacy_to {
        let legacy: LegacyEthClientExecutorInput =
            bincode::deserialize(&input_data).expect("failed to deserialize legacy input");
        converted = EthClientExecutorInput::from(legacy);
        let bytes = bincode::serialize(&converted).expect("failed to serialize converted input");
        std::fs::write(out, &bytes).expect("failed to write converted input");
        info!("converted legacy input ({} bytes) -> flat input ({} bytes)", input_data.len(), bytes.len());
        converted.clone()
    } else {
        bincode::deserialize(&input_data).expect("failed to deserialize input")
    };
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
