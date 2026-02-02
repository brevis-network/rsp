#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use clap::Parser;
use execute::PersistExecutionReport;
use futures::{future::join_all, stream, StreamExt};
use rsp_host_executor::{
    build_executor, create_eth_block_execution_strategy_factory, BlockExecutor,
    EthExecutorComponents, OpExecutorComponents,
};
use rsp_provider::create_provider;
use std::{fs, path::Path, sync::Arc};
use tracing_subscriber::{
    filter::EnvFilter, fmt, prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt,
};

mod execute;

mod cli;
use cli::HostArgs;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // Initialize the environment variables.
    dotenv::dotenv().ok();

    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }

    // Initialize the logger.
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::from_default_env()
                .add_directive("sp1_core_machine=warn".parse().unwrap())
                .add_directive("sp1_core_executor::executor=warn".parse().unwrap())
                .add_directive("sp1_prover=warn".parse().unwrap()),
        )
        .init();

    // Parse the command line arguments.
    let args = Arc::new(HostArgs::parse());
    let report_path = args.report_path.clone();
    let config = args.as_config().await?;
    let cache_dir = config.cache_dir.clone().unwrap();
    let persist_execution_report = PersistExecutionReport::new(
        config.chain.id(),
        report_path,
        args.precompile_tracking,
        args.opcode_tracking,
    );

    let block_execution_strategy_factory =
        create_eth_block_execution_strategy_factory(&config.genesis, config.custom_beneficiary);
    let provider = config.rpc_url.as_ref().map(|url| create_provider(url.clone()));

    let executor = Arc::new(
        build_executor::<EthExecutorComponents<_>, _>(
            provider,
            block_execution_strategy_factory,
            persist_execution_report,
            config,
        )
        .await
        .unwrap(),
    );

    let blocks: Vec<_> = (args.begin_block..args.end_block).step_by(args.step_size).collect();
    let batch_size = args.batch_size;

    let mut handles = vec![];
    for batch in blocks.chunks(batch_size).into_iter() {
        // temp comment out
        fs::remove_dir_all(&cache_dir);

        if batch.is_empty() {
            break;
        }
        let batch_begin_block = batch[0];
        stream::iter(batch)
            .map(|block| {
                let executor = executor.clone();
                async move {
                    executor.execute(*block, None).await.unwrap();
                }
            })
            .buffer_unordered(args.par_size)
            .collect::<Vec<_>>()
            .await;

        let dump_dir = cache_dir.join("input/1");

        let zip_buf = zip_dir(&dump_dir)?;

        let s3_key = format!("rsp-24000000/{batch_begin_block}+{batch_size}.tar.gz");
        let handle = tokio::task::spawn_blocking(move || {
            upload_to_s3(&s3_key, zip_buf, "application/x-gzip");
        });
        handles.push(handle);
    }

    let _ = join_all(handles).await;

    Ok(())
}

fn zip_dir<P: AsRef<Path>>(dir: P) -> eyre::Result<Vec<u8>> {
    use flate2::{write::GzEncoder, Compression};

    let mut buf = vec![];

    let enc = GzEncoder::new(&mut buf, Compression::default());
    let mut tar = tar::Builder::new(enc);
    tar.append_dir_all(".", &dir)?;
    tar.finish()?;
    let enc = tar.into_inner()?;
    enc.finish()?;

    Ok(buf)
}

fn upload_to_s3(key: &str, buffer: Vec<u8>, content_type: &str) {
    use futures::AsyncWriteExt;
    use object_store::{
        aws::AmazonS3Builder, path::Path, ClientOptions, MultipartId, ObjectStore, WriteMultipart,
    };
    use std::{env, io::Cursor, sync::Arc, time::Duration};
    use tokio::runtime::Runtime;

    let access_key_id = env::var("AWS_ACCESS_KEY_ID").unwrap();
    let secret_access_key = env::var("AWS_SECRET_ACCESS_KEY").unwrap();
    let region = env::var("AWS_REGION").unwrap();
    let bucket = env::var("AWS_S3_BUCKET").unwrap();

    let client_options = ClientOptions::default()
        .with_timeout(Duration::from_secs(120))
        .with_default_content_type(content_type);

    let builder = AmazonS3Builder::new()
        .with_region(region)
        .with_bucket_name(bucket)
        .with_client_options(client_options);

    let builder =
        builder.with_access_key_id(access_key_id).with_secret_access_key(secret_access_key);

    let s3 = Arc::new(builder.build().unwrap());
    let path = Path::from(key);

    let rt = Runtime::new().unwrap();
    rt.block_on(async move {
        let upload = s3.put_multipart(&path).await.unwrap();
        let mut writer = WriteMultipart::new(upload);
        for chunk in buffer.chunks(1_000_000_000) {
            writer.write(chunk);
        }
        writer.finish().await.unwrap();
    });
}
