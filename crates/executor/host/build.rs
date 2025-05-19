fn main() {
    tonic_build::compile_protos("proto/prover_server.proto").unwrap();
}
