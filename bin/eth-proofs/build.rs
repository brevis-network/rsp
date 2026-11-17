fn main() {
    // The service brevis-vm's provers expose (its `distributed-vm/proto/prover_server.proto`).
    tonic_build::compile_protos("proto/prover_server.proto").unwrap();
}
