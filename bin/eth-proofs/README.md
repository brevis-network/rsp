# eth-proofs

Follows a chain and has its blocks proved by the `PicoProver` gRPC service of brevis-vm
(`mle-ethproofs-single-node`). This package builds two binaries:

- **`eth-proofs`**: the long-running service. It subscribes to new blocks over WebSocket, takes
  every block whose number is a multiple of `--block-interval`, fetches it and its execution
  witness, executes it to build the client input, and hands the input to the prover. One block
  is with the prover at a time; the others wait in memory, up to `--max-queued-blocks`. A proof
  is reported to [ethproofs](https://ethproofs.org) when `ETH_PROOFS_ENDPOINT` and
  `ETH_PROOFS_API_TOKEN` are both set; without them the service proves and logs.
- **`native-prover`**: a stand-in for the prover that serves the same gRPC service, runs the
  client natively on each input and proves nothing. With it the fetching side runs against a
  node on a machine without a GPU, and every block that passes is an input the guest accepts.

## Building

The host has to emit the witness the guest ELF in the prover was built to read, and that is a
cargo feature here as it is for the guest. The Pico guest that ships is built with `flat`:

```bash
cargo build --release -p eth-proofs --features flat
```

`--features arena` emits the arena witness, and neither feature the pointer-based state of the
default backend. The service logs which one it was built with when it starts.

The build needs `protoc` with its `google/protobuf` include files.

## Running

```bash
# the prover: brevis-vm's ethproofs_single_node_mle.sh, or the stand-in
native-prover --grpc-addr 127.0.0.1:50051

# the service
HTTP_RPC_URL=http://<node>:8545 WS_RPC_URL=ws://<node>:8546 \
WITNESS_GETAWAY_ENDPOINT=http://127.0.0.1:50051 \
eth-proofs --chain sepolia --block-interval 1
```

`--chain` is `mainnet` (the default) or `sepolia`, and the service refuses to start on a node of
another chain. `--stdin-dir` keeps every input as the prover was sent it (`<block>.bin`, which
brevis-vm's `MLE_INPUT_DIR` reads), `--proof-dir` every proof, and `--max-blocks` stops after
that many proofs.

The input is one buffer: the witness blob, the bincode-serialized client input, and the blob's
length (`rsp_client_executor::stdin`).

The node has to serve `debug_executionWitness` (`--state-backend execution-witness`, the
default). On reth that request is cheap at the head and grows with the block's distance from
it, so a service that falls far behind should skip blocks, which `--max-queued-blocks` does,
rather than ask for old ones.
