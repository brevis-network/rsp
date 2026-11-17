use std::iter::once;

use alloy_consensus::{Block, BlockHeader, Header};
use alloy_primitives::map::HashMap;
use alloy_primitives::{keccak256, Address, B256, U256};
use evm2::{
    bytecode::Bytecode,
    evm::{AccountInfo, DatabaseRef},
};
use itertools::Itertools;
use reth_errors::ProviderError;
use reth_ethereum_primitives::EthPrimitives;
use reth_primitives_traits::{NodePrimitives, SealedHeader};
use reth_trie_common::EMPTY_ROOT_HASH;
use rsp_mpt::StateTries;
use rsp_primitives::genesis::Genesis;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

#[cfg(not(any(feature = "arena", feature = "flat")))]
use rsp_mpt::EthereumState;
#[cfg(feature = "flat")]
use rsp_mpt::{FlatEthereumState, FlatTries};
#[cfg(feature = "arena")]
use {bumpalo::Bump, rsp_mpt::ArenaTries};

use crate::error::ClientError;

/// Bincode-compatible serde for [`Block`] via RLP.
///
/// The default serde representation of [`Block`] is not bincode-compatible (the inner transaction
/// envelope serializes via `#[serde(flatten)]`, which makes bincode fail with
/// `SequenceMustHaveLength`). reth makes blocks bincode-serializable by round-tripping through RLP
/// bytes; we mirror that here because the previous
/// `reth_primitives_traits::serde_bincode_compat::Block` wrapper was removed when
/// `reth-primitives-traits` was published to crates.io.
mod block_rlp {
    use alloy_consensus::Block;
    use alloy_primitives::Bytes;
    use alloy_rlp::{Decodable, Encodable};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use serde_with::{DeserializeAs, SerializeAs};

    pub(crate) struct BlockRlp;

    impl<T: Encodable> SerializeAs<Block<T>> for BlockRlp {
        fn serialize_as<S: Serializer>(
            source: &Block<T>,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            Bytes::from(alloy_rlp::encode(source)).serialize(serializer)
        }
    }

    impl<'de, T: Decodable> DeserializeAs<'de, Block<T>> for BlockRlp {
        fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<Block<T>, D::Error> {
            let bytes = Bytes::deserialize(deserializer)?;
            Block::<T>::decode(&mut bytes.as_ref()).map_err(serde::de::Error::custom)
        }
    }
}

/// Serde for contract code as its raw bytes.
///
/// The wire form is the preimage of the code hash and nothing else. What kind of code it is (an
/// EIP-7702 delegation or plain code), the address a delegation names and the jump table are all
/// derived from those bytes in the client, so nothing about how the code executes is taken from
/// the host: two values with the same hash are the same bytes, and so the same [`Bytecode`]. Our
/// `pico-rv64` branch found what the alternative costs, where an analysed form on the wire let a
/// forged jump table or delegation address execute under a legitimate contract's hash.
///
/// Bytes that start with the EIP-7702 magic and are not a delegation designator are refused, as
/// an error of the decode.
mod raw_bytecode {
    use alloy_primitives::Bytes;
    use evm2::bytecode::Bytecode;
    use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
    use serde_with::{DeserializeAs, SerializeAs};

    pub(crate) struct RawBytecode;

    impl SerializeAs<Bytecode> for RawBytecode {
        fn serialize_as<S: Serializer>(
            source: &Bytecode,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            source.original_bytes().serialize(serializer)
        }
    }

    impl<'de> DeserializeAs<'de, Bytecode> for RawBytecode {
        fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<Bytecode, D::Error> {
            Bytecode::new_raw_checked(Bytes::deserialize(deserializer)?).map_err(D::Error::custom)
        }
    }
}

/// Domain tag for [`ClientExecutorInput::config_digest`]'s keccak preimage. Versioned, so that
/// changing what the digest covers does not silently keep the old value valid.
pub const CONFIG_DIGEST_DOMAIN: &[u8] = b"rsp/config-digest/v1";

/// The digest over the three wire fields that change execution and leave no trace in the
/// committed header. See [`CommittedHeader`].
///
/// Two distinct properties. Bincode's tuple framing length-delimits the fields, so no two
/// configurations share an encoding -- but that says nothing about anything *outside* this
/// preimage, and the guest keccaks trie nodes, bytecodes and trie keys with the same function.
/// [`CONFIG_DIGEST_DOMAIN`] supplies the separation; being fixed-length, the split between tag
/// and payload is unambiguous.
///
/// Fallible because `Genesis::Custom(ChainConfig)` carries prover-supplied structure.
pub fn config_digest(
    genesis: &Genesis,
    custom_beneficiary: &Option<Address>,
    opcode_tracking: bool,
) -> Result<B256, ClientError> {
    let mut preimage = Vec::from(CONFIG_DIGEST_DOMAIN);
    bincode::serialize_into(&mut preimage, &(genesis, custom_beneficiary, opcode_tracking))?;
    Ok(keccak256(preimage))
}

pub type EthClientExecutorInput = ClientExecutorInput<EthPrimitives>;

/// The input for the client to execute a block and fully verify the STF (state transition
/// function).
///
/// Instead of passing in the entire state, we only pass in the state roots along with merkle proofs
/// for the storage slots that were modified and accessed.
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClientExecutorInput<P: NodePrimitives> {
    /// The current block (which will be executed inside the client).
    #[serde_as(as = "block_rlp::BlockRlp")]
    pub current_block: Block<P::SignedTx>,
    /// The previous block headers starting from the most recent. There must be at least one header
    /// to provide the parent state root.
    #[serde_as(as = "Vec<alloy_consensus::serde_bincode_compat::Header>")]
    pub ancestor_headers: Vec<Header>,
    /// Network state as of the parent block.
    ///
    /// - default (legacy) backend: the full [`EthereumState`] (pointer-based MPT), serialized
    ///   inline via bincode.
    /// - `arena` backend: the arena-codec witness blob, shipped as a *separate* SP1 stdin item and
    ///   `#[serde(skip)]`ped here so bincode only handles the small header; the guest fills this
    ///   field from `sp1_zkvm::io::read_vec()` before executing and decodes it zero-copy (see
    ///   [`ArenaTries::decode`]).
    /// - `flat` backend: the flat-RLP witness ([`FlatEthereumState`], bincode), a separate stdin
    ///   item like the arena blob; [`Self::flat_state`] borrows the node blobs out of it.
    #[cfg(not(any(feature = "arena", feature = "flat")))]
    pub parent_state: EthereumState,
    #[cfg(any(feature = "arena", feature = "flat"))]
    #[serde(skip, default)]
    pub parent_state: Vec<u8>,
    /// Account bytecodes, on the wire as raw bytes.
    #[serde_as(as = "Vec<raw_bytecode::RawBytecode>")]
    pub bytecodes: Vec<Bytecode>,
    /// The genesis block, as a json string.
    pub genesis: Genesis,
    /// The genesis block, as a json string.
    pub custom_beneficiary: Option<Address>,
    /// Whether to track the cycle count of opcodes.
    pub opcode_tracking: bool,
}

impl<P: NodePrimitives> ClientExecutorInput<P> {
    /// Gets the immediate parent block's header.
    #[inline(always)]
    pub fn parent_header(&self) -> &Header {
        &self.ancestor_headers[0]
    }

    /// The digest of the fields of this input that configure the run: [`config_digest`] over
    /// `genesis`, `custom_beneficiary` and `opcode_tracking`.
    pub fn config_digest(&self) -> Result<B256, ClientError> {
        config_digest(&self.genesis, &self.custom_beneficiary, self.opcode_tracking)
    }

    /// Creates a [`TrieDB`] backed by the legacy pointer-based [`EthereumState`].
    #[cfg(not(any(feature = "arena", feature = "flat")))]
    #[inline(always)]
    pub fn witness_db(
        &self,
        sealed_headers: &[SealedHeader],
    ) -> Result<TrieDB<'_, EthereumState>, ClientError> {
        build_trie_db(&self.parent_state, self.state_anchor(), self.bytecodes(), sealed_headers)
    }

    /// Decodes the arena witness blob into bump-scoped tries (zero-copy, hash-verifying). The
    /// bump and returned tries must outlive the [`TrieDB`] built from them by
    /// [`Self::witness_db`].
    #[cfg(feature = "arena")]
    #[inline(always)]
    pub fn tries<'a>(&'a self, bump: &'a Bump) -> Result<ArenaTries<'a>, ClientError> {
        ArenaTries::decode(bump, &self.parent_state).map_err(|_| ClientError::MismatchedStateRoot)
    }

    /// Reads the flat witness out of `parent_state`. Its node blobs are borrowed from that
    /// buffer, and the returned state must outlive the [`FlatTries`] built over it.
    #[cfg(feature = "flat")]
    #[inline(always)]
    pub fn flat_state(&self) -> Result<FlatEthereumState<'_>, ClientError> {
        bincode::deserialize(&self.parent_state).map_err(|_| ClientError::MismatchedStateRoot)
    }

    /// Creates a [`TrieDB`] backed by the flat tries built over [`Self::flat_state`].
    #[cfg(feature = "flat")]
    #[inline(always)]
    pub fn witness_db<'a, 'b>(
        &'a self,
        tries: &'a FlatTries<'b>,
        sealed_headers: &[SealedHeader],
    ) -> Result<TrieDB<'a, FlatTries<'b>>, ClientError> {
        build_trie_db(tries, self.state_anchor(), self.bytecodes(), sealed_headers)
    }

    /// Creates a [`TrieDB`] backed by the arena tries decoded via [`Self::tries`].
    #[cfg(feature = "arena")]
    #[inline(always)]
    pub fn witness_db<'a, 'b>(
        &'a self,
        tries: &'a ArenaTries<'b>,
        sealed_headers: &[SealedHeader],
    ) -> Result<TrieDB<'a, ArenaTries<'b>>, ClientError> {
        build_trie_db(tries, self.state_anchor(), self.bytecodes(), sealed_headers)
    }
}

impl<P: NodePrimitives> WitnessInput for ClientExecutorInput<P> {
    #[inline(always)]
    fn state_anchor(&self) -> B256 {
        self.parent_header().state_root()
    }

    #[inline(always)]
    fn bytecodes(&self) -> impl Iterator<Item = &Bytecode> {
        self.bytecodes.iter()
    }

    #[inline(always)]
    fn sealed_headers(&self) -> impl Iterator<Item = SealedHeader> {
        once(SealedHeader::seal_slow(self.current_block.header.clone()))
            .chain(self.ancestor_headers.iter().map(|h| SealedHeader::seal_slow(h.clone())))
    }
}

/// What the guest commits: the derived header, **and a digest of the wire fields that change
/// execution but are invisible in it**.
///
/// The header alone does not identify the statement proved. Every field but the state root is
/// copied verbatim from the prover's block header, so the header says "this input is
/// self-consistent", not "this is mainnet block N". Pinning the header hash against a canonical
/// block closes most of that, but three wire fields leave no trace in it at all:
///
/// * [`ClientExecutorInput::genesis`] becomes the whole `ChainSpec`: a chain id with the fork
///   ladder lowered or raised runs the block under other rules, and the header it derives can be
///   the same.
/// * [`ClientExecutorInput::custom_beneficiary`] replaces the block's beneficiary for the whole
///   execution, while the committed header takes `beneficiary` from the block header. (This
///   build refuses one; the field is on the wire and stays in the digest.)
/// * [`ClientExecutorInput::opcode_tracking`] selects another execution path. (Refused by this
///   build too.)
///
/// The digest says which configuration ran; what a verifier should *accept* is integration
/// policy. The layout is the one our `pico-rv64` branch commits, and the digest of a mainnet run
/// is the same value there and here.
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CommittedHeader {
    #[serde_as(as = "alloy_consensus::serde_bincode_compat::Header")]
    pub header: Header,
    /// `keccak256` of [`CONFIG_DIGEST_DOMAIN`] followed by the bincode encoding of
    /// `(genesis, custom_beneficiary, opcode_tracking)`; see [`config_digest`].
    pub config_digest: B256,
}

impl CommittedHeader {
    /// Builds the committed value from a derived header and the configuration it ran under.
    pub fn new(header: Header, config_digest: B256) -> Self {
        Self { header, config_digest }
    }
}

/// Witness-backed database the EVM reads from during execution. Generic over the state-trie backend
/// `T` (the legacy [`EthereumState`] or the arena [`ArenaTries`]) via the [`StateTries`] trait.
#[derive(Debug)]
pub struct TrieDB<'a, T: StateTries> {
    inner: &'a T,
    block_hashes: HashMap<u64, B256>,
    bytecode_by_hash: HashMap<B256, &'a Bytecode>,
}

impl<'a, T: StateTries> TrieDB<'a, T> {
    pub fn new(
        inner: &'a T,
        block_hashes: HashMap<u64, B256>,
        bytecode_by_hash: HashMap<B256, &'a Bytecode>,
    ) -> Self {
        Self { inner, block_hashes, bytecode_by_hash }
    }
}

impl<T: StateTries> DatabaseRef for TrieDB<'_, T> {
    /// The database error type.
    type Error = ProviderError;

    /// Get basic account information.
    fn get_account(&self, address: &Address) -> Result<Option<AccountInfo>, Self::Error> {
        let account_in_trie = self.inner.account(keccak256(address));

        let account = account_in_trie.map(|account_in_trie| {
            AccountInfo::new(
                account_in_trie.balance,
                account_in_trie.nonce,
                account_in_trie.code_hash,
                None,
            )
        });

        Ok(account)
    }

    /// Get account code by its hash.
    fn get_code_by_hash(&self, hash: &B256) -> Result<Bytecode, Self::Error> {
        Ok(self.bytecode_by_hash.get(hash).map(|code| (*code).clone()).unwrap())
    }

    /// Get storage value of address at index.
    fn get_storage(&self, address: &Address, index: &U256) -> Result<U256, Self::Error> {
        Ok(self.inner.storage_value(keccak256(address), keccak256(index.to_be_bytes::<32>())))
    }

    /// Get block hash by block number.
    fn get_block_hash(&self, number: &U256) -> Result<B256, Self::Error> {
        Ok(*self
            .block_hashes
            .get(&number.to::<u64>())
            .expect("A block hash must be provided for each block number"))
    }
}

/// Verifies the state/storage roots, ancestor headers and account bytecodes of a [`StateTries`]
/// backend, and constructs the [`TrieDB`] the EVM reads against during execution.
///
/// NOTE: For some unknown reasons, calling this via a bare trait method (rather than from a method
/// on the input type) causes a zkVM run to cost over 5M cycles more. The per-backend `witness_db`
/// inherent methods on [`ClientExecutorInput`] preserve that call shape.
#[inline(always)]
pub fn build_trie_db<'a, T: StateTries>(
    tries: &'a T,
    state_anchor: B256,
    bytecodes: impl Iterator<Item = &'a Bytecode>,
    sealed_headers: &[SealedHeader],
) -> Result<TrieDB<'a, T>, ClientError> {
    if state_anchor != tries.state_root() {
        return Err(ClientError::MismatchedStateRoot);
    }

    for (hashed_address, storage_root) in tries.storage_roots() {
        let account_storage_root =
            tries.account(hashed_address).map_or(EMPTY_ROOT_HASH, |a| a.storage_root);
        if account_storage_root != storage_root {
            return Err(ClientError::MismatchedStorageRoot);
        }
    }

    let bytecodes_by_hash =
        bytecodes.map(|code| (code.hash_slow(), code)).collect::<HashMap<_, _>>();

    // Verify and build block hashes
    let mut block_hashes: HashMap<u64, B256> = HashMap::with_hasher(Default::default());
    for (child_header, parent_header) in sealed_headers.iter().tuple_windows() {
        if parent_header.number() != child_header.number() - 1 {
            return Err(ClientError::InvalidHeaderBlockNumber(
                parent_header.number() + 1,
                child_header.number(),
            ));
        }

        let parent_header_hash = parent_header.hash();
        if parent_header_hash != child_header.parent_hash() {
            return Err(ClientError::InvalidHeaderParentHash(
                parent_header_hash,
                child_header.parent_hash(),
            ));
        }

        block_hashes.insert(parent_header.number(), child_header.parent_hash());
    }

    Ok(TrieDB::new(tries, block_hashes, bytecodes_by_hash))
}

/// A trait for the backend-independent inputs used to construct a [`TrieDB`].
pub trait WitnessInput {
    /// Gets the state trie root hash that the backing state must conform to.
    fn state_anchor(&self) -> B256;

    /// Gets an iterator over account bytecodes.
    fn bytecodes(&self) -> impl Iterator<Item = &Bytecode>;

    /// Gets an iterator over references to a consecutive, reverse-chronological block headers
    /// starting from the current block header.
    fn sealed_headers(&self) -> impl Iterator<Item = SealedHeader>;
}

#[cfg(test)]
mod raw_bytecode_tests {
    use alloy_primitives::{address, bytes, Bytes};

    use super::*;

    /// A list of codes on the wire, as [`ClientExecutorInput::bytecodes`] is.
    #[serde_as]
    #[derive(Serialize, Deserialize)]
    struct Codes(#[serde_as(as = "Vec<raw_bytecode::RawBytecode>")] Vec<Bytecode>);

    fn over_the_wire(codes: Vec<Bytecode>) -> Result<Vec<Bytecode>, bincode::Error> {
        let wire = bincode::serialize(&Codes(codes))?;
        bincode::deserialize::<Codes>(&wire).map(|codes| codes.0)
    }

    /// What arrives is a function of the bytes the code hash covers: the wire carries them and
    /// nothing else, so there is nothing to vary while holding the hash fixed.
    #[test]
    fn the_wire_form_is_the_hashed_preimage() {
        // PUSH1 4 JUMP INVALID JUMPDEST STOP, then a PUSH32 cut short at the end of the code
        let plain = bytes!("600456fe5b007f0102");
        let delegation =
            Bytecode::new_eip7702(address!("00000000000000000000000000000000cafebabe"));
        let codes = vec![
            Bytecode::new_raw(plain.clone()),
            delegation.clone(),
            Bytecode::new_raw(Bytes::new()),
        ];

        // On the wire: a count, then each code as its length and its bytes.
        let wire = bincode::serialize(&Codes(codes.clone())).unwrap();
        let mut expected = 3u64.to_le_bytes().to_vec();
        for code in &codes {
            let raw = code.original_bytes();
            expected.extend_from_slice(&(raw.len() as u64).to_le_bytes());
            expected.extend_from_slice(&raw);
        }
        assert_eq!(wire, expected);

        let received = over_the_wire(codes.clone()).unwrap();
        assert_eq!(received.len(), 3);
        for (sent, got) in codes.iter().zip(&received) {
            assert_eq!(got.original_bytes(), sent.original_bytes());
            assert_eq!(got.hash_slow(), keccak256(sent.original_bytes()));
        }
        // The kind and the delegated address come out of the bytes.
        assert_eq!(received[0].eip7702_address(), None);
        assert_eq!(received[1].eip7702_address(), delegation.eip7702_address());
        assert!(received[1].eip7702_address().is_some());
    }

    /// Bytes with the EIP-7702 magic that are not a delegation designator are refused by the
    /// decode, as an error rather than a panic.
    #[test]
    fn a_malformed_7702_designator_is_refused() {
        let designator =
            Bytecode::new_eip7702(address!("00000000000000000000000000000000cafebabe"))
                .original_bytes();
        // one byte short, one byte long, and an unknown version
        let mut wrong_version = designator.to_vec();
        wrong_version[2] = 1;
        for malformed in [
            designator.slice(..designator.len() - 1),
            [&designator[..], &[0u8][..]].concat().into(),
            Bytes::from(wrong_version),
        ] {
            // `Bytecode::new_legacy` takes any bytes, which is how one gets onto the wire.
            let sent = vec![Bytecode::new_legacy(malformed)];
            assert!(over_the_wire(sent).is_err());
        }
    }
}

#[cfg(test)]
mod committed_header_tests {
    use alloy_primitives::address;

    use super::*;

    /// Each of the three wire fields the committed header cannot show must move the digest.
    ///
    /// This is the property that makes the commitment say *which chain's rules ran*: without
    /// it, `genesis` could be swapped for one whose fork ladder differs and the committed header
    /// would be identical.
    #[test]
    fn the_config_digest_separates_every_field_it_covers() {
        let base = config_digest(&Genesis::Mainnet, &None, false).unwrap();

        assert_ne!(config_digest(&Genesis::Sepolia, &None, false).unwrap(), base, "genesis");
        assert_ne!(
            config_digest(&Genesis::Linea, &None, false).unwrap(),
            base,
            "genesis (the custom-chain escape hatch)"
        );

        let beneficiary = Some(address!("00000000000000000000000000000000cafebabe"));
        let with_beneficiary = config_digest(&Genesis::Mainnet, &beneficiary, false).unwrap();
        assert_ne!(with_beneficiary, base, "custom_beneficiary");
        let other = Some(address!("00000000000000000000000000000000cafebabf"));
        assert_ne!(
            config_digest(&Genesis::Mainnet, &other, false).unwrap(),
            with_beneficiary,
            "custom_beneficiary value"
        );

        assert_ne!(config_digest(&Genesis::Mainnet, &None, true).unwrap(), base, "opcode_tracking");
    }

    /// The preimage is domain-separated. Bincode length-delimits *within* the tuple; it cannot
    /// stop the encoding from also being a valid preimage somewhere else in a guest that keccaks
    /// trie nodes, bytecodes and trie keys with the same function.
    #[test]
    fn the_config_digest_preimage_is_domain_tagged() {
        let bare = bincode::serialize(&(&Genesis::Mainnet, &None::<Address>, false)).unwrap();
        let digest = config_digest(&Genesis::Mainnet, &None, false).unwrap();
        assert_ne!(digest, keccak256(&bare), "the digest is keccak of the untagged encoding");

        let mut tagged = Vec::from(CONFIG_DIGEST_DOMAIN);
        tagged.extend_from_slice(&bare);
        assert_eq!(digest, keccak256(&tagged));
    }

    /// The digest of a mainnet run is the value our `pico-rv64` branch commits for one, so
    /// whatever pins it there keeps holding. (Its `Genesis` has `OpMainnet` between `Mainnet` and
    /// `Sepolia`; a Sepolia run therefore has another digest here than there.)
    #[test]
    fn the_mainnet_digest_is_the_one_our_pico_rv64_branch_commits() {
        // variant 0 of the enum, no beneficiary, no opcode tracking
        let mut preimage = Vec::from(CONFIG_DIGEST_DOMAIN);
        preimage.extend_from_slice(&[0u8; 4 + 1 + 1]);
        assert_eq!(config_digest(&Genesis::Mainnet, &None, false).unwrap(), keccak256(&preimage));
    }
}
