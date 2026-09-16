use std::iter::once;

use alloy_consensus::{Block, BlockHeader, Header};
use alloy_primitives::map::{B256Map, HashMap};
use alloy_rlp::Decodable;
use itertools::Itertools;
use reth_errors::ProviderError;
use reth_ethereum_primitives::EthPrimitives;
use reth_primitives_traits::{NodePrimitives, SealedHeader};
use reth_trie::{TrieAccount, EMPTY_ROOT_HASH};
use revm::{
    state::{AccountInfo, Bytecode},
    DatabaseRef,
};
use revm_primitives::{keccak256, Address, B256, U256};
use rsp_mpt::{FlatEthereumState, FlatStateViews};
use rsp_primitives::genesis::Genesis;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::error::ClientError;

/// Domain tag for [`ClientExecutorInput::config_digest`]'s keccak preimage. Versioned, so that
/// changing what the digest covers does not silently keep the old value valid.
pub const CONFIG_DIGEST_DOMAIN: &[u8] = b"rsp/config-digest/v1";

/// The digest over the three wire fields that change execution and leave no trace in the
/// committed header. See [`CommittedHeader`].
///
/// Free-standing so that the value the guest commits is computed from the fields that
/// *configured the run* -- `ClientExecutor` holds its own copies and digests those -- while
/// [`ClientExecutorInput::config_digest`] stays available to host tooling holding an input.
pub fn config_digest(
    genesis: &Genesis,
    custom_beneficiary: &Option<Address>,
    opcode_tracking: bool,
) -> Result<B256, ClientError> {
    let mut preimage = Vec::from(CONFIG_DIGEST_DOMAIN);
    bincode::serialize_into(&mut preimage, &(genesis, custom_beneficiary, opcode_tracking))?;
    Ok(keccak256(preimage))
}

pub type EthClientExecutorInput<'a> = ClientExecutorInput<'a, EthPrimitives>;

#[cfg(feature = "optimism")]
pub type OpClientExecutorInput<'a> =
    ClientExecutorInput<'a, reth_optimism_primitives::OpPrimitives>;

/// The input for the client to execute a block and fully verify the STF (state transition
/// function).
///
/// Instead of passing in the entire state, we only pass in the state roots along with merkle proofs
/// for the storage slots that were modified and accessed. The tries are shipped in the flat RLP
/// wire format (see [`FlatEthereumState`]) and, in the zkVM, are borrowed zero-copy from the raw
/// input buffer.
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClientExecutorInput<'a, P: NodePrimitives> {
    /// The current block (which will be executed inside the client).
    #[serde_as(
        as = "reth_primitives_traits::serde_bincode_compat::Block<'_, P::SignedTx, Header>"
    )]
    pub current_block: Block<P::SignedTx>,
    /// The previous block headers starting from the most recent. There must be at least one header
    /// to provide the parent state root.
    #[serde_as(as = "Vec<alloy_consensus::serde_bincode_compat::Header>")]
    pub ancestor_headers: Vec<Header>,
    /// Network state as of the parent block, as flat RLP trie blobs.
    #[serde(borrow)]
    pub parent_state: FlatEthereumState<'a>,
    /// Account bytecodes.
    #[serde(with = "wire_bytecodes")]
    pub bytecodes: Vec<Bytecode>,
    /// The genesis block, as a json string.
    pub genesis: Genesis,
    /// The genesis block, as a json string.
    pub custom_beneficiary: Option<Address>,
    /// Whether to track the cycle count of opcodes.
    pub opcode_tracking: bool,
}

impl<P: NodePrimitives> ClientExecutorInput<'_, P> {
    /// Gets the immediate parent block's header.
    #[inline(always)]
    pub fn parent_header(&self) -> &Header {
        &self.ancestor_headers[0]
    }

    /// Parses and verifies the witness tries against the parent state root; see
    /// [`WitnessInput::verified_views`].
    #[allow(clippy::type_complexity)]
    pub fn verified_views(
        &self,
    ) -> Result<(FlatStateViews<'_>, B256Map<Option<WitnessedAccount>>), ClientError> {
        <Self as WitnessInput>::verified_views(self)
    }

    /// Verifies bytecodes and ancestor headers; see [`WitnessInput::witness_aux`].
    #[allow(clippy::type_complexity)]
    pub fn witness_aux(
        &self,
        sealed_headers: &[SealedHeader],
    ) -> Result<(HashMap<u64, B256>, B256Map<&Bytecode>), ClientError> {
        <Self as WitnessInput>::witness_aux(self, sealed_headers)
    }

    /// A digest over every wire field that changes execution and is invisible in the committed
    /// header. See [`CommittedHeader`].
    ///
    /// Two distinct properties. Bincode's tuple framing length-delimits the fields, so no two
    /// configurations share an encoding -- but that says nothing about anything *outside* this
    /// preimage, and the guest keccaks trie node blobs, bytecodes and trie keys with the same
    /// function. [`CONFIG_DIGEST_DOMAIN`] supplies the separation; being fixed-length, the split
    /// between tag and payload is unambiguous.
    ///
    /// Fallible because `Genesis::Custom(ChainConfig)` carries prover-supplied structure.
    pub fn config_digest(&self) -> Result<B256, ClientError> {
        config_digest(&self.genesis, &self.custom_beneficiary, self.opcode_tracking)
    }

    /// Converts any borrowed wire bytes into owned buffers.
    pub fn into_owned(self) -> ClientExecutorInput<'static, P> {
        ClientExecutorInput {
            current_block: self.current_block,
            ancestor_headers: self.ancestor_headers,
            parent_state: self.parent_state.into_owned(),
            bytecodes: self.bytecodes,
            genesis: self.genesis,
            custom_beneficiary: self.custom_beneficiary,
            opcode_tracking: self.opcode_tracking,
        }
    }
}

impl<P: NodePrimitives> WitnessInput for ClientExecutorInput<'_, P> {
    #[inline(always)]
    fn state(&self) -> &FlatEthereumState<'_> {
        &self.parent_state
    }

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
/// * [`ClientExecutorInput::genesis`] becomes the whole `ChainSpec`. Executed: `chainId: 1` with
///   the fork ladder lowered flips all five forks (PRAGUE gas 23605 against ISTANBUL 21705, the
///   difference landing in the beneficiary's balance), and `chainId 59144` reaches
///   `handle_custom_chains`, which turns `ExtraDataExceedsMax` and `TheMergeDifficultyIsNotZero`
///   into `Ok(())` -- a wire field that switches off two consensus rejection conditions.
/// * [`ClientExecutorInput::custom_beneficiary`] overwrites `block_env.beneficiary` for the whole
///   block while the committed header takes `beneficiary` from the block header. Executed: the
///   entire tip moves to an address of the prover's choosing and nothing compares the two.
/// * [`ClientExecutorInput::opcode_tracking`] selects an inspector frame, a different execution
///   path.
///
/// The digest says which configuration ran; what a verifier should *accept* is integration
/// policy. This changes the public values, and therefore the verifier contract -- unavoidably,
/// since by construction these fields leave no trace anywhere else.
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CommittedHeader {
    #[serde_as(as = "alloy_consensus::serde_bincode_compat::Header")]
    pub header: Header,
    /// `keccak256` of the bincode encoding of `(genesis, custom_beneficiary, opcode_tracking)`;
    /// see [`ClientExecutorInput::config_digest`].
    pub config_digest: B256,
}

impl CommittedHeader {
    /// Builds the committed value from a derived header and the configuration it ran under.
    pub fn new(header: Header, config_digest: B256) -> Self {
        Self { header, config_digest }
    }
}

/// The fields of a witnessed account that [`TrieDB::basic_ref`] hands back, small and `Copy`.
///
/// `AccountInfo` itself carries an `Option<Bytecode>`, so caching it would make every hit
/// clone a `Bytes` handle it never uses.
#[derive(Debug, Clone, Copy)]
pub struct WitnessedAccount {
    nonce: u64,
    balance: U256,
    code_hash: B256,
}

#[derive(Debug)]
pub struct TrieDB<'a> {
    views: &'a FlatStateViews<'a>,
    /// Accounts already read out of the state trie by [`WitnessInput::verified_views`], which
    /// walks to one per witnessed storage trie to bind its root. `basic_ref` is called for the
    /// same set (250 accounts and 250 storage tries on mainnet block 24006677), and a walk
    /// costs ~1,460 retired instructions against ~40 for this lookup. A miss simply falls
    /// through to the trie, so the two sets need not agree.
    accounts: B256Map<Option<WitnessedAccount>>,
    block_hashes: HashMap<u64, B256>,
    /// Keyed by the code hash, so the map's hasher must not re-hash it: `B256Map`'s
    /// `FbBuildHasher<32>` reads eight bytes of the (already uniformly distributed) digest,
    /// where the default `foldhash` builder runs its whole 32-byte mixing chain per lookup.
    bytecode_by_hash: B256Map<&'a Bytecode>,
}

impl<'a> TrieDB<'a> {
    pub fn new(
        views: &'a FlatStateViews<'a>,
        accounts: B256Map<Option<WitnessedAccount>>,
        block_hashes: HashMap<u64, B256>,
        bytecode_by_hash: B256Map<&'a Bytecode>,
    ) -> Self {
        Self { views, accounts, block_hashes, bytecode_by_hash }
    }
}

impl DatabaseRef for TrieDB<'_> {
    /// The database error type.
    type Error = ProviderError;

    /// Get basic account information.
    fn basic_ref(&self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        let hashed_address = keccak256(address);

        if let Some(cached) = self.accounts.get(&hashed_address) {
            return Ok(cached.map(|a| AccountInfo {
                balance: a.balance,
                nonce: a.nonce,
                code_hash: a.code_hash,
                code: None,
            }));
        }

        // `get` answers `Err(NodeNotResolved)` for a key whose path leaves the witnessed region
        // -- a subtree the prover chose to omit, so a rejection rather than a crash.
        let account_in_trie = self
            .views
            .state
            .get(hashed_address.as_slice())
            .map_err(|e| ProviderError::TrieWitnessError(e.to_string()))?;

        let account = account_in_trie.map(|mut bytes| {
            let account_in_trie = TrieAccount::decode(&mut bytes).unwrap();
            AccountInfo {
                balance: account_in_trie.balance,
                nonce: account_in_trie.nonce,
                code_hash: account_in_trie.code_hash,
                code: None,
            }
        });

        Ok(account)
    }

    /// Get account code by its hash.
    fn code_by_hash_ref(&self, hash: B256) -> Result<Bytecode, Self::Error> {
        Ok(self.bytecode_by_hash.get(&hash).map(|code| (*code).clone()).unwrap())
    }

    /// Get storage value of address at index.
    fn storage_ref(&self, address: Address, index: U256) -> Result<U256, Self::Error> {
        let hashed_address = keccak256(address);

        let storage_view = self
            .views
            .storage
            .get(&hashed_address)
            .expect("A storage trie must be provided for each account");

        // As in `basic_ref`: an unwitnessed path is a rejection, not a panic.
        Ok(storage_view
            .get(keccak256(index.to_be_bytes::<32>()).as_slice())
            .map_err(|e| ProviderError::TrieWitnessError(e.to_string()))?
            .map(|mut bytes| U256::decode(&mut bytes).unwrap())
            .unwrap_or_default())
    }

    /// Get block hash by block number.
    fn block_hash_ref(&self, number: u64) -> Result<B256, Self::Error> {
        Ok(*self
            .block_hashes
            .get(&number)
            .expect("A block hash must be provided for each block number"))
    }
}

/// A trait for constructing [`TrieDB`].
pub trait WitnessInput {
    /// Gets a reference to the flat state from which account info and storage slots are loaded.
    fn state(&self) -> &FlatEthereumState<'_>;

    /// Gets the state trie root hash that the state referenced by
    /// [state()](trait.WitnessInput#tymethod.state) must conform to.
    fn state_anchor(&self) -> B256;

    /// Gets an iterator over account bytecodes.
    fn bytecodes(&self) -> impl Iterator<Item = &Bytecode>;

    /// Gets an iterator over references to a consecutive, reverse-chronological block headers
    /// starting from the current block header.
    fn sealed_headers(&self) -> impl Iterator<Item = SealedHeader>;

    /// Parses the flat witness tries, verifying every node against the state root anchor and
    /// each storage trie against its account's storage root.
    ///
    /// NOTE: For some unknown reasons, calling this trait method directly from outside of the type
    /// implementing this trait causes a zkVM run to cost over 5M cycles more. To avoid this, define
    /// a method inside the type that calls this trait method instead.
    #[inline(always)]
    #[allow(clippy::type_complexity)]
    fn verified_views(
        &self,
    ) -> Result<(FlatStateViews<'_>, B256Map<Option<WitnessedAccount>>), ClientError> {
        let views = self.state().views()?;

        if self.state_anchor() != views.state.root_hash {
            return Err(ClientError::MismatchedStateRoot);
        }

        // Every walk below reads exactly the row `basic_ref` will want later, so keep it.
        let mut accounts: B256Map<Option<WitnessedAccount>> =
            B256Map::with_capacity_and_hasher(views.storage.len(), Default::default());
        for (hashed_address, storage_view) in views.storage.iter() {
            let account = views
                .state
                .get(hashed_address.as_slice())?
                .map(|mut bytes| TrieAccount::decode(&mut bytes))
                .transpose()
                .map_err(rsp_mpt::Error::from)?;
            let storage_root = account.map_or(EMPTY_ROOT_HASH, |a| a.storage_root);
            if storage_root != storage_view.root_hash {
                return Err(ClientError::MismatchedStorageRoot);
            }
            accounts.insert(
                *hashed_address,
                account.map(|a| WitnessedAccount {
                    nonce: a.nonce,
                    balance: a.balance,
                    code_hash: a.code_hash,
                }),
            );
        }

        Ok((views, accounts))
    }

    /// Verifies the account bytecodes and the ancestor header chain, returning the block hash
    /// and bytecode lookup tables for [`TrieDB`].
    #[allow(clippy::type_complexity)]
    #[inline(always)]
    fn witness_aux(
        &self,
        sealed_headers: &[SealedHeader],
    ) -> Result<(HashMap<u64, B256>, B256Map<&Bytecode>), ClientError> {
        let bytecodes_by_hash =
            self.bytecodes().map(|code| (code.hash_slow(), code)).collect::<B256Map<_>>();

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

        Ok((block_hashes, bytecodes_by_hash))
    }
}

/// Compact wire format for account bytecodes.
///
/// revm's `Bytecode` serde round-trips the jumpdest table through `bitvec`, which bincode
/// decodes element-wise (~1.85M cycles for a mainnet block's contracts). Ship the raw code
/// bytes and the raw jump-table words instead and rebuild the analyzed bytecode with two
/// memcpys per contract. Non-legacy variants (EIP-7702) fall back to their normal encoding.
///
/// # Only the hashed preimage is trusted
///
/// `hash_slow()` covers `code[..original_len]` and nothing else, so `jump_bit_len`,
/// `jump_table`, the padding in `code[original_len..]` and the variant tag all cross the wire
/// unauthenticated under one unchanged `code_hash`. Executed, when they were believed: a
/// forged jump table turned an `InvalidJump` halt into `SUCCESS`, and one byte of the unhashed
/// padding became arbitrary-length attacker code under a legitimate contract's hash.
///
/// [`deserialize`] therefore reads **only the preimage** and re-derives everything else
/// through `Bytecode::new_raw_checked`, which is `analyze_legacy` plus the EIP-7702 prefix
/// test. That closes RSP-S0-1/2/4/6, and with them the wire half of REVM-S1-7: the analysis's
/// post-conditions hold because the analysis produced them, rather than because a constructor
/// restated two of the three.
///
/// **The wire format is deliberately unchanged.** The redundant fields are still serialized
/// and still ignored, so every previously serialized `EthClientExecutorInput` -- including the
/// committed `perf/bench_data/rv64/reth-*.bin` bench fixtures -- still decodes. That is the
/// difference from `fa4bb04`, which shipped the preimage alone and broke all of them; it is
/// also why this is not yet the whole of brevis-network/rsp#24, which additionally drops the
/// dead fields and the witness bytes they cost.
///
/// # What it costs, measured
///
/// **+202.8 M retired instructions across the thirteen `perf/bench_data/rv64` blocks, +6.5 %**
/// (+5.8 M / +1.9 % on block 24006677; worst case +17.8 % on 18884864, where the witness is
/// small and the contracts are not). Emulated on `validation/rv64-emu`, same fixtures, same
/// rustflags, both arms.
///
/// That is much more than the ~1.85M the old shape saved on the `bitvec` decode, because the
/// decode was never the whole of it: the jump table was *computed by the host*, and the guest
/// now walks every contract byte itself. **It is not avoidable by preferring #24** -- shipping
/// the preimage alone pays exactly the same walk, and differs only in also saving the witness
/// bytes the dead fields cost. The price is authenticating the bytecode at all.
///
/// Inherited from `succinctlabs/rsp`, where the unauthenticated form is still present at
/// `upstream/main` `2013b56`.
mod wire_bytecodes {
    use std::borrow::Cow;

    use revm::{primitives::Bytes, state::Bytecode};
    use rsp_mpt::serde_cow_bytes;
    use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    enum WireBytecode<'a> {
        Legacy {
            #[serde(with = "serde_cow_bytes", borrow)]
            code: Cow<'a, [u8]>,
            original_len: u64,
            jump_bit_len: u64,
            #[serde(with = "serde_cow_bytes", borrow)]
            jump_table: Cow<'a, [u8]>,
        },
        Other(Bytecode),
    }

    pub(super) fn serialize<S: Serializer>(v: &[Bytecode], s: S) -> Result<S::Ok, S::Error> {
        let wire: Vec<WireBytecode<'_>> = v
            .iter()
            .map(|b| match b {
                Bytecode::LegacyAnalyzed(a) => WireBytecode::Legacy {
                    code: Cow::Borrowed(a.bytecode().as_ref()),
                    original_len: a.original_len() as u64,
                    jump_bit_len: a.jump_table().len() as u64,
                    jump_table: Cow::Borrowed(a.jump_table().as_slice()),
                },
                other => WireBytecode::Other(other.clone()),
            })
            .collect();
        wire.serialize(s)
    }

    /// Rebuilds each bytecode **from its hashed preimage alone**.
    ///
    /// Only `code[..original_len]` is read. `jump_bit_len`, `jump_table`, the padding in
    /// `code[original_len..]` and the variant tag are all still on the wire -- the format is
    /// unchanged, so previously serialized witnesses still decode -- but none of them is
    /// trusted: `Bytecode::new_raw_checked` re-derives the jump table, the padding and the
    /// variant from the preimage, which `code_hash` binds.
    ///
    /// `original_len` is not an exception. It selects the preimage, and the preimage is what
    /// `hash_slow()` hashes, so a wrong `original_len` yields a bytecode whose hash does not
    /// match the `code_hash` the lookup used.
    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Bytecode>, D::Error> {
        let wire: Vec<WireBytecode<'de>> = Vec::deserialize(d)?;
        wire.into_iter()
            .map(|w| {
                let preimage: Bytes = match w {
                    WireBytecode::Legacy { code, original_len, .. } => {
                        let n = original_len as usize;
                        if n > code.len() {
                            return Err(D::Error::custom("original_len exceeds the code buffer"));
                        }
                        Bytes::from(code[..n].to_vec())
                    }
                    // The preimage of every variant, and the only thing `hash_slow()` covers.
                    WireBytecode::Other(b) => Bytes::from(b.original_byte_slice().to_vec()),
                };
                Bytecode::new_raw_checked(preimage).map_err(D::Error::custom)
            })
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use bincode::Options;

        use super::*;
        use revm_primitives::{bytes, Address};

        #[test]
        fn wire_bytecode_roundtrip() {
            // a couple of real-shaped legacy bytecodes (with jumpdests) + default
            let codes = vec![
                Bytecode::new_raw(bytes!("6001600255005b600056")),
                Bytecode::new_raw(bytes!("5b5b5b5b")),
                Bytecode::default(),
            ];
            let mut buf = Vec::new();
            let mut ser = bincode::Serializer::new(
                &mut buf,
                bincode::options()
                    .with_fixint_encoding()
                    .allow_trailing_bytes(),
            );
            serialize(&codes, &mut ser).unwrap();
            let mut de = bincode::Deserializer::from_slice(
                &buf,
                bincode::options()
                    .with_fixint_encoding()
                    .allow_trailing_bytes(),
            );
            let back = deserialize(&mut de).unwrap();
            assert_eq!(codes, back);
        }

        fn decode_wire(wire: Vec<WireBytecode<'_>>) -> Vec<Bytecode> {
            let mut buf = Vec::new();
            let mut ser = bincode::Serializer::new(
                &mut buf,
                bincode::options().with_fixint_encoding().allow_trailing_bytes(),
            );
            wire.serialize(&mut ser).unwrap();
            let mut de = bincode::Deserializer::from_slice(
                &buf,
                bincode::options().with_fixint_encoding().allow_trailing_bytes(),
            );
            deserialize(&mut de).unwrap()
        }

        /// Everything outside the hashed preimage is ignored.
        ///
        /// This is RSP-S0-1/2/6 as a test: the same `code_hash` with a forged jump table used
        /// to turn an `InvalidJump` halt into `SUCCESS`, and attacker bytes in the unhashed
        /// padding used to execute under a legitimate contract's hash. Both records below
        /// carry exactly those forgeries and must decode to what the preimage analyses to.
        #[test]
        fn only_the_hashed_preimage_survives_the_wire() {
            let preimage = bytes!("5b600056");
            let honest = Bytecode::new_raw(preimage.clone());

            // Junk padding standing in for attacker code, plus a jump table asserting that
            // every one of the first eight positions is a valid JUMPDEST.
            let mut code = preimage.to_vec();
            code.extend_from_slice(&[0x5b, 0x60, 0xff, 0x00]);
            let forged = WireBytecode::Legacy {
                code: Cow::Owned(code),
                original_len: preimage.len() as u64,
                jump_bit_len: 8,
                jump_table: Cow::Owned(vec![0xff]),
            };
            assert_eq!(decode_wire(vec![forged]), vec![honest.clone()]);

            // The variant tag is not the prover's either: the same preimage sent through the
            // `Other` arm derives the same legacy bytecode, because `new_raw_checked` picks
            // the variant from the preimage's own prefix.
            assert_eq!(decode_wire(vec![WireBytecode::Other(honest.clone())]), vec![honest]);

            // And an EIP-7702 preimage still derives EIP-7702, by that same prefix.
            let delegated = Bytecode::new_eip7702(Address::repeat_byte(0xab));
            assert_eq!(decode_wire(vec![WireBytecode::Other(delegated.clone())]), vec![delegated]);
        }

        /// `original_len` past the buffer is a rejection, not a panic or a wild slice.
        #[test]
        fn an_original_len_past_the_buffer_is_rejected() {
            let wire = vec![WireBytecode::Legacy {
                code: Cow::Owned(vec![0x00, 0x00]),
                original_len: 99,
                jump_bit_len: 0,
                jump_table: Cow::Owned(vec![]),
            }];
            let mut buf = Vec::new();
            let mut ser = bincode::Serializer::new(
                &mut buf,
                bincode::options().with_fixint_encoding().allow_trailing_bytes(),
            );
            wire.serialize(&mut ser).unwrap();
            let mut de = bincode::Deserializer::from_slice(
                &buf,
                bincode::options().with_fixint_encoding().allow_trailing_bytes(),
            );
            assert!(deserialize(&mut de).is_err());
        }
    }
}

/// The legacy wire format of [`ClientExecutorInput`], where the witness tries were shipped as a
/// bincode-serialized [`rsp_mpt::EthereumState`] node graph. Kept for converting previously
/// generated inputs/fixtures to the flat format (host tooling only).
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyClientExecutorInput<P: NodePrimitives> {
    #[serde_as(
        as = "reth_primitives_traits::serde_bincode_compat::Block<'_, P::SignedTx, Header>"
    )]
    pub current_block: Block<P::SignedTx>,
    #[serde_as(as = "Vec<alloy_consensus::serde_bincode_compat::Header>")]
    pub ancestor_headers: Vec<Header>,
    pub parent_state: rsp_mpt::EthereumState,
    pub bytecodes: Vec<Bytecode>,
    pub genesis: Genesis,
    pub custom_beneficiary: Option<Address>,
    pub opcode_tracking: bool,
}

pub type LegacyEthClientExecutorInput = LegacyClientExecutorInput<EthPrimitives>;

impl<P: NodePrimitives> From<LegacyClientExecutorInput<P>> for ClientExecutorInput<'static, P> {
    fn from(legacy: LegacyClientExecutorInput<P>) -> Self {
        ClientExecutorInput {
            current_block: legacy.current_block,
            ancestor_headers: legacy.ancestor_headers,
            parent_state: FlatEthereumState::from_state(&legacy.parent_state),
            bytecodes: legacy.bytecodes,
            genesis: legacy.genesis,
            custom_beneficiary: legacy.custom_beneficiary,
            opcode_tracking: legacy.opcode_tracking,
        }
    }
}

#[cfg(test)]
mod committed_header_tests {
    use super::*;
    use revm_primitives::address;

    fn input() -> EthClientExecutorInput<'static> {
        EthClientExecutorInput {
            current_block: Default::default(),
            ancestor_headers: vec![Default::default()],
            parent_state: rsp_mpt::FlatEthereumState {
                state_nodes: std::borrow::Cow::Owned(Vec::new()),
                storage_tries: Vec::new(),
            },
            bytecodes: vec![],
            genesis: Genesis::Mainnet,
            custom_beneficiary: None,
            opcode_tracking: false,
        }
    }

    /// Each of the three wire fields the committed header cannot show must move the digest.
    ///
    /// This is the property that makes the commitment say *which chain's rules ran*. Before
    /// it, `genesis` could be swapped for one whose fork ladder is lowered -- or, at
    /// `chainId 59144`, one that routes through `handle_custom_chains` and turns two consensus
    /// rejection conditions into `Ok(())` -- and the committed header would be identical.
    #[test]
    fn the_config_digest_separates_every_field_it_covers() {
        let base = input().config_digest().unwrap();

        let mut g = input();
        g.genesis = Genesis::Sepolia;
        assert_ne!(g.config_digest().unwrap(), base, "genesis");

        let mut g = input();
        g.genesis = Genesis::Linea;
        assert_ne!(g.config_digest().unwrap(), base, "genesis (the custom-chain escape hatch)");

        let mut b = input();
        b.custom_beneficiary = Some(address!("00000000000000000000000000000000cafebabe"));
        assert_ne!(b.config_digest().unwrap(), base, "custom_beneficiary");

        let mut b2 = input();
        b2.custom_beneficiary = Some(address!("00000000000000000000000000000000cafebabf"));
        assert_ne!(
            b2.config_digest().unwrap(),
            b.config_digest().unwrap(),
            "custom_beneficiary value"
        );

        let mut t = input();
        t.opcode_tracking = true;
        assert_ne!(t.config_digest().unwrap(), base, "opcode_tracking");

        // And it is a function of those fields only -- nothing else in the input perturbs it.
        let mut unrelated = input();
        unrelated.ancestor_headers = vec![Default::default(), Default::default()];
        assert_eq!(unrelated.config_digest().unwrap(), base);
    }

    /// The preimage is domain-separated, which is the property the doc used to claim of
    /// bincode's framing. Bincode length-delimits *within* the tuple; it cannot stop the
    /// encoding from also being a valid preimage somewhere else in a guest that keccaks trie
    /// blobs, bytecodes and trie keys with the same function.
    #[test]
    fn the_config_digest_preimage_is_domain_tagged() {
        let bare = bincode::serialize(&(&Genesis::Mainnet, &None::<Address>, false)).unwrap();
        assert_ne!(
            input().config_digest().unwrap(),
            keccak256(&bare),
            "the digest is keccak of the untagged encoding"
        );

        let mut tagged = Vec::from(CONFIG_DIGEST_DOMAIN);
        tagged.extend_from_slice(&bare);
        assert_eq!(input().config_digest().unwrap(), keccak256(&tagged));
    }
}
