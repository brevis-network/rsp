//! The client's input as one buffer.
//!
//! SP1 hands its guest the serialized input and the witness blob as two stdin items. brevis-vm's
//! provers hand the guest of a block one byte vector, so for Pico they travel in one, with the
//! contract codes in a section of their own:
//!
//! ```text
//! <witness blob> <zeros up to a multiple of 8> <codes> <serialized input>
//! <length of the codes: u64, little-endian> <length of the witness blob: u64, little-endian>
//! ```
//!
//! The guest keeps the buffer as one [`Bytes`] and takes the blob and every code as a slice of
//! it, so nothing copies them. The codes are laid out as [`crate::codes`] describes, and the
//! serialized input carries none. The default backend has no blob (its state is inside the
//! serialized input).

use alloy_primitives::Bytes;
use reth_primitives_traits::NodePrimitives;
use serde::de::DeserializeOwned;

use crate::{
    codes::{self, CodesError},
    io::ClientExecutorInput,
};

const TRAILER_LEN: usize = 2 * core::mem::size_of::<u64>();

/// The parts of a buffer made by [`join`], each a slice of it.
#[derive(Debug)]
pub struct Parts {
    /// The serialized input, which carries no codes.
    pub input: Bytes,
    /// The witness blob.
    pub witness: Bytes,
    /// The codes, as [`codes::write`] laid them out.
    pub codes: Bytes,
}

/// Why a buffer is not a client input.
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error("the buffer is not a joined input")]
    NotJoined,
    #[error("the input does not decode: {0}")]
    Input(#[from] bincode::Error),
    #[error("the codes do not decode: {0}")]
    Codes(#[from] CodesError),
    #[error("the buffer carries a witness blob, which the default backend does not take")]
    UnexpectedWitness,
}

/// Joins a serialized input, its witness blob and its codes into one buffer.
pub fn join(serialized_input: &[u8], witness: &[u8], codes: &[u8]) -> Vec<u8> {
    let codes_start = witness.len().next_multiple_of(8);
    let mut buffer =
        Vec::with_capacity(codes_start + codes.len() + serialized_input.len() + TRAILER_LEN);
    buffer.extend_from_slice(witness);
    buffer.resize(codes_start, 0);
    buffer.extend_from_slice(codes);
    buffer.extend_from_slice(serialized_input);
    buffer.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    buffer.extend_from_slice(&(witness.len() as u64).to_le_bytes());
    buffer
}

/// Takes a buffer made by [`join`] apart.
///
/// `None` if the buffer is not one that [`join`] makes: it is shorter than the trailer, the
/// trailer names more bytes than come before it, or the bytes between the blob and the codes are
/// not zeros.
pub fn split(buffer: Bytes) -> Option<Parts> {
    let body_len = buffer.len().checked_sub(TRAILER_LEN)?;
    let length = |at: usize| {
        let bytes: [u8; 8] = buffer[at..at + 8].try_into().unwrap();
        usize::try_from(u64::from_le_bytes(bytes)).ok()
    };
    let (codes_len, witness_len) = (length(body_len)?, length(body_len + 8)?);
    let codes_start = witness_len.checked_next_multiple_of(8)?;
    let codes_end = codes_start.checked_add(codes_len).filter(|end| *end <= body_len)?;
    if buffer[witness_len..codes_start].iter().any(|byte| *byte != 0) {
        return None;
    }
    Some(Parts {
        input: buffer.slice(codes_end..body_len),
        witness: buffer.slice(..witness_len),
        codes: buffer.slice(codes_start..codes_end),
    })
}

/// Reads the client input out of a buffer made by [`join`]: the serialized input, with the codes
/// and the witness blob given to it as slices of the buffer.
pub fn read<P>(buffer: Bytes) -> Result<ClientExecutorInput<P>, ReadError>
where
    P: NodePrimitives,
    ClientExecutorInput<P>: DeserializeOwned,
{
    let parts = split(buffer).ok_or(ReadError::NotJoined)?;
    let mut input = bincode::deserialize::<ClientExecutorInput<P>>(&parts.input)?;
    input.bytecodes = codes::read(&parts.codes)?;
    #[cfg(any(feature = "arena", feature = "flat"))]
    {
        input.parent_state = parts.witness;
    }
    // A blob here means the host was built with another witness backend than this client.
    #[cfg(not(any(feature = "arena", feature = "flat")))]
    if !parts.witness.is_empty() {
        return Err(ReadError::UnexpectedWitness);
    }
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::{join, split, Bytes, TRAILER_LEN};

    #[test]
    fn split_returns_what_was_joined() {
        for input in [&b"serialized input"[..], b""] {
            for witness in [&b"witness blob"[..], b"eight by", b""] {
                for codes in [&b"the codes of the block"[..], b""] {
                    let buffer = join(input, witness, codes);
                    let gap = witness.len().next_multiple_of(8) - witness.len();
                    assert_eq!(
                        buffer.len(),
                        witness.len() + gap + codes.len() + input.len() + TRAILER_LEN
                    );
                    let parts = split(buffer.into()).unwrap();
                    assert_eq!(&parts.input[..], input);
                    assert_eq!(&parts.witness[..], witness);
                    assert_eq!(&parts.codes[..], codes);
                }
            }
        }
    }

    #[test]
    fn the_parts_are_slices_of_the_buffer_and_the_codes_start_at_a_multiple_of_8() {
        for witness_len in [0, 1, 7, 8, 9, 1000, 1001] {
            let buffer = Bytes::from(join(&[7; 100], &vec![9; witness_len], &[5; 64]));
            let start = buffer.as_ptr() as usize;
            let parts = split(buffer.clone()).unwrap();
            assert_eq!(parts.witness.as_ptr() as usize, start);
            let codes_at = parts.codes.as_ptr() as usize - start;
            assert_eq!(codes_at, witness_len.next_multiple_of(8));
            assert_eq!(parts.input.as_ptr() as usize - start, codes_at + 64);
            assert_eq!(&parts.witness[..], &vec![9; witness_len][..]);
            assert_eq!(&parts.codes[..], &[5; 64][..]);
            assert_eq!(&parts.input[..], &[7; 100][..]);
        }
    }

    #[test]
    fn a_buffer_join_did_not_make_is_refused() {
        // shorter than the trailer
        assert!(split(vec![0; TRAILER_LEN - 1].into()).is_none());
        let joined = join(b"input", b"witness", b"codes");
        let (codes_at, witness_at) = (joined.len() - 16, joined.len() - 8);
        let with = |at: usize, length: u64| {
            let mut buffer = joined.clone();
            buffer[at..at + 8].copy_from_slice(&length.to_le_bytes());
            split(buffer.into())
        };
        // a witness or codes one byte longer than what is before the trailer, and lengths that
        // only fit 64 bits
        assert!(with(witness_at, 19).is_none());
        assert!(with(codes_at, 11).is_none());
        assert!(with(witness_at, u64::MAX).is_none());
        assert!(with(witness_at, u64::MAX - 7).is_none());
        assert!(with(codes_at, u64::MAX).is_none());
        // the largest codes that are still inside the buffer
        let parts = with(codes_at, 10).unwrap();
        assert_eq!((&parts.codes[..], &parts.input[..]), (&b"codesinput"[..], &b""[..]));
        // a byte between the blob and the codes that is not zero
        let mut buffer = joined.clone();
        buffer[7] = 1;
        assert!(split(buffer.into()).is_none());
        assert!(split(joined.into()).is_some());
    }
}
