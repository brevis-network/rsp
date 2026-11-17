//! The client's input as one buffer.
//!
//! SP1 hands its guest the serialized input and the witness blob as two stdin items. brevis-vm's
//! provers hand the guest of a block one byte vector, so for Pico the two travel in one:
//!
//! ```text
//! <witness blob> <serialized input> <length of the witness blob: u64, little-endian>
//! ```
//!
//! The witness blob comes first so that the guest keeps it without copying it: once the
//! serialized input behind it has been read, cutting the buffer down to the blob's length leaves
//! the blob. The default backend has no blob (its state is inside the serialized input), and its
//! buffer is the serialized input followed by eight zero bytes.

const TRAILER_LEN: usize = core::mem::size_of::<u64>();

/// Joins a serialized input and its witness blob into one buffer.
pub fn join(serialized_input: &[u8], witness: &[u8]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(witness.len() + serialized_input.len() + TRAILER_LEN);
    buffer.extend_from_slice(witness);
    buffer.extend_from_slice(serialized_input);
    buffer.extend_from_slice(&(witness.len() as u64).to_le_bytes());
    buffer
}

/// Takes a buffer made by [`join`] apart. `read` is given the serialized input; the buffer, cut
/// down to the witness blob, comes back next to what `read` returned.
///
/// `None` if the buffer is not one that [`join`] makes: it is shorter than the trailer, or the
/// trailer names more bytes than come before it.
pub fn split<T>(mut buffer: Vec<u8>, read: impl FnOnce(&[u8]) -> T) -> Option<(T, Vec<u8>)> {
    let body_len = buffer.len().checked_sub(TRAILER_LEN)?;
    let witness_len = u64::from_le_bytes(buffer[body_len..].try_into().unwrap());
    let witness_len = usize::try_from(witness_len).ok().filter(|len| *len <= body_len)?;
    let value = read(&buffer[witness_len..body_len]);
    buffer.truncate(witness_len);
    Some((value, buffer))
}

#[cfg(test)]
mod tests {
    use super::{join, split};

    #[test]
    fn split_returns_what_was_joined() {
        for (input, witness) in [
            (&b"serialized input"[..], &b"witness blob"[..]),
            (b"serialized input", b""),
            (b"", b"witness blob"),
            (b"", b""),
        ] {
            let buffer = join(input, witness);
            assert_eq!(buffer.len(), input.len() + witness.len() + 8);
            let (seen, rest) = split(buffer, |bytes| bytes.to_vec()).unwrap();
            assert_eq!(seen, input);
            assert_eq!(rest, witness);
        }
    }

    #[test]
    fn the_witness_is_not_copied() {
        let buffer = join(&[7; 100], &[9; 1000]);
        let (start, capacity) = (buffer.as_ptr(), buffer.capacity());
        let (_, witness) = split(buffer, |_| ()).unwrap();
        assert_eq!((witness.as_ptr(), witness.capacity()), (start, capacity));
        assert_eq!(witness, [9; 1000]);
    }

    #[test]
    fn a_buffer_join_did_not_make_is_refused() {
        // shorter than the trailer
        assert!(split(vec![0; 7], |_| ()).is_none());
        // the trailer names one byte more than there is before it
        let mut buffer = join(b"input", b"witness");
        let at = buffer.len() - 8;
        buffer[at..].copy_from_slice(&13u64.to_le_bytes());
        assert!(split(buffer, |_| ()).is_none());
        // a length that only fits 64 bits
        let mut buffer = join(b"input", b"witness");
        buffer[at..].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(split(buffer, |_| ()).is_none());
        // and the largest length that is still inside the buffer
        let mut buffer = join(b"input", b"witness");
        buffer[at..].copy_from_slice(&12u64.to_le_bytes());
        let (seen, rest) = split(buffer, |bytes| bytes.to_vec()).unwrap();
        assert_eq!((seen.as_slice(), rest.as_slice()), (&b""[..], &b"witnessinput"[..]));
    }
}
