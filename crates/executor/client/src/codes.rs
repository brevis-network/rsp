//! The contract codes of a block on the wire, laid out so that the guest runs them where they
//! arrive.
//!
//! evm2 runs a code from a buffer in which 33 zero bytes follow it: a `PUSH32` that is the last
//! byte of a code reads that far, and a `STOP` ends it. Given the code alone it makes that buffer
//! by copying the code. Here the zeros travel with the code:
//!
//! ```text
//! <length: u64, little-endian> <code> <33 to 40 zero bytes, to a multiple of 8>
//! ```
//!
//! so the buffer evm2 runs a code from is the slice of the input the code is in. The section
//! starts at a multiple of 8 of the input ([`crate::stdin`]) and every entry ends at one, which
//! puts every code at a multiple of 8 as well.
//!
//! The zeros are part of the input and are checked, not assumed. What a code's hash covers is the
//! code alone, as before.

use alloy_primitives::Bytes;
use evm2::{
    bytecode::{analyze_legacy, Bytecode, BytecodeDecodeError},
    constants::EIP7702_MAGIC_BYTES,
};

/// The zero bytes evm2 puts after a legacy code.
const PADDING: usize = 33;

/// Why a section is not the codes of a block.
#[derive(Debug, thiserror::Error)]
pub enum CodesError {
    #[error("the codes end inside an entry")]
    Truncated,
    #[error("a code is followed by bytes that are not zero")]
    Padding,
    #[error(transparent)]
    Code(#[from] BytecodeDecodeError),
}

/// The zero bytes after a code of `len` bytes: at least [`PADDING`], to a multiple of 8.
const fn padding(len: usize) -> usize {
    (len + PADDING).next_multiple_of(8) - len
}

/// Lays `codes` out one after the other.
pub fn write<'a>(codes: impl IntoIterator<Item = &'a Bytecode>) -> Vec<u8> {
    let mut section = Vec::new();
    for code in codes {
        let code = code.original_bytes();
        section.extend_from_slice(&(code.len() as u64).to_le_bytes());
        section.extend_from_slice(&code);
        section.resize(section.len() + padding(code.len()), 0);
    }
    section
}

/// The codes of a section that [`write`] laid out, each a slice of it.
pub fn read(section: &Bytes) -> Result<Vec<Bytecode>, CodesError> {
    let mut codes = Vec::new();
    let mut at = 0;
    while at < section.len() {
        let len = section.get(at..at + 8).ok_or(CodesError::Truncated)?;
        let len = usize::try_from(u64::from_le_bytes(len.try_into().unwrap()))
            .map_err(|_| CodesError::Truncated)?;
        // A code longer than the section is none, and its padding is then not computed.
        if len > section.len() {
            return Err(CodesError::Truncated);
        }
        let start = at + 8;
        let (end, next) = (start + len, start + len + padding(len));
        let zeros = section.get(end..next).ok_or(CodesError::Truncated)?;
        if zeros.iter().any(|byte| *byte != 0) {
            return Err(CodesError::Padding);
        }
        codes.push(bytecode(section.slice(start..next), len)?);
        at = next;
    }
    Ok(codes)
}

/// The code in the first `len` bytes of `padded`, the rest of which is zeros, at least
/// [`PADDING`] of them.
fn bytecode(padded: Bytes, len: usize) -> Result<Bytecode, CodesError> {
    let code = padded.slice(..len);
    // What `Bytecode::new_raw_checked` makes of a code by its first bytes. An account without code
    // and a delegation designator are not run, and take no padding.
    if code.is_empty() || code.starts_with(EIP7702_MAGIC_BYTES) {
        return Ok(Bytecode::new_raw_checked(code)?);
    }
    let jump_table = analyze_legacy(&code);
    // SAFETY: At least 33 zero bytes follow the code, which is the padding `Bytecode::new_legacy`
    // gives a code, and the jump table is that of the code.
    Ok(unsafe { Bytecode::new_analyzed(padded, len, jump_table) })
}

#[cfg(test)]
mod tests {
    use alloy_primitives::{address, bytes, keccak256};

    use super::*;

    /// Codes of every length around a multiple of 8, one that ends in a `PUSH32`, one that ends
    /// in zeros of its own, an account without code and a delegation designator.
    fn codes() -> Vec<Bytecode> {
        let mut codes: Vec<Bytecode> = (1..=70usize)
            .map(|len| Bytecode::new_raw((0..len).map(|i| (i * 7 + len) as u8).collect()))
            .collect();
        // PUSH1 4 JUMP INVALID JUMPDEST STOP, then a PUSH32 cut short at the end of the code
        codes.push(Bytecode::new_raw(bytes!("600456fe5b007f0102")));
        codes.push(Bytecode::new_raw([&[0x5b][..], &[0; 40]].concat().into()));
        codes.push(Bytecode::new_raw(Bytes::new()));
        codes.push(Bytecode::new_eip7702(address!("00000000000000000000000000000000cafebabe")));
        codes
    }

    #[test]
    fn what_is_read_is_what_was_written() {
        let sent = codes();
        let section = Bytes::from(write(&sent));
        let received = read(&section).unwrap();
        assert_eq!(received.len(), sent.len());
        for (sent, got) in sent.iter().zip(&received) {
            assert_eq!(got.original_bytes(), sent.original_bytes());
            assert_eq!(got.hash_slow(), keccak256(sent.original_bytes()));
            assert_eq!(got.eip7702_address(), sent.eip7702_address());
            assert_eq!(got.legacy_jump_table().is_some(), sent.legacy_jump_table().is_some());
            for target in 0..sent.original_bytes().len() + 2 {
                assert_eq!(
                    got.legacy_jump_table().map(|table| table.is_valid(target)),
                    sent.legacy_jump_table().map(|table| table.is_valid(target))
                );
            }
        }
    }

    #[test]
    fn a_code_is_a_slice_of_the_section_at_a_multiple_of_8_with_its_zeros_behind_it() {
        let sent = codes();
        let section = Bytes::from(write(&sent));
        assert_eq!(section.len() % 8, 0);
        let (start, end) = (section.as_ptr() as usize, section.as_ptr() as usize + section.len());
        for (sent, got) in sent.iter().zip(read(&section).unwrap()) {
            let code = got.original_bytes();
            if code.is_empty() {
                continue;
            }
            let at = code.as_ptr() as usize;
            assert!(start <= at && at + code.len() <= end, "the code was copied");
            assert_eq!((at - start) % 8, 0);
            if sent.eip7702_address().is_none() {
                // SAFETY: an entry of the section holds its code and at least 33 bytes more.
                let behind =
                    unsafe { core::slice::from_raw_parts(code.as_ptr().add(code.len()), 33) };
                assert_eq!(behind, [0; 33]);
            }
        }
    }

    #[test]
    fn a_section_that_was_not_written_is_refused() {
        let section = write(&[Bytecode::new_raw(bytes!("600456fe5b00"))]);
        assert_eq!(section.len(), 8 + 6 + 34);
        assert!(read(&section.clone().into()).is_ok());
        // cut anywhere
        for len in 1..section.len() {
            assert!(
                matches!(
                    read(&Bytes::copy_from_slice(&section[..len])),
                    Err(CodesError::Truncated)
                ),
                "{len}"
            );
        }
        // any of the zeros changed
        for at in 8 + 6..section.len() {
            let mut section = section.clone();
            section[at] = 1;
            assert!(matches!(read(&section.into()), Err(CodesError::Padding)), "{at}");
        }
        // a length that only fits 64 bits
        for len in [u64::MAX, u64::MAX - 40, 1 << 40] {
            let mut section = section.clone();
            section[..8].copy_from_slice(&len.to_le_bytes());
            assert!(matches!(read(&section.into()), Err(CodesError::Truncated)), "{len}");
        }
        // bytes with the EIP-7702 magic that are no delegation designator
        let designator =
            Bytecode::new_eip7702(address!("00000000000000000000000000000000cafebabe"))
                .original_bytes();
        let short = Bytecode::new_raw(bytes!("600456fe5b00"));
        let mut section = write(&[short]);
        section[8..10].copy_from_slice(&designator[..2]);
        assert!(matches!(read(&section.into()), Err(CodesError::Code(_))));
    }
}
