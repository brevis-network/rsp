//! EIP-152's compression function F, its rounds on Pico's `BLAKE2B_ROUNDS` precompile in the
//! guest, and revm's software implementation elsewhere.
//!
//! revm runs F's rounds in software, about 250 instructions each, and F takes a caller-chosen round
//! count at one gas a round: mainnet block 26426155 called it once with 10.5M rounds, 2.6 billion
//! instructions that no proving budget had room for. In the guest the rounds go to the precompile,
//! one trace row each; F's initialization and finalization stay in software, a few dozen
//! instructions.
//!
//! The syscall is issued here rather than through pico-sdk: the guest's pinned SDK predates it,
//! and the call is one `ecall`. Its ABI (Pico's `BLAKE2B_ROUNDS`, `0x00_00_01_35`): `a0` points
//! at 32 dwords, 8-byte aligned, the working vector `v` then the message block `m`; `a1` is a round
//! count in `1..=1000`. It runs rounds `0..a1` of BLAKE2b's schedule on `v` (round `i` reads `m`
//! in the order `SIGMA[i % 10]`) and writes `v` back.

/// BLAKE2b's initialization vector (RFC 7693 §2.6).
#[cfg(target_os = "zkvm")]
const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

/// F with `rounds` rounds on the state `h`, message block `m`, offset counter `t` and final-block
/// flag `f`, as revm's `blake2::algo::compress` computes it.
pub(crate) fn compress(rounds: u32, h: &mut [u64; 8], m: &[u64; 16], t: &[u64; 2], f: bool) {
    #[cfg(target_os = "zkvm")]
    {
        // RFC 7693 §3.2: the working vector, the rounds, then the fold into `h`.
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(h);
        v[8..].copy_from_slice(&IV);
        v[12] ^= t[0];
        v[13] ^= t[1];
        if f {
            v[14] = !v[14];
        }
        run_rounds(&mut v, m, rounds);
        for i in 0..8 {
            h[i] ^= v[i] ^ v[i + 8];
        }
    }

    #[cfg(not(target_os = "zkvm"))]
    revm::precompile::blake2::algo::compress(rounds as usize, h, *m, *t, f);
}

/// Rounds `0..rounds` of the schedule on `v`, as calls of at most [`MAX_ROUNDS_PER_CALL`] rounds:
/// a multiple of ten, so each call starts at round 0 of the schedule, as the syscall does.
#[cfg(target_os = "zkvm")]
fn run_rounds(v: &mut [u64; 16], m: &[u64; 16], rounds: u32) {
    const BLAKE2B_ROUNDS: u32 = 0x00_00_01_35;
    const MAX_ROUNDS_PER_CALL: u32 = 1000;
    let mut buffer = [0u64; 32];
    buffer[..16].copy_from_slice(v);
    buffer[16..].copy_from_slice(m);
    let mut remaining = rounds;
    while remaining > 0 {
        let n = remaining.min(MAX_ROUNDS_PER_CALL);
        // SAFETY: `buffer` is 32 live, 8-byte aligned dwords; the syscall reads all of them and
        // writes the first 16, and the asm block may touch memory, so `buffer` is read back after.
        unsafe {
            core::arch::asm!(
                "ecall",
                in("t0") BLAKE2B_ROUNDS,
                in("a0") buffer.as_mut_ptr(),
                in("a1") u64::from(n),
            );
        }
        remaining -= n;
    }
    v.copy_from_slice(&buffer[..16]);
}

#[cfg(test)]
mod tests {
    /// EIP-152's test vector 5: BLAKE2b-512("abc") after 12 rounds, the input revm's own tests use.
    #[test]
    fn compress_matches_eip_152_vector_5() {
        let words = |hex: &str| -> Vec<u64> {
            alloy_primitives::hex::decode(hex)
                .unwrap()
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let mut h: [u64; 8] = words(
            "48c9bdf267e6096a3ba7ca8485ae67bb2bf894fe72f36e3cf1361d5f3af54fa5\
             d182e6ad7f520e511f6c3e2b8c68059b6bbd41fbabd9831f79217e1319cde05b",
        )
        .try_into()
        .unwrap();
        let mut m = [0u64; 16];
        m[0] = u64::from_le_bytes(*b"abc\0\0\0\0\0");
        super::compress(12, &mut h, &m, &[3, 0], true);
        let out: Vec<u8> = h.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(
            alloy_primitives::hex::encode(out),
            "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
             7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
        );
    }
}
