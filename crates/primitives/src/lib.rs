//! Types shared by the host and the guest.
//!
//! [`genesis::Genesis`] is the one that matters for soundness: it is a wire field that becomes the
//! entire `ChainSpec` -- chain id, every hardfork activation, blob params -- and it leaves no
//! trace in the committed header, which is why the guest commits a digest over it.

#![cfg_attr(not(test), warn(unused_crate_dependencies))]

pub mod account_proof;
pub mod chain_spec;
pub mod error;
pub mod genesis;
