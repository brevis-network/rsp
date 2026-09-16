//! Block execution as the guest runs it.
//!
//! [`executor::ClientExecutor`] takes an [`io::ClientExecutorInput`] -- a block plus the witness
//! for the state it touches -- executes it, and returns an [`io::CommittedHeader`]: the derived
//! header together with a digest of the wire fields that change execution but leave no trace in
//! it. Nothing here trusts the input; see [`io::WitnessInput::verified_views`] for what anchors
//! it, and [`error::ClientError`] for what a witness that does not hold up comes back as.

#![cfg_attr(not(test), warn(unused_crate_dependencies))]

/// Client program input data types.
pub mod io;
#[macro_use]
pub mod utils;
pub mod custom;
pub mod error;
pub mod executor;
pub mod tracking;

mod into_primitives;
pub use into_primitives::{BlockValidator, FromInput, IntoInput, IntoPrimitives};
