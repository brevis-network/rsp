#![cfg_attr(not(test), warn(unused_crate_dependencies))]

/// Client program input data types.
pub mod io;
#[macro_use]
pub mod utils;
pub mod custom;
mod eip2537;
pub mod error;
pub mod executor;
pub mod tracking;

mod into_primitives;
pub use into_primitives::{BlockValidator, FromInput, IntoInput, IntoPrimitives};
