//! Protocol translators: canonical Chat hub <-> each wire format.
//!
//! `common` holds shared framing/usage/text primitives; each protocol
//! module owns its `<wire> <-> chat` pair (request, response, stream,
//! parameter loss/fatal analysis).

pub mod anthropic;
pub mod common;
pub mod google;
pub mod openai;
pub mod openai_compatible;

#[cfg(test)]
mod tests;

pub use anthropic::*;
pub use common::*;
pub use openai::*;
pub use openai_compatible::*;
