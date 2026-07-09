//! AI Gateway — a loopback sidecar that serves the Anthropic Messages API
//! (`POST /v1/messages`) on top of an OpenAI-compatible upstream.
//!
//! It lets a bare OpenAI/local-model key back a Claude Code session offline:
//! Claude Code speaks Anthropic wire, the gateway translates to the configured
//! upstream, and translates the response (including streaming SSE and tool
//! calls) back to Anthropic wire.
//!
//! - [`config`] — the TOML configuration ([`GatewayConfig`]).
//! - [`bridge`] — the Anthropic ↔ OpenAI-compatible request/response glue
//!   ([`Upstream`]).
//! - [`server`] — the axum router and server ([`router`], [`serve`],
//!   [`AppState`]).

#![forbid(unsafe_code)]

pub mod bridge;
pub mod config;
pub mod server;

pub use aigw_openai_compat::Quirks;
pub use bridge::Upstream;
pub use config::{GatewayConfig, UpstreamConfig, Wire};
pub use server::{AppState, router, serve};
