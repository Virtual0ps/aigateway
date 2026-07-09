//! Translation layer: canonical ↔ Anthropic Messages API.
//!
//! Unlike the OpenAI translator (near-passthrough), Anthropic translation
//! requires significant restructuring:
//! - System messages are extracted to a top-level field
//! - Tool definitions are unwrapped from the OpenAI `function` wrapper
//! - Tool results are restructured into user messages with content blocks
//! - Streaming events use a different granularity (block-level vs choice-level)

pub mod cache_control;
pub mod native_bridge;
pub mod request;
pub mod response;
pub mod stream;
pub mod thinking;
pub mod tools;

pub use cache_control::{
    CacheControlStrategy, DefaultCacheControlStrategy, MAX_CACHE_BREAKPOINTS,
    NoCacheControlStrategy, ephemeral_marker, ephemeral_marker_with_ttl,
};
pub use native_bridge::{
    AnthropicSseFrame, SseContext as NativeSseContext, chat_response_to_messages,
    messages_request_to_canonical, stream_event_to_anthropic_sse,
};
pub use request::AnthropicRequestTranslator;
pub use response::AnthropicResponseTranslator;
pub use stream::AnthropicStreamParser;
pub use thinking::{AnthropicThinkingProjector, AnthropicThinkingTarget};

use crate::types::StopReason;
use aigw_core::model::FinishReason;

/// Canonical conversion: Anthropic stop reason → canonical finish reason.
///
/// Used by both `response.rs` and `stream.rs` to avoid duplication.
impl From<StopReason> for FinishReason {
    fn from(reason: StopReason) -> Self {
        match reason {
            StopReason::EndTurn | StopReason::StopSequence => FinishReason::Stop,
            StopReason::MaxTokens => FinishReason::Length,
            StopReason::ToolUse => FinishReason::ToolCalls,
            StopReason::Other(s) => FinishReason::Unknown(s),
        }
    }
}

/// Inverse conversion: canonical finish reason → Anthropic stop reason.
///
/// Used by the native bridge (`native_bridge.rs`) when a gateway serves the
/// Anthropic-native wire protocol on top of a non-Anthropic backend.
///
/// `ContentFilter` has no Anthropic stop-reason equivalent and collapses to
/// `EndTurn`; unknown reasons pass through verbatim via `Other`.
impl From<FinishReason> for StopReason {
    fn from(reason: FinishReason) -> Self {
        match reason {
            FinishReason::Stop | FinishReason::ContentFilter => StopReason::EndTurn,
            FinishReason::Length => StopReason::MaxTokens,
            FinishReason::ToolCalls => StopReason::ToolUse,
            FinishReason::Unknown(s) => StopReason::Other(s),
        }
    }
}
