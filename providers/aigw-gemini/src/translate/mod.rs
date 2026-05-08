//! Translation layer: canonical ↔ Gemini `generateContent` wire types.
//!
//! Translations:
//! - **Request**: canonical [`ChatRequest`] → [`GenerateContentRequest`].
//!   System messages → `system_instruction`; consecutive same-role turns
//!   merged (Gemini rejects them); tool definitions → `function_declarations`;
//!   tool results → `function_response` parts on a `user`-role turn.
//! - **Response**: [`GenerateContentResponse`] → canonical [`ChatResponse`].
//!   Text parts joined; `function_call` parts → tool calls; `thought=true`
//!   parts → canonical thinking content with [`ThinkingSource::Gemini`].
//! - **Stream**: each Gemini SSE event is a complete response object;
//!   the parser diffs successive snapshots to emit incremental canonical
//!   events.
//!
//! [`ChatRequest`]: aigw_core::model::ChatRequest
//! [`ChatResponse`]: aigw_core::model::ChatResponse
//! [`GenerateContentRequest`]: crate::types::GenerateContentRequest
//! [`GenerateContentResponse`]: crate::types::GenerateContentResponse
//! [`ThinkingSource::Gemini`]: aigw_core::model::ThinkingSource::Gemini

pub mod native_bridge;
pub mod request;
pub mod response;
pub mod stream;
pub mod thinking;

pub use native_bridge::{
    SseContext as NativeSseContext, chat_response_to_gemini, gemini_request_to_canonical,
    stream_event_to_gemini_sse,
};
pub use request::{
    GeminiRequestTranslator, build_generate_content_request,
    build_generate_content_request_with_projector,
};
pub use response::GeminiResponseTranslator;
pub use stream::GeminiStreamParser;
pub use thinking::{GeminiThinkingProjector, GeminiThinkingTarget};
