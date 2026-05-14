//! Canonical streaming event types.
//!
//! These are the intermediate events produced by [`StreamParser`](crate::translate::StreamParser)
//! implementations. The gateway assembles them into OpenAI-format `ChatCompletionChunk`
//! objects for the client.
//!
//! The design is deliberately more granular than any single provider's event model.
//! For example, Anthropic's `content_block_start(tool_use)` + `content_block_delta(input_json)`
//! maps to `ToolCallStart` + `ToolCallDelta`, while OpenAI's single delta chunk with
//! `tool_calls[].function.name` and `tool_calls[].function.arguments` maps to the same
//! pair.

use super::response::{FinishReason, Usage};
use super::thinking::ThinkingSource;

/// A canonical streaming event.
///
/// Stream parsers produce a sequence of these events. The gateway consumes
/// them to build OpenAI-format `ChatCompletionChunk` responses.
///
/// Typical event order:
/// ```text
/// ResponseMeta → ContentDelta* → (ToolCallStart → ToolCallDelta*)* → Finish → Usage → Done
/// ```
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// First event — establishes the response identity.
    /// Captured from OpenAI's first chunk, Anthropic's `message_start`,
    /// or generated for Gemini.
    ResponseMeta {
        /// Response ID (e.g. `"chatcmpl-xxx"`, `"msg_xxx"`).
        id: String,
        /// Model identifier.
        model: String,
    },

    /// Incremental text content.
    ContentDelta(String),

    /// A new reasoning / thinking block begins.
    ///
    /// `index` lets multi-block reasoning (rare but Anthropic-supported) be
    /// tracked separately when assembling [`TypedContentPart::Thinking`]
    /// parts on the consumer side.
    ///
    /// [`TypedContentPart::Thinking`]: crate::model::TypedContentPart::Thinking
    ReasoningStart {
        index: u32,
        source: Option<ThinkingSource>,
    },

    /// Incremental reasoning/thinking summary text (e.g. from reasoning models
    /// like o3, o4-mini via the Responses API `response.reasoning_summary_text.delta`).
    ReasoningDelta(String),

    /// A reasoning block finalizes. Carries the integrity signature.
    ///
    /// Consumers assemble the accumulated `ReasoningDelta`s + this signature
    /// into a [`TypedContentPart::Thinking`].
    ///
    /// [`TypedContentPart::Thinking`]: crate::model::TypedContentPart::Thinking
    ReasoningEnd { index: u32, signature: String },

    /// Opaque reasoning signature.
    ///
    /// Deprecated in favor of [`StreamEvent::ReasoningEnd`], which also
    /// carries the block index. Kept for parsers that haven't migrated yet.
    #[deprecated(
        since = "0.5.0",
        note = "use ReasoningEnd { index, signature } instead"
    )]
    ReasoningSignature(String),

    /// A new tool call begins.
    ///
    /// `index` is the zero-based position in the `tool_calls` array.
    /// For Gemini, the `id` is a generated UUID since Gemini doesn't provide one.
    ToolCallStart {
        index: u32,
        id: String,
        name: String,
    },

    /// Incremental tool call arguments (partial JSON string).
    ToolCallDelta { index: u32, arguments: String },

    /// The model has finished generating.
    Finish(FinishReason),

    /// Token usage statistics (typically arrives at/near end of stream).
    Usage(Usage),

    /// Stream is complete — no more events will follow.
    Done,
}
