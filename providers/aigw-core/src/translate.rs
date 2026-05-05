//! Translator traits and intermediate types.
//!
//! The translation layer is a pure data-mapping boundary:
//! - **No IO**: translators don't make HTTP calls or touch the network.
//! - **No decisions**: no routing, retrying, or rate limiting.
//! - **No HTTP client dependency**: the output is [`TranslatedRequest`], a
//!   client-agnostic intermediate that the gateway's HTTP layer consumes.
//!
//! # Architecture
//!
//! ```text
//! Client request (OpenAI format)
//!     │
//!     ▼
//! ChatRequest (canonical model)
//!     │
//!     ├── RequestTranslator::translate_request()
//!     ▼
//! TranslatedRequest { url, method, headers, body }
//!     │
//!     ├── HTTP layer sends request
//!     ▼
//! Raw HTTP response
//!     │
//!     ├── ResponseTranslator::translate_response()  (non-streaming)
//!     ├── StreamParser::parse_event()                (streaming)
//!     ▼
//! ChatResponse / Vec<StreamEvent> (canonical model)
//!     │
//!     ▼
//! Client response (OpenAI format)
//! ```

use bytes::Bytes;
use http::{HeaderMap, Method, StatusCode};

use crate::error::{ProviderError, TranslateError};
use crate::model::{ChatRequest, ChatResponse, StreamEvent, ThinkingRequest};

// ─── TranslatedRequest ──────────────────────────────────────────────────────

/// An HTTP-client-agnostic intermediate produced by [`RequestTranslator`].
///
/// Contains everything needed to build an HTTP request, but doesn't reference
/// `reqwest`, `hyper`, or any specific client. The gateway's HTTP layer
/// converts this into the concrete request type.
///
/// The `body` is pre-serialized JSON bytes, giving the translator full control
/// over serialization (e.g., Gemini's camelCase, Anthropic's snake_case).
#[derive(Debug, Clone)]
pub struct TranslatedRequest {
    /// Full URL including path (e.g. `"https://api.anthropic.com/v1/messages"`).
    pub url: String,
    /// HTTP method — almost always `POST`.
    pub method: Method,
    /// Provider-specific headers (auth, version, content-type, etc.).
    pub headers: HeaderMap,
    /// Pre-serialized JSON request body.
    pub body: Bytes,
}

// ─── RequestTranslator ──────────────────────────────────────────────────────

/// Translates a canonical [`ChatRequest`] into a provider-native HTTP request.
///
/// Implementors hold provider configuration (base URL, API version, default
/// headers, model-specific defaults like Anthropic's required `max_tokens`).
///
/// This trait is **stateless** — the same translator instance is reused
/// across requests.
pub trait RequestTranslator: Send + Sync {
    /// Translate a canonical request into the provider's native format.
    ///
    /// The implementation should:
    /// 1. Map canonical fields to provider-native fields
    /// 2. Extract provider-specific parameters from `req.extra`
    /// 3. Apply defaults for required fields (e.g. `max_tokens` for Anthropic)
    /// 4. Serialize the native request body into `TranslatedRequest::body`
    /// 5. Build the full URL and provider-specific headers
    fn translate_request(&self, req: &ChatRequest) -> Result<TranslatedRequest, TranslateError>;

    /// Translate a streaming request.
    ///
    /// Default: delegates to [`translate_request`](Self::translate_request).
    /// Override when the streaming endpoint differs (e.g. Gemini uses
    /// `streamGenerateContent?alt=sse` vs `generateContent`).
    fn translate_stream_request(
        &self,
        req: &ChatRequest,
    ) -> Result<TranslatedRequest, TranslateError> {
        self.translate_request(req)
    }
}

// ─── ResponseTranslator ─────────────────────────────────────────────────────

/// Translates provider-native HTTP responses back into canonical types.
///
/// Implementors hold provider configuration but are otherwise **stateless**.
/// For streaming, use [`stream_parser`](Self::stream_parser) to obtain a
/// per-request stateful [`StreamParser`].
pub trait ResponseTranslator: Send + Sync {
    /// Translate a complete (non-streaming) response body.
    ///
    /// Called when the full response body is available. The implementation
    /// deserializes the provider-native response and maps it to [`ChatResponse`].
    fn translate_response(
        &self,
        status: StatusCode,
        body: &[u8],
    ) -> Result<ChatResponse, TranslateError>;

    /// Create a new stateful stream parser for a single streaming request.
    ///
    /// Each streaming request gets its own parser instance, which maintains
    /// state across SSE events (e.g. accumulated tool call index, message ID,
    /// usage counters).
    fn stream_parser(&self) -> Box<dyn StreamParser>;

    /// Translate an error response into a structured [`ProviderError`].
    ///
    /// The implementation parses the provider's error body format and maps
    /// HTTP status codes to semantic error variants.
    fn translate_error(
        &self,
        status: StatusCode,
        headers: &HeaderMap,
        body: &[u8],
    ) -> ProviderError;
}

// ─── StreamParser ───────────────────────────────────────────────────────────

/// Stateful parser that translates provider-native SSE events into canonical
/// [`StreamEvent`]s.
///
/// Created per-request via [`ResponseTranslator::stream_parser`]. The parser
/// maintains state across events — for example:
///
/// - **Anthropic**: tracks `tool_call_index` (incremented on each
///   `content_block_start(tool_use)`), captures `id` and `model` from
///   `message_start`.
/// - **Gemini**: diffs consecutive snapshot responses to extract incremental
///   text deltas.
/// - **OpenAI**: mostly stateless (near-passthrough), but captures `id`/`model`
///   from the first chunk.
///
/// # Why `parse_event` instead of implementing `Stream`?
///
/// A single provider SSE event can map to zero, one, or multiple canonical
/// events. For example, Anthropic's `content_block_start(tool_use)` produces
/// a `ToolCallStart`, while `ping` produces nothing. The `parse_event` +
/// `Vec<StreamEvent>` model handles this one-to-many mapping naturally.
pub trait StreamParser: Send {
    /// Process a single SSE event and produce zero or more canonical events.
    ///
    /// # Parameters
    /// - `event_type`: The SSE `event:` field (e.g. `"message_start"` for
    ///   Anthropic, `""` for OpenAI's unnamed events).
    /// - `data`: The SSE `data:` payload (JSON string, or `"[DONE]"`).
    ///
    /// # Returns
    /// Zero or more canonical [`StreamEvent`]s. An empty vec means the SSE
    /// event was consumed but produced no output (e.g. `ping`).
    fn parse_event(
        &mut self,
        event_type: &str,
        data: &str,
    ) -> Result<Vec<StreamEvent>, TranslateError>;

    /// Signal that the stream has ended.
    ///
    /// Returns any buffered final events — typically a [`StreamEvent::Usage`]
    /// summary if the provider reports usage incrementally, or
    /// [`StreamEvent::Done`] if not already emitted.
    fn finish(&mut self) -> Result<Vec<StreamEvent>, TranslateError>;
}

// ─── ProviderTranslator ─────────────────────────────────────────────────────

/// Convenience façade combining request and response translators.
///
/// This is the type the gateway holds per configured provider.
pub struct ProviderTranslator<Req, Res> {
    pub request: Req,
    pub response: Res,
}

impl<Req, Res> ProviderTranslator<Req, Res>
where
    Req: RequestTranslator,
    Res: ResponseTranslator,
{
    pub fn new(request: Req, response: Res) -> Self {
        Self { request, response }
    }
}

// ─── ThinkingProjector ──────────────────────────────────────────────────────

/// Projects a canonical [`ThinkingRequest`] onto provider-native mutable state.
///
/// Each provider crate exposes its own `Target` type — a small struct holding
/// the subset of the native request body that thinking touches (e.g. for
/// Anthropic: `thinking`, `max_tokens`, `output_config`). The provider's
/// [`RequestTranslator`] constructs a `Target`, hands it to the projector,
/// then unpacks the mutated `Target` into the native request being built.
///
/// # Why a separate trait
///
/// The mapping rules are non-trivial (capability-dependent dispatch,
/// budget-vs-level conversion, max_tokens headroom for Anthropic, etc.) and
/// users may want to override them — different deployments tune the
/// `level → budget` table or the "is this an adaptive model" predicate. A
/// separate trait makes that override pluggable without touching the
/// translator.
///
/// # Object safety
///
/// The trait is **object-safe**: `Target` is a generic on the trait itself
/// (concrete at every `dyn` site), and `apply` has no method-level generics
/// and no `Self` in return position. Translators hold
/// `Box<dyn ThinkingProjector<TheirTarget>>` so users can swap projectors at
/// runtime.
pub trait ThinkingProjector<Target>: Send + Sync {
    /// Mutate `target` according to the canonical request.
    ///
    /// When `req` is `None`, projectors typically leave `target` untouched
    /// (the translator falls back to whatever it has — usually legacy
    /// `extra["thinking"]` passthrough).
    ///
    /// `model` is the request's model identifier; projectors inspect it to
    /// decide capability-dependent details (Claude 4.6 adaptive vs. legacy,
    /// Gemini 3 levels vs. Gemini 2.5 budget, etc.).
    fn apply(&self, model: &str, req: Option<&ThinkingRequest>, target: &mut Target);
}

#[cfg(test)]
mod thinking_projector_tests {
    use super::*;

    /// Compile-time assertion that the trait is object-safe.
    #[allow(dead_code)]
    fn assert_object_safe(_: Box<dyn ThinkingProjector<()>>) {}

    struct NoopProjector;
    impl ThinkingProjector<u32> for NoopProjector {
        fn apply(&self, _model: &str, _req: Option<&ThinkingRequest>, target: &mut u32) {
            *target = 42;
        }
    }

    #[test]
    fn dyn_dispatch_works() {
        let p: Box<dyn ThinkingProjector<u32>> = Box::new(NoopProjector);
        let mut target = 0u32;
        p.apply("any-model", None, &mut target);
        assert_eq!(target, 42);
    }
}
