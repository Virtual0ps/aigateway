//! The request/response glue tying the inbound Anthropic protocol to an
//! outbound OpenAI-compatible upstream.
//!
//! Flow per request:
//! `Anthropic /v1/messages` → [`messages_request_to_canonical`] → canonical
//! `ChatRequest` → [`OpenAICompatRequestTranslator`] → `reqwest` → upstream →
//! (unary) [`chat_response_to_messages`] or (stream) OpenAI SSE →
//! `OpenAIStreamParser` → [`stream_event_to_anthropic_sse`] → Anthropic SSE.

use std::time::Duration;

use aigw_anthropic::translate::{
    NativeSseContext, chat_response_to_messages, messages_request_to_canonical,
    stream_event_to_anthropic_sse,
};
use aigw_anthropic::types::MessagesRequest;
use aigw_core::error::ProviderError;
use aigw_core::model::{ChatRequest, Message, StreamEvent};
use aigw_core::translate::{
    RequestTranslator, ResponseTranslator, StreamParser, TranslatedRequest,
};
use aigw_openai::translate::{
    OpenAIResponseTranslator, ResponsesRequestTranslator, ResponsesResponseTranslator,
};
use aigw_openai::{
    DEFAULT_TIMEOUT_SECONDS, HttpTransportConfig, OpenAIAuthConfig, OpenAITransport,
    OpenAITransportConfig,
};
use aigw_openai_compat::translate::OpenAICompatRequestTranslator;
use aigw_openai_compat::{OpenAICompatConfig, OpenAICompatProvider};
use axum::Json;
use axum::body::Body;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use http::StatusCode;
use serde_json::json;

use crate::config::{UpstreamConfig, Wire};

/// Idle gap after which a keepalive `ping` SSE event is emitted during
/// streaming. Well under typical client/proxy idle timeouts.
const PING_INTERVAL: Duration = Duration::from_secs(15);

/// A configured upstream: the translators, HTTP client, and model mapping
/// needed to serve one inbound Anthropic request against one OpenAI-compatible
/// backend. Cheap to share behind an `Arc`.
pub struct Upstream {
    client: reqwest::Client,
    request: Box<dyn RequestTranslator>,
    response: Box<dyn ResponseTranslator>,
    config: UpstreamConfig,
}

impl Upstream {
    /// Build an [`Upstream`] from validated [`UpstreamConfig`].
    ///
    /// # Errors
    ///
    /// - The upstream base URL or API key is invalid.
    /// - The configured proxy URL is invalid, or the HTTP client can't build.
    pub fn new(config: UpstreamConfig) -> anyhow::Result<Self> {
        let timeout_seconds = config.timeout_seconds.unwrap_or(DEFAULT_TIMEOUT_SECONDS);
        let http = HttpTransportConfig {
            base_url: config.base_url.clone(),
            timeout_seconds,
            default_headers: config.default_headers.clone(),
        };
        let auth = OpenAIAuthConfig {
            api_key: config.api_key.clone(),
            organization: None,
            project: None,
        };

        let (request, response): (Box<dyn RequestTranslator>, Box<dyn ResponseTranslator>) =
            match config.wire {
                Wire::OpenaiChat => {
                    let provider = OpenAICompatProvider::new(OpenAICompatConfig {
                        name: "upstream".to_owned(),
                        http,
                        auth,
                        quirks: config.quirks.clone(),
                    })
                    .map_err(|e| anyhow::anyhow!("invalid upstream config: {e}"))?;
                    let request = OpenAICompatRequestTranslator::new(&provider)
                        .map_err(|e| anyhow::anyhow!("building upstream translator: {e}"))?;
                    // Compat upstreams return OpenAI-shaped responses, so the
                    // plain OpenAI response translator handles unary + SSE.
                    (Box::new(request), Box::new(OpenAIResponseTranslator))
                }
                Wire::OpenaiResponses => {
                    let transport = OpenAITransport::new(OpenAITransportConfig { http, auth })
                        .map_err(|e| anyhow::anyhow!("invalid upstream config: {e}"))?;
                    (
                        Box::new(ResponsesRequestTranslator::new(transport)),
                        Box::new(ResponsesResponseTranslator),
                    )
                }
            };

        let client = build_client(&config, timeout_seconds)?;

        Ok(Self {
            client,
            request,
            response,
            config,
        })
    }

    /// Handle one inbound Anthropic request end-to-end, producing the HTTP
    /// response to return to the client (unary JSON or streaming SSE).
    pub async fn handle(&self, req: MessagesRequest) -> Response {
        let streaming = req.stream.unwrap_or(false);

        let mut canonical = match messages_request_to_canonical(req) {
            Ok(c) => c,
            Err(e) => {
                return anthropic_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    &e.to_string(),
                );
            }
        };
        // Echo the client's requested model back in the response, not the
        // upstream's — Claude Code keys behavior on the model name.
        let requested_model = canonical.model.clone();
        canonical.model = self.config.resolve_model(&canonical.model);
        // Let the chosen translate method own the `stream` flag.
        canonical.stream = None;

        if streaming {
            self.handle_streaming(&canonical, &requested_model).await
        } else {
            self.handle_unary(&canonical, &requested_model).await
        }
    }

    /// The inbound Anthropic model names the gateway advertises on
    /// `GET /v1/models` — the configured `[upstream.models]` keys.
    pub fn advertised_models(&self) -> Vec<String> {
        self.config.models.keys().cloned().collect()
    }

    async fn handle_unary(&self, canonical: &ChatRequest, requested_model: &str) -> Response {
        let translated = match self.request.translate_request(canonical) {
            Ok(t) => t,
            Err(e) => {
                return anthropic_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    &e.to_string(),
                );
            }
        };

        let resp = match self.send(translated).await {
            Ok(r) => r,
            Err(e) => return upstream_unreachable(&e),
        };
        let status = resp.status();
        let headers = resp.headers().clone();
        let body = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => return upstream_unreachable(&e),
        };

        if !status.is_success() {
            return provider_error(self.response.translate_error(status, &headers, &body));
        }

        let chat = match self.response.translate_response(status, &body) {
            Ok(c) => c,
            Err(e) => {
                return anthropic_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    &format!("failed to parse upstream response: {e}"),
                );
            }
        };
        match chat_response_to_messages(chat) {
            Ok(mut messages) => {
                messages.model = requested_model.to_owned();
                (StatusCode::OK, Json(messages)).into_response()
            }
            Err(e) => anthropic_error(StatusCode::BAD_GATEWAY, "api_error", &e.to_string()),
        }
    }

    async fn handle_streaming(&self, canonical: &ChatRequest, requested_model: &str) -> Response {
        let translated = match self.request.translate_stream_request(canonical) {
            Ok(t) => t,
            Err(e) => {
                return anthropic_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    &e.to_string(),
                );
            }
        };

        let resp = match self.send(translated).await {
            Ok(r) => r,
            Err(e) => return upstream_unreachable(&e),
        };
        let status = resp.status();
        if !status.is_success() {
            let headers = resp.headers().clone();
            let body = resp.bytes().await.unwrap_or_default();
            return provider_error(self.response.translate_error(status, &headers, &body));
        }

        let parser = self.response.stream_parser();
        let ctx = NativeSseContext::with_pinned_model(requested_model);
        let byte_stream = resp.bytes_stream();
        let sse = anthropic_sse_stream(byte_stream, parser, ctx);

        Response::builder()
            .status(StatusCode::OK)
            .header(http::header::CONTENT_TYPE, "text/event-stream")
            .header(http::header::CACHE_CONTROL, "no-cache")
            .body(Body::from_stream(sse))
            .unwrap_or_else(|e| {
                anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    &format!("failed to build streaming response: {e}"),
                )
            })
    }

    async fn send(&self, translated: TranslatedRequest) -> reqwest::Result<reqwest::Response> {
        self.client
            .request(translated.method, translated.url)
            .headers(translated.headers)
            .body(translated.body)
            .send()
            .await
    }
}

/// Build the outbound HTTP client.
///
/// Uses an idle read timeout (reset on each received chunk) rather than a total
/// request timeout, so long-lived streams are never cut off. Proxy handling:
/// an explicit `upstream.proxy` if configured, otherwise bypass the ambient
/// system proxy — a loopback sidecar talks directly to its configured upstream
/// (often a local model), and honoring the system proxy would silently
/// MITM/break localhost upstreams.
fn build_client(config: &UpstreamConfig, timeout_seconds: u64) -> anyhow::Result<reqwest::Client> {
    let builder = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(timeout_seconds));
    let builder = match &config.proxy {
        Some(url) => builder.proxy(
            reqwest::Proxy::all(url)
                .map_err(|e| anyhow::anyhow!("invalid upstream proxy '{url}': {e}"))?,
        ),
        None => builder.no_proxy(),
    };
    builder
        .build()
        .map_err(|e| anyhow::anyhow!("building HTTP client: {e}"))
}

/// Transform an upstream OpenAI SSE byte stream into an Anthropic SSE byte
/// stream, driving the canonical [`StreamParser`] and the block-lifecycle
/// [`NativeSseContext`] statefully.
fn anthropic_sse_stream(
    byte_stream: impl futures_util::Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
    mut parser: Box<dyn StreamParser>,
    mut ctx: NativeSseContext,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static {
    async_stream::stream! {
        // `eventsource()` needs an `Unpin` stream; reqwest's `bytes_stream()`
        // is not, so pin it on the heap.
        let mut events = Box::pin(byte_stream).eventsource();
        // Keepalive: emit an Anthropic `ping` after each idle gap so a slow or
        // local upstream doesn't idle-timeout the client before the first
        // token. The interval resets on every upstream event, so pings only
        // fire during genuine silence.
        let mut keepalive = tokio::time::interval(PING_INTERVAL);
        keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        keepalive.tick().await; // consume the immediate first tick
        loop {
            let event = tokio::select! {
                biased;
                item = events.next() => match item {
                    None => break,
                    Some(Ok(event)) => event,
                    // Upstream transport dropped mid-stream — surface it and stop.
                    Some(Err(e)) => {
                        yield Ok(error_sse_bytes(&e.to_string()));
                        return;
                    }
                },
                _ = keepalive.tick() => {
                    yield Ok(Bytes::from_static(b"event: ping\ndata: {\"type\":\"ping\"}\n\n"));
                    continue;
                }
            };
            match parser.parse_event("", &event.data) {
                Ok(canonical_events) => {
                    for cev in &canonical_events {
                        for frame in stream_event_to_anthropic_sse(cev, &mut ctx) {
                            yield Ok(Bytes::from(frame.to_sse_bytes()));
                        }
                    }
                }
                Err(e) => {
                    yield Ok(error_sse_bytes(&e.to_string()));
                    return;
                }
            }
            keepalive.reset(); // pings only after genuine idle
        }
        // Flush any events the parser buffered for end-of-stream.
        if let Ok(tail) = parser.finish() {
            for cev in &tail {
                for frame in stream_event_to_anthropic_sse(cev, &mut ctx) {
                    yield Ok(Bytes::from(frame.to_sse_bytes()));
                }
            }
        }
        // Safety net: guarantee a terminal `message_delta` + `message_stop`
        // even if the upstream closed without a `[DONE]` sentinel (some
        // providers just close the socket). Idempotent — a no-op if the stream
        // already emitted `message_stop`.
        for frame in stream_event_to_anthropic_sse(&StreamEvent::Done, &mut ctx) {
            yield Ok(Bytes::from(frame.to_sse_bytes()));
        }
    }
}

fn error_sse_bytes(message: &str) -> Bytes {
    let data = json!({
        "type": "error",
        "error": { "type": "api_error", "message": message },
    });
    Bytes::from(format!("event: error\ndata: {data}\n\n"))
}

/// Heuristic input-token estimate for `POST /v1/messages/count_tokens`.
///
/// OpenAI-compatible upstreams have no token-count endpoint, so this is a rough
/// `~chars/4` estimate over message text, tool-call arguments, and tool schemas.
/// It is **advisory** — Claude Code uses it for the context indicator and
/// degrades gracefully when it's imprecise.
pub(crate) fn estimate_input_tokens(req: &MessagesRequest) -> u64 {
    let Ok(canonical) = messages_request_to_canonical(req.clone()) else {
        return 0;
    };
    let mut chars: usize = canonical.messages.iter().map(message_text_len).sum();
    for tool in canonical.tools.iter().flatten() {
        chars += tool.function.name.len();
        chars += tool.function.description.as_deref().map_or(0, str::len);
        if let Some(params) = &tool.function.parameters {
            chars += params.to_string().len();
        }
    }
    // ~4 chars per token, rounded up.
    chars.div_ceil(4) as u64
}

fn message_text_len(msg: &Message) -> usize {
    use aigw_core::model::{ContentPart, MessageContent, TypedContentPart};
    let mut len = 0;
    match &msg.content {
        Some(MessageContent::Text(s)) => len += s.len(),
        Some(MessageContent::Parts(parts)) => {
            for part in parts {
                if let ContentPart::Known(TypedContentPart::Text { text, .. }) = part {
                    len += text.len();
                }
            }
        }
        None => {}
    }
    for tc in msg.tool_calls.iter().flatten() {
        len += tc.function.name.len() + tc.function.arguments.len();
    }
    len
}

/// Build an Anthropic-shaped error response body with the given status.
pub(crate) fn anthropic_error(status: StatusCode, err_type: &str, message: &str) -> Response {
    let body = json!({
        "type": "error",
        "error": { "type": err_type, "message": message },
    });
    (status, Json(body)).into_response()
}

fn upstream_unreachable(err: &reqwest::Error) -> Response {
    tracing::warn!(error = %err, "upstream request failed");
    anthropic_error(
        StatusCode::BAD_GATEWAY,
        "api_error",
        &format!("upstream request failed: {err}"),
    )
}

/// Map a canonical [`ProviderError`] onto an Anthropic-shaped HTTP error.
fn provider_error(err: ProviderError) -> Response {
    tracing::warn!(error = %err, "upstream returned an error");
    let (status, err_type) = match &err {
        ProviderError::RateLimited { .. } => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
        ProviderError::AuthenticationFailed { .. } => {
            (StatusCode::UNAUTHORIZED, "authentication_error")
        }
        ProviderError::PermissionDenied { .. } => (StatusCode::FORBIDDEN, "permission_error"),
        ProviderError::ModelNotFound { .. } => (StatusCode::NOT_FOUND, "not_found_error"),
        ProviderError::ContextLengthExceeded { .. } | ProviderError::InvalidRequest { .. } => {
            (StatusCode::BAD_REQUEST, "invalid_request_error")
        }
        ProviderError::Overloaded { .. } => (
            StatusCode::from_u16(529).unwrap_or(StatusCode::SERVICE_UNAVAILABLE),
            "overloaded_error",
        ),
        ProviderError::ServerError { status, .. } => (
            StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
            "api_error",
        ),
        ProviderError::Unknown { status, .. } => (
            StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
            "api_error",
        ),
    };
    anthropic_error(status, err_type, &err.to_string())
}
