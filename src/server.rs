//! The axum HTTP server: an inbound Anthropic Messages endpoint bound to
//! loopback, plus a health check.
//!
//! Inbound authentication is intentionally ignored — Claude Code sends a
//! placeholder token, and the real upstream key is injected from config by the
//! [`Upstream`]. The server only ever binds loopback in practice (the CLI
//! defaults to `127.0.0.1`).

use std::sync::Arc;

use aigw_anthropic::types::MessagesRequest;
use axum::Json;
use axum::extract::{DefaultBodyLimit, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Router, body::Bytes};
use http::StatusCode;
use serde_json::{Value, json};

use crate::bridge::{Upstream, anthropic_error, estimate_input_tokens};

/// Inbound request body cap. Claude Code sends large histories (long
/// transcripts, tool results, base64 images), so the default 2 MB `Bytes`
/// limit is far too small; 64 MiB is generous for a trusted loopback sidecar.
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Shared application state handed to every request handler.
#[derive(Clone)]
pub struct AppState {
    /// The single configured upstream.
    pub upstream: Arc<Upstream>,
}

impl AppState {
    /// Wrap an [`Upstream`] into shared state.
    #[must_use]
    pub fn new(upstream: Upstream) -> Self {
        Self {
            upstream: Arc::new(upstream),
        }
    }
}

/// Build the gateway router: `POST /v1/messages` and `GET /health`.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/messages", post(messages_handler))
        .route("/v1/messages/count_tokens", post(count_tokens_handler))
        .route("/v1/models", get(models_handler))
        .route("/health", get(health_handler))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

/// Serve the gateway on an already-bound listener until a shutdown signal.
///
/// # Errors
///
/// Returns any error from the underlying `axum::serve`.
pub async fn serve(listener: tokio::net::TcpListener, state: AppState) -> anyhow::Result<()> {
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// Inbound Anthropic `POST /v1/messages`.
///
/// The raw body is taken as [`Bytes`] (rather than an `axum::Json` extractor)
/// so parse failures produce an Anthropic-shaped error instead of axum's
/// default plain-text rejection.
async fn messages_handler(State(state): State<AppState>, body: Bytes) -> Response {
    match serde_json::from_slice(&body) {
        Ok(req) => state.upstream.handle(req).await,
        Err(e) => anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!("failed to parse request body: {e}"),
        ),
    }
}

/// `POST /v1/messages/count_tokens` — Anthropic's token-count endpoint. Returns
/// a heuristic estimate since OpenAI upstreams have no equivalent (advisory;
/// Claude Code degrades gracefully).
async fn count_tokens_handler(body: Bytes) -> Response {
    match serde_json::from_slice::<MessagesRequest>(&body) {
        Ok(req) => Json(json!({ "input_tokens": estimate_input_tokens(&req) })).into_response(),
        Err(e) => anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!("failed to parse request body: {e}"),
        ),
    }
}

/// `GET /v1/models` — advertise the inbound Anthropic model names the gateway
/// accepts (the configured `[upstream.models]` keys). Claude Code probes this
/// at startup.
async fn models_handler(State(state): State<AppState>) -> Response {
    let data: Vec<Value> = state
        .upstream
        .advertised_models()
        .into_iter()
        .map(|id| {
            json!({
                "type": "model",
                "id": id,
                "display_name": id,
                "created_at": "2025-01-01T00:00:00Z",
            })
        })
        .collect();
    let first = data.first().map(|m| m["id"].clone());
    let last = data.last().map(|m| m["id"].clone());
    Json(json!({
        "data": data,
        "has_more": false,
        "first_id": first,
        "last_id": last,
    }))
    .into_response()
}

/// `GET /health` — a liveness probe for the spawning daemon.
async fn health_handler() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// Complete when the process receives Ctrl-C or (on Unix) SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
}
