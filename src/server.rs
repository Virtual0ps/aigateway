//! The axum HTTP server: an inbound Anthropic Messages endpoint bound to
//! loopback, plus a health check.
//!
//! Inbound authentication is intentionally ignored — Claude Code sends a
//! placeholder token, and the real upstream key is injected from config by the
//! [`Upstream`]. The server only ever binds loopback in practice (the CLI
//! defaults to `127.0.0.1`).

use std::sync::Arc;

use axum::Json;
use axum::extract::{DefaultBodyLimit, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Router, body::Bytes};
use http::StatusCode;
use serde_json::json;

use crate::bridge::Upstream;

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
        .route("/health", get(health_handler))
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
    let req = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => {
            let err = json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": format!("failed to parse request body: {e}"),
                },
            });
            return (StatusCode::BAD_REQUEST, Json(err)).into_response();
        }
    };
    state.upstream.handle(req).await
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
