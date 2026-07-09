//! End-to-end acceptance test for the loopback gateway.
//!
//! Spins up a mock OpenAI-compatible upstream and the real gateway, then drives
//! `POST /v1/messages` (streaming-with-tools and unary) and asserts the gateway
//! emits correct Anthropic wire output — proving the full
//! `Anthropic-in → canonical → OpenAI-compat → OpenAI-SSE → Anthropic-SSE` loop.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use aigateway::{AppState, Upstream, UpstreamConfig, Wire, serve};
use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use secrecy::SecretString;
use serde_json::Value;
use tokio::net::TcpListener;

/// What the mock upstream recorded about the request the gateway sent it.
#[derive(Default)]
struct Recorded {
    model: Option<String>,
    authorization: Option<String>,
}

type Recorder = Arc<Mutex<Recorded>>;

const STREAM_SSE: &str = concat!(
    "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"location\\\":\\\"SF\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"gpt-4.1\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":7,\"total_tokens\":17}}\n\n",
    "data: [DONE]\n\n",
);

const UNARY_JSON: &str = r#"{"id":"chatcmpl-mock","object":"chat.completion","created":0,"model":"gpt-4.1","choices":[{"index":0,"message":{"role":"assistant","content":"Hi there!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":4,"total_tokens":12}}"#;

async fn chat_completions(
    State(recorder): State<Recorder>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Response {
    let v: Value = serde_json::from_slice(&body).expect("upstream body is JSON");
    {
        let mut rec = recorder.lock().unwrap();
        rec.model = v["model"].as_str().map(str::to_owned);
        rec.authorization = headers
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .map(str::to_owned);
    }

    if v["stream"].as_bool().unwrap_or(false) {
        ([(header::CONTENT_TYPE, "text/event-stream")], STREAM_SSE).into_response()
    } else {
        ([(header::CONTENT_TYPE, "application/json")], UNARY_JSON).into_response()
    }
}

/// Bind a mock OpenAI-compatible upstream on loopback. Returns its base URL
/// (with the `/v1` prefix the gateway appends `/chat/completions` to) and the
/// request recorder.
async fn spawn_mock_upstream() -> (String, Recorder) {
    let recorder: Recorder = Arc::new(Mutex::new(Recorded::default()));
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(recorder.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}/v1"), recorder)
}

/// Bind the real gateway on loopback, pointing at `upstream_base`. Returns the
/// gateway's base URL.
async fn spawn_gateway(upstream_base: String) -> String {
    let config = UpstreamConfig {
        base_url: upstream_base,
        api_key: SecretString::from("sk-upstream"),
        wire: Wire::OpenaiChat,
        timeout_seconds: None,
        default_headers: BTreeMap::new(),
        models: BTreeMap::from([("claude-sonnet-4-20250514".to_owned(), "gpt-4.1".to_owned())]),
        default_model: None,
    };
    let upstream = Upstream::new(config).unwrap();
    let state = AppState::new(upstream);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve(listener, state).await.unwrap();
    });
    format!("http://{addr}")
}

/// Parse an SSE body into `(event, data-json)` frames.
fn parse_sse(text: &str) -> Vec<(String, Value)> {
    let mut frames = Vec::new();
    for block in text.split("\n\n") {
        let mut event = None;
        let mut data = None;
        for line in block.lines() {
            if let Some(rest) = line.strip_prefix("event: ") {
                event = Some(rest.to_owned());
            } else if let Some(rest) = line.strip_prefix("data: ") {
                data = Some(rest.to_owned());
            }
        }
        if let (Some(event), Some(data)) = (event, data) {
            frames.push((
                event,
                serde_json::from_str(&data).expect("frame data is JSON"),
            ));
        }
    }
    frames
}

fn client() -> reqwest::Client {
    // Bypass any ambient system proxy so we hit loopback directly.
    reqwest::Client::builder().no_proxy().build().unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_with_tools_produces_anthropic_sse() {
    let (upstream_base, recorder) = spawn_mock_upstream().await;
    let gateway = spawn_gateway(upstream_base).await;

    let body = serde_json::json!({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 256,
        "stream": true,
        "messages": [{ "role": "user", "content": "What's the weather in SF?" }],
        "tools": [{
            "name": "get_weather",
            "description": "Get the weather",
            "input_schema": { "type": "object", "properties": { "location": { "type": "string" } } }
        }]
    });

    let resp = client()
        .post(format!("{gateway}/v1/messages"))
        // A placeholder inbound token — the gateway must ignore it.
        .header("x-api-key", "placeholder-token")
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok()),
        Some("text/event-stream")
    );

    let text = resp.text().await.unwrap();
    let frames = parse_sse(&text);
    let names: Vec<&str> = frames.iter().map(|(e, _)| e.as_str()).collect();
    assert_eq!(
        names,
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ],
        "unexpected frame sequence: {names:?}"
    );

    // Text block.
    assert_eq!(frames[1].1["content_block"]["type"], "text");
    assert_eq!(frames[2].1["delta"]["text"], "Hello");
    assert_eq!(frames[3].1["delta"]["text"], " world");

    // Tool-use block: distinct block index, id/name, streamed args.
    assert_eq!(frames[5].1["content_block"]["type"], "tool_use");
    assert_eq!(frames[5].1["content_block"]["id"], "call_1");
    assert_eq!(frames[5].1["content_block"]["name"], "get_weather");
    assert_eq!(frames[5].1["index"], 1);
    assert_eq!(frames[6].1["delta"]["type"], "input_json_delta");
    assert_eq!(frames[6].1["delta"]["partial_json"], r#"{"location":"SF"}"#);

    // Terminal frames.
    assert_eq!(frames[8].1["delta"]["stop_reason"], "tool_use");
    assert_eq!(frames[8].1["usage"]["output_tokens"], 7);

    // Upstream saw the mapped model and the injected upstream key (not the
    // inbound placeholder).
    let rec = recorder.lock().unwrap();
    assert_eq!(rec.model.as_deref(), Some("gpt-4.1"));
    assert_eq!(rec.authorization.as_deref(), Some("Bearer sk-upstream"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unary_produces_anthropic_message() {
    let (upstream_base, _recorder) = spawn_mock_upstream().await;
    let gateway = spawn_gateway(upstream_base).await;

    let body = serde_json::json!({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 256,
        "messages": [{ "role": "user", "content": "hi" }]
    });

    let resp = client()
        .post(format!("{gateway}/v1/messages"))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], "Hi there!");
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(v["usage"]["input_tokens"], 8);
    assert_eq!(v["usage"]["output_tokens"], 4);
    assert_eq!(v["model"], "gpt-4.1");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn health_check_ok() {
    let (upstream_base, _recorder) = spawn_mock_upstream().await;
    let gateway = spawn_gateway(upstream_base).await;

    let resp = client()
        .get(format!("{gateway}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["status"], "ok");
}
