//! Live OAuth E2E tests for subscription-backed provider plans.
//!
//! These tests intentionally do not run in normal CI. Run them manually with:
//!
//! ```text
//! cargo test --test oauth_e2e -- --ignored --nocapture
//! ```
//!
//! Token environment variables:
//! - Anthropic Claude Code: `AIGW_E2E_ANTHROPIC_OAUTH_TOKEN`
//! - OpenAI Codex: `AIGW_E2E_OPENAI_CODEX_OAUTH_TOKEN`
//! - Google Gemini: `AIGW_E2E_GOOGLE_OAUTH_TOKEN`
//!
//! Optional model/base-url overrides are documented near each test.

use std::collections::BTreeMap;
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

use aigw::anthropic::{
    AuthMode as AnthropicAuthMode, Client as AnthropicClient, ContentBlock, ContentDelta,
    Message as AnthropicMessage, MessageContent as AnthropicMessageContent,
    MessagesRequest as AnthropicMessagesRequest, Role as AnthropicRole,
    StreamEvent as AnthropicStreamEvent, Transport as AnthropicTransport,
    TransportConfig as AnthropicTransportConfig, TypedContentBlock,
};
use aigw::openai::{
    HttpTransportConfig, OpenAIAuthConfig, OpenAIClient, OpenAITransportConfig, RequestOptions,
    ResponsesRequestConfig, build_responses_create_request,
    wire_types::{
        ChatCompletionRequest, ChatContentPart, ChatMessage, ChatMessageContent, ChatMessageRole,
        ResponseCreateRequest, TypedChatContentPart,
    },
};
use aigw_core::model::{
    ChatRequest as CanonicalChatRequest, Message as CanonicalMessage,
    MessageContent as CanonicalMessageContent, Role as CanonicalRole,
};
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use secrecy::SecretString;

const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-5";
const DEFAULT_ANTHROPIC_BETA: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,redact-thinking-2026-02-12,context-management-2025-06-27,prompt-caching-scope-2026-01-05,advanced-tool-use-2025-11-20,effort-2025-11-24,structured-outputs-2025-12-15,fast-mode-2026-02-01,token-efficient-tools-2026-03-28";
const DEFAULT_CLAUDE_CODE_USER_AGENT: &str = "claude-cli/2.1.109 (external, cli)";

const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const DEFAULT_CODEX_MODEL: &str = "gpt-5.1-codex-mini";
const DEFAULT_CODEX_USER_AGENT: &str =
    "codex-tui/0.120.0 (Mac OS 26.0.1; arm64) Apple_Terminal/464";

const DEFAULT_GOOGLE_OPENAI_BASE_URL: &str =
    "https://generativelanguage.googleapis.com/v1beta/openai";
const DEFAULT_GOOGLE_MODEL: &str = "gemini-2.0-flash";

#[tokio::test]
#[ignore = "requires a real Anthropic Claude Code OAuth token"]
async fn anthropic_claude_code_oauth_messages_round_trip() {
    // Required token:
    //   AIGW_E2E_ANTHROPIC_OAUTH_TOKEN
    // Aliases:
    //   AIGW_E2E_CLAUDE_CODE_OAUTH_TOKEN, AIGW_ANTHROPIC_OAUTH_TOKEN,
    //   ANTHROPIC_OAUTH_TOKEN, CLAUDE_CODE_OAUTH_TOKEN
    // Optional:
    //   AIGW_E2E_ANTHROPIC_MODEL, AIGW_E2E_ANTHROPIC_BASE_URL,
    //   AIGW_E2E_ANTHROPIC_BETA, AIGW_E2E_CLAUDE_CODE_USER_AGENT
    let token = oauth_token(
        "AIGW_E2E_ANTHROPIC_OAUTH_TOKEN",
        &[
            "AIGW_E2E_CLAUDE_CODE_OAUTH_TOKEN",
            "AIGW_ANTHROPIC_OAUTH_TOKEN",
            "ANTHROPIC_OAUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
        ],
    );
    let model = env_or("AIGW_E2E_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL);
    let client = anthropic_claude_code_client(token);

    let response = client
        .messages(&anthropic_request(&model, false))
        .await
        .expect("Anthropic Claude Code OAuth messages request should succeed");
    assert_eq!(response.body.r#type, "message");
    assert_eq!(response.body.role, AnthropicRole::Assistant);
    assert!(
        anthropic_blocks_have_text(&response.body.content),
        "Anthropic response should contain at least one non-empty text block: {:?}",
        response.body.content
    );

    let anthropic_stream_request = anthropic_request(&model, true);
    let stream = client
        .messages_stream(&anthropic_stream_request)
        .await
        .expect("Anthropic Claude Code OAuth streaming request should start")
        .body;
    futures_util::pin_mut!(stream);

    let mut saw_text_delta = false;
    let mut saw_stop = false;
    while let Some(event) = stream.next().await {
        match event.expect("Anthropic stream event should decode") {
            AnthropicStreamEvent::ContentBlockDelta {
                delta: ContentDelta::TextDelta { text },
                ..
            } => {
                saw_text_delta |= !text.is_empty();
            }
            AnthropicStreamEvent::MessageStop => {
                saw_stop = true;
                break;
            }
            AnthropicStreamEvent::Error { error } => {
                panic!("Anthropic stream returned an in-band error: {error:?}");
            }
            _ => {}
        }
    }

    assert!(saw_text_delta, "Anthropic stream should emit text deltas");
    assert!(saw_stop, "Anthropic stream should emit message_stop");
}

#[tokio::test]
#[ignore = "requires a real OpenAI Codex OAuth token"]
async fn openai_codex_oauth_responses_stream_round_trip() {
    // Required token:
    //   AIGW_E2E_OPENAI_CODEX_OAUTH_TOKEN
    // Aliases:
    //   AIGW_E2E_OPENAI_OAUTH_TOKEN, AIGW_CODEX_OAUTH_TOKEN,
    //   AIGW_OPENAI_OAUTH_TOKEN, OPENAI_CODEX_OAUTH_TOKEN,
    //   OPENAI_OAUTH_TOKEN, CODEX_OAUTH_TOKEN
    // Optional:
    //   AIGW_E2E_CODEX_MODEL, AIGW_E2E_CODEX_BASE_URL, AIGW_E2E_CODEX_USER_AGENT
    let token = oauth_token(
        "AIGW_E2E_OPENAI_CODEX_OAUTH_TOKEN",
        &[
            "AIGW_E2E_OPENAI_OAUTH_TOKEN",
            "AIGW_CODEX_OAUTH_TOKEN",
            "AIGW_OPENAI_OAUTH_TOKEN",
            "OPENAI_CODEX_OAUTH_TOKEN",
            "OPENAI_OAUTH_TOKEN",
            "CODEX_OAUTH_TOKEN",
        ],
    );
    let model = env_or("AIGW_E2E_CODEX_MODEL", DEFAULT_CODEX_MODEL);
    let client = openai_client(
        token,
        &env_or("AIGW_E2E_CODEX_BASE_URL", DEFAULT_CODEX_BASE_URL),
    );

    let request = build_codex_response_request(&model);
    let mut stream = client
        .stream_response(&request, &codex_request_options())
        .await
        .expect("OpenAI Codex OAuth Responses stream should start")
        .body;

    let mut saw_delta = false;
    let mut saw_completed = false;
    while let Some(event) = stream.next().await {
        let event = event.expect("Codex Responses stream event should decode");
        saw_delta |= event.event_type.ends_with(".delta");
        if event.event_type == "response.completed" {
            let status = event
                .extra
                .get("response")
                .and_then(|response| response.get("status"))
                .and_then(serde_json::Value::as_str);
            assert_eq!(status, Some("completed"));
            saw_completed = true;
            break;
        }
    }

    assert!(
        saw_delta,
        "Codex stream should emit at least one delta event"
    );
    assert!(saw_completed, "Codex stream should emit response.completed");
}

#[tokio::test]
#[ignore = "requires a real Google Gemini OAuth token"]
async fn google_gemini_oauth_openai_compat_round_trip() {
    // Required token:
    //   AIGW_E2E_GOOGLE_OAUTH_TOKEN
    // Aliases:
    //   AIGW_E2E_GEMINI_OAUTH_TOKEN, AIGW_GOOGLE_OAUTH_TOKEN,
    //   AIGW_GEMINI_OAUTH_TOKEN, GOOGLE_OAUTH_TOKEN, GEMINI_OAUTH_TOKEN
    // Optional:
    //   AIGW_E2E_GOOGLE_MODEL, AIGW_E2E_GOOGLE_OPENAI_BASE_URL
    //
    // This uses Google's OpenAI-compatible Gemini endpoint because it accepts
    // subscription OAuth tokens as `Authorization: Bearer <token>`.
    let token = oauth_token(
        "AIGW_E2E_GOOGLE_OAUTH_TOKEN",
        &[
            "AIGW_E2E_GEMINI_OAUTH_TOKEN",
            "AIGW_GOOGLE_OAUTH_TOKEN",
            "AIGW_GEMINI_OAUTH_TOKEN",
            "GOOGLE_OAUTH_TOKEN",
            "GEMINI_OAUTH_TOKEN",
        ],
    );
    let model = env_or("AIGW_E2E_GOOGLE_MODEL", DEFAULT_GOOGLE_MODEL);
    let client = openai_client(
        token,
        &env_or(
            "AIGW_E2E_GOOGLE_OPENAI_BASE_URL",
            DEFAULT_GOOGLE_OPENAI_BASE_URL,
        ),
    );

    let response = client
        .create_chat_completion(&chat_request(&model), &RequestOptions::default())
        .await
        .expect("Google Gemini OAuth OpenAI-compatible chat request should succeed");
    assert!(
        response.body.choices.iter().any(|choice| choice
            .message
            .content
            .as_ref()
            .is_some_and(openai_content_has_text)),
        "Google Gemini response should contain at least one non-empty assistant message: {:?}",
        response.body
    );

    let mut stream = client
        .stream_chat_completion(&chat_request(&model), &RequestOptions::default())
        .await
        .expect("Google Gemini OAuth OpenAI-compatible streaming request should start")
        .body;

    let mut saw_delta = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("Google Gemini stream chunk should decode");
        saw_delta |= chunk.choices.iter().any(|choice| {
            choice
                .delta
                .content
                .as_deref()
                .is_some_and(|s| !s.is_empty())
        });
        if chunk
            .choices
            .iter()
            .any(|choice| choice.finish_reason.is_some())
        {
            break;
        }
    }

    assert!(
        saw_delta,
        "Google Gemini stream should emit at least one content delta"
    );
}

fn anthropic_claude_code_client(token: String) -> AnthropicClient {
    let mut headers = HeaderMap::new();
    headers.insert(
        "anthropic-dangerous-direct-browser-access",
        HeaderValue::from_static("true"),
    );
    headers.insert("x-app", HeaderValue::from_static("cli"));
    headers.insert(
        USER_AGENT,
        HeaderValue::from_str(&env_or(
            "AIGW_E2E_CLAUDE_CODE_USER_AGENT",
            DEFAULT_CLAUDE_CODE_USER_AGENT,
        ))
        .expect("Claude Code user-agent must be a valid HTTP header value"),
    );

    let transport = AnthropicTransport::new(AnthropicTransportConfig {
        api_key: SecretString::from(token),
        auth_mode: AnthropicAuthMode::Bearer,
        base_url: env_or("AIGW_E2E_ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL),
        beta: Some(env_or("AIGW_E2E_ANTHROPIC_BETA", DEFAULT_ANTHROPIC_BETA)),
        extra_headers: headers,
        ..Default::default()
    })
    .expect("Anthropic OAuth transport config should be valid");

    AnthropicClient::new(transport).expect("Anthropic OAuth client should build")
}

fn anthropic_request(model: &str, stream: bool) -> AnthropicMessagesRequest {
    AnthropicMessagesRequest::builder()
        .model(model)
        .messages(vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: AnthropicMessageContent::Text(
                "Reply with a short acknowledgement for an AI gateway OAuth E2E test.".to_owned(),
            ),
        }])
        .max_tokens(64_u64)
        .stream(stream)
        .build()
}

fn anthropic_blocks_have_text(blocks: &[ContentBlock]) -> bool {
    blocks.iter().any(|block| match block {
        ContentBlock::Typed(TypedContentBlock::Text { text, .. }) => !text.is_empty(),
        _ => false,
    })
}

fn openai_client(token: String, base_url: &str) -> OpenAIClient {
    OpenAIClient::new(OpenAITransportConfig {
        http: HttpTransportConfig {
            base_url: base_url.to_owned(),
            timeout_seconds: 600,
            default_headers: BTreeMap::new(),
        },
        auth: OpenAIAuthConfig {
            api_key: SecretString::from(token),
            organization: None,
            project: None,
        },
    })
    .expect("OpenAI-compatible OAuth client config should be valid")
}

fn build_codex_response_request(model: &str) -> ResponseCreateRequest {
    let message = CanonicalMessage::builder()
        .role(CanonicalRole::User)
        .content(CanonicalMessageContent::Text(
            "Reply with a short acknowledgement for an AI gateway Codex OAuth E2E test.".to_owned(),
        ))
        .build();
    let canonical = CanonicalChatRequest::builder()
        .model(model)
        .messages(vec![message])
        .max_tokens(64_u64)
        .build();

    build_responses_create_request(&canonical, &ResponsesRequestConfig::codex())
        .expect("canonical request should translate to Codex Responses request")
}

fn codex_request_options() -> RequestOptions {
    RequestOptions {
        extra_headers: BTreeMap::from([
            (
                "Session_id".to_owned(),
                format!("aigw-e2e-{}", unique_suffix()),
            ),
            (
                "User-Agent".to_owned(),
                env_or("AIGW_E2E_CODEX_USER_AGENT", DEFAULT_CODEX_USER_AGENT),
            ),
            ("Originator".to_owned(), "codex_cli_rs".to_owned()),
            ("Connection".to_owned(), "Keep-Alive".to_owned()),
        ]),
    }
}

fn chat_request(model: &str) -> ChatCompletionRequest {
    ChatCompletionRequest::builder()
        .model(model)
        .messages(vec![ChatMessage {
            role: ChatMessageRole::User,
            content: Some(ChatMessageContent::Text(
                "Reply with a short acknowledgement for an AI gateway OAuth E2E test.".to_owned(),
            )),
            name: None,
            refusal: None,
            tool_call_id: None,
            tool_calls: None,
            extra: Default::default(),
        }])
        .max_tokens(64_u32)
        .build()
}

fn openai_content_has_text(content: &ChatMessageContent) -> bool {
    match content {
        ChatMessageContent::Text(text) => !text.is_empty(),
        ChatMessageContent::Parts(parts) => parts.iter().any(|part| match part {
            ChatContentPart::Typed(TypedChatContentPart::Text { text, .. }) => !text.is_empty(),
            _ => false,
        }),
    }
}

fn oauth_token(primary: &str, aliases: &[&str]) -> String {
    std::iter::once(primary)
        .chain(aliases.iter().copied())
        .find_map(|name| {
            env::var(name)
                .ok()
                .map(|value| normalize_bearer_token(&value))
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| {
            panic!(
                "missing OAuth token; set {primary} (aliases: {})",
                aliases.join(", ")
            )
        })
}

fn normalize_bearer_token(value: &str) -> String {
    let trimmed = value.trim();
    trimmed
        .strip_prefix("Bearer ")
        .or_else(|| trimmed.strip_prefix("bearer "))
        .unwrap_or(trimmed)
        .trim()
        .to_owned()
}

fn env_or(name: &str, default: &str) -> String {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_owned())
}

fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after UNIX_EPOCH")
        .as_nanos()
}
