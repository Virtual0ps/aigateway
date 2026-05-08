//! Request translation: canonical `ChatRequest` → Anthropic Messages API.
//!
//! Key transformations:
//! - System messages extracted to top-level `system` field
//! - Consecutive tool messages merged into a single user message
//! - Tool definitions unwrapped from OpenAI `function` wrapper
//! - Tool call arguments (JSON string) parsed into JSON objects

use aigw_core::error::TranslateError;
use aigw_core::model::{
    ChatRequest, ContentPart, Message, MessageContent, Role, ThinkingSource, TypedContentPart,
};
use aigw_core::translate::{RequestTranslator, ThinkingProjector, TranslatedRequest};
use bytes::Bytes;
use http::{HeaderMap, Method};

use crate::types::{
    ContentBlock, ImageSource, Message as AnthropicMessage, MessageContent as AnthropicContent,
    MessagesRequest, Metadata, Role as AnthropicRole, SystemPrompt, TypedContentBlock,
};

use super::cache_control::{
    CacheControlStrategy, DefaultCacheControlStrategy, enforce_breakpoint_cap,
    normalize_ttl_ordering,
};
use super::thinking::{AnthropicThinkingProjector, AnthropicThinkingTarget};
use super::tools;

const DEFAULT_MAX_TOKENS: u64 = 4096;

/// Translates canonical requests into Anthropic Messages API requests.
pub struct AnthropicRequestTranslator {
    headers: HeaderMap,
    url: String,
    default_max_tokens: u64,
    thinking: Box<dyn ThinkingProjector<AnthropicThinkingTarget>>,
    cache_control: Box<dyn CacheControlStrategy>,
}

impl AnthropicRequestTranslator {
    pub fn new(transport: &crate::Transport, default_max_tokens: Option<u64>) -> Self {
        Self {
            headers: transport.headers().clone(),
            url: transport.url("/v1/messages"),
            default_max_tokens: default_max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            thinking: Box::new(AnthropicThinkingProjector::default()),
            cache_control: Box::new(DefaultCacheControlStrategy::default()),
        }
    }

    /// Replace the thinking projector. Use with a custom
    /// [`AnthropicThinkingProjector`] (different adaptive-model matcher,
    /// level→budget table, etc.) or any other implementation of
    /// [`ThinkingProjector<AnthropicThinkingTarget>`].
    #[must_use]
    pub fn with_thinking_projector(
        mut self,
        projector: Box<dyn ThinkingProjector<AnthropicThinkingTarget>>,
    ) -> Self {
        self.thinking = projector;
        self
    }

    /// Replace the cache-control strategy. Use a different positioning
    /// policy or [`super::cache_control::NoCacheControlStrategy`] to
    /// disable automatic injection while still keeping the always-on
    /// 4-breakpoint cap and TTL ordering enforcement.
    #[must_use]
    pub fn with_cache_control_strategy(
        mut self,
        strategy: Box<dyn CacheControlStrategy>,
    ) -> Self {
        self.cache_control = strategy;
        self
    }
}

impl RequestTranslator for AnthropicRequestTranslator {
    fn translate_request(&self, req: &ChatRequest) -> Result<TranslatedRequest, TranslateError> {
        // Reject unsupported features.
        if let Some(n) = req.n
            && n > 1
        {
            return Err(TranslateError::UnsupportedFeature {
                provider: "anthropic",
                feature: "n > 1".into(),
            });
        }

        let system = extract_system(&req.messages);
        let messages = translate_messages(&req.messages)?;

        let mut extra = serde_json::Map::new();
        // Pass through Anthropic-specific extra fields.
        for (k, v) in &req.extra {
            extra.insert(k.clone(), v.clone());
        }

        // Resolve thinking config:
        // 1. Canonical `req.thinking` (preferred) → projector → target.
        // 2. Legacy fallback: `extra["thinking"]` (deprecated, will be
        //    removed in 0.6.0). Only consulted when canonical is absent.
        let mut target = AnthropicThinkingTarget {
            max_tokens: req.max_tokens.unwrap_or(self.default_max_tokens),
            ..Default::default()
        };
        if req.thinking.is_some() {
            self.thinking
                .apply(&req.model, req.thinking.as_ref(), &mut target);
            // Canonical took precedence — drop any extra-passthrough key.
            extra.remove("thinking");
        } else if let Some(legacy) = extra.remove("thinking") {
            target.thinking = serde_json::from_value(legacy).ok();
        }

        // Apply output_config decisions from the projector.
        if let Some(effort) = target.output_config_effort {
            extra.insert(
                "output_config".into(),
                serde_json::json!({ "effort": effort }),
            );
        } else if target.clear_output_config {
            extra.remove("output_config");
        }

        let mut native = MessagesRequest::builder()
            .model(&req.model)
            .messages(messages)
            .max_tokens(target.max_tokens)
            .maybe_system(system)
            .maybe_temperature(req.temperature)
            .maybe_top_p(req.top_p)
            .maybe_stop_sequences(req.stop.as_ref().map(|s| s.to_vec()))
            .maybe_stream(req.stream)
            .maybe_tools(req.tools.as_ref().map(|t| tools::translate_tools(t)))
            .maybe_tool_choice(req.tool_choice.as_ref().map(tools::translate_tool_choice))
            .maybe_metadata(req.user.as_ref().map(|u| Metadata {
                user_id: Some(u.clone()),
            }))
            .maybe_thinking(target.thinking)
            .extra(extra)
            .build();

        // Cache-control pipeline: configurable strategy first, then the
        // always-on Anthropic API correctness rules.
        self.cache_control.apply(&mut native);
        enforce_breakpoint_cap(&mut native);
        normalize_ttl_ordering(&mut native);

        let body = serde_json::to_vec(&native)?;

        Ok(TranslatedRequest {
            url: self.url.clone(),
            method: Method::POST,
            headers: self.headers.clone(),
            body: Bytes::from(body),
        })
    }
}

// ─── System message extraction ──────────────────────────────────────────────

/// Extract all system/developer messages and join them into a single `SystemPrompt`.
fn extract_system(messages: &[Message]) -> Option<SystemPrompt> {
    let mut all_texts = Vec::new();

    for msg in messages {
        if !matches!(msg.role, Role::System | Role::Developer) {
            continue;
        }
        match &msg.content {
            Some(MessageContent::Text(s)) => all_texts.push(s.clone()),
            Some(MessageContent::Parts(parts)) => {
                let text: String = parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Known(TypedContentPart::Text { text, .. }) => {
                            Some(text.as_str())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                if !text.is_empty() {
                    all_texts.push(text);
                }
            }
            None => {}
        }
    }

    if all_texts.is_empty() {
        None
    } else {
        Some(SystemPrompt::Text(all_texts.join("\n\n")))
    }
}

// ─── Message translation ────────────────────────────────────────────────────

/// Translate canonical messages to Anthropic format.
///
/// Filters out system/developer messages (already extracted) and merges
/// consecutive tool messages into a single user message with tool_result blocks.
fn translate_messages(messages: &[Message]) -> Result<Vec<AnthropicMessage>, TranslateError> {
    let non_system: Vec<&Message> = messages
        .iter()
        .filter(|m| !matches!(m.role, Role::System | Role::Developer))
        .collect();

    let mut result = Vec::new();
    let mut i = 0;

    while i < non_system.len() {
        let msg = non_system[i];
        match msg.role {
            Role::User => {
                result.push(translate_user_message(msg)?);
                i += 1;
            }
            Role::Assistant => {
                result.push(translate_assistant_message(msg)?);
                i += 1;
            }
            Role::Tool => {
                // Merge consecutive tool messages into a single user message.
                let mut tool_blocks = Vec::new();
                while i < non_system.len() && non_system[i].role == Role::Tool {
                    tool_blocks.push(translate_tool_result(non_system[i])?);
                    i += 1;
                }
                result.push(AnthropicMessage {
                    role: AnthropicRole::User,
                    content: AnthropicContent::Blocks(tool_blocks),
                });
            }
            _ => {
                // Unknown roles: treat as user.
                result.push(translate_user_message(msg)?);
                i += 1;
            }
        }
    }

    Ok(result)
}

fn translate_user_message(msg: &Message) -> Result<AnthropicMessage, TranslateError> {
    let content = match &msg.content {
        Some(MessageContent::Text(s)) => AnthropicContent::Text(s.clone()),
        Some(MessageContent::Parts(parts)) => {
            let mut blocks = Vec::with_capacity(parts.len());
            for part in parts {
                if let Some(block) = translate_content_part(part)? {
                    blocks.push(block);
                }
            }
            AnthropicContent::Blocks(blocks)
        }
        None => AnthropicContent::Text(String::new()),
    };

    Ok(AnthropicMessage {
        role: AnthropicRole::User,
        content,
    })
}

fn translate_assistant_message(msg: &Message) -> Result<AnthropicMessage, TranslateError> {
    let mut blocks = Vec::new();

    // Text content.
    match &msg.content {
        Some(MessageContent::Text(s)) if !s.is_empty() => {
            blocks.push(ContentBlock::Typed(TypedContentBlock::Text {
                text: s.clone(),
                cache_control: None,
            }));
        }
        Some(MessageContent::Parts(parts)) => {
            for part in parts {
                if let Some(block) = translate_content_part(part)? {
                    blocks.push(block);
                }
            }
        }
        _ => {}
    }

    // Tool calls → ToolUse blocks.
    if let Some(tool_calls) = &msg.tool_calls {
        for tc in tool_calls {
            let input: serde_json::Value =
                serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));
            blocks.push(ContentBlock::Typed(TypedContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input,
                cache_control: None,
            }));
        }
    }

    let content = if blocks.is_empty() {
        AnthropicContent::Text(String::new())
    } else {
        AnthropicContent::Blocks(blocks)
    };

    Ok(AnthropicMessage {
        role: AnthropicRole::Assistant,
        content,
    })
}

fn translate_tool_result(msg: &Message) -> Result<ContentBlock, TranslateError> {
    let tool_use_id = msg
        .tool_call_id
        .clone()
        .ok_or(TranslateError::MissingField {
            field: "tool_call_id",
        })?;

    let content = msg.content.as_ref().map(|c| match c {
        MessageContent::Text(s) => crate::types::ToolResultContent::Text(s.clone()),
        MessageContent::Parts(_) => {
            // Serialize parts content as text fallback.
            crate::types::ToolResultContent::Text(serde_json::to_string(c).unwrap_or_default())
        }
    });

    Ok(ContentBlock::Typed(TypedContentBlock::ToolResult {
        tool_use_id,
        content,
        is_error: None,
        cache_control: None,
    }))
}

/// Translate a single canonical content part into an Anthropic content block.
///
/// Returns `Ok(Some(_))` for parts that should appear in the wire request,
/// `Ok(None)` for parts that should be silently dropped (e.g. thinking
/// blocks generated by a different provider — Anthropic rejects signatures
/// it didn't produce).
fn translate_content_part(part: &ContentPart) -> Result<Option<ContentBlock>, TranslateError> {
    match part {
        ContentPart::Known(TypedContentPart::Text { text, .. }) => {
            Ok(Some(ContentBlock::Typed(TypedContentBlock::Text {
                text: text.clone(),
                cache_control: None,
            })))
        }
        ContentPart::Known(TypedContentPart::ImageUrl { image_url, .. }) => {
            let source = translate_image_source(&image_url.url)?;
            Ok(Some(ContentBlock::Typed(TypedContentBlock::Image {
                source,
                cache_control: None,
            })))
        }
        ContentPart::Known(TypedContentPart::Thinking {
            thinking,
            signature,
            source,
            ..
        }) => Ok(forward_thinking_to_anthropic(*source).then(|| {
            ContentBlock::Typed(TypedContentBlock::Thinking {
                thinking: thinking.clone(),
                signature: signature.clone(),
            })
        })),
        ContentPart::Known(TypedContentPart::RedactedThinking { data, source, .. }) => {
            Ok(forward_thinking_to_anthropic(*source).then(|| {
                ContentBlock::Typed(TypedContentBlock::RedactedThinking { data: data.clone() })
            }))
        }
        ContentPart::Raw(obj) => {
            // Both sides are now serde_json::Map — direct clone.
            Ok(Some(ContentBlock::Raw(obj.clone())))
        }
        _ => Err(TranslateError::IncompatibleContent {
            reason: "unsupported content part type for Anthropic".into(),
        }),
    }
}

/// Returns `true` if a thinking block with the given source should be
/// forwarded to Anthropic.
///
/// Anthropic rejects thinking blocks whose signature it did not produce, so
/// blocks tagged as having come from another provider must be dropped before
/// forwarding. Untagged blocks (`source == None`) are forwarded for
/// backward compatibility — pre-source-tagging clients still work.
const fn forward_thinking_to_anthropic(source: Option<ThinkingSource>) -> bool {
    matches!(source, None | Some(ThinkingSource::Anthropic))
}

fn translate_image_source(url: &str) -> Result<ImageSource, TranslateError> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (header, data) =
            rest.split_once(',')
                .ok_or_else(|| TranslateError::IncompatibleContent {
                    reason: "malformed data: URI".into(),
                })?;
        let media_type =
            header
                .strip_suffix(";base64")
                .ok_or_else(|| TranslateError::IncompatibleContent {
                    reason: "data: URI must be base64-encoded".into(),
                })?;
        Ok(ImageSource::Base64 {
            media_type: media_type.to_owned(),
            data: data.to_owned(),
        })
    } else {
        Ok(ImageSource::Url {
            url: url.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aigw_core::model::{FunctionCall, ImageUrl, ToolCall};

    fn user_msg(text: &str) -> Message {
        Message {
            role: Role::User,
            content: Some(MessageContent::Text(text.into())),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            extra: Default::default(),
        }
    }

    fn system_msg(text: &str) -> Message {
        Message {
            role: Role::System,
            content: Some(MessageContent::Text(text.into())),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            extra: Default::default(),
        }
    }

    fn tool_msg(tool_call_id: &str, content: &str) -> Message {
        Message {
            role: Role::Tool,
            content: Some(MessageContent::Text(content.into())),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn extract_no_system() {
        let msgs = vec![user_msg("hi")];
        assert!(extract_system(&msgs).is_none());
    }

    #[test]
    fn extract_single_system() {
        let msgs = vec![system_msg("You are helpful"), user_msg("hi")];
        match extract_system(&msgs) {
            Some(SystemPrompt::Text(s)) => assert_eq!(s, "You are helpful"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn extract_multiple_system_messages() {
        let msgs = vec![
            system_msg("You are helpful"),
            system_msg("Be concise"),
            user_msg("hi"),
        ];
        match extract_system(&msgs) {
            Some(SystemPrompt::Text(s)) => assert_eq!(s, "You are helpful\n\nBe concise"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn translate_messages_filters_system() {
        let msgs = vec![system_msg("system"), user_msg("hello")];
        let result = translate_messages(&msgs).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, AnthropicRole::User);
    }

    #[test]
    fn consecutive_tool_messages_merged() {
        let msgs = vec![
            user_msg("check weather"),
            // assistant with tool call would be here in real conversation
            tool_msg("call_1", "72F sunny"),
            tool_msg("call_2", "65F cloudy"),
        ];
        let result = translate_messages(&msgs).unwrap();
        // user + merged tool results (as single user message)
        assert_eq!(result.len(), 2);
        assert_eq!(result[1].role, AnthropicRole::User);
        match &result[1].content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                // Both should be ToolResult blocks.
                for block in blocks {
                    assert!(matches!(
                        block,
                        ContentBlock::Typed(TypedContentBlock::ToolResult { .. })
                    ));
                }
            }
            _ => panic!("expected Blocks"),
        }
    }

    #[test]
    fn assistant_message_with_tool_calls() {
        let msg = Message {
            role: Role::Assistant,
            content: Some(MessageContent::Text("Let me check.".into())),
            name: None,
            tool_call_id: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: r#"{"location":"SF"}"#.into(),
                    extra: Default::default(),
                },
                extra: Default::default(),
            }]),
            extra: Default::default(),
        };

        let result = translate_assistant_message(&msg).unwrap();
        assert_eq!(result.role, AnthropicRole::Assistant);
        match &result.content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2); // text + tool_use
                assert!(matches!(
                    &blocks[0],
                    ContentBlock::Typed(TypedContentBlock::Text { text, .. }) if text == "Let me check."
                ));
                assert!(matches!(
                    &blocks[1],
                    ContentBlock::Typed(TypedContentBlock::ToolUse { name, .. }) if name == "get_weather"
                ));
            }
            _ => panic!("expected Blocks"),
        }
    }

    #[test]
    fn image_url_translation() {
        let source = translate_image_source("https://example.com/img.png").unwrap();
        assert!(matches!(source, ImageSource::Url { url } if url == "https://example.com/img.png"));
    }

    #[test]
    fn image_data_uri_translation() {
        let source = translate_image_source("data:image/png;base64,iVBOR...").unwrap();
        match source {
            ImageSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "image/png");
                assert_eq!(data, "iVBOR...");
            }
            _ => panic!("expected Base64"),
        }
    }

    #[test]
    fn image_content_in_user_message() {
        let msg = Message {
            role: Role::User,
            content: Some(MessageContent::Parts(vec![
                ContentPart::Known(TypedContentPart::Text {
                    text: "What's in this image?".into(),
                    extra: Default::default(),
                }),
                ContentPart::Known(TypedContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/cat.jpg".into(),
                        detail: None,
                        extra: Default::default(),
                    },
                    extra: Default::default(),
                }),
            ])),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            extra: Default::default(),
        };

        let result = translate_user_message(&msg).unwrap();
        match &result.content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(matches!(
                    &blocks[0],
                    ContentBlock::Typed(TypedContentBlock::Text { .. })
                ));
                assert!(matches!(
                    &blocks[1],
                    ContentBlock::Typed(TypedContentBlock::Image { .. })
                ));
            }
            _ => panic!("expected Blocks"),
        }
    }

    #[test]
    fn anthropic_thinking_part_round_trips_to_native() {
        let msg = Message {
            role: Role::Assistant,
            content: Some(MessageContent::Parts(vec![
                ContentPart::Known(TypedContentPart::Thinking {
                    thinking: "Let me think...".into(),
                    signature: "ErWj123".into(),
                    source: Some(ThinkingSource::Anthropic),
                    extra: Default::default(),
                }),
                ContentPart::Known(TypedContentPart::Text {
                    text: "Answer is 42.".into(),
                    extra: Default::default(),
                }),
            ])),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            extra: Default::default(),
        };

        let result = translate_assistant_message(&msg).unwrap();
        match &result.content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(matches!(
                    &blocks[0],
                    ContentBlock::Typed(TypedContentBlock::Thinking { signature, .. })
                        if signature == "ErWj123"
                ));
                assert!(matches!(
                    &blocks[1],
                    ContentBlock::Typed(TypedContentBlock::Text { .. })
                ));
            }
            _ => panic!("expected Blocks"),
        }
    }

    #[test]
    fn gemini_sourced_thinking_part_is_dropped() {
        let msg = Message {
            role: Role::Assistant,
            content: Some(MessageContent::Parts(vec![
                ContentPart::Known(TypedContentPart::Thinking {
                    thinking: "from gemini".into(),
                    signature: "garbage".into(),
                    source: Some(ThinkingSource::Gemini),
                    extra: Default::default(),
                }),
                ContentPart::Known(TypedContentPart::Text {
                    text: "kept".into(),
                    extra: Default::default(),
                }),
            ])),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            extra: Default::default(),
        };

        let result = translate_assistant_message(&msg).unwrap();
        match &result.content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1, "gemini-sourced thinking must be dropped");
                assert!(matches!(
                    &blocks[0],
                    ContentBlock::Typed(TypedContentBlock::Text { text, .. }) if text == "kept"
                ));
            }
            _ => panic!("expected Blocks"),
        }
    }

    #[test]
    fn untagged_thinking_part_is_forwarded() {
        // No `source`: assume legacy producer, forward by default.
        let msg = Message {
            role: Role::Assistant,
            content: Some(MessageContent::Parts(vec![ContentPart::Known(
                TypedContentPart::Thinking {
                    thinking: "no source".into(),
                    signature: "Ezzz".into(),
                    source: None,
                    extra: Default::default(),
                },
            )])),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            extra: Default::default(),
        };

        let result = translate_assistant_message(&msg).unwrap();
        match &result.content {
            AnthropicContent::Blocks(blocks) => assert_eq!(blocks.len(), 1),
            _ => panic!("expected Blocks"),
        }
    }

    #[test]
    fn redacted_thinking_anthropic_source_forwarded() {
        let msg = Message {
            role: Role::Assistant,
            content: Some(MessageContent::Parts(vec![ContentPart::Known(
                TypedContentPart::RedactedThinking {
                    data: "blob".into(),
                    source: Some(ThinkingSource::Anthropic),
                    extra: Default::default(),
                },
            )])),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            extra: Default::default(),
        };

        let result = translate_assistant_message(&msg).unwrap();
        match &result.content {
            AnthropicContent::Blocks(blocks) => {
                assert!(matches!(
                    &blocks[0],
                    ContentBlock::Typed(TypedContentBlock::RedactedThinking { data })
                        if data == "blob"
                ));
            }
            _ => panic!("expected Blocks"),
        }
    }

    #[test]
    fn canonical_thinking_field_invokes_projector() {
        // ChatRequest.thinking = Level::High on adaptive model →
        // thinking: {type:"adaptive"} + extra.output_config.effort = "high".
        use aigw_core::model::{ThinkingLevel, ThinkingRequest};

        let transport = crate::Transport::new(crate::TransportConfig {
            api_key: secrecy::SecretString::from("sk-ant-test"),
            ..Default::default()
        })
        .unwrap();
        let translator = AnthropicRequestTranslator::new(&transport, None);

        let req = ChatRequest::builder()
            .model("claude-opus-4-6")
            .messages(vec![user_msg("hi")])
            .thinking(ThinkingRequest::Level {
                level: ThinkingLevel::High,
            })
            .build();

        let translated = translator.translate_request(&req).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&translated.body).unwrap();

        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "high");
    }

    #[test]
    fn legacy_extra_thinking_still_works_when_canonical_absent() {
        let transport = crate::Transport::new(crate::TransportConfig {
            api_key: secrecy::SecretString::from("sk-ant-test"),
            ..Default::default()
        })
        .unwrap();
        let translator = AnthropicRequestTranslator::new(&transport, None);

        let mut extra = serde_json::Map::new();
        extra.insert(
            "thinking".into(),
            serde_json::json!({"type":"enabled","budget_tokens":5000}),
        );

        let req = ChatRequest::builder()
            .model("claude-opus-4-5")
            .messages(vec![user_msg("hi")])
            .extra(extra)
            .build();

        let translated = translator.translate_request(&req).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&translated.body).unwrap();

        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 5000);
    }

    #[test]
    fn canonical_thinking_overrides_extra_thinking() {
        use aigw_core::model::ThinkingRequest;

        let transport = crate::Transport::new(crate::TransportConfig {
            api_key: secrecy::SecretString::from("sk-ant-test"),
            ..Default::default()
        })
        .unwrap();
        let translator = AnthropicRequestTranslator::new(&transport, None);

        let mut extra = serde_json::Map::new();
        extra.insert(
            "thinking".into(),
            serde_json::json!({"type":"enabled","budget_tokens":99999}),
        );

        let req = ChatRequest::builder()
            .model("claude-opus-4-5")
            .messages(vec![user_msg("hi")])
            .thinking(ThinkingRequest::Budget {
                budget_tokens: 8_000,
            })
            .extra(extra)
            .build();

        let translated = translator.translate_request(&req).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&translated.body).unwrap();

        assert_eq!(body["thinking"]["budget_tokens"], 8_000);
    }

    #[test]
    fn translate_injects_cache_control_on_default() {
        let transport = crate::Transport::new(crate::TransportConfig {
            api_key: secrecy::SecretString::from("sk-ant-test"),
            ..Default::default()
        })
        .unwrap();
        let translator = AnthropicRequestTranslator::new(&transport, None);

        let req = ChatRequest::builder()
            .model("claude-opus-4-5")
            .messages(vec![
                user_msg("first"),
                Message {
                    role: Role::Assistant,
                    content: Some(MessageContent::Text("ok".into())),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                    extra: Default::default(),
                },
                user_msg("second"),
            ])
            .build();

        let translated = translator.translate_request(&req).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&translated.body).unwrap();
        // First user message (= second-to-last user) should have the
        // breakpoint injected on its first content block.
        let cc = &body["messages"][0]["content"][0]["cache_control"];
        assert_eq!(cc["type"], "ephemeral");
    }

    #[test]
    fn translate_with_no_cache_control_strategy() {
        use super::super::cache_control::NoCacheControlStrategy;
        let transport = crate::Transport::new(crate::TransportConfig {
            api_key: secrecy::SecretString::from("sk-ant-test"),
            ..Default::default()
        })
        .unwrap();
        let translator = AnthropicRequestTranslator::new(&transport, None)
            .with_cache_control_strategy(Box::new(NoCacheControlStrategy));

        let req = ChatRequest::builder()
            .model("claude-opus-4-5")
            .messages(vec![user_msg("first"), user_msg("second")])
            .build();

        let translated = translator.translate_request(&req).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&translated.body).unwrap();
        // Strategy disabled → no breakpoints injected.
        // (Messages may still serialize as plain strings.)
        let messages = body["messages"].as_array().unwrap();
        for msg in messages {
            // String content has no cache_control hook to begin with.
            // If it's been promoted to an array, none of the blocks should
            // carry a cache_control field.
            if let Some(content) = msg["content"].as_array() {
                for block in content {
                    assert!(block.get("cache_control").is_none());
                }
            }
        }
    }

    #[test]
    fn tool_result_missing_tool_call_id() {
        let msg = Message {
            role: Role::Tool,
            content: Some(MessageContent::Text("result".into())),
            name: None,
            tool_call_id: None, // missing!
            tool_calls: None,
            extra: Default::default(),
        };

        let err = translate_tool_result(&msg).unwrap_err();
        assert!(matches!(
            err,
            TranslateError::MissingField {
                field: "tool_call_id"
            }
        ));
    }
}
