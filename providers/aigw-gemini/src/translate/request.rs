//! Request translation: canonical [`ChatRequest`] → Gemini
//! [`GenerateContentRequest`].
//!
//! Key transformations:
//! - System/Developer messages → `systemInstruction` (Gemini has no
//!   `system` role).
//! - Consecutive same-role turns are merged (Gemini rejects them).
//! - `tool_calls` on an assistant message → `functionCall` parts.
//! - `tool` role messages → `user` role with `functionResponse` parts
//!   (modern Gemini 2.5+ convention; the legacy `function` role is also
//!   accepted but deprecated).
//! - `response_format::JsonObject` → `responseMimeType: "application/json"`.
//! - `response_format::JsonSchema` → `responseMimeType` + `responseSchema`.
//! - Canonical `req.thinking` → `generationConfig.thinkingConfig` via the
//!   provider's [`ThinkingProjector`].
//!
//! [`ChatRequest`]: aigw_core::model::ChatRequest
//! [`GenerateContentRequest`]: crate::types::GenerateContentRequest

use std::collections::HashMap;

use aigw_core::error::TranslateError;
use aigw_core::model::{
    ChatRequest, ContentPart, Message, MessageContent, ResponseFormat, Role, ThinkingSource, Tool,
    ToolCall, ToolChoice, ToolChoiceMode, TypedContentPart,
};
use aigw_core::translate::{RequestTranslator, ThinkingProjector, TranslatedRequest};
use bytes::Bytes;
use http::{HeaderMap, HeaderValue, Method};
use reqwest::header::CONTENT_TYPE;
use secrecy::{ExposeSecret, SecretString};

use super::thinking::{GeminiThinkingProjector, GeminiThinkingTarget};
use crate::types::{
    Blob, Content, FileData, FunctionCall as NativeFunctionCall, FunctionCallingConfig,
    FunctionCallingMode, FunctionDeclaration, FunctionResponse, GenerateContentRequest,
    GenerationConfig, Part, Role as NativeRole, Tool as NativeTool, ToolConfig,
};

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const DEFAULT_API_VERSION: &str = "v1beta";

/// Translates canonical requests into Gemini `generateContent` requests.
pub struct GeminiRequestTranslator {
    base_url: String,
    api_version: String,
    headers: HeaderMap,
    thinking: Box<dyn ThinkingProjector<GeminiThinkingTarget>>,
}

impl GeminiRequestTranslator {
    /// Construct a translator with default base URL
    /// (`https://generativelanguage.googleapis.com`) and API version
    /// (`v1beta`). Uses [`GeminiThinkingProjector::default`] for thinking
    /// translation.
    ///
    /// # Errors
    ///
    /// Returns [`TranslateError::Other`] if the api key contains characters
    /// that aren't valid in an HTTP header.
    pub fn new(api_key: &SecretString) -> Result<Self, TranslateError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-goog-api-key",
            HeaderValue::from_str(api_key.expose_secret())
                .map_err(|e| TranslateError::Other(format!("invalid api key: {e}")))?,
        );
        Ok(Self {
            base_url: DEFAULT_BASE_URL.to_owned(),
            api_version: DEFAULT_API_VERSION.to_owned(),
            headers,
            thinking: Box::new(GeminiThinkingProjector::default()),
        })
    }

    /// Override the base URL (e.g. a regional endpoint or proxy).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_owned();
        self
    }

    /// Override the API version path segment (default `v1beta`).
    #[must_use]
    pub fn with_api_version(mut self, ver: impl Into<String>) -> Self {
        self.api_version = ver.into();
        self
    }

    /// Replace the thinking projector.
    #[must_use]
    pub fn with_thinking_projector(
        mut self,
        projector: Box<dyn ThinkingProjector<GeminiThinkingTarget>>,
    ) -> Self {
        self.thinking = projector;
        self
    }

    fn url(&self, model: &str, streaming: bool) -> String {
        let action = if streaming {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        format!(
            "{}/{}/models/{}:{}",
            self.base_url, self.api_version, model, action
        )
    }
}

impl RequestTranslator for GeminiRequestTranslator {
    fn translate_request(&self, req: &ChatRequest) -> Result<TranslatedRequest, TranslateError> {
        let body = build_body(req, self.thinking.as_ref())?;
        let body_bytes = serde_json::to_vec(&body)?;
        Ok(TranslatedRequest {
            url: self.url(&req.model, false),
            method: Method::POST,
            headers: self.headers.clone(),
            body: Bytes::from(body_bytes),
        })
    }

    fn translate_stream_request(
        &self,
        req: &ChatRequest,
    ) -> Result<TranslatedRequest, TranslateError> {
        let body = build_body(req, self.thinking.as_ref())?;
        let body_bytes = serde_json::to_vec(&body)?;
        Ok(TranslatedRequest {
            url: self.url(&req.model, true),
            method: Method::POST,
            headers: self.headers.clone(),
            body: Bytes::from(body_bytes),
        })
    }
}

// ─── Public body builder ───────────────────────────────────────────────────

/// Build a [`GenerateContentRequest`] from a canonical [`ChatRequest`]
/// without constructing a [`GeminiRequestTranslator`].
///
/// Use this when you only need the translated body (for example, as a
/// downstream gateway that wraps the Gemini request in a custom envelope
/// — Antigravity's `/v1internal:generateContent` is one such case). For
/// the full [`TranslatedRequest`] with URL/headers included, use
/// [`GeminiRequestTranslator`].
///
/// Defaults to [`GeminiThinkingProjector::default`] for canonical
/// thinking handling. Pass a custom projector via
/// [`build_generate_content_request_with_projector`] if needed.
///
/// # Errors
///
/// Returns [`TranslateError`] if any field in `req` fails to translate.
pub fn build_generate_content_request(
    req: &ChatRequest,
) -> Result<GenerateContentRequest, TranslateError> {
    let projector = GeminiThinkingProjector::default();
    build_body(req, &projector)
}

/// Like [`build_generate_content_request`] but with a caller-provided
/// thinking projector.
///
/// # Errors
///
/// Returns [`TranslateError`] if any field in `req` fails to translate.
pub fn build_generate_content_request_with_projector(
    req: &ChatRequest,
    projector: &dyn ThinkingProjector<GeminiThinkingTarget>,
) -> Result<GenerateContentRequest, TranslateError> {
    build_body(req, projector)
}

// ─── Body builder ───────────────────────────────────────────────────────────

fn build_body(
    req: &ChatRequest,
    thinking: &dyn ThinkingProjector<GeminiThinkingTarget>,
) -> Result<GenerateContentRequest, TranslateError> {
    if let Some(n) = req.n
        && n > 1
    {
        return Err(TranslateError::UnsupportedFeature {
            provider: "gemini",
            feature: "n > 1".into(),
        });
    }

    // Build a id→name map from assistant tool_calls so we can populate
    // FunctionResponse.name when translating tool-role messages (canonical
    // tool messages only carry tool_call_id, not the function name).
    let tool_call_names = build_tool_call_name_map(&req.messages);

    let (system_instruction, contents) = translate_messages(&req.messages, &tool_call_names)?;

    let tools = req
        .tools
        .as_ref()
        .map(|t| vec![translate_tools_into_one(t)]);
    let tool_config = req.tool_choice.as_ref().map(translate_tool_choice);

    let generation_config = build_generation_config(req, thinking);

    let mut extra = serde_json::Map::new();
    for (k, v) in &req.extra {
        // The canonical thinking field is the single source of truth for
        // thinkingConfig; anything else passes through.
        if k == "thinking" {
            continue;
        }
        extra.insert(k.clone(), v.clone());
    }

    Ok(GenerateContentRequest {
        model: req.model.clone(),
        contents,
        tools,
        tool_config,
        safety_settings: None,
        system_instruction,
        generation_config,
        cached_content: None,
        extra,
    })
}

fn build_tool_call_name_map(messages: &[Message]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for msg in messages {
        if msg.role == Role::Assistant
            && let Some(tcs) = &msg.tool_calls
        {
            for tc in tcs {
                map.insert(tc.id.clone(), tc.function.name.clone());
            }
        }
    }
    map
}

// ─── System extraction + message translation ───────────────────────────────

/// Translate canonical messages to Gemini contents.
///
/// Returns `(systemInstruction, contents)`. `contents` is post-processed to
/// merge consecutive same-role turns (Gemini rejects them).
fn translate_messages(
    messages: &[Message],
    tool_call_names: &HashMap<String, String>,
) -> Result<(Option<Content>, Vec<Content>), TranslateError> {
    // Collect system text first.
    let mut system_texts: Vec<String> = Vec::new();
    for msg in messages {
        if matches!(msg.role, Role::System | Role::Developer) {
            if let Some(t) = extract_text(&msg.content) {
                system_texts.push(t);
            }
        }
    }
    let system_instruction = if system_texts.is_empty() {
        None
    } else {
        Some(Content {
            role: None,
            parts: vec![Part::text(system_texts.join("\n\n"))],
        })
    };

    // Translate non-system messages.
    let mut raw_contents: Vec<Content> = Vec::new();
    for msg in messages {
        if matches!(msg.role, Role::System | Role::Developer) {
            continue;
        }
        match msg.role {
            Role::User => {
                raw_contents.push(Content {
                    role: Some(NativeRole::User),
                    parts: translate_user_parts(msg)?,
                });
            }
            Role::Assistant => {
                raw_contents.push(Content {
                    role: Some(NativeRole::Model),
                    parts: translate_assistant_parts(msg)?,
                });
            }
            Role::Tool => {
                raw_contents.push(Content {
                    role: Some(NativeRole::User),
                    parts: vec![translate_tool_result(msg, tool_call_names)?],
                });
            }
            _ => {
                // Unknown roles fall through to user.
                raw_contents.push(Content {
                    role: Some(NativeRole::User),
                    parts: translate_user_parts(msg)?,
                });
            }
        }
    }

    Ok((system_instruction, merge_adjacent(raw_contents)))
}

fn translate_user_parts(msg: &Message) -> Result<Vec<Part>, TranslateError> {
    match &msg.content {
        Some(MessageContent::Text(s)) => Ok(vec![Part::text(s.clone())]),
        Some(MessageContent::Parts(parts)) => {
            let mut out = Vec::with_capacity(parts.len());
            for part in parts {
                if let Some(p) = translate_content_part(part)? {
                    out.push(p);
                }
            }
            Ok(out)
        }
        None => Ok(vec![]),
    }
}

fn translate_assistant_parts(msg: &Message) -> Result<Vec<Part>, TranslateError> {
    let mut parts: Vec<Part> = Vec::new();

    match &msg.content {
        Some(MessageContent::Text(s)) if !s.is_empty() => {
            parts.push(Part::text(s.clone()));
        }
        Some(MessageContent::Parts(content_parts)) => {
            for part in content_parts {
                if let Some(p) = translate_content_part(part)? {
                    parts.push(p);
                }
            }
        }
        _ => {}
    }

    if let Some(tool_calls) = &msg.tool_calls {
        for tc in tool_calls {
            parts.push(translate_tool_call(tc));
        }
    }

    Ok(parts)
}

fn translate_tool_result(
    msg: &Message,
    tool_call_names: &HashMap<String, String>,
) -> Result<Part, TranslateError> {
    let tool_use_id = msg
        .tool_call_id
        .clone()
        .ok_or(TranslateError::MissingField {
            field: "tool_call_id",
        })?;
    let name = tool_call_names
        .get(&tool_use_id)
        .cloned()
        .unwrap_or_default();

    let response_value = match &msg.content {
        Some(MessageContent::Text(s)) => {
            // Try to parse as JSON; fall back to wrapping in a {"result": ...}.
            serde_json::from_str(s).unwrap_or_else(|_| serde_json::json!({ "result": s }))
        }
        Some(MessageContent::Parts(_)) => {
            serde_json::json!({ "result": serde_json::to_string(&msg.content).unwrap_or_default() })
        }
        None => serde_json::json!({}),
    };

    Ok(Part {
        function_response: Some(FunctionResponse {
            name,
            response: response_value,
            id: Some(tool_use_id),
            extra: serde_json::Map::new(),
        }),
        ..Default::default()
    })
}

fn translate_content_part(part: &ContentPart) -> Result<Option<Part>, TranslateError> {
    match part {
        ContentPart::Known(TypedContentPart::Text { text, .. }) => Ok(Some(Part::text(text))),
        ContentPart::Known(TypedContentPart::ImageUrl { image_url, .. }) => {
            Ok(Some(translate_image(&image_url.url)?))
        }
        ContentPart::Known(TypedContentPart::Thinking {
            thinking,
            signature,
            source,
            ..
        }) => Ok(forward_thinking_to_gemini(*source).then(|| Part {
            text: Some(thinking.clone()),
            thought: Some(true),
            thought_signature: if signature.is_empty() {
                None
            } else {
                Some(signature.clone())
            },
            ..Default::default()
        })),
        ContentPart::Known(TypedContentPart::RedactedThinking { source, .. }) => {
            // Redacted blocks have only opaque data; if we forward them at
            // all, we round-trip via a thought=true part with empty text.
            // For now drop them (no native redacted_thinking on Gemini).
            Ok(forward_thinking_to_gemini(*source).then(|| Part {
                text: Some(String::new()),
                thought: Some(true),
                ..Default::default()
            }))
        }
        ContentPart::Raw(_) => Ok(None),
        _ => Ok(None),
    }
}

fn translate_image(url: &str) -> Result<Part, TranslateError> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (header, data) =
            rest.split_once(',')
                .ok_or_else(|| TranslateError::IncompatibleContent {
                    reason: "malformed data: URI".into(),
                })?;
        let mime_type =
            header
                .strip_suffix(";base64")
                .ok_or_else(|| TranslateError::IncompatibleContent {
                    reason: "data: URI must be base64-encoded".into(),
                })?;
        Ok(Part::inline_data(Blob {
            mime_type: mime_type.to_owned(),
            data: data.to_owned(),
            extra: serde_json::Map::new(),
        }))
    } else {
        Ok(Part::file_data(FileData {
            // Gemini requires a MIME type; use the generic image fallback.
            mime_type: "image/*".to_owned(),
            file_uri: url.to_owned(),
            extra: serde_json::Map::new(),
        }))
    }
}

fn translate_tool_call(tc: &ToolCall) -> Part {
    let args: serde_json::Value =
        serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));
    Part {
        function_call: Some(NativeFunctionCall {
            name: tc.function.name.clone(),
            args,
            id: Some(tc.id.clone()),
            extra: serde_json::Map::new(),
        }),
        ..Default::default()
    }
}

const fn forward_thinking_to_gemini(source: Option<ThinkingSource>) -> bool {
    matches!(source, None | Some(ThinkingSource::Gemini))
}

/// Merge consecutive same-role contents. Gemini rejects requests with
/// adjacent same-role turns; the canonical model has no such restriction.
fn merge_adjacent(contents: Vec<Content>) -> Vec<Content> {
    let mut merged: Vec<Content> = Vec::with_capacity(contents.len());
    for c in contents {
        match merged.last_mut() {
            Some(prev) if prev.role == c.role => {
                prev.parts.extend(c.parts);
            }
            _ => merged.push(c),
        }
    }
    merged
}

fn extract_text(content: &Option<MessageContent>) -> Option<String> {
    match content {
        Some(MessageContent::Text(s)) => Some(s.clone()),
        Some(MessageContent::Parts(parts)) => {
            let texts: Vec<&str> = parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Known(TypedContentPart::Text { text, .. }) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join(""))
            }
        }
        None => None,
    }
}

// ─── Tool translation ──────────────────────────────────────────────────────

/// Bundle all canonical [`Tool`]s into a single Gemini [`NativeTool`] with
/// `function_declarations`. Gemini accepts multiple `Tool` objects, but
/// using a single one is the simplest and most common shape.
fn translate_tools_into_one(tools: &[Tool]) -> NativeTool {
    let decls: Vec<FunctionDeclaration> = tools
        .iter()
        .filter(|t| t.kind == "function")
        .map(|t| FunctionDeclaration {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            parameters: t.function.parameters.clone(),
            extra: serde_json::Map::new(),
        })
        .collect();
    NativeTool {
        function_declarations: if decls.is_empty() { None } else { Some(decls) },
        google_search: None,
        code_execution: None,
        url_context: None,
        extra: serde_json::Map::new(),
    }
}

fn translate_tool_choice(tc: &ToolChoice) -> ToolConfig {
    let (mode, allowed) = match tc {
        ToolChoice::Mode(ToolChoiceMode::None) => (Some(FunctionCallingMode::None), None),
        ToolChoice::Mode(ToolChoiceMode::Auto) => (Some(FunctionCallingMode::Auto), None),
        ToolChoice::Mode(ToolChoiceMode::Required) => (Some(FunctionCallingMode::Any), None),
        ToolChoice::Mode(ToolChoiceMode::Unknown(_)) => (None, None),
        ToolChoice::Named(named) => (
            Some(FunctionCallingMode::Any),
            Some(vec![named.function.name.clone()]),
        ),
        ToolChoice::Raw(_) => (None, None),
    };
    ToolConfig {
        function_calling_config: Some(FunctionCallingConfig {
            mode,
            allowed_function_names: allowed,
        }),
    }
}

// ─── Generation config ─────────────────────────────────────────────────────

fn build_generation_config(
    req: &ChatRequest,
    thinking: &dyn ThinkingProjector<GeminiThinkingTarget>,
) -> Option<GenerationConfig> {
    let mut cfg = GenerationConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        max_output_tokens: req.max_tokens,
        stop_sequences: req.stop.as_ref().map(|s| s.to_vec()),
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        seed: req.seed,
        ..Default::default()
    };

    // response_format → response_mime_type / response_schema.
    if let Some(rf) = &req.response_format {
        match rf {
            ResponseFormat::Text { .. } => {
                cfg.response_mime_type = Some("text/plain".into());
            }
            ResponseFormat::JsonObject { .. } => {
                cfg.response_mime_type = Some("application/json".into());
            }
            ResponseFormat::JsonSchema { json_schema, .. } => {
                cfg.response_mime_type = Some("application/json".into());
                cfg.response_schema = json_schema.schema.clone();
            }
        }
    }

    // thinking via projector.
    let mut target = GeminiThinkingTarget::default();
    if req.thinking.is_some() {
        thinking.apply(&req.model, req.thinking.as_ref(), &mut target);
    }
    cfg.thinking_config = target.config;

    // Omit if everything is None (no point sending an empty object).
    if generation_config_is_empty(&cfg) {
        None
    } else {
        Some(cfg)
    }
}

fn generation_config_is_empty(cfg: &GenerationConfig) -> bool {
    cfg.temperature.is_none()
        && cfg.top_p.is_none()
        && cfg.top_k.is_none()
        && cfg.candidate_count.is_none()
        && cfg.max_output_tokens.is_none()
        && cfg.stop_sequences.is_none()
        && cfg.presence_penalty.is_none()
        && cfg.frequency_penalty.is_none()
        && cfg.response_mime_type.is_none()
        && cfg.response_schema.is_none()
        && cfg.response_modalities.is_none()
        && cfg.seed.is_none()
        && cfg.response_logprobs.is_none()
        && cfg.logprobs.is_none()
        && cfg.thinking_config.is_none()
        && cfg.extra.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aigw_core::model::{
        ChatRequest, FunctionCall, FunctionDefinition, ImageUrl, JsonSchema, Message,
        MessageContent, ThinkingLevel, ThinkingRequest,
    };

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

    fn assistant_msg(text: &str) -> Message {
        Message {
            role: Role::Assistant,
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

    fn translator() -> GeminiRequestTranslator {
        GeminiRequestTranslator::new(&SecretString::from("AIza-test")).unwrap()
    }

    fn translate(req: &ChatRequest) -> GenerateContentRequest {
        let projector = GeminiThinkingProjector::default();
        build_body(req, &projector).unwrap()
    }

    #[test]
    fn url_for_non_streaming() {
        let t = translator();
        assert_eq!(
            t.url("gemini-2.5-flash", false),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
    }

    #[test]
    fn url_for_streaming() {
        let t = translator();
        assert_eq!(
            t.url("gemini-2.5-flash", true),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn translator_attaches_api_key_header() {
        let t = translator();
        assert_eq!(
            t.headers
                .get("x-goog-api-key")
                .unwrap()
                .to_str()
                .unwrap(),
            "AIza-test"
        );
    }

    #[test]
    fn system_message_becomes_system_instruction() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![system_msg("You are helpful."), user_msg("Hi")])
            .build();
        let body = translate(&req);
        let sys = body.system_instruction.unwrap();
        assert!(sys.role.is_none());
        assert_eq!(sys.parts[0].text.as_deref(), Some("You are helpful."));
    }

    #[test]
    fn multiple_system_messages_join_with_double_newline() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![
                system_msg("Be terse."),
                system_msg("Be polite."),
                user_msg("Hi"),
            ])
            .build();
        let body = translate(&req);
        assert_eq!(
            body.system_instruction.unwrap().parts[0].text.as_deref(),
            Some("Be terse.\n\nBe polite.")
        );
    }

    #[test]
    fn assistant_role_maps_to_model() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi"), assistant_msg("hello")])
            .build();
        let body = translate(&req);
        assert_eq!(body.contents.len(), 2);
        assert_eq!(body.contents[1].role, Some(NativeRole::Model));
    }

    #[test]
    fn adjacent_user_messages_merged() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("first"), user_msg("second")])
            .build();
        let body = translate(&req);
        assert_eq!(body.contents.len(), 1);
        assert_eq!(body.contents[0].parts.len(), 2);
        assert_eq!(body.contents[0].parts[0].text.as_deref(), Some("first"));
        assert_eq!(body.contents[0].parts[1].text.as_deref(), Some("second"));
    }

    #[test]
    fn assistant_with_tool_calls_translates_to_function_call_parts() {
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
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("weather?"), msg])
            .build();
        let body = translate(&req);
        let model_parts = &body.contents[1].parts;
        assert_eq!(model_parts.len(), 2);
        assert_eq!(model_parts[0].text.as_deref(), Some("Let me check."));
        let fc = model_parts[1].function_call.as_ref().unwrap();
        assert_eq!(fc.name, "get_weather");
        assert_eq!(fc.id.as_deref(), Some("call_1"));
        assert_eq!(fc.args["location"], "SF");
    }

    #[test]
    fn tool_role_becomes_function_response_user_part() {
        // Build a 3-message sequence: user → assistant(tool_call) → tool result
        let user = user_msg("weather?");
        let assistant = Message {
            role: Role::Assistant,
            content: None,
            name: None,
            tool_call_id: None,
            tool_calls: Some(vec![ToolCall {
                id: "c1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: "{}".into(),
                    extra: Default::default(),
                },
                extra: Default::default(),
            }]),
            extra: Default::default(),
        };
        let tool = Message {
            role: Role::Tool,
            content: Some(MessageContent::Text(r#"{"temp":72}"#.into())),
            name: None,
            tool_call_id: Some("c1".into()),
            tool_calls: None,
            extra: Default::default(),
        };
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user, assistant, tool])
            .build();
        let body = translate(&req);
        // The tool message becomes a user-role content with a functionResponse part.
        let tool_content = body.contents.last().unwrap();
        assert_eq!(tool_content.role, Some(NativeRole::User));
        let fr = tool_content.parts[0].function_response.as_ref().unwrap();
        assert_eq!(fr.name, "get_weather"); // resolved via tool_call_names
        assert_eq!(fr.id.as_deref(), Some("c1"));
        assert_eq!(fr.response["temp"], 72);
    }

    #[test]
    fn tools_translated_to_function_declarations() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi")])
            .tools(vec![Tool {
                kind: "function".into(),
                function: FunctionDefinition {
                    name: "search".into(),
                    description: Some("Search the web".into()),
                    parameters: Some(serde_json::json!({"type":"object"})),
                    strict: None,
                    extra: Default::default(),
                },
                extra: Default::default(),
            }])
            .build();
        let body = translate(&req);
        let tools = body.tools.unwrap();
        assert_eq!(tools.len(), 1);
        let decls = tools[0].function_declarations.as_ref().unwrap();
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].name, "search");
        assert_eq!(decls[0].description.as_deref(), Some("Search the web"));
    }

    #[test]
    fn tool_choice_required_maps_to_any() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi")])
            .tool_choice(ToolChoice::Mode(ToolChoiceMode::Required))
            .build();
        let body = translate(&req);
        let cfg = body.tool_config.unwrap().function_calling_config.unwrap();
        assert!(matches!(cfg.mode, Some(FunctionCallingMode::Any)));
    }

    #[test]
    fn tool_choice_named_sets_allowed_function_names() {
        use aigw_core::model::{NamedToolChoice, NamedToolChoiceFunction};
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi")])
            .tool_choice(ToolChoice::Named(NamedToolChoice {
                kind: "function".into(),
                function: NamedToolChoiceFunction {
                    name: "search".into(),
                    extra: Default::default(),
                },
                extra: Default::default(),
            }))
            .build();
        let body = translate(&req);
        let cfg = body.tool_config.unwrap().function_calling_config.unwrap();
        assert!(matches!(cfg.mode, Some(FunctionCallingMode::Any)));
        assert_eq!(cfg.allowed_function_names.unwrap(), vec!["search"]);
    }

    #[test]
    fn generation_config_carries_basic_params() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi")])
            .temperature(0.7)
            .max_tokens(1024_u64)
            .top_p(0.9)
            .build();
        let body = translate(&req);
        let cfg = body.generation_config.unwrap();
        assert_eq!(cfg.temperature, Some(0.7));
        assert_eq!(cfg.max_output_tokens, Some(1024));
        assert_eq!(cfg.top_p, Some(0.9));
    }

    #[test]
    fn json_object_response_format_sets_mime_type() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi")])
            .response_format(ResponseFormat::JsonObject {
                extra: Default::default(),
            })
            .build();
        let body = translate(&req);
        let cfg = body.generation_config.unwrap();
        assert_eq!(cfg.response_mime_type.as_deref(), Some("application/json"));
        assert!(cfg.response_schema.is_none());
    }

    #[test]
    fn json_schema_response_format_sets_mime_and_schema() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi")])
            .response_format(ResponseFormat::JsonSchema {
                json_schema: JsonSchema {
                    name: "person".into(),
                    description: None,
                    schema: Some(serde_json::json!({"type":"object"})),
                    strict: None,
                    extra: Default::default(),
                },
                extra: Default::default(),
            })
            .build();
        let body = translate(&req);
        let cfg = body.generation_config.unwrap();
        assert_eq!(cfg.response_mime_type.as_deref(), Some("application/json"));
        assert_eq!(cfg.response_schema.unwrap()["type"], "object");
    }

    #[test]
    fn canonical_thinking_writes_thinking_config() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi")])
            .thinking(ThinkingRequest::Budget {
                budget_tokens: 4096,
            })
            .build();
        let body = translate(&req);
        let cfg = body.generation_config.unwrap();
        assert_eq!(cfg.thinking_config.unwrap().thinking_budget, Some(4096));
    }

    #[test]
    fn canonical_thinking_disabled_emits_budget_zero() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi")])
            .thinking(ThinkingRequest::Disabled)
            .build();
        let body = translate(&req);
        let cfg = body.generation_config.unwrap();
        assert_eq!(cfg.thinking_config.unwrap().thinking_budget, Some(0));
    }

    #[test]
    fn canonical_level_g3_uses_thinking_level() {
        let req = ChatRequest::builder()
            .model("gemini-3-pro")
            .messages(vec![user_msg("hi")])
            .thinking(ThinkingRequest::Level {
                level: ThinkingLevel::High,
            })
            .build();
        let body = translate(&req);
        let cfg = body.generation_config.unwrap();
        let tc = cfg.thinking_config.unwrap();
        assert!(tc.thinking_budget.is_none());
        assert_eq!(
            tc.thinking_level,
            Some(crate::types::ThinkingLevel::High)
        );
    }

    #[test]
    fn n_greater_than_one_rejected() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi")])
            .n(2_u32)
            .build();
        let projector = GeminiThinkingProjector::default();
        let err = build_body(&req, &projector).unwrap_err();
        assert!(matches!(err, TranslateError::UnsupportedFeature { .. }));
    }

    #[test]
    fn extra_thinking_key_dropped_canonical_takes_priority() {
        // extra["thinking"] is a legacy passthrough — when canonical
        // req.thinking is present the extra version must not leak through.
        let mut extra = serde_json::Map::new();
        extra.insert(
            "thinking".into(),
            serde_json::json!({"thinkingBudget":99999}),
        );
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi")])
            .thinking(ThinkingRequest::Budget {
                budget_tokens: 1024,
            })
            .extra(extra)
            .build();
        let body = translate(&req);
        let cfg = body.generation_config.clone().unwrap();
        assert_eq!(cfg.thinking_config.unwrap().thinking_budget, Some(1024));
        // And the wire body shouldn't carry a stray "thinking" passthrough.
        let wire = serde_json::to_value(&body).unwrap();
        assert!(wire.get("thinking").is_none());
    }

    #[test]
    fn image_url_translates_to_file_data_part() {
        let part = translate_image("https://example.com/cat.jpg").unwrap();
        let fd = part.file_data.unwrap();
        assert_eq!(fd.file_uri, "https://example.com/cat.jpg");
    }

    #[test]
    fn image_data_uri_translates_to_inline_data() {
        let part = translate_image("data:image/png;base64,iVBOR").unwrap();
        let blob = part.inline_data.unwrap();
        assert_eq!(blob.mime_type, "image/png");
        assert_eq!(blob.data, "iVBOR");
    }

    #[test]
    fn gemini_sourced_thinking_part_round_trips() {
        let part_in = ContentPart::Known(TypedContentPart::Thinking {
            thinking: "considering...".into(),
            signature: "sig123".into(),
            source: Some(ThinkingSource::Gemini),
            extra: Default::default(),
        });
        let part_out = translate_content_part(&part_in).unwrap().unwrap();
        assert_eq!(part_out.text.as_deref(), Some("considering..."));
        assert_eq!(part_out.thought, Some(true));
        assert_eq!(part_out.thought_signature.as_deref(), Some("sig123"));
    }

    #[test]
    fn anthropic_sourced_thinking_part_dropped() {
        let part_in = ContentPart::Known(TypedContentPart::Thinking {
            thinking: "from claude".into(),
            signature: "ErWj123".into(),
            source: Some(ThinkingSource::Anthropic),
            extra: Default::default(),
        });
        let part_out = translate_content_part(&part_in).unwrap();
        assert!(part_out.is_none());
    }

    #[test]
    fn user_image_in_multipart_translates() {
        let user = Message {
            role: Role::User,
            content: Some(MessageContent::Parts(vec![
                ContentPart::Known(TypedContentPart::Text {
                    text: "what's this?".into(),
                    extra: Default::default(),
                }),
                ContentPart::Known(TypedContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/x.png".into(),
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
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user])
            .build();
        let body = translate(&req);
        let parts = &body.contents[0].parts;
        assert_eq!(parts.len(), 2);
        assert!(parts[0].text.is_some());
        assert!(parts[1].file_data.is_some());
    }

    #[test]
    fn translate_request_produces_correct_url_and_method() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi")])
            .build();
        let t = translator();
        let translated = t.translate_request(&req).unwrap();
        assert_eq!(translated.method, Method::POST);
        assert!(translated.url.contains("gemini-2.5-flash:generateContent"));
        let body: serde_json::Value = serde_json::from_slice(&translated.body).unwrap();
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hi");
    }

    #[test]
    fn translate_stream_request_uses_stream_endpoint() {
        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![user_msg("hi")])
            .build();
        let t = translator();
        let translated = t.translate_stream_request(&req).unwrap();
        assert!(
            translated
                .url
                .contains("gemini-2.5-flash:streamGenerateContent?alt=sse")
        );
    }
}
