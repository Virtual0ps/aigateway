//! Native-protocol bridge — the inverse direction of
//! [`GeminiRequestTranslator`] / [`GeminiResponseTranslator`].
//!
//! [`GeminiRequestTranslator`] takes a canonical [`ChatRequest`] and emits a
//! Gemini-native [`GenerateContentRequest`]; this module goes the other
//! way, so a gateway can serve clients that speak the Gemini-native API
//! while routing to a non-Gemini backend (e.g. an OpenAI Chat Completions
//! upstream).
//!
//! Three helpers cover the three directions:
//!
//! - [`gemini_request_to_canonical`] — Gemini-native [`GenerateContentRequest`]
//!   → canonical [`ChatRequest`]. Used by the receiving end of a
//!   `generateContent` HTTP call.
//! - [`chat_response_to_gemini`] — canonical [`ChatResponse`] →
//!   Gemini-native [`GenerateContentResponse`]. Used to format a
//!   non-streaming response back to the client.
//! - [`stream_event_to_gemini_sse`] — single canonical [`StreamEvent`] →
//!   Gemini-native SSE chunk bytes. Stateful via [`SseContext`] so
//!   per-stream metadata (response id, model name, accumulated tool call
//!   args) survives across events.
//!
//! [`ChatRequest`]: aigw_core::model::ChatRequest
//! [`ChatResponse`]: aigw_core::model::ChatResponse
//! [`StreamEvent`]: aigw_core::model::StreamEvent
//! [`GenerateContentRequest`]: crate::types::GenerateContentRequest
//! [`GenerateContentResponse`]: crate::types::GenerateContentResponse
//! [`GeminiRequestTranslator`]: super::request::GeminiRequestTranslator
//! [`GeminiResponseTranslator`]: super::response::GeminiResponseTranslator

use std::collections::{HashMap, VecDeque};

use aigw_core::ForwardCompatible;
use aigw_core::error::TranslateError;
use aigw_core::model::{
    ChatRequest, ChatResponse, ContentPart, FinishReason as CanonicalFinishReason, FunctionCall,
    ImageUrl, JsonSchema, Message, MessageContent, NamedToolChoice, NamedToolChoiceFunction,
    ResponseFormat, Role, StreamEvent, ThinkingLevel as CanonicalThinkingLevel, ThinkingRequest,
    ThinkingSource, Tool as CanonicalTool, ToolCall, ToolChoice, ToolChoiceMode, TypedContentPart,
    Usage,
};
use serde_json::{Value, json};

use crate::types::{
    Candidate, Content, FinishReason as NativeFinishReason, FunctionCall as NativeFunctionCall,
    FunctionCallingConfig, FunctionCallingMode, FunctionResponse, GenerateContentRequest,
    GenerateContentResponse, GenerationConfig, Part, Role as NativeRole,
    ThinkingConfig as NativeThinkingConfig, ThinkingLevel as NativeThinkingLevel, ToolConfig,
    UsageMetadata,
};

// ─── Gemini → canonical request ────────────────────────────────────────────

/// Convert a Gemini-native [`GenerateContentRequest`] into a canonical
/// [`ChatRequest`].
///
/// Used by gateways that accept Gemini-native traffic and forward it to a
/// non-Gemini backend (e.g. OpenAI Chat Completions). The inverse of
/// [`build_generate_content_request`](super::request::build_generate_content_request).
///
/// # Errors
///
/// Currently always returns `Ok` — the conversion is total. The
/// `Result` is kept for forward-compatibility (future schema additions
/// may surface errors).
pub fn gemini_request_to_canonical(
    req: GenerateContentRequest,
) -> Result<ChatRequest, TranslateError> {
    reject_unsupported_native_request_features(&req)?;

    let mut messages: Vec<Message> = Vec::new();
    let mut pending_tool_call_ids: HashMap<String, VecDeque<String>> = HashMap::new();

    // System instruction → role: System message at the front.
    if let Some(sys) = req.system_instruction {
        let texts: Vec<String> = sys
            .parts
            .into_iter()
            .filter_map(|p| p.text)
            .collect::<Vec<_>>();
        if !texts.is_empty() {
            messages.push(Message {
                role: Role::System,
                content: Some(MessageContent::Text(texts.join("\n\n"))),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                extra: Default::default(),
            });
        }
    }

    // Walk contents.
    for content in req.contents {
        let role = native_role_to_canonical(content.role.as_ref());
        let parts = content.parts;
        match role {
            Role::Tool => {
                // Gemini uses `user` role with functionResponse parts; we
                // surface each functionResponse as its own canonical
                // tool-role message so the canonical contract holds.
                for p in parts {
                    if let Some(fr) = p.function_response {
                        messages.push(message_from_function_response(
                            fr,
                            &mut pending_tool_call_ids,
                        ));
                    }
                }
            }
            Role::Assistant => {
                let (content_field, tool_calls) = split_assistant_parts(parts);
                remember_pending_tool_calls(&tool_calls, &mut pending_tool_call_ids);
                messages.push(Message {
                    role: Role::Assistant,
                    content: content_field,
                    name: None,
                    tool_call_id: None,
                    tool_calls,
                    extra: Default::default(),
                });
            }
            // User (default) — function_response parts on a "user"-role
            // turn become tool-role canonical messages too (modern Gemini
            // 2.5+ wire convention).
            _ => {
                let (tool_results, user_parts) = partition_function_responses(parts);
                for fr in tool_results {
                    messages.push(message_from_function_response(
                        fr,
                        &mut pending_tool_call_ids,
                    ));
                }
                if !user_parts.is_empty() {
                    let content_field = user_parts_to_content(user_parts);
                    messages.push(Message {
                        role: Role::User,
                        content: content_field,
                        name: None,
                        tool_call_id: None,
                        tool_calls: None,
                        extra: Default::default(),
                    });
                }
            }
        }
    }

    let tool_config = req.tool_config;
    let allowed_function_names = constrained_allowed_function_names(tool_config.as_ref());

    // Tools.
    let tools: Option<Vec<CanonicalTool>> = req.tools.map(|gemini_tools| {
        gemini_tools
            .into_iter()
            .flat_map(|t| t.function_declarations.unwrap_or_default())
            .filter(|d| match allowed_function_names {
                Some(allowed) => allowed.iter().any(|name| name == &d.name),
                None => true,
            })
            .map(|d| CanonicalTool {
                kind: "function".to_owned(),
                function: aigw_core::model::FunctionDefinition {
                    name: d.name,
                    description: d.description,
                    parameters: d.parameters.as_ref().map(schema_types_to_canonical),
                    strict: None,
                    extra: Default::default(),
                },
                extra: Default::default(),
            })
            .collect()
    });

    // Tool choice.
    let tool_choice = tool_config.and_then(|tc| {
        tc.function_calling_config
            .map(native_tool_choice_to_canonical)
    });

    // Generation config.
    let (
        temperature,
        top_p,
        max_tokens,
        stop,
        seed,
        frequency_penalty,
        presence_penalty,
        response_format,
        thinking,
    ) = req.generation_config.as_ref().map_or(
        (None, None, None, None, None, None, None, None, None),
        |g| {
            (
                g.temperature,
                g.top_p,
                g.max_output_tokens,
                g.stop_sequences.clone().map(|v| {
                    if v.len() == 1 {
                        aigw_core::OneOrMany::One(v.into_iter().next().unwrap())
                    } else {
                        aigw_core::OneOrMany::Many(v)
                    }
                }),
                g.seed,
                g.frequency_penalty,
                g.presence_penalty,
                native_response_format(g),
                native_thinking_to_canonical(g.thinking_config.as_ref()),
            )
        },
    );

    Ok(ChatRequest::builder()
        .model(req.model)
        .messages(messages)
        .maybe_temperature(temperature)
        .maybe_top_p(top_p)
        .maybe_max_tokens(max_tokens)
        .maybe_stop(stop)
        .maybe_tools(tools)
        .maybe_tool_choice(tool_choice)
        .maybe_response_format(response_format)
        .maybe_frequency_penalty(frequency_penalty)
        .maybe_presence_penalty(presence_penalty)
        .maybe_seed(seed)
        .maybe_thinking(thinking)
        .build())
}

fn reject_unsupported_native_request_features(
    req: &GenerateContentRequest,
) -> Result<(), TranslateError> {
    if req.cached_content.is_some() {
        return Err(TranslateError::UnsupportedFeature {
            provider: "canonical",
            feature: "Gemini cachedContent cannot be represented for non-Gemini backends"
                .to_owned(),
        });
    }

    if req
        .safety_settings
        .as_ref()
        .is_some_and(|settings| !settings.is_empty())
    {
        return Err(TranslateError::UnsupportedFeature {
            provider: "canonical",
            feature: "Gemini safetySettings cannot be represented for non-Gemini backends"
                .to_owned(),
        });
    }

    for tool in req.tools.iter().flatten() {
        let unsupported = unsupported_native_tool_features(tool);
        if !unsupported.is_empty() {
            return Err(TranslateError::UnsupportedFeature {
                provider: "canonical",
                feature: format!(
                    "Gemini built-in tool(s) cannot be represented for non-Gemini backends: {}",
                    unsupported.join(", ")
                ),
            });
        }
    }

    Ok(())
}

fn unsupported_native_tool_features(tool: &crate::types::Tool) -> Vec<String> {
    let mut unsupported = Vec::new();
    if tool.google_search.is_some() {
        unsupported.push("googleSearch".to_owned());
    }
    if tool.code_execution.is_some() {
        unsupported.push("codeExecution".to_owned());
    }
    if tool.url_context.is_some() {
        unsupported.push("urlContext".to_owned());
    }
    unsupported.extend(tool.extra.keys().cloned());
    unsupported
}

fn constrained_allowed_function_names(tool_config: Option<&ToolConfig>) -> Option<&[String]> {
    let fcc = tool_config?.function_calling_config.as_ref()?;
    match fcc.mode {
        Some(FunctionCallingMode::Any | FunctionCallingMode::Validated) => fcc
            .allowed_function_names
            .as_deref()
            .filter(|names| !names.is_empty()),
        _ => None,
    }
}

fn native_tool_choice_to_canonical(fcc: FunctionCallingConfig) -> ToolChoice {
    match fcc.mode {
        Some(FunctionCallingMode::None) => ToolChoice::Mode(ToolChoiceMode::None),
        Some(FunctionCallingMode::Auto) | None => ToolChoice::Mode(ToolChoiceMode::Auto),
        Some(FunctionCallingMode::Any) => match fcc.allowed_function_names.as_deref() {
            Some([name]) => ToolChoice::Named(NamedToolChoice {
                kind: "function".to_owned(),
                function: NamedToolChoiceFunction {
                    name: name.clone(),
                    extra: Default::default(),
                },
                extra: Default::default(),
            }),
            _ => ToolChoice::Mode(ToolChoiceMode::Required),
        },
        Some(FunctionCallingMode::Validated) => ToolChoice::Mode(ToolChoiceMode::Auto),
        Some(FunctionCallingMode::Unknown(_)) => ToolChoice::Mode(ToolChoiceMode::Auto),
    }
}

fn native_response_format(g: &GenerationConfig) -> Option<ResponseFormat> {
    match g.response_mime_type.as_deref() {
        Some("application/json") => {
            if let Some(schema) = &g.response_schema {
                Some(ResponseFormat::JsonSchema {
                    json_schema: JsonSchema {
                        name: "response".to_owned(),
                        description: None,
                        schema: Some(schema_types_to_canonical(schema)),
                        strict: None,
                        extra: Default::default(),
                    },
                    extra: Default::default(),
                })
            } else {
                Some(ResponseFormat::JsonObject {
                    extra: Default::default(),
                })
            }
        }
        Some("text/plain") => Some(ResponseFormat::Text {
            extra: Default::default(),
        }),
        _ => None,
    }
}

fn schema_types_to_canonical(schema: &Value) -> Value {
    transform_schema_types(schema, |t| match t {
        "OBJECT" => "object",
        "STRING" => "string",
        "NUMBER" => "number",
        "INTEGER" => "integer",
        "BOOLEAN" => "boolean",
        "ARRAY" => "array",
        "NULL" => "null",
        other => other,
    })
}

fn transform_schema_types(schema: &Value, map_type: fn(&str) -> &str) -> Value {
    match schema {
        Value::Object(obj) => {
            let mut out = serde_json::Map::with_capacity(obj.len());
            for (key, value) in obj {
                let mapped = if key == "type" {
                    match value {
                        Value::String(t) => Value::String(map_type(t).to_owned()),
                        Value::Array(types) => Value::Array(
                            types
                                .iter()
                                .map(|v| match v {
                                    Value::String(t) => Value::String(map_type(t).to_owned()),
                                    other => transform_schema_types(other, map_type),
                                })
                                .collect(),
                        ),
                        other => transform_schema_types(other, map_type),
                    }
                } else {
                    transform_schema_types(value, map_type)
                };
                out.insert(key.clone(), mapped);
            }
            Value::Object(out)
        }
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(|v| transform_schema_types(v, map_type))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn native_thinking_to_canonical(
    thinking: Option<&NativeThinkingConfig>,
) -> Option<ThinkingRequest> {
    let thinking = thinking?;
    if let Some(budget) = thinking.thinking_budget {
        return match budget {
            0 => Some(ThinkingRequest::Disabled),
            -1 => Some(ThinkingRequest::Auto),
            n if n > 0 => Some(ThinkingRequest::Budget {
                budget_tokens: u32::try_from(n).unwrap_or(u32::MAX),
            }),
            _ => None,
        };
    }

    thinking
        .thinking_level
        .as_ref()
        .and_then(native_level_to_canonical)
        .map(|level| ThinkingRequest::Level { level })
}

fn native_level_to_canonical(level: &NativeThinkingLevel) -> Option<CanonicalThinkingLevel> {
    match level {
        NativeThinkingLevel::Minimal => Some(CanonicalThinkingLevel::Minimal),
        NativeThinkingLevel::Low => Some(CanonicalThinkingLevel::Low),
        NativeThinkingLevel::Medium => Some(CanonicalThinkingLevel::Medium),
        NativeThinkingLevel::High => Some(CanonicalThinkingLevel::High),
        NativeThinkingLevel::Other(s) => match s.to_ascii_uppercase().as_str() {
            "MINIMAL" => Some(CanonicalThinkingLevel::Minimal),
            "LOW" => Some(CanonicalThinkingLevel::Low),
            "MEDIUM" => Some(CanonicalThinkingLevel::Medium),
            "HIGH" => Some(CanonicalThinkingLevel::High),
            "XHIGH" | "X_HIGH" => Some(CanonicalThinkingLevel::XHigh),
            "MAX" => Some(CanonicalThinkingLevel::Max),
            _ => None,
        },
    }
}

fn native_role_to_canonical(role: Option<&NativeRole>) -> Role {
    match role {
        Some(NativeRole::User) | None => Role::User,
        Some(NativeRole::Model) => Role::Assistant,
        Some(NativeRole::Function) => Role::Tool,
        Some(NativeRole::Other(s)) => Role::Unknown(s.clone()),
    }
}

fn remember_pending_tool_calls(
    tool_calls: &Option<Vec<ToolCall>>,
    pending_tool_call_ids: &mut HashMap<String, VecDeque<String>>,
) {
    if let Some(tool_calls) = tool_calls {
        for tool_call in tool_calls {
            pending_tool_call_ids
                .entry(tool_call.function.name.clone())
                .or_default()
                .push_back(tool_call.id.clone());
        }
    }
}

fn message_from_function_response(
    fr: FunctionResponse,
    pending_tool_call_ids: &mut HashMap<String, VecDeque<String>>,
) -> Message {
    let tool_call_id = match fr.id {
        Some(id) => {
            if let Some(ids) = pending_tool_call_ids.get_mut(&fr.name)
                && let Some(pos) = ids.iter().position(|pending_id| pending_id == &id)
            {
                ids.remove(pos);
            }
            Some(id)
        }
        None => pending_tool_call_ids
            .get_mut(&fr.name)
            .and_then(VecDeque::pop_front),
    };
    let content = match fr.response {
        Value::String(s) => s,
        v => v.to_string(),
    };
    Message {
        role: Role::Tool,
        content: Some(MessageContent::Text(content)),
        name: Some(fr.name),
        tool_call_id,
        tool_calls: None,
        extra: Default::default(),
    }
}

fn partition_function_responses(parts: Vec<Part>) -> (Vec<FunctionResponse>, Vec<Part>) {
    let mut tool_results = Vec::new();
    let mut other = Vec::new();
    for p in parts {
        if let Some(fr) = p.function_response {
            tool_results.push(fr);
        } else {
            other.push(p);
        }
    }
    (tool_results, other)
}

fn user_parts_to_content(parts: Vec<Part>) -> Option<MessageContent> {
    let mut out = Vec::new();
    let mut saw_non_text = false;
    let mut text_pieces = Vec::new();

    for part in parts {
        let is_plain_text = part.text.is_some()
            && part.inline_data.is_none()
            && part.file_data.is_none()
            && part.function_call.is_none()
            && part.function_response.is_none()
            && !part.thought.unwrap_or(false);

        if is_plain_text {
            let text = part.text.unwrap_or_default();
            text_pieces.push(text.clone());
            out.push(ForwardCompatible::Known(TypedContentPart::Text {
                text,
                extra: Default::default(),
            }));
            continue;
        }

        if let Some(content_part) = native_part_to_content_part(part) {
            saw_non_text = true;
            out.push(content_part);
        }
    }

    if out.is_empty() {
        None
    } else if saw_non_text {
        Some(MessageContent::Parts(out))
    } else {
        Some(MessageContent::Text(text_pieces.join("")))
    }
}

fn native_part_to_content_part(part: Part) -> Option<ContentPart> {
    let is_thought = part.thought.unwrap_or(false);
    if is_thought {
        return Some(ForwardCompatible::Known(TypedContentPart::Thinking {
            thinking: part.text.unwrap_or_default(),
            signature: part.thought_signature.unwrap_or_default(),
            source: Some(ThinkingSource::Gemini),
            extra: Default::default(),
        }));
    }

    if let Some(blob) = part.inline_data {
        if blob.mime_type.starts_with("image/") {
            return Some(ForwardCompatible::Known(TypedContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: format!("data:{};base64,{}", blob.mime_type, blob.data),
                    detail: None,
                    extra: Default::default(),
                },
                extra: Default::default(),
            }));
        }
        return Some(ForwardCompatible::Known(TypedContentPart::File {
            file: json!({
                "mime_type": blob.mime_type,
                "data": blob.data,
            }),
            extra: Default::default(),
        }));
    }

    if let Some(file) = part.file_data {
        if file.mime_type.starts_with("image/") {
            return Some(ForwardCompatible::Known(TypedContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: file.file_uri,
                    detail: None,
                    extra: Default::default(),
                },
                extra: Default::default(),
            }));
        }
        return Some(ForwardCompatible::Known(TypedContentPart::File {
            file: json!({
                "mime_type": file.mime_type,
                "file_uri": file.file_uri,
            }),
            extra: Default::default(),
        }));
    }

    None
}

fn split_assistant_parts(parts: Vec<Part>) -> (Option<MessageContent>, Option<Vec<ToolCall>>) {
    let mut content_parts: Vec<ContentPart> = Vec::new();
    let mut text_pieces: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for part in parts {
        let is_thought = part.thought.unwrap_or(false);
        if let Some(fc) = part.function_call {
            let id = fc
                .id
                .unwrap_or_else(|| format!("call_{}", tool_calls.len()));
            let arguments = serde_json::to_string(&fc.args).unwrap_or_else(|_| "{}".to_owned());
            tool_calls.push(ToolCall {
                id,
                kind: "function".to_owned(),
                function: FunctionCall {
                    name: fc.name,
                    arguments,
                    extra: Default::default(),
                },
                extra: Default::default(),
            });
        } else if is_thought {
            content_parts.push(ForwardCompatible::Known(TypedContentPart::Thinking {
                thinking: part.text.unwrap_or_default(),
                signature: part.thought_signature.unwrap_or_default(),
                source: Some(ThinkingSource::Gemini),
                extra: Default::default(),
            }));
        } else if let Some(t) = part.text {
            text_pieces.push(t);
        }
    }

    let content = if content_parts.is_empty() {
        if text_pieces.is_empty() {
            None
        } else {
            Some(MessageContent::Text(text_pieces.join("")))
        }
    } else {
        if !text_pieces.is_empty() {
            content_parts.push(ForwardCompatible::Known(TypedContentPart::Text {
                text: text_pieces.join(""),
                extra: Default::default(),
            }));
        }
        Some(MessageContent::Parts(content_parts))
    };

    let tool_calls_opt = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };
    (content, tool_calls_opt)
}

// ─── Canonical → Gemini response ───────────────────────────────────────────

/// Convert a canonical [`ChatResponse`] into a Gemini-native
/// [`GenerateContentResponse`].
///
/// Used by gateways that route Gemini-native client requests to a
/// non-Gemini backend and need to format the canonical response back into
/// Gemini wire format.
///
/// # Errors
///
/// Currently always returns `Ok`. Reserved for forward-compatibility.
pub fn chat_response_to_gemini(
    resp: ChatResponse,
) -> Result<GenerateContentResponse, TranslateError> {
    let model = resp.model.clone();
    let usage = resp.usage.map(canonical_usage_to_native);

    let candidates = resp
        .choices
        .into_iter()
        .map(|c| {
            let parts = message_to_gemini_parts(&c.message);
            let content = if parts.is_empty() {
                None
            } else {
                Some(Content {
                    role: Some(NativeRole::Model),
                    parts,
                })
            };
            Candidate {
                content,
                finish_reason: c.finish_reason.map(canonical_finish_to_native),
                safety_ratings: Vec::new(),
                citation_metadata: None,
                grounding_metadata: None,
                index: Some(c.index),
                avg_logprobs: None,
                extra: Default::default(),
            }
        })
        .collect();

    Ok(GenerateContentResponse {
        candidates,
        prompt_feedback: None,
        usage_metadata: usage,
        model_version: Some(model),
        response_id: Some(resp.id),
        extra: Default::default(),
    })
}

fn message_to_gemini_parts(msg: &Message) -> Vec<Part> {
    let mut parts: Vec<Part> = Vec::new();
    match &msg.content {
        Some(MessageContent::Text(s)) if !s.is_empty() => parts.push(Part::text(s.clone())),
        Some(MessageContent::Parts(content_parts)) => {
            for cp in content_parts {
                if let ContentPart::Known(typed) = cp {
                    match typed {
                        TypedContentPart::Text { text, .. } => parts.push(Part::text(text)),
                        TypedContentPart::Thinking {
                            thinking,
                            signature,
                            ..
                        } => parts.push(Part {
                            text: Some(thinking.clone()),
                            thought: Some(true),
                            thought_signature: if signature.is_empty() {
                                None
                            } else {
                                Some(signature.clone())
                            },
                            ..Default::default()
                        }),
                        // ImageUrl/RedactedThinking/etc.: skip for MVP.
                        _ => {}
                    }
                }
            }
        }
        _ => {}
    }
    if let Some(tcs) = &msg.tool_calls {
        for tc in tcs {
            let args: Value = serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));
            parts.push(Part {
                function_call: Some(NativeFunctionCall {
                    name: tc.function.name.clone(),
                    args,
                    id: Some(tc.id.clone()),
                    extra: Default::default(),
                }),
                ..Default::default()
            });
        }
    }
    parts
}

fn canonical_finish_to_native(reason: CanonicalFinishReason) -> NativeFinishReason {
    match reason {
        CanonicalFinishReason::Stop => NativeFinishReason::Stop,
        CanonicalFinishReason::Length => NativeFinishReason::MaxTokens,
        CanonicalFinishReason::ToolCalls => NativeFinishReason::Stop,
        CanonicalFinishReason::ContentFilter => NativeFinishReason::Safety,
        CanonicalFinishReason::Unknown(s) => NativeFinishReason::Unknown(s),
    }
}

fn canonical_usage_to_native(u: Usage) -> UsageMetadata {
    UsageMetadata {
        prompt_token_count: u.prompt_tokens,
        candidates_token_count: u.completion_tokens,
        total_token_count: u.total_tokens,
        cached_content_token_count: u
            .extra
            .get("cached_content_token_count")
            .and_then(Value::as_u64),
        thoughts_token_count: u.extra.get("thoughts_token_count").and_then(Value::as_u64),
        extra: Default::default(),
    }
}

// ─── Stream → Gemini SSE ───────────────────────────────────────────────────

/// Per-stream state for the canonical → Gemini-native SSE bridge.
///
/// Holds the response identity captured from the first
/// [`StreamEvent::ResponseMeta`], plus partial tool-call argument buffers
/// (Gemini emits whole `functionCall` objects per chunk while canonical
/// streams emit name + incremental JSON deltas).
#[derive(Debug, Default)]
pub struct SseContext {
    /// Model name surfaced in each Gemini chunk's `modelVersion`.
    pub model: String,
    /// Response id captured from the first [`StreamEvent::ResponseMeta`].
    /// Surfaced in each Gemini chunk's `responseId`.
    pub response_id: String,
    /// Per-tool-call accumulated args (index → state).
    tool_calls: HashMap<u32, ToolCallBuf>,
    /// Active reasoning index, if any.
    open_reasoning: Option<u32>,
}

#[derive(Debug, Default)]
struct ToolCallBuf {
    id: String,
    name: String,
    arguments: String,
}

impl SseContext {
    /// Build a new context with a fixed model name (used as a fallback
    /// when the canonical stream's `ResponseMeta` doesn't include one).
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }
}

/// Convert a single canonical [`StreamEvent`] to Gemini-native SSE bytes.
///
/// Returns `None` when the event produces no client-visible output (e.g.
/// `ResponseMeta` only updates context; `ReasoningStart` carries no
/// payload distinct from the deltas that follow). The returned bytes
/// already include the `data: ` prefix and the `\r\n\r\n` separator.
///
/// The function is stateful via `ctx` because Gemini's wire format
/// differs from canonical streams in two ways:
/// 1. Each chunk carries the full `functionCall` object — canonical
///    streams emit name and incremental args separately, so we buffer
///    until a logical boundary (finish or the stream's `Done`) and flush
///    as a single Gemini chunk.
/// 2. Reasoning blocks have explicit start/end events on the canonical
///    side; on the Gemini side they're individual `thought=true` parts
///    with an optional `thoughtSignature`. We mirror Anthropic's bridge
///    convention: emit a thought part on each delta, attach the signature
///    on `ReasoningEnd`.
pub fn stream_event_to_gemini_sse(event: &StreamEvent, ctx: &mut SseContext) -> Option<Vec<u8>> {
    match event {
        StreamEvent::ResponseMeta { id, model } => {
            ctx.response_id = id.clone();
            if !model.is_empty() {
                ctx.model = model.clone();
            }
            None
        }

        StreamEvent::ContentDelta(text) if !text.is_empty() => {
            Some(emit_chunk(ctx, vec![Part::text(text)], None, None))
        }
        StreamEvent::ContentDelta(_) => None,

        StreamEvent::ReasoningStart { index, .. } => {
            ctx.open_reasoning = Some(*index);
            None
        }
        StreamEvent::ReasoningDelta(text) if !text.is_empty() => {
            let part = Part {
                text: Some(text.clone()),
                thought: Some(true),
                ..Default::default()
            };
            Some(emit_chunk(ctx, vec![part], None, None))
        }
        StreamEvent::ReasoningDelta(_) => None,
        StreamEvent::ReasoningEnd { signature, .. } => {
            ctx.open_reasoning = None;
            if signature.is_empty() {
                return None;
            }
            let part = Part {
                thought: Some(true),
                thought_signature: Some(signature.clone()),
                ..Default::default()
            };
            Some(emit_chunk(ctx, vec![part], None, None))
        }
        #[allow(deprecated)]
        StreamEvent::ReasoningSignature(signature) if !signature.is_empty() => {
            let part = Part {
                thought: Some(true),
                thought_signature: Some(signature.clone()),
                ..Default::default()
            };
            Some(emit_chunk(ctx, vec![part], None, None))
        }
        #[allow(deprecated)]
        StreamEvent::ReasoningSignature(_) => None,

        StreamEvent::ToolCallStart { index, id, name } => {
            ctx.tool_calls.insert(
                *index,
                ToolCallBuf {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                },
            );
            None
        }
        StreamEvent::ToolCallDelta { index, arguments } => {
            if let Some(buf) = ctx.tool_calls.get_mut(index) {
                buf.arguments.push_str(arguments);
            }
            None
        }

        StreamEvent::Finish(reason) => {
            // Drain any pending tool-call buffers as Gemini functionCall parts.
            let parts = drain_tool_call_parts(ctx);
            let native_reason = canonical_finish_to_native(reason.clone());
            Some(emit_chunk(ctx, parts, Some(native_reason), None))
        }

        StreamEvent::Usage(u) => {
            let usage = canonical_usage_to_native(u.clone());
            Some(emit_chunk(ctx, Vec::new(), None, Some(usage)))
        }

        // Gemini's streamGenerateContent doesn't emit a [DONE] sentinel —
        // the connection just closes. If an upstream parser only surfaced
        // tool-call deltas and then Done (with no Finish), flush the
        // buffered functionCall parts before suppressing the sentinel.
        StreamEvent::Done => {
            let parts = drain_tool_call_parts(ctx);
            if parts.is_empty() {
                None
            } else {
                Some(emit_chunk(ctx, parts, None, None))
            }
        }
    }
}

fn drain_tool_call_parts(ctx: &mut SseContext) -> Vec<Part> {
    let mut parts = Vec::new();
    let mut indices: Vec<u32> = ctx.tool_calls.keys().copied().collect();
    indices.sort_unstable();
    for idx in indices {
        if let Some(buf) = ctx.tool_calls.remove(&idx) {
            let args: Value = serde_json::from_str(&buf.arguments).unwrap_or(json!({}));
            parts.push(Part {
                function_call: Some(NativeFunctionCall {
                    name: buf.name,
                    args,
                    id: Some(buf.id),
                    extra: Default::default(),
                }),
                ..Default::default()
            });
        }
    }
    parts
}

fn emit_chunk(
    ctx: &SseContext,
    parts: Vec<Part>,
    finish_reason: Option<NativeFinishReason>,
    usage: Option<UsageMetadata>,
) -> Vec<u8> {
    let candidate = if parts.is_empty() && finish_reason.is_none() {
        Vec::new()
    } else {
        let content = if parts.is_empty() {
            None
        } else {
            Some(Content {
                role: Some(NativeRole::Model),
                parts,
            })
        };
        vec![Candidate {
            content,
            finish_reason,
            safety_ratings: Vec::new(),
            citation_metadata: None,
            grounding_metadata: None,
            index: Some(0),
            avg_logprobs: None,
            extra: Default::default(),
        }]
    };

    let chunk = GenerateContentResponse {
        candidates: candidate,
        prompt_feedback: None,
        usage_metadata: usage,
        model_version: if ctx.model.is_empty() {
            None
        } else {
            Some(ctx.model.clone())
        },
        response_id: if ctx.response_id.is_empty() {
            None
        } else {
            Some(ctx.response_id.clone())
        },
        extra: Default::default(),
    };

    let body = serde_json::to_string(&chunk).unwrap_or_default();
    format!("data: {body}\r\n\r\n").into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── gemini_request_to_canonical ──────────────────────────────────────

    #[test]
    fn gemini_request_with_user_message_round_trips() {
        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![Content {
                role: Some(NativeRole::User),
                parts: vec![Part::text("Hello")],
            }])
            .build();
        let canonical = gemini_request_to_canonical(native).unwrap();
        assert_eq!(canonical.model, "gemini-2.5-flash");
        assert_eq!(canonical.messages.len(), 1);
        assert_eq!(canonical.messages[0].role, Role::User);
        match &canonical.messages[0].content {
            Some(MessageContent::Text(s)) => assert_eq!(s, "Hello"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn gemini_request_system_instruction_becomes_system_message() {
        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![Content {
                role: Some(NativeRole::User),
                parts: vec![Part::text("hi")],
            }])
            .system_instruction(Content {
                role: None,
                parts: vec![Part::text("You are helpful.")],
            })
            .build();
        let canonical = gemini_request_to_canonical(native).unwrap();
        assert_eq!(canonical.messages[0].role, Role::System);
        assert!(matches!(
            canonical.messages[0].content,
            Some(MessageContent::Text(ref s)) if s == "You are helpful."
        ));
    }

    #[test]
    fn gemini_request_function_call_becomes_assistant_tool_call() {
        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![Content {
                role: Some(NativeRole::Model),
                parts: vec![Part {
                    function_call: Some(NativeFunctionCall {
                        name: "get_weather".into(),
                        args: json!({"location": "SF"}),
                        id: Some("fc1".into()),
                        extra: Default::default(),
                    }),
                    ..Default::default()
                }],
            }])
            .build();
        let canonical = gemini_request_to_canonical(native).unwrap();
        let msg = &canonical.messages[0];
        assert_eq!(msg.role, Role::Assistant);
        let tcs = msg.tool_calls.as_ref().unwrap();
        assert_eq!(tcs[0].function.name, "get_weather");
        assert_eq!(tcs[0].id, "fc1");
        assert_eq!(tcs[0].function.arguments, r#"{"location":"SF"}"#);
    }

    #[test]
    fn gemini_request_function_response_becomes_tool_message() {
        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![Content {
                role: Some(NativeRole::User),
                parts: vec![Part {
                    function_response: Some(FunctionResponse {
                        name: "get_weather".into(),
                        response: json!({"temp": 72}),
                        id: Some("fc1".into()),
                        extra: Default::default(),
                    }),
                    ..Default::default()
                }],
            }])
            .build();
        let canonical = gemini_request_to_canonical(native).unwrap();
        let msg = &canonical.messages[0];
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("fc1"));
        assert_eq!(msg.name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn gemini_request_missing_function_response_id_reuses_pending_call_id() {
        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![
                Content {
                    role: Some(NativeRole::Model),
                    parts: vec![Part {
                        function_call: Some(NativeFunctionCall {
                            name: "get_weather".into(),
                            args: json!({"location": "SF"}),
                            id: None,
                            extra: Default::default(),
                        }),
                        ..Default::default()
                    }],
                },
                Content {
                    role: Some(NativeRole::User),
                    parts: vec![Part {
                        function_response: Some(FunctionResponse {
                            name: "get_weather".into(),
                            response: json!({"temp": 72}),
                            id: None,
                            extra: Default::default(),
                        }),
                        ..Default::default()
                    }],
                },
            ])
            .build();

        let canonical = gemini_request_to_canonical(native).unwrap();
        let call_id = canonical.messages[0].tool_calls.as_ref().unwrap()[0]
            .id
            .as_str();
        assert_eq!(call_id, "call_0");
        assert_eq!(canonical.messages[1].role, Role::Tool);
        assert_eq!(canonical.messages[1].tool_call_id.as_deref(), Some(call_id));
    }

    #[test]
    fn gemini_request_inline_image_becomes_image_url_part() {
        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![Content {
                role: Some(NativeRole::User),
                parts: vec![
                    Part::text("describe"),
                    Part {
                        inline_data: Some(crate::types::Blob {
                            mime_type: "image/png".into(),
                            data: "aW1n".into(),
                            extra: Default::default(),
                        }),
                        ..Default::default()
                    },
                ],
            }])
            .build();

        let canonical = gemini_request_to_canonical(native).unwrap();
        let parts = match canonical.messages[0].content.as_ref().unwrap() {
            MessageContent::Parts(parts) => parts,
            other => panic!("expected Parts, got {other:?}"),
        };
        assert!(matches!(
            &parts[0],
            ContentPart::Known(TypedContentPart::Text { text, .. }) if text == "describe"
        ));
        assert!(matches!(
            &parts[1],
            ContentPart::Known(TypedContentPart::ImageUrl { image_url, .. })
                if image_url.url == "data:image/png;base64,aW1n"
        ));
    }

    #[test]
    fn gemini_request_generation_config_maps_thinking_and_response_format() {
        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![Content {
                role: Some(NativeRole::User),
                parts: vec![Part::text("json please")],
            }])
            .generation_config(GenerationConfig {
                thinking_config: Some(NativeThinkingConfig {
                    thinking_budget: Some(-1),
                    thinking_level: None,
                    include_thoughts: Some(true),
                }),
                response_mime_type: Some("application/json".into()),
                response_schema: Some(json!({
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"}
                    }
                })),
                ..Default::default()
            })
            .build();

        let canonical = gemini_request_to_canonical(native).unwrap();
        assert_eq!(canonical.thinking, Some(ThinkingRequest::Auto));
        match canonical.response_format.as_ref().unwrap() {
            ResponseFormat::JsonSchema { json_schema, .. } => {
                assert_eq!(json_schema.name, "response");
                assert_eq!(
                    json_schema.schema.as_ref().unwrap(),
                    &json!({
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"}
                        }
                    })
                );
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
    }

    #[test]
    fn gemini_function_declaration_schema_becomes_canonical_json_schema() {
        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![Content {
                role: Some(NativeRole::User),
                parts: vec![Part::text("search")],
            }])
            .tools(vec![crate::types::Tool {
                function_declarations: Some(vec![crate::types::FunctionDeclaration {
                    name: "search".into(),
                    description: Some("Search".into()),
                    parameters: Some(json!({
                        "type": "OBJECT",
                        "properties": {
                            "query": {"type": "STRING"},
                            "limit": {"type": ["INTEGER", "NULL"]}
                        }
                    })),
                    extra: Default::default(),
                }]),
                google_search: None,
                code_execution: None,
                url_context: None,
                extra: Default::default(),
            }])
            .build();

        let canonical = gemini_request_to_canonical(native).unwrap();
        let params = canonical.tools.as_ref().unwrap()[0]
            .function
            .parameters
            .as_ref()
            .unwrap();
        assert_eq!(params["type"], "object");
        assert_eq!(params["properties"]["query"]["type"], "string");
        assert_eq!(
            params["properties"]["limit"]["type"],
            json!(["integer", "null"])
        );
    }

    #[test]
    fn gemini_request_thinking_level_maps_to_canonical_level() {
        let native = GenerateContentRequest::builder()
            .model("gemini-3-pro")
            .contents(vec![Content {
                role: Some(NativeRole::User),
                parts: vec![Part::text("hi")],
            }])
            .generation_config(GenerationConfig {
                thinking_config: Some(NativeThinkingConfig {
                    thinking_budget: None,
                    thinking_level: Some(NativeThinkingLevel::High),
                    include_thoughts: None,
                }),
                ..Default::default()
            })
            .build();

        let canonical = gemini_request_to_canonical(native).unwrap();
        assert_eq!(
            canonical.thinking,
            Some(ThinkingRequest::Level {
                level: CanonicalThinkingLevel::High
            })
        );
    }

    #[test]
    fn gemini_request_single_allowed_tool_becomes_named_tool_choice() {
        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![Content {
                role: Some(NativeRole::User),
                parts: vec![Part::text("call a tool")],
            }])
            .tool_config(crate::types::ToolConfig {
                function_calling_config: Some(FunctionCallingConfig {
                    mode: Some(FunctionCallingMode::Any),
                    allowed_function_names: Some(vec!["get_weather".into()]),
                }),
            })
            .build();

        let canonical = gemini_request_to_canonical(native).unwrap();
        assert!(matches!(
            canonical.tool_choice,
            Some(ToolChoice::Named(named)) if named.function.name == "get_weather"
        ));
    }

    #[test]
    fn gemini_request_multiple_allowed_tools_filters_declarations() {
        let declarations = ["get_weather", "search", "delete_all"]
            .into_iter()
            .map(|name| crate::types::FunctionDeclaration {
                name: name.to_owned(),
                description: None,
                parameters: None,
                extra: Default::default(),
            })
            .collect();

        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![Content {
                role: Some(NativeRole::User),
                parts: vec![Part::text("call an allowed tool")],
            }])
            .tools(vec![crate::types::Tool {
                function_declarations: Some(declarations),
                google_search: None,
                code_execution: None,
                url_context: None,
                extra: Default::default(),
            }])
            .tool_config(crate::types::ToolConfig {
                function_calling_config: Some(FunctionCallingConfig {
                    mode: Some(FunctionCallingMode::Any),
                    allowed_function_names: Some(vec!["get_weather".into(), "search".into()]),
                }),
            })
            .build();

        let canonical = gemini_request_to_canonical(native).unwrap();
        assert!(matches!(
            canonical.tool_choice,
            Some(ToolChoice::Mode(ToolChoiceMode::Required))
        ));
        let names: Vec<_> = canonical
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect();
        assert_eq!(names, ["get_weather", "search"]);
    }

    #[test]
    fn gemini_request_validated_mode_stays_auto_but_filters_tools() {
        let declarations = ["get_weather", "delete_all"]
            .into_iter()
            .map(|name| crate::types::FunctionDeclaration {
                name: name.to_owned(),
                description: None,
                parameters: None,
                extra: Default::default(),
            })
            .collect();

        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![Content {
                role: Some(NativeRole::User),
                parts: vec![Part::text("answer or call a tool")],
            }])
            .tools(vec![crate::types::Tool {
                function_declarations: Some(declarations),
                google_search: None,
                code_execution: None,
                url_context: None,
                extra: Default::default(),
            }])
            .tool_config(crate::types::ToolConfig {
                function_calling_config: Some(FunctionCallingConfig {
                    mode: Some(FunctionCallingMode::Validated),
                    allowed_function_names: Some(vec!["get_weather".into()]),
                }),
            })
            .build();

        let canonical = gemini_request_to_canonical(native).unwrap();
        assert!(matches!(
            canonical.tool_choice,
            Some(ToolChoice::Mode(ToolChoiceMode::Auto))
        ));
        let names: Vec<_> = canonical
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect();
        assert_eq!(names, ["get_weather"]);
    }

    #[test]
    fn gemini_request_builtin_tool_is_rejected_for_canonical_bridge() {
        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![Content {
                role: Some(NativeRole::User),
                parts: vec![Part::text("search")],
            }])
            .tools(vec![crate::types::Tool {
                function_declarations: None,
                google_search: Some(json!({})),
                code_execution: None,
                url_context: None,
                extra: Default::default(),
            }])
            .build();

        let err = gemini_request_to_canonical(native).unwrap_err();
        assert!(matches!(
            err,
            TranslateError::UnsupportedFeature { feature, .. }
                if feature.contains("googleSearch")
        ));
    }

    #[test]
    fn gemini_request_cached_content_is_rejected_for_canonical_bridge() {
        let native = GenerateContentRequest::builder()
            .model("gemini-2.5-flash")
            .contents(vec![Content {
                role: Some(NativeRole::User),
                parts: vec![Part::text("use cache")],
            }])
            .cached_content("cachedContents/abc")
            .build();

        let err = gemini_request_to_canonical(native).unwrap_err();
        assert!(matches!(
            err,
            TranslateError::UnsupportedFeature { feature, .. }
                if feature.contains("cachedContent")
        ));
    }

    // ── chat_response_to_gemini ──────────────────────────────────────────

    #[test]
    fn chat_response_text_becomes_gemini_candidate() {
        use aigw_core::model::Choice;
        let resp = ChatResponse {
            id: "chatcmpl-x".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "gpt-4".into(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: Some(MessageContent::Text("Hi there!".into())),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                    extra: Default::default(),
                },
                finish_reason: Some(CanonicalFinishReason::Stop),
                extra: Default::default(),
            }],
            usage: Some(Usage {
                prompt_tokens: Some(8),
                completion_tokens: Some(4),
                total_tokens: Some(12),
                extra: Default::default(),
            }),
            extra: Default::default(),
        };
        let gemini = chat_response_to_gemini(resp).unwrap();
        assert_eq!(gemini.model_version.as_deref(), Some("gpt-4"));
        assert_eq!(gemini.response_id.as_deref(), Some("chatcmpl-x"));
        let cand = &gemini.candidates[0];
        assert_eq!(
            cand.content.as_ref().unwrap().parts[0].text.as_deref(),
            Some("Hi there!")
        );
        assert_eq!(cand.finish_reason, Some(NativeFinishReason::Stop));
        let usage = gemini.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, Some(8));
        assert_eq!(usage.candidates_token_count, Some(4));
    }

    #[test]
    fn chat_response_finish_reason_length_maps_to_max_tokens() {
        use aigw_core::model::Choice;
        let resp = ChatResponse {
            id: "x".into(),
            object: "chat.completion".into(),
            created: 0,
            model: String::new(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: Some(MessageContent::Text("...".into())),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                    extra: Default::default(),
                },
                finish_reason: Some(CanonicalFinishReason::Length),
                extra: Default::default(),
            }],
            usage: None,
            extra: Default::default(),
        };
        let gemini = chat_response_to_gemini(resp).unwrap();
        assert_eq!(
            gemini.candidates[0].finish_reason,
            Some(NativeFinishReason::MaxTokens)
        );
    }

    // ── stream_event_to_gemini_sse ───────────────────────────────────────

    fn extract_data(bytes: &[u8]) -> Value {
        let s = std::str::from_utf8(bytes).unwrap();
        let line = s.strip_prefix("data: ").unwrap();
        let line = line.trim_end();
        serde_json::from_str(line).unwrap()
    }

    #[test]
    fn response_meta_updates_context_no_output() {
        let mut ctx = SseContext::default();
        let out = stream_event_to_gemini_sse(
            &StreamEvent::ResponseMeta {
                id: "r1".into(),
                model: "gpt-4".into(),
            },
            &mut ctx,
        );
        assert!(out.is_none());
        assert_eq!(ctx.response_id, "r1");
        assert_eq!(ctx.model, "gpt-4");
    }

    #[test]
    fn content_delta_emits_text_part() {
        let mut ctx = SseContext::with_model("gemini-2.5-pro");
        let out = stream_event_to_gemini_sse(&StreamEvent::ContentDelta("Hello".into()), &mut ctx)
            .unwrap();
        let v = extract_data(&out);
        assert_eq!(v["candidates"][0]["content"]["parts"][0]["text"], "Hello");
        assert_eq!(v["modelVersion"], "gemini-2.5-pro");
    }

    #[test]
    fn reasoning_delta_emits_thought_part() {
        let mut ctx = SseContext::with_model("gemini-2.5-pro");
        stream_event_to_gemini_sse(
            &StreamEvent::ReasoningStart {
                index: 0,
                source: None,
            },
            &mut ctx,
        );
        let out =
            stream_event_to_gemini_sse(&StreamEvent::ReasoningDelta("thinking".into()), &mut ctx)
                .unwrap();
        let v = extract_data(&out);
        assert_eq!(
            v["candidates"][0]["content"]["parts"][0]["text"],
            "thinking"
        );
        assert_eq!(v["candidates"][0]["content"]["parts"][0]["thought"], true);
    }

    #[test]
    fn reasoning_end_emits_signature_part() {
        let mut ctx = SseContext::with_model("gemini-2.5-pro");
        stream_event_to_gemini_sse(
            &StreamEvent::ReasoningStart {
                index: 0,
                source: None,
            },
            &mut ctx,
        );
        let out = stream_event_to_gemini_sse(
            &StreamEvent::ReasoningEnd {
                index: 0,
                signature: "sig".into(),
            },
            &mut ctx,
        )
        .unwrap();
        let v = extract_data(&out);
        assert_eq!(v["candidates"][0]["content"]["parts"][0]["thought"], true);
        assert_eq!(
            v["candidates"][0]["content"]["parts"][0]["thoughtSignature"],
            "sig"
        );
    }

    #[test]
    fn tool_call_buffer_flushes_on_finish() {
        let mut ctx = SseContext::with_model("gemini-2.5-pro");
        stream_event_to_gemini_sse(
            &StreamEvent::ToolCallStart {
                index: 0,
                id: "fc1".into(),
                name: "get_weather".into(),
            },
            &mut ctx,
        );
        stream_event_to_gemini_sse(
            &StreamEvent::ToolCallDelta {
                index: 0,
                arguments: r#"{"location":"SF"}"#.into(),
            },
            &mut ctx,
        );
        let out = stream_event_to_gemini_sse(
            &StreamEvent::Finish(CanonicalFinishReason::ToolCalls),
            &mut ctx,
        )
        .unwrap();
        let v = extract_data(&out);
        let parts = v["candidates"][0]["content"]["parts"].as_array().unwrap();
        let fc = &parts[0]["functionCall"];
        assert_eq!(fc["name"], "get_weather");
        assert_eq!(fc["args"]["location"], "SF");
        // ToolCalls maps to STOP on the Gemini side (no dedicated tool finishReason).
        assert_eq!(v["candidates"][0]["finishReason"], "STOP");
    }

    #[test]
    fn usage_event_emits_usage_metadata() {
        let mut ctx = SseContext::with_model("gemini-2.5-pro");
        let out = stream_event_to_gemini_sse(
            &StreamEvent::Usage(Usage {
                prompt_tokens: Some(10),
                completion_tokens: Some(5),
                total_tokens: Some(15),
                extra: Default::default(),
            }),
            &mut ctx,
        )
        .unwrap();
        let v = extract_data(&out);
        assert_eq!(v["usageMetadata"]["promptTokenCount"], 10);
        assert_eq!(v["usageMetadata"]["candidatesTokenCount"], 5);
        assert_eq!(v["usageMetadata"]["totalTokenCount"], 15);
    }

    #[test]
    fn done_event_suppressed() {
        let mut ctx = SseContext::with_model("gemini-2.5-pro");
        let out = stream_event_to_gemini_sse(&StreamEvent::Done, &mut ctx);
        assert!(out.is_none());
    }

    #[test]
    fn done_event_flushes_pending_tool_call() {
        let mut ctx = SseContext::with_model("gemini-2.5-pro");
        stream_event_to_gemini_sse(
            &StreamEvent::ToolCallStart {
                index: 0,
                id: "fc1".into(),
                name: "get_weather".into(),
            },
            &mut ctx,
        );
        stream_event_to_gemini_sse(
            &StreamEvent::ToolCallDelta {
                index: 0,
                arguments: r#"{"location":"SF"}"#.into(),
            },
            &mut ctx,
        );

        let out = stream_event_to_gemini_sse(&StreamEvent::Done, &mut ctx).unwrap();
        let v = extract_data(&out);
        let parts = v["candidates"][0]["content"]["parts"].as_array().unwrap();
        let fc = &parts[0]["functionCall"];
        assert_eq!(fc["id"], "fc1");
        assert_eq!(fc["name"], "get_weather");
        assert_eq!(fc["args"]["location"], "SF");

        let out = stream_event_to_gemini_sse(&StreamEvent::Done, &mut ctx);
        assert!(out.is_none());
    }
}
