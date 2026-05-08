//! Stream translation: Gemini SSE events → canonical [`StreamEvent`]s.
//!
//! Each Gemini SSE event is a complete [`GenerateContentResponse`] whose
//! `candidates[0].content.parts[]` carries **incremental** parts (Gemini
//! does not emit cumulative snapshots in `streamGenerateContent?alt=sse`).
//! The parser walks those parts and emits canonical events:
//!
//! - `text` parts (non-thought) → [`StreamEvent::ContentDelta`].
//! - `text` parts with `thought=true` → [`StreamEvent::ReasoningStart`]
//!   (first thought of a block) + [`StreamEvent::ReasoningDelta`] for the
//!   incremental text. [`StreamEvent::ReasoningEnd`] is emitted when a
//!   `thoughtSignature` arrives or when a non-thought part follows.
//! - `function_call` parts → [`StreamEvent::ToolCallStart`] + a single
//!   [`StreamEvent::ToolCallDelta`] carrying the JSON-serialized args
//!   (Gemini emits the entire call atomically; there's no progressive
//!   args delta).
//! - The first event whose response carries `responseId`/`modelVersion`
//!   produces a [`StreamEvent::ResponseMeta`].
//! - When a candidate's `finishReason` is set, the parser emits
//!   [`StreamEvent::Finish`] + [`StreamEvent::Usage`] +
//!   [`StreamEvent::Done`].
//!
//! [`StreamEvent`]: aigw_core::model::StreamEvent
//! [`StreamEvent::ContentDelta`]: aigw_core::model::StreamEvent::ContentDelta
//! [`StreamEvent::ReasoningStart`]: aigw_core::model::StreamEvent::ReasoningStart
//! [`StreamEvent::ReasoningDelta`]: aigw_core::model::StreamEvent::ReasoningDelta
//! [`StreamEvent::ReasoningEnd`]: aigw_core::model::StreamEvent::ReasoningEnd
//! [`StreamEvent::ToolCallStart`]: aigw_core::model::StreamEvent::ToolCallStart
//! [`StreamEvent::ToolCallDelta`]: aigw_core::model::StreamEvent::ToolCallDelta
//! [`GenerateContentResponse`]: crate::types::GenerateContentResponse

use aigw_core::error::TranslateError;
use aigw_core::model::{StreamEvent, ThinkingSource, Usage};
use aigw_core::translate::StreamParser;

use super::response::map_finish_reason;
use crate::types::{GenerateContentResponse, Part};

/// Currently-open reasoning block state.
#[derive(Debug, Clone)]
struct OpenReasoning {
    canonical_index: u32,
    /// Buffered signature bytes emitted by `thoughtSignature` deltas.
    signature: String,
}

/// Stateful parser for Gemini SSE streams.
pub struct GeminiStreamParser {
    meta_emitted: bool,
    tool_call_index: u32,
    reasoning_index: u32,
    open_reasoning: Option<OpenReasoning>,
    done: bool,
}

impl Default for GeminiStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl GeminiStreamParser {
    pub fn new() -> Self {
        Self {
            meta_emitted: false,
            tool_call_index: 0,
            reasoning_index: 0,
            open_reasoning: None,
            done: false,
        }
    }

    fn close_reasoning(&mut self) -> Option<StreamEvent> {
        self.open_reasoning
            .take()
            .map(|r| StreamEvent::ReasoningEnd {
                index: r.canonical_index,
                signature: r.signature,
            })
    }

    /// Ensure a reasoning block is open. Returns `(index, just_opened)`
    /// — `just_opened` is `true` exactly on the first call after no block
    /// was open, so the caller can emit `ReasoningStart`.
    fn open_or_get_reasoning(&mut self) -> (u32, bool) {
        if let Some(r) = &self.open_reasoning {
            return (r.canonical_index, false);
        }
        let idx = self.reasoning_index;
        self.reasoning_index += 1;
        self.open_reasoning = Some(OpenReasoning {
            canonical_index: idx,
            signature: String::new(),
        });
        (idx, true)
    }

    fn emit_meta(&mut self, resp: &GenerateContentResponse, out: &mut Vec<StreamEvent>) {
        if self.meta_emitted {
            return;
        }
        if resp.response_id.is_some() || resp.model_version.is_some() {
            out.push(StreamEvent::ResponseMeta {
                id: resp.response_id.clone().unwrap_or_default(),
                model: resp.model_version.clone().unwrap_or_default(),
            });
            self.meta_emitted = true;
        }
    }
}

impl StreamParser for GeminiStreamParser {
    fn parse_event(
        &mut self,
        _event_type: &str,
        data: &str,
    ) -> Result<Vec<StreamEvent>, TranslateError> {
        if data.trim() == "[DONE]" {
            self.done = true;
            return Ok(vec![StreamEvent::Done]);
        }

        let resp: GenerateContentResponse =
            serde_json::from_str(data).map_err(|e| TranslateError::StreamParse {
                message: format!("failed to parse Gemini stream event: {e}"),
            })?;

        let mut out = Vec::new();
        self.emit_meta(&resp, &mut out);

        // Walk parts on candidate 0 (Gemini sends one candidate by default).
        let mut finish_reason = None;
        if let Some(c) = resp.candidates.first() {
            finish_reason = c.finish_reason.clone();
            if let Some(content) = &c.content {
                for part in &content.parts {
                    handle_part(part, self, &mut out);
                }
            }
        }

        // Finalisation: if this event reports a finish reason, emit
        // Finish + Usage + Done. Close any still-open reasoning block first.
        if let Some(reason) = finish_reason {
            if let Some(end) = self.close_reasoning() {
                out.push(end);
            }
            out.push(StreamEvent::Finish(map_finish_reason(reason)));
            if let Some(u) = resp.usage_metadata {
                let mut extra = serde_json::Map::new();
                if let Some(t) = u.cached_content_token_count {
                    extra.insert(
                        "cached_content_token_count".into(),
                        serde_json::Value::Number(t.into()),
                    );
                }
                if let Some(t) = u.thoughts_token_count {
                    extra.insert(
                        "thoughts_token_count".into(),
                        serde_json::Value::Number(t.into()),
                    );
                }
                out.push(StreamEvent::Usage(Usage {
                    prompt_tokens: u.prompt_token_count,
                    completion_tokens: u.candidates_token_count,
                    total_tokens: u.total_token_count,
                    extra,
                }));
            }
            out.push(StreamEvent::Done);
            self.done = true;
        }

        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, TranslateError> {
        if self.done {
            return Ok(vec![]);
        }
        // Defensive: emit a final Done if the upstream stream ended without
        // a finishReason event.
        let mut out = Vec::new();
        if let Some(end) = self.close_reasoning() {
            out.push(end);
        }
        out.push(StreamEvent::Done);
        self.done = true;
        Ok(out)
    }
}

fn handle_part(part: &Part, parser: &mut GeminiStreamParser, out: &mut Vec<StreamEvent>) {
    let is_thought = part.thought.unwrap_or(false);

    if let Some(fc) = &part.function_call {
        // Close any open reasoning block before starting tool calls.
        if let Some(end) = parser.close_reasoning() {
            out.push(end);
        }
        let id = fc
            .id
            .clone()
            .unwrap_or_else(|| format!("call_{}", parser.tool_call_index));
        let index = parser.tool_call_index;
        parser.tool_call_index += 1;
        out.push(StreamEvent::ToolCallStart {
            index,
            id,
            name: fc.name.clone(),
        });
        // Gemini emits the full args in a single chunk; surface as one delta.
        let arguments = serde_json::to_string(&fc.args).unwrap_or_else(|_| "{}".to_owned());
        if !arguments.is_empty() && arguments != "null" {
            out.push(StreamEvent::ToolCallDelta { index, arguments });
        }
        return;
    }

    if is_thought {
        let (idx, just_opened) = parser.open_or_get_reasoning();
        if just_opened {
            out.push(StreamEvent::ReasoningStart {
                index: idx,
                source: Some(ThinkingSource::Gemini),
            });
        }
        if let Some(text) = &part.text
            && !text.is_empty()
        {
            out.push(StreamEvent::ReasoningDelta(text.clone()));
        }
        // Buffer signature for emission at close.
        if let Some(sig) = &part.thought_signature
            && !sig.is_empty()
            && let Some(open) = parser.open_reasoning.as_mut()
        {
            open.signature.push_str(sig);
        }
        return;
    }

    // Non-thought, non-function: text part.
    // Close reasoning block before plain text starts.
    if let Some(end) = parser.close_reasoning() {
        out.push(end);
    }
    if let Some(text) = &part.text
        && !text.is_empty()
    {
        out.push(StreamEvent::ContentDelta(text.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aigw_core::model::FinishReason;

    fn parser() -> GeminiStreamParser {
        GeminiStreamParser::new()
    }

    #[test]
    fn first_event_emits_response_meta() {
        let mut p = parser();
        let data = r#"{
            "candidates": [{ "content": { "role": "model", "parts": [{"text":"hi"}] } }],
            "responseId": "r1",
            "modelVersion": "gemini-2.5-flash"
        }"#;
        let events = p.parse_event("", data).unwrap();
        assert!(matches!(
            &events[0],
            StreamEvent::ResponseMeta { id, model }
            if id == "r1" && model == "gemini-2.5-flash"
        ));
        assert!(matches!(&events[1], StreamEvent::ContentDelta(s) if s == "hi"));
    }

    #[test]
    fn text_delta_emitted() {
        let mut p = parser();
        // First chunk: just meta + text.
        let chunk = r#"{
            "candidates": [{ "content": { "role": "model", "parts": [{"text":"Hello "}] } }],
            "responseId": "r1"
        }"#;
        let e1 = p.parse_event("", chunk).unwrap();
        let chunk2 = r#"{
            "candidates": [{ "content": { "role": "model", "parts": [{"text":"world"}] } }]
        }"#;
        let e2 = p.parse_event("", chunk2).unwrap();
        assert!(
            e1.iter()
                .any(|e| matches!(e, StreamEvent::ContentDelta(s) if s == "Hello "))
        );
        assert!(
            e2.iter()
                .any(|e| matches!(e, StreamEvent::ContentDelta(s) if s == "world"))
        );
    }

    #[test]
    fn finish_reason_emits_finish_usage_done() {
        let mut p = parser();
        let chunk = r#"{
            "candidates": [{
                "content": { "role": "model", "parts": [{"text":"done."}] },
                "finishReason": "STOP"
            }],
            "responseId": "r1",
            "usageMetadata": { "promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7 }
        }"#;
        let events = p.parse_event("", chunk).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::ResponseMeta { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::ContentDelta(s) if s == "done."))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::Finish(FinishReason::Stop)))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::Usage(u) if u.prompt_tokens == Some(5)))
        );
        assert!(events.iter().any(|e| matches!(e, StreamEvent::Done)));
    }

    #[test]
    fn function_call_emits_tool_start_and_delta() {
        let mut p = parser();
        let chunk = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": { "name": "get_weather", "args": {"location":"SF"}, "id": "fc1" }
                    }]
                }
            }],
            "responseId": "r1"
        }"#;
        let events = p.parse_event("", chunk).unwrap();
        let tcs: Vec<&StreamEvent> = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    StreamEvent::ToolCallStart { .. } | StreamEvent::ToolCallDelta { .. }
                )
            })
            .collect();
        assert_eq!(tcs.len(), 2);
        assert!(matches!(
            tcs[0],
            StreamEvent::ToolCallStart { index: 0, id, name }
            if id == "fc1" && name == "get_weather"
        ));
        match tcs[1] {
            StreamEvent::ToolCallDelta { index, arguments } => {
                assert_eq!(*index, 0);
                let v: serde_json::Value = serde_json::from_str(arguments).unwrap();
                assert_eq!(v["location"], "SF");
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
    }

    #[test]
    fn thought_part_emits_reasoning_delta() {
        let mut p = parser();
        let chunk = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        { "text": "let me think...", "thought": true },
                        { "text": "and also...", "thought": true }
                    ]
                }
            }],
            "responseId": "r1"
        }"#;
        let events = p.parse_event("", chunk).unwrap();
        let rds: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ReasoningDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(rds, vec!["let me think...", "and also..."]);
    }

    #[test]
    fn thought_then_text_closes_reasoning_block() {
        let mut p = parser();
        let chunk = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        { "text": "thinking", "thought": true, "thoughtSignature": "sig" },
                        { "text": "answer" }
                    ]
                }
            }],
            "responseId": "r1"
        }"#;
        let events = p.parse_event("", chunk).unwrap();
        // Order should be: ReasoningDelta → ReasoningEnd → ContentDelta
        let kinds: Vec<&StreamEvent> = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    StreamEvent::ReasoningDelta(_)
                        | StreamEvent::ReasoningEnd { .. }
                        | StreamEvent::ContentDelta(_)
                )
            })
            .collect();
        assert!(matches!(kinds[0], StreamEvent::ReasoningDelta(_)));
        assert!(matches!(
            kinds[1],
            StreamEvent::ReasoningEnd { signature, .. } if signature == "sig"
        ));
        assert!(matches!(kinds[2], StreamEvent::ContentDelta(s) if s == "answer"));
    }

    #[test]
    fn done_marker() {
        let mut p = parser();
        let events = p.parse_event("", "[DONE]").unwrap();
        assert!(matches!(&events[0], StreamEvent::Done));
    }

    #[test]
    fn finish_emits_done_if_not_already() {
        let mut p = parser();
        let events = p.finish().unwrap();
        assert!(events.iter().any(|e| matches!(e, StreamEvent::Done)));
        // Idempotent.
        let events = p.finish().unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn finish_closes_open_reasoning() {
        let mut p = parser();
        let chunk = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{ "text": "thinking", "thought": true, "thoughtSignature": "sig" }]
                }
            }],
            "responseId": "r1"
        }"#;
        p.parse_event("", chunk).unwrap();
        let events = p.finish().unwrap();
        // Should emit ReasoningEnd before Done.
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::ReasoningEnd { signature, .. } if signature == "sig"
        )));
        assert!(events.iter().any(|e| matches!(e, StreamEvent::Done)));
    }
}
