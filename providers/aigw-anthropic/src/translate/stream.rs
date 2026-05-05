//! Stream translation: Anthropic SSE events → canonical `StreamEvent`s.
//!
//! Anthropic uses named SSE events with block-level granularity. The parser
//! maintains state across events to track tool call indices and combine
//! input_tokens (from message_start) with output_tokens (from message_delta).

use aigw_core::error::TranslateError;
use aigw_core::model::{StreamEvent as CanonicalStreamEvent, ThinkingSource, Usage};
use aigw_core::translate::StreamParser;

use crate::types::{
    ContentBlock, ContentDelta, StreamEvent as AnthropicStreamEvent, TypedContentBlock,
};

/// What kind of content block is currently open in the Anthropic SSE
/// stream, so the parser knows how to interpret deltas and stop events.
#[derive(Debug, Clone)]
enum OpenBlock {
    /// No block is currently open (between blocks, or before the first).
    None,
    /// Text block (no extra state needed).
    Text,
    /// Tool use block — the canonical tool_call index has already been
    /// emitted via `ToolCallStart`.
    ToolUse,
    /// Thinking block — accumulating signature deltas to emit as
    /// `ReasoningEnd { signature }` on close.
    Thinking {
        canonical_index: u32,
        signature_buf: String,
    },
    /// Redacted thinking — currently not surfaced as canonical stream
    /// events. TODO(aigw): emit a dedicated canonical event once
    /// `StreamEvent` has a `ReasoningRedacted { index, data }` variant;
    /// non-streaming responses already round-trip redacted blocks.
    RedactedThinking,
}

/// Stateful parser for Anthropic SSE streams.
///
/// Created per-request via [`AnthropicResponseTranslator::stream_parser()`].
pub struct AnthropicStreamParser {
    /// Incremented on each `content_block_start` with `tool_use` type.
    tool_call_index: u32,
    /// Incremented on each `content_block_start` with `thinking` type.
    reasoning_index: u32,
    /// Currently-open block kind (set on `content_block_start`, cleared on
    /// `content_block_stop`).
    open_block: OpenBlock,
    /// Input token count from `message_start`.
    input_tokens: Option<u64>,
    /// Whether `Done` has been emitted.
    done: bool,
}

impl Default for AnthropicStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicStreamParser {
    pub fn new() -> Self {
        Self {
            tool_call_index: 0,
            reasoning_index: 0,
            open_block: OpenBlock::None,
            input_tokens: None,
            done: false,
        }
    }
}

impl StreamParser for AnthropicStreamParser {
    fn parse_event(
        &mut self,
        _event_type: &str,
        data: &str,
    ) -> Result<Vec<CanonicalStreamEvent>, TranslateError> {
        let native: AnthropicStreamEvent =
            serde_json::from_str(data).map_err(|e| TranslateError::StreamParse {
                message: format!("failed to parse Anthropic stream event: {e}"),
            })?;

        match native {
            AnthropicStreamEvent::MessageStart { message } => {
                self.input_tokens = Some(message.usage.input_tokens);
                Ok(vec![CanonicalStreamEvent::ResponseMeta {
                    id: message.id,
                    model: message.model,
                }])
            }

            AnthropicStreamEvent::ContentBlockStart { content_block, .. } => {
                match &content_block {
                    ContentBlock::Typed(TypedContentBlock::ToolUse { id, name, .. }) => {
                        let idx = self.tool_call_index;
                        self.tool_call_index += 1;
                        self.open_block = OpenBlock::ToolUse;
                        Ok(vec![CanonicalStreamEvent::ToolCallStart {
                            index: idx,
                            id: id.clone(),
                            name: name.clone(),
                        }])
                    }
                    ContentBlock::Typed(TypedContentBlock::Thinking { .. }) => {
                        let idx = self.reasoning_index;
                        self.reasoning_index += 1;
                        self.open_block = OpenBlock::Thinking {
                            canonical_index: idx,
                            signature_buf: String::new(),
                        };
                        Ok(vec![CanonicalStreamEvent::ReasoningStart {
                            index: idx,
                            source: Some(ThinkingSource::Anthropic),
                        }])
                    }
                    ContentBlock::Typed(TypedContentBlock::RedactedThinking { .. }) => {
                        // TODO(aigw): emit ReasoningRedacted { index, data }
                        // once that variant exists. For now skip in streaming;
                        // non-streaming responses still surface the block.
                        self.open_block = OpenBlock::RedactedThinking;
                        Ok(vec![])
                    }
                    ContentBlock::Typed(TypedContentBlock::Text { .. }) => {
                        self.open_block = OpenBlock::Text;
                        Ok(vec![])
                    }
                    _ => {
                        self.open_block = OpenBlock::None;
                        Ok(vec![])
                    }
                }
            }

            AnthropicStreamEvent::ContentBlockDelta { delta, .. } => match delta {
                ContentDelta::TextDelta { text } => {
                    Ok(vec![CanonicalStreamEvent::ContentDelta(text)])
                }
                ContentDelta::InputJsonDelta { partial_json } => {
                    let tool_idx = self.tool_call_index.saturating_sub(1);
                    Ok(vec![CanonicalStreamEvent::ToolCallDelta {
                        index: tool_idx,
                        arguments: partial_json,
                    }])
                }
                ContentDelta::ThinkingDelta { thinking } => {
                    Ok(vec![CanonicalStreamEvent::ReasoningDelta(thinking)])
                }
                ContentDelta::SignatureDelta { signature } => {
                    if let OpenBlock::Thinking { signature_buf, .. } = &mut self.open_block {
                        signature_buf.push_str(&signature);
                    }
                    // Anthropic accumulates the signature server-side and
                    // emits it as one or more deltas; we surface it only at
                    // block close via `ReasoningEnd`.
                    Ok(vec![])
                }
                // Unknown: skip.
                _ => Ok(vec![]),
            },

            AnthropicStreamEvent::ContentBlockStop { .. } => {
                let prev = std::mem::replace(&mut self.open_block, OpenBlock::None);
                if let OpenBlock::Thinking {
                    canonical_index,
                    signature_buf,
                } = prev
                {
                    Ok(vec![CanonicalStreamEvent::ReasoningEnd {
                        index: canonical_index,
                        signature: signature_buf,
                    }])
                } else {
                    Ok(vec![])
                }
            }

            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                let mut events = Vec::new();

                if let Some(stop_reason) = delta.stop_reason {
                    events.push(CanonicalStreamEvent::Finish(stop_reason.into()));
                }

                let output_tokens = usage.output_tokens;
                let input_tokens = self.input_tokens.unwrap_or(0);
                events.push(CanonicalStreamEvent::Usage(Usage {
                    prompt_tokens: Some(input_tokens),
                    completion_tokens: Some(output_tokens),
                    total_tokens: Some(input_tokens + output_tokens),
                    extra: Default::default(),
                }));

                Ok(events)
            }

            AnthropicStreamEvent::MessageStop => {
                self.done = true;
                Ok(vec![CanonicalStreamEvent::Done])
            }

            AnthropicStreamEvent::Ping => Ok(vec![]),

            AnthropicStreamEvent::Error { error } => Err(TranslateError::StreamParse {
                message: format!(
                    "Anthropic stream error: [{}] {}",
                    error.r#type, error.message
                ),
            }),

            AnthropicStreamEvent::Unknown => Ok(vec![]),
        }
    }

    fn finish(&mut self) -> Result<Vec<CanonicalStreamEvent>, TranslateError> {
        if !self.done {
            self.done = true;
            Ok(vec![CanonicalStreamEvent::Done])
        } else {
            Ok(vec![])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aigw_core::model::FinishReason;

    fn parser() -> AnthropicStreamParser {
        AnthropicStreamParser::new()
    }

    #[test]
    fn message_start_emits_response_meta() {
        let mut p = parser();
        let data = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": null,
                "stop_sequence": null,
                "usage": { "input_tokens": 25, "output_tokens": 0 }
            }
        }"#;

        let events = p.parse_event("message_start", data).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::ResponseMeta { id, model }
            if id == "msg_01" && model == "claude-sonnet-4-20250514"
        ));
        assert_eq!(p.input_tokens, Some(25));
    }

    #[test]
    fn text_content_delta() {
        let mut p = parser();

        // content_block_start (text) → no output
        let start = r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}"#;
        let events = p.parse_event("content_block_start", start).unwrap();
        assert!(events.is_empty());

        // content_block_delta (text_delta) → ContentDelta
        let delta = r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}"#;
        let events = p.parse_event("content_block_delta", delta).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::ContentDelta(s) if s == "Hello"
        ));
    }

    #[test]
    fn tool_call_streaming() {
        let mut p = parser();

        // content_block_start (tool_use) → ToolCallStart
        let start = r#"{
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "get_weather",
                "input": {}
            }
        }"#;
        let events = p.parse_event("content_block_start", start).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::ToolCallStart { index: 0, id, name }
            if id == "toolu_01" && name == "get_weather"
        ));

        // content_block_delta (input_json_delta) → ToolCallDelta
        let delta = r#"{"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"loc"}}"#;
        let events = p.parse_event("content_block_delta", delta).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::ToolCallDelta { index: 0, arguments }
            if arguments == "{\"loc"
        ));
    }

    #[test]
    fn multiple_tool_calls_increment_index() {
        let mut p = parser();

        // First tool
        let start1 = r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "fn1", "input": {}}}"#;
        let events = p.parse_event("", start1).unwrap();
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::ToolCallStart { index: 0, .. }
        ));

        // Second tool
        let start2 = r#"{"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t2", "name": "fn2", "input": {}}}"#;
        let events = p.parse_event("", start2).unwrap();
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::ToolCallStart { index: 1, .. }
        ));
    }

    #[test]
    fn message_delta_emits_finish_and_usage() {
        let mut p = parser();
        p.input_tokens = Some(25);

        let data = r#"{
            "type": "message_delta",
            "delta": { "stop_reason": "end_turn", "stop_sequence": null },
            "usage": { "output_tokens": 15 }
        }"#;

        let events = p.parse_event("message_delta", data).unwrap();
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::Finish(FinishReason::Stop)
        ));
        match &events[1] {
            CanonicalStreamEvent::Usage(u) => {
                assert_eq!(u.prompt_tokens, Some(25));
                assert_eq!(u.completion_tokens, Some(15));
                assert_eq!(u.total_tokens, Some(40));
            }
            other => panic!("expected Usage, got {other:?}"),
        }
    }

    #[test]
    fn message_stop_emits_done() {
        let mut p = parser();
        let data = r#"{"type": "message_stop"}"#;
        let events = p.parse_event("message_stop", data).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], CanonicalStreamEvent::Done));
        assert!(p.done);
    }

    #[test]
    fn ping_is_ignored() {
        let mut p = parser();
        let data = r#"{"type": "ping"}"#;
        let events = p.parse_event("ping", data).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn error_event_returns_err() {
        let mut p = parser();
        let data =
            r#"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#;
        let err = p.parse_event("error", data).unwrap_err();
        assert!(matches!(err, TranslateError::StreamParse { .. }));
    }

    #[test]
    fn finish_emits_done_if_not_already() {
        let mut p = parser();
        let events = p.finish().unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], CanonicalStreamEvent::Done));

        // Second call: no duplicate.
        let events = p.finish().unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn thinking_block_streams_reasoning_events() {
        let mut p = parser();

        // open thinking block
        let start = r#"{
            "type": "content_block_start",
            "index": 0,
            "content_block": { "type": "thinking", "thinking": "", "signature": "" }
        }"#;
        let events = p.parse_event("content_block_start", start).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::ReasoningStart { index: 0, source: Some(ThinkingSource::Anthropic) }
        ));

        // thinking_delta → ReasoningDelta
        let d1 = r#"{
            "type": "content_block_delta", "index": 0,
            "delta": { "type": "thinking_delta", "thinking": "Let me " }
        }"#;
        let events = p.parse_event("content_block_delta", d1).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::ReasoningDelta(s) if s == "Let me "
        ));

        let d2 = r#"{
            "type": "content_block_delta", "index": 0,
            "delta": { "type": "thinking_delta", "thinking": "think." }
        }"#;
        let events = p.parse_event("content_block_delta", d2).unwrap();
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::ReasoningDelta(s) if s == "think."
        ));

        // signature_delta is buffered, no output yet
        let sig = r#"{
            "type": "content_block_delta", "index": 0,
            "delta": { "type": "signature_delta", "signature": "ErWj" }
        }"#;
        let events = p.parse_event("content_block_delta", sig).unwrap();
        assert!(events.is_empty(), "signature_delta is buffered");

        let sig2 = r#"{
            "type": "content_block_delta", "index": 0,
            "delta": { "type": "signature_delta", "signature": "Kl123" }
        }"#;
        let events = p.parse_event("content_block_delta", sig2).unwrap();
        assert!(events.is_empty());

        // content_block_stop emits ReasoningEnd with concatenated signature
        let stop = r#"{ "type": "content_block_stop", "index": 0 }"#;
        let events = p.parse_event("content_block_stop", stop).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::ReasoningEnd { index: 0, signature } if signature == "ErWjKl123"
        ));
    }

    #[test]
    fn thinking_then_text_uses_separate_indices() {
        let mut p = parser();

        // First a thinking block
        let s1 = r#"{
            "type":"content_block_start","index":0,
            "content_block":{"type":"thinking","thinking":"","signature":""}
        }"#;
        let e = p.parse_event("", s1).unwrap();
        assert!(matches!(
            e[0],
            CanonicalStreamEvent::ReasoningStart { index: 0, .. }
        ));

        let stop1 = r#"{"type":"content_block_stop","index":0}"#;
        let e = p.parse_event("", stop1).unwrap();
        assert!(matches!(
            e[0],
            CanonicalStreamEvent::ReasoningEnd { index: 0, .. }
        ));

        // Now a tool_use block — should start at tool_call_index 0 (separate counter)
        let s2 = r#"{
            "type":"content_block_start","index":1,
            "content_block":{"type":"tool_use","id":"t1","name":"fn","input":{}}
        }"#;
        let e = p.parse_event("", s2).unwrap();
        assert!(matches!(
            e[0],
            CanonicalStreamEvent::ToolCallStart { index: 0, .. }
        ));
    }

    #[test]
    fn redacted_thinking_skipped_in_stream() {
        let mut p = parser();
        let start = r#"{
            "type":"content_block_start","index":0,
            "content_block":{"type":"redacted_thinking","data":"blob"}
        }"#;
        let e = p.parse_event("", start).unwrap();
        assert!(e.is_empty(), "redacted_thinking is currently skipped in stream");

        let stop = r#"{"type":"content_block_stop","index":0}"#;
        let e = p.parse_event("", stop).unwrap();
        assert!(e.is_empty(), "stop without thinking emits nothing");
    }

    #[test]
    fn full_stream_replay() {
        let mut p = parser();
        let mut all_events = Vec::new();

        let sequence = [
            r#"{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":0}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}"#,
            r#"{"type":"message_stop"}"#,
        ];

        for data in sequence {
            all_events.extend(p.parse_event("", data).unwrap());
        }

        // Verify the sequence.
        assert!(matches!(
            &all_events[0],
            CanonicalStreamEvent::ResponseMeta { .. }
        ));
        assert!(matches!(&all_events[1], CanonicalStreamEvent::ContentDelta(s) if s == "Hello"));
        assert!(matches!(&all_events[2], CanonicalStreamEvent::ContentDelta(s) if s == " world"));
        assert!(matches!(
            &all_events[3],
            CanonicalStreamEvent::Finish(FinishReason::Stop)
        ));
        assert!(matches!(&all_events[4], CanonicalStreamEvent::Usage(_)));
        assert!(matches!(&all_events[5], CanonicalStreamEvent::Done));
    }
}
