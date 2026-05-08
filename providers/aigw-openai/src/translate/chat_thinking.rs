//! OpenAI Chat Completions / openai-compat–side projection of canonical
//! [`ThinkingRequest`].
//!
//! Chat Completions exposes thinking via a single top-level
//! `reasoning_effort` field (`"minimal"|"low"|"medium"|"high"`). Unlike
//! the Responses API there's no nested `reasoning` object — `summary`
//! and other reasoning controls aren't supported on this surface.
//!
//! Two integration points:
//!
//! - [`OpenAIChatThinkingProjector`] wires into a typed
//!   [`OpenAIRequestTranslator`] (or any
//!   [`ThinkingProjector<OpenAIChatThinkingTarget>`]) for callers that go
//!   through the typed translator.
//! - [`apply_thinking_to_chat_body`] is the Value-level helper for
//!   callers that hold a raw JSON body (e.g. proxy gateways serving
//!   provider-specific upstreams that don't deserialize through aigw's
//!   typed model). It mutates `body` in place: writes
//!   `reasoning_effort` and removes the canonical `thinking` field
//!   (which OpenAI's Chat Completions API doesn't accept).
//!
//! [`ThinkingRequest`]: aigw_core::model::ThinkingRequest
//! [`OpenAIRequestTranslator`]: super::request::OpenAIRequestTranslator

use aigw_core::model::{ThinkingLevel, ThinkingRequest};
use aigw_core::translate::ThinkingProjector;
use serde_json::Value;

/// Mutable target the Chat Completions translator constructs while
/// assembling its request body. The translator unpacks the projector's
/// output into the wire body.
#[derive(Debug, Clone, Default)]
pub struct OpenAIChatThinkingTarget {
    /// Final value of the `reasoning_effort` wire field. `None` = leave
    /// unset (let the API/model decide).
    pub reasoning_effort: Option<String>,
    /// If `true`, the translator must omit the field entirely. Used when
    /// canonical `Disabled` is requested and the caller wants to suppress
    /// any inherited default.
    pub disable: bool,
}

/// Default Chat Completions thinking projector.
///
/// Maps canonical [`ThinkingRequest`] onto the `reasoning_effort` axis.
/// Behaviour matches [`OpenAIResponsesThinkingProjector`] except the
/// target writes a flat `reasoning_effort` field instead of a nested
/// `reasoning.effort`.
///
/// [`OpenAIResponsesThinkingProjector`]: super::responses_thinking::OpenAIResponsesThinkingProjector
pub struct OpenAIChatThinkingProjector {
    /// Budget→effort thresholds: `(low_le, medium_le)`. Values ≤ `low_le`
    /// → `"low"`; ≤ `medium_le` → `"medium"`; otherwise `"high"`.
    pub budget_thresholds: (u32, u32),
}

impl Default for OpenAIChatThinkingProjector {
    fn default() -> Self {
        Self {
            budget_thresholds: (1_024, 8_192),
        }
    }
}

impl OpenAIChatThinkingProjector {
    /// Builder: replace the budget→effort thresholds.
    #[must_use]
    pub fn with_budget_thresholds(mut self, low_le: u32, medium_le: u32) -> Self {
        self.budget_thresholds = (low_le, medium_le);
        self
    }

    fn budget_to_effort(&self, budget: u32) -> &'static str {
        let (lo, mid) = self.budget_thresholds;
        if budget <= lo {
            "low"
        } else if budget <= mid {
            "medium"
        } else {
            "high"
        }
    }
}

impl ThinkingProjector<OpenAIChatThinkingTarget> for OpenAIChatThinkingProjector {
    fn apply(
        &self,
        _model: &str,
        req: Option<&ThinkingRequest>,
        target: &mut OpenAIChatThinkingTarget,
    ) {
        let Some(req) = req else { return };
        match req {
            ThinkingRequest::Disabled => {
                target.disable = true;
            }
            ThinkingRequest::Auto => {
                // Leave effort unset; Chat Completions has no explicit
                // "auto" — the API picks the model default.
            }
            ThinkingRequest::Budget { budget_tokens } => {
                target.reasoning_effort = Some(self.budget_to_effort(*budget_tokens).to_owned());
            }
            ThinkingRequest::Level { level } => {
                target.reasoning_effort = Some(level_to_chat_effort(*level).to_owned());
            }
        }
    }
}

/// Map a canonical level to the Chat Completions `reasoning_effort`
/// vocabulary (`minimal`/`low`/`medium`/`high`). `XHigh` and `Max`
/// collapse to `"high"` since the wire vocabulary stops there.
const fn level_to_chat_effort(l: ThinkingLevel) -> &'static str {
    match l {
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High | ThinkingLevel::XHigh | ThinkingLevel::Max => "high",
    }
}

// ─── Value-level helper ─────────────────────────────────────────────────────

/// Project a canonical [`ThinkingRequest`] onto a Chat Completions /
/// openai-compat JSON body.
///
/// Mutates `body` in place: writes `reasoning_effort` (when the
/// projector produced one) and unconditionally removes the canonical
/// `thinking` field — OpenAI's Chat Completions API rejects unknown
/// top-level fields, and `thinking` is canonical-only.
///
/// `model` is forwarded to the projector so a custom projector can do
/// model-specific dispatch; the default projector ignores it.
///
/// Use this when you hold a raw JSON body (a proxy that doesn't go
/// through [`OpenAIRequestTranslator`]) and need the same projection
/// the typed translator would apply.
///
/// [`OpenAIRequestTranslator`]: super::request::OpenAIRequestTranslator
pub fn apply_thinking_to_chat_body(body: &mut Value, req: &ThinkingRequest, model: &str) {
    let projector = OpenAIChatThinkingProjector::default();
    let mut target = OpenAIChatThinkingTarget::default();
    projector.apply(model, Some(req), &mut target);

    if let Some(obj) = body.as_object_mut() {
        // Canonical-only field — never sent on the wire.
        obj.remove("thinking");

        if target.disable {
            obj.remove("reasoning_effort");
            return;
        }
        if let Some(effort) = target.reasoning_effort {
            obj.insert("reasoning_effort".into(), Value::String(effort));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn projector() -> OpenAIChatThinkingProjector {
        OpenAIChatThinkingProjector::default()
    }

    #[test]
    fn no_request_is_noop() {
        let mut t = OpenAIChatThinkingTarget::default();
        projector().apply("gpt-5", None, &mut t);
        assert!(t.reasoning_effort.is_none());
        assert!(!t.disable);
    }

    #[test]
    fn disabled_sets_flag() {
        let mut t = OpenAIChatThinkingTarget::default();
        projector().apply("gpt-5", Some(&ThinkingRequest::Disabled), &mut t);
        assert!(t.disable);
    }

    #[test]
    fn auto_leaves_effort_unset() {
        let mut t = OpenAIChatThinkingTarget::default();
        projector().apply("gpt-5", Some(&ThinkingRequest::Auto), &mut t);
        assert!(t.reasoning_effort.is_none());
    }

    #[test]
    fn budget_threshold_low() {
        let mut t = OpenAIChatThinkingTarget::default();
        projector().apply(
            "gpt-5",
            Some(&ThinkingRequest::Budget { budget_tokens: 500 }),
            &mut t,
        );
        assert_eq!(t.reasoning_effort.as_deref(), Some("low"));
    }

    #[test]
    fn budget_threshold_medium() {
        let mut t = OpenAIChatThinkingTarget::default();
        projector().apply(
            "gpt-5",
            Some(&ThinkingRequest::Budget {
                budget_tokens: 4_000,
            }),
            &mut t,
        );
        assert_eq!(t.reasoning_effort.as_deref(), Some("medium"));
    }

    #[test]
    fn budget_threshold_high() {
        let mut t = OpenAIChatThinkingTarget::default();
        projector().apply(
            "gpt-5",
            Some(&ThinkingRequest::Budget {
                budget_tokens: 50_000,
            }),
            &mut t,
        );
        assert_eq!(t.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn level_minimal_maps_to_minimal() {
        let mut t = OpenAIChatThinkingTarget::default();
        projector().apply(
            "gpt-5",
            Some(&ThinkingRequest::Level {
                level: ThinkingLevel::Minimal,
            }),
            &mut t,
        );
        assert_eq!(t.reasoning_effort.as_deref(), Some("minimal"));
    }

    #[test]
    fn level_max_collapses_to_high() {
        let mut t = OpenAIChatThinkingTarget::default();
        projector().apply(
            "gpt-5",
            Some(&ThinkingRequest::Level {
                level: ThinkingLevel::Max,
            }),
            &mut t,
        );
        assert_eq!(t.reasoning_effort.as_deref(), Some("high"));
    }

    // ── Value-level helper ───────────────────────────────────────────────

    #[test]
    fn apply_to_body_writes_reasoning_effort() {
        let mut body = json!({"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]});
        apply_thinking_to_chat_body(
            &mut body,
            &ThinkingRequest::Level {
                level: ThinkingLevel::High,
            },
            "gpt-5",
        );
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn apply_to_body_strips_canonical_thinking() {
        let mut body = json!({
            "model": "gpt-5",
            "messages": [],
            "thinking": {"mode": "level", "level": "high"}
        });
        apply_thinking_to_chat_body(
            &mut body,
            &ThinkingRequest::Level {
                level: ThinkingLevel::High,
            },
            "gpt-5",
        );
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn apply_to_body_disabled_strips_existing_effort() {
        let mut body = json!({
            "model": "gpt-5",
            "messages": [],
            "reasoning_effort": "high"
        });
        apply_thinking_to_chat_body(&mut body, &ThinkingRequest::Disabled, "gpt-5");
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn apply_to_body_auto_omits_effort_but_strips_canonical() {
        let mut body = json!({
            "model": "gpt-5",
            "messages": [],
            "thinking": {"mode": "auto"}
        });
        apply_thinking_to_chat_body(&mut body, &ThinkingRequest::Auto, "gpt-5");
        assert!(body.get("thinking").is_none());
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn dyn_dispatch_compiles() {
        let _: Box<dyn ThinkingProjector<OpenAIChatThinkingTarget>> =
            Box::new(OpenAIChatThinkingProjector::default());
    }
}
