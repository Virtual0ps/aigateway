//! OpenAI Responses–side projection of canonical [`ThinkingRequest`].
//!
//! The Responses API exposes thinking via `reasoning.effort`
//! (`"minimal"|"low"|"medium"|"high"`) and an orthogonal `reasoning.summary`
//! (`"auto"|"concise"|"detailed"`). The projector maps the canonical
//! [`ThinkingRequest`] onto the `effort` axis; `summary` is left untouched
//! (translator config defaults handle that).
//!
//! [`ThinkingRequest`]: aigw_core::model::ThinkingRequest

use aigw_core::model::{ThinkingLevel, ThinkingRequest};
use aigw_core::translate::ThinkingProjector;

/// Mutable target the Responses translator constructs while assembling the
/// `reasoning` object. The translator merges this with `extra.reasoning` /
/// `extra.reasoning_effort` / config defaults; canonical (this target)
/// takes priority.
#[derive(Debug, Clone, Default)]
pub struct ResponsesThinkingTarget {
    /// Final `reasoning.effort` value. `Some(_)` = set by projector;
    /// `None` = projector deferred to extra/defaults.
    pub effort: Option<String>,
    /// If `true`, the translator must omit the `reasoning` field entirely.
    /// Used when canonical `Disabled` is requested — no effort, no summary,
    /// no extra-passthrough.
    pub disable: bool,
}

/// Default Responses-API thinking projector.
///
/// Behaviour summary:
/// - `Disabled`: set `disable = true`. Translator drops the `reasoning`
///   field entirely (closest the API offers to "off").
/// - `Auto`: leave `effort` unset. Translator falls through to extra /
///   config defaults — equivalent to "let the API/config decide".
/// - `Budget(n)`: threshold-based — `≤1024 → "low"`, `≤8192 → "medium"`,
///   `>8192 → "high"`. Override via [`Self::with_budget_thresholds`].
/// - `Level(l)`: [`ThinkingLevel::default_effort`] (Minimal/Low → `"low"`,
///   Medium → `"medium"`, High/XHigh/Max → `"high"`). The Responses API
///   wire vocabulary doesn't have a `"max"` effort, so `Max` collapses to
///   `"high"` like the upstream `convert.go` table.
pub struct OpenAIResponsesThinkingProjector {
    /// Budget→effort thresholds. `(low_le, medium_le)` — values ≤ `low_le`
    /// map to `"low"`, ≤ `medium_le` map to `"medium"`, otherwise `"high"`.
    pub budget_thresholds: (u32, u32),
}

impl Default for OpenAIResponsesThinkingProjector {
    fn default() -> Self {
        Self {
            budget_thresholds: (1_024, 8_192),
        }
    }
}

impl OpenAIResponsesThinkingProjector {
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

impl ThinkingProjector<ResponsesThinkingTarget> for OpenAIResponsesThinkingProjector {
    fn apply(
        &self,
        _model: &str,
        req: Option<&ThinkingRequest>,
        target: &mut ResponsesThinkingTarget,
    ) {
        let Some(req) = req else { return };
        match req {
            ThinkingRequest::Disabled => {
                target.disable = true;
            }
            ThinkingRequest::Auto => {
                // Leave effort unset; translator falls through to defaults.
            }
            ThinkingRequest::Budget { budget_tokens } => {
                target.effort = Some(self.budget_to_effort(*budget_tokens).to_owned());
            }
            ThinkingRequest::Level { level } => {
                target.effort = Some(level_to_responses_effort(*level).to_owned());
            }
        }
    }
}

/// Maps a canonical level to the Responses API effort string.
///
/// Identical to [`ThinkingLevel::default_effort`] today; defined separately
/// so it can diverge later if the Responses API gains a `"max"` effort.
const fn level_to_responses_effort(l: ThinkingLevel) -> &'static str {
    match l {
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High | ThinkingLevel::XHigh | ThinkingLevel::Max => "high",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projector() -> OpenAIResponsesThinkingProjector {
        OpenAIResponsesThinkingProjector::default()
    }

    #[test]
    fn no_request_is_noop() {
        let mut t = ResponsesThinkingTarget::default();
        projector().apply("o4-mini", None, &mut t);
        assert!(t.effort.is_none());
        assert!(!t.disable);
    }

    #[test]
    fn disabled_sets_flag() {
        let mut t = ResponsesThinkingTarget::default();
        projector().apply("o4-mini", Some(&ThinkingRequest::Disabled), &mut t);
        assert!(t.disable);
    }

    #[test]
    fn auto_leaves_effort_unset() {
        let mut t = ResponsesThinkingTarget::default();
        projector().apply("o4-mini", Some(&ThinkingRequest::Auto), &mut t);
        assert!(t.effort.is_none());
    }

    #[test]
    fn budget_threshold_low() {
        let mut t = ResponsesThinkingTarget::default();
        projector().apply(
            "o4-mini",
            Some(&ThinkingRequest::Budget { budget_tokens: 500 }),
            &mut t,
        );
        assert_eq!(t.effort.as_deref(), Some("low"));
    }

    #[test]
    fn budget_threshold_medium() {
        let mut t = ResponsesThinkingTarget::default();
        projector().apply(
            "o4-mini",
            Some(&ThinkingRequest::Budget {
                budget_tokens: 4_000,
            }),
            &mut t,
        );
        assert_eq!(t.effort.as_deref(), Some("medium"));
    }

    #[test]
    fn budget_threshold_high() {
        let mut t = ResponsesThinkingTarget::default();
        projector().apply(
            "o4-mini",
            Some(&ThinkingRequest::Budget {
                budget_tokens: 50_000,
            }),
            &mut t,
        );
        assert_eq!(t.effort.as_deref(), Some("high"));
    }

    #[test]
    fn level_minimal_maps_to_minimal() {
        let mut t = ResponsesThinkingTarget::default();
        projector().apply(
            "o4-mini",
            Some(&ThinkingRequest::Level {
                level: ThinkingLevel::Minimal,
            }),
            &mut t,
        );
        assert_eq!(t.effort.as_deref(), Some("minimal"));
    }

    #[test]
    fn level_max_collapses_to_high() {
        let mut t = ResponsesThinkingTarget::default();
        projector().apply(
            "o4-mini",
            Some(&ThinkingRequest::Level {
                level: ThinkingLevel::Max,
            }),
            &mut t,
        );
        assert_eq!(t.effort.as_deref(), Some("high"));
    }

    #[test]
    fn custom_thresholds() {
        let p = projector().with_budget_thresholds(2_000, 16_000);
        let mut t = ResponsesThinkingTarget::default();
        p.apply(
            "o4-mini",
            Some(&ThinkingRequest::Budget {
                budget_tokens: 3_000,
            }),
            &mut t,
        );
        assert_eq!(t.effort.as_deref(), Some("medium"));
    }

    #[test]
    fn dyn_dispatch_compiles() {
        let _: Box<dyn ThinkingProjector<ResponsesThinkingTarget>> =
            Box::new(OpenAIResponsesThinkingProjector::default());
    }
}
