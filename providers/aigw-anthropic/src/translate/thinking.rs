//! Anthropic-side projection of canonical [`ThinkingRequest`] into the
//! provider-native shape.
//!
//! The [`AnthropicRequestTranslator`] holds an
//! [`AnthropicThinkingProjector`] (boxed as `dyn ThinkingProjector<…>`) and
//! invokes it during request translation. Users can swap the default
//! projector for one with a custom level→budget table, hard cap, or
//! adaptive-model matcher.
//!
//! [`ThinkingRequest`]: aigw_core::model::ThinkingRequest
//! [`AnthropicRequestTranslator`]: super::request::AnthropicRequestTranslator

use std::sync::Arc;

use aigw_core::model::{LevelBudgetTable, ThinkingLevel, ThinkingRequest};
use aigw_core::translate::ThinkingProjector;

use crate::types::ThinkingConfig;

/// Mutable target an Anthropic translator constructs while building its
/// `MessagesRequest`. Lives only for the duration of `translate_request`.
///
/// The projector mutates this; the translator unpacks it into the native
/// builder.
#[derive(Debug, Clone, Default)]
pub struct AnthropicThinkingTarget {
    /// Final value of `MessagesRequest.thinking`.
    pub thinking: Option<ThinkingConfig>,
    /// Final value of `MessagesRequest.max_tokens`. Translator initializes
    /// this from `req.max_tokens.unwrap_or(default_max_tokens)`; projector
    /// may bump it to fit a budget + headroom.
    pub max_tokens: u64,
    /// Effort string for `output_config: { effort }` (sibling of `thinking`).
    /// `Some(_)` = inject; `None` = leave whatever was there.
    pub output_config_effort: Option<&'static str>,
    /// If `true`, translator must remove any inherited `output_config` from
    /// `extra` (used in Disabled and adaptive-without-effort modes).
    pub clear_output_config: bool,
}

/// Default Anthropic thinking projector.
///
/// Behaviour summary:
/// - `Disabled`: `thinking: {type:"disabled"}`, scrub `output_config`.
/// - `Auto` on adaptive models (Claude 4.6+): `thinking: {type:"adaptive"}`,
///   no `output_config.effort`.
/// - `Auto` on legacy models: `thinking: {type:"enabled", budget_tokens}`
///   with `auto_default_budget`, `max_tokens` bumped if too small.
/// - `Budget(n)`: `enabled` with `n.min(hard_cap)`, `max_tokens` bumped.
/// - `Level(l)` on adaptive models: `adaptive` + `output_config.effort`
///   from level.
/// - `Level(l)` on legacy models: `enabled` with `levels.budget(l)`,
///   `max_tokens` bumped.
pub struct AnthropicThinkingProjector {
    /// Returns `true` if the model supports `thinking.type:"adaptive"`.
    /// Default matcher accepts `claude-opus-4-6` and `claude-sonnet-4-6`
    /// prefixes. Override with [`Self::with_adaptive_matcher`] for other
    /// model lists.
    pub is_adaptive_model: Arc<dyn Fn(&str) -> bool + Send + Sync>,
    /// Hard cap on `budget_tokens` (Anthropic API: 128_000).
    pub hard_cap: u32,
    /// Default budget for `Auto` on legacy (non-adaptive) models. The
    /// Anthropic API requires `budget_tokens` whenever `type == "enabled"`.
    pub auto_default_budget: u32,
    /// Headroom (tokens) added on top of `budget_tokens` when bumping
    /// `max_tokens`. Default: `max(budget / 10, 1024)`.
    pub headroom: fn(u32) -> u32,
    /// Level→budget table (used in legacy mode and as fallback when adaptive
    /// effort isn't a level).
    pub levels: LevelBudgetTable,
}

impl Default for AnthropicThinkingProjector {
    fn default() -> Self {
        Self {
            is_adaptive_model: Arc::new(|m| {
                m.starts_with("claude-opus-4-6") || m.starts_with("claude-sonnet-4-6")
            }),
            hard_cap: 128_000,
            auto_default_budget: 10_000,
            headroom: |b| (b / 10).max(1024),
            levels: LevelBudgetTable::default(),
        }
    }
}

impl AnthropicThinkingProjector {
    /// Builder: replace the adaptive-model matcher.
    #[must_use]
    pub fn with_adaptive_matcher<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.is_adaptive_model = Arc::new(f);
        self
    }

    /// Builder: replace the level→budget table.
    #[must_use]
    pub fn with_levels(mut self, levels: LevelBudgetTable) -> Self {
        self.levels = levels;
        self
    }

    fn set_legacy(&self, target: &mut AnthropicThinkingTarget, budget: u32) {
        let needed = u64::from(budget.saturating_add((self.headroom)(budget)));
        if target.max_tokens < needed {
            target.max_tokens = needed;
        }
        target.thinking = Some(ThinkingConfig::Enabled {
            budget_tokens: u64::from(budget),
        });
    }
}

impl ThinkingProjector<AnthropicThinkingTarget> for AnthropicThinkingProjector {
    fn apply(
        &self,
        model: &str,
        req: Option<&ThinkingRequest>,
        target: &mut AnthropicThinkingTarget,
    ) {
        let Some(req) = req else { return };
        let adaptive = (self.is_adaptive_model)(model);

        match req {
            ThinkingRequest::Disabled => {
                target.thinking = Some(ThinkingConfig::Disabled);
                target.clear_output_config = true;
            }
            ThinkingRequest::Auto if adaptive => {
                target.thinking = Some(ThinkingConfig::Adaptive);
                target.clear_output_config = true;
            }
            ThinkingRequest::Auto => {
                let b = self.auto_default_budget.min(self.hard_cap);
                self.set_legacy(target, b);
            }
            ThinkingRequest::Budget { budget_tokens } => {
                let b = (*budget_tokens).min(self.hard_cap);
                self.set_legacy(target, b);
            }
            ThinkingRequest::Level { level } if adaptive => {
                target.thinking = Some(ThinkingConfig::Adaptive);
                target.output_config_effort = Some(level_to_claude_effort(*level));
            }
            ThinkingRequest::Level { level } => {
                let b = self.levels.budget(*level).min(self.hard_cap);
                self.set_legacy(target, b);
            }
        }
    }
}

/// Maps a thinking level to a Claude adaptive effort string.
///
/// Claude 4.6 supports `low`, `medium`, `high`, `max`. `Minimal`/`Low` both
/// collapse to `"low"`; `High`/`XHigh` collapse to `"high"`; `Max` is the
/// only level that maps to `"max"`.
const fn level_to_claude_effort(l: ThinkingLevel) -> &'static str {
    match l {
        ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High | ThinkingLevel::XHigh => "high",
        ThinkingLevel::Max => "max",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projector() -> AnthropicThinkingProjector {
        AnthropicThinkingProjector::default()
    }

    fn target_with_max(mt: u64) -> AnthropicThinkingTarget {
        AnthropicThinkingTarget {
            max_tokens: mt,
            ..Default::default()
        }
    }

    #[test]
    fn no_request_is_noop() {
        let mut t = target_with_max(8192);
        projector().apply("claude-opus-4-5", None, &mut t);
        assert!(t.thinking.is_none());
        assert_eq!(t.max_tokens, 8192);
        assert!(!t.clear_output_config);
        assert!(t.output_config_effort.is_none());
    }

    #[test]
    fn budget_legacy_bumps_max_tokens() {
        let mut t = target_with_max(100);
        projector().apply(
            "claude-opus-4-5",
            Some(&ThinkingRequest::Budget {
                budget_tokens: 10_000,
            }),
            &mut t,
        );
        match t.thinking {
            Some(ThinkingConfig::Enabled { budget_tokens }) => {
                assert_eq!(budget_tokens, 10_000);
            }
            other => panic!("expected Enabled(10_000), got {other:?}"),
        }
        // headroom = max(10_000/10, 1024) = 1024 → max_tokens = 11_024
        assert_eq!(t.max_tokens, 11_024);
    }

    #[test]
    fn budget_clamps_to_hard_cap() {
        let mut t = target_with_max(0);
        projector().apply(
            "claude-opus-4-5",
            Some(&ThinkingRequest::Budget {
                budget_tokens: 200_000,
            }),
            &mut t,
        );
        match t.thinking {
            Some(ThinkingConfig::Enabled { budget_tokens }) => {
                assert_eq!(budget_tokens, 128_000);
            }
            other => panic!("expected Enabled(128_000), got {other:?}"),
        }
    }

    #[test]
    fn level_legacy_uses_table_budget() {
        let mut t = target_with_max(1_000_000);
        projector().apply(
            "claude-opus-4-5",
            Some(&ThinkingRequest::Level {
                level: ThinkingLevel::High,
            }),
            &mut t,
        );
        match t.thinking {
            Some(ThinkingConfig::Enabled { budget_tokens }) => {
                assert_eq!(budget_tokens, 24_576);
            }
            other => panic!("expected Enabled, got {other:?}"),
        }
        // max_tokens already 1_000_000 — should NOT be lowered.
        assert_eq!(t.max_tokens, 1_000_000);
    }

    #[test]
    fn auto_legacy_uses_default_budget() {
        let mut t = target_with_max(0);
        projector().apply("claude-opus-4-5", Some(&ThinkingRequest::Auto), &mut t);
        match t.thinking {
            Some(ThinkingConfig::Enabled { budget_tokens }) => {
                assert_eq!(budget_tokens, 10_000);
            }
            other => panic!("expected Enabled(10_000), got {other:?}"),
        }
    }

    #[test]
    fn auto_adaptive_emits_adaptive_no_effort() {
        let mut t = target_with_max(8192);
        projector().apply("claude-opus-4-6", Some(&ThinkingRequest::Auto), &mut t);
        assert!(matches!(t.thinking, Some(ThinkingConfig::Adaptive)));
        assert!(t.output_config_effort.is_none());
        assert!(t.clear_output_config);
        assert_eq!(t.max_tokens, 8192);
    }

    #[test]
    fn level_adaptive_sets_effort() {
        let mut t = target_with_max(8192);
        projector().apply(
            "claude-sonnet-4-6",
            Some(&ThinkingRequest::Level {
                level: ThinkingLevel::High,
            }),
            &mut t,
        );
        assert!(matches!(t.thinking, Some(ThinkingConfig::Adaptive)));
        assert_eq!(t.output_config_effort, Some("high"));
        assert!(!t.clear_output_config);
    }

    #[test]
    fn level_adaptive_max_maps_to_max_effort() {
        let mut t = target_with_max(8192);
        projector().apply(
            "claude-opus-4-6",
            Some(&ThinkingRequest::Level {
                level: ThinkingLevel::Max,
            }),
            &mut t,
        );
        assert_eq!(t.output_config_effort, Some("max"));
    }

    #[test]
    fn disabled_scrubs_output_config() {
        let mut t = AnthropicThinkingTarget {
            max_tokens: 8192,
            output_config_effort: Some("high"),
            ..Default::default()
        };
        projector().apply(
            "claude-opus-4-6",
            Some(&ThinkingRequest::Disabled),
            &mut t,
        );
        assert!(matches!(t.thinking, Some(ThinkingConfig::Disabled)));
        assert!(t.clear_output_config);
    }

    #[test]
    fn custom_adaptive_matcher() {
        let p = projector().with_adaptive_matcher(|m| m == "my-special-model");
        let mut t = target_with_max(0);
        p.apply("my-special-model", Some(&ThinkingRequest::Auto), &mut t);
        assert!(matches!(t.thinking, Some(ThinkingConfig::Adaptive)));
    }

    #[test]
    fn dyn_dispatch_compiles() {
        let _: Box<dyn ThinkingProjector<AnthropicThinkingTarget>> =
            Box::new(AnthropicThinkingProjector::default());
    }
}
