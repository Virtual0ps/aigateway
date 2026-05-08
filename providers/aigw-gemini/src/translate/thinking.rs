//! Gemini-side projection of canonical [`ThinkingRequest`].
//!
//! Gemini exposes thinking via two mutually-related fields under
//! `generationConfig.thinkingConfig`:
//!
//! - **`thinkingBudget`** (Gemini 2.5): integer token budget. Special
//!   values: `-1` = dynamic thinking (let the API decide), `0` = disabled,
//!   otherwise `128`–`32768`.
//! - **`thinkingLevel`** (Gemini 3): one of `MINIMAL`/`LOW`/`MEDIUM`/`HIGH`.
//!
//! The two fields don't co-exist; the projector picks based on a configurable
//! `is_gemini_3_model` matcher.
//!
//! [`ThinkingRequest`]: aigw_core::model::ThinkingRequest

use std::sync::Arc;

use aigw_core::model::{LevelBudgetTable, ThinkingLevel as CanonicalLevel, ThinkingRequest};
use aigw_core::translate::ThinkingProjector;

use crate::types::{ThinkingConfig as NativeThinkingConfig, ThinkingLevel as NativeThinkingLevel};

/// Mutable target the Gemini translator constructs while building its
/// `GenerationConfig`. The translator unpacks `config` into
/// `generation_config.thinking_config`.
#[derive(Debug, Clone, Default)]
pub struct GeminiThinkingTarget {
    /// Final value of `GenerationConfig.thinking_config`.
    pub config: Option<NativeThinkingConfig>,
}

/// Default Gemini thinking projector.
///
/// Behaviour summary (canonical → native):
/// - `Disabled`: `thinking_budget: 0`. Works on both Gemini 2.5 and 3.
/// - `Auto`: `thinking_budget: -1` (dynamic) on Gemini 2.5; **omit** on
///   Gemini 3 (the API picks a default level).
/// - `Budget(n)`: `thinking_budget: n` (clamped to `i64::MAX` on overflow).
///   Honored as-is on both API generations — Gemini 2.5 documents
///   `128..=32768`, but the API itself accepts the field on G3 too as a
///   compatibility shim.
/// - `Level(l)`: on Gemini 3 → `thinking_level` enum; on Gemini 2.5 →
///   `thinking_budget` derived from [`LevelBudgetTable`].
pub struct GeminiThinkingProjector {
    /// Returns `true` if the model accepts the Gemini-3-style
    /// `thinking_level` field. Default matcher: any model id starting with
    /// `gemini-3` or `gemini-thinking-3` (forward-compat).
    pub is_gemini_3_model: Arc<dyn Fn(&str) -> bool + Send + Sync>,
    /// Level→budget table for Gemini 2.5 fallback.
    pub levels: LevelBudgetTable,
}

impl Default for GeminiThinkingProjector {
    fn default() -> Self {
        Self {
            is_gemini_3_model: Arc::new(|m| {
                m.starts_with("gemini-3") || m.starts_with("gemini-thinking-3")
            }),
            levels: LevelBudgetTable::default(),
        }
    }
}

impl GeminiThinkingProjector {
    /// Builder: replace the Gemini-3 matcher.
    #[must_use]
    pub fn with_gemini_3_matcher<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.is_gemini_3_model = Arc::new(f);
        self
    }

    /// Builder: replace the level→budget table used for Gemini 2.5 fallback.
    #[must_use]
    pub fn with_levels(mut self, levels: LevelBudgetTable) -> Self {
        self.levels = levels;
        self
    }
}

impl ThinkingProjector<GeminiThinkingTarget> for GeminiThinkingProjector {
    fn apply(
        &self,
        model: &str,
        req: Option<&ThinkingRequest>,
        target: &mut GeminiThinkingTarget,
    ) {
        let Some(req) = req else { return };
        let g3 = (self.is_gemini_3_model)(model);

        match req {
            ThinkingRequest::Disabled => {
                target.config = Some(NativeThinkingConfig {
                    thinking_budget: Some(0),
                    thinking_level: None,
                    include_thoughts: None,
                });
            }
            ThinkingRequest::Auto if g3 => {
                // Gemini 3 has no explicit "auto"; omitting thinkingConfig
                // entirely lets the API pick. Leave target.config = None.
            }
            ThinkingRequest::Auto => {
                target.config = Some(NativeThinkingConfig {
                    thinking_budget: Some(-1),
                    thinking_level: None,
                    include_thoughts: None,
                });
            }
            ThinkingRequest::Budget { budget_tokens } => {
                target.config = Some(NativeThinkingConfig {
                    thinking_budget: Some(i64::from(*budget_tokens)),
                    thinking_level: None,
                    include_thoughts: None,
                });
            }
            ThinkingRequest::Level { level } if g3 => {
                target.config = Some(NativeThinkingConfig {
                    thinking_budget: None,
                    thinking_level: Some(canonical_level_to_native(*level)),
                    include_thoughts: None,
                });
            }
            ThinkingRequest::Level { level } => {
                let budget = self.levels.budget(*level);
                target.config = Some(NativeThinkingConfig {
                    thinking_budget: Some(i64::from(budget)),
                    thinking_level: None,
                    include_thoughts: None,
                });
            }
        }
    }
}

/// Map canonical [`CanonicalLevel`] to native [`NativeThinkingLevel`].
///
/// Gemini's wire enum doesn't have `XHigh` or `Max`; both collapse to
/// `High` (the wire vocabulary stops there).
fn canonical_level_to_native(l: CanonicalLevel) -> NativeThinkingLevel {
    match l {
        CanonicalLevel::Minimal => NativeThinkingLevel::Minimal,
        CanonicalLevel::Low => NativeThinkingLevel::Low,
        CanonicalLevel::Medium => NativeThinkingLevel::Medium,
        CanonicalLevel::High | CanonicalLevel::XHigh | CanonicalLevel::Max => {
            NativeThinkingLevel::High
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projector() -> GeminiThinkingProjector {
        GeminiThinkingProjector::default()
    }

    #[test]
    fn no_request_is_noop() {
        let mut t = GeminiThinkingTarget::default();
        projector().apply("gemini-2.5-flash", None, &mut t);
        assert!(t.config.is_none());
    }

    #[test]
    fn disabled_sets_budget_zero() {
        let mut t = GeminiThinkingTarget::default();
        projector().apply(
            "gemini-2.5-flash",
            Some(&ThinkingRequest::Disabled),
            &mut t,
        );
        let c = t.config.unwrap();
        assert_eq!(c.thinking_budget, Some(0));
        assert!(c.thinking_level.is_none());
    }

    #[test]
    fn auto_g25_sets_budget_minus_one() {
        let mut t = GeminiThinkingTarget::default();
        projector().apply("gemini-2.5-pro", Some(&ThinkingRequest::Auto), &mut t);
        let c = t.config.unwrap();
        assert_eq!(c.thinking_budget, Some(-1));
    }

    #[test]
    fn auto_g3_omits_thinking_config() {
        let mut t = GeminiThinkingTarget::default();
        projector().apply("gemini-3-pro", Some(&ThinkingRequest::Auto), &mut t);
        assert!(
            t.config.is_none(),
            "Gemini 3 Auto must omit thinking_config"
        );
    }

    #[test]
    fn budget_passes_through() {
        let mut t = GeminiThinkingTarget::default();
        projector().apply(
            "gemini-2.5-flash",
            Some(&ThinkingRequest::Budget {
                budget_tokens: 16_384,
            }),
            &mut t,
        );
        let c = t.config.unwrap();
        assert_eq!(c.thinking_budget, Some(16_384));
    }

    #[test]
    fn level_g3_sets_native_level() {
        let mut t = GeminiThinkingTarget::default();
        projector().apply(
            "gemini-3-pro",
            Some(&ThinkingRequest::Level {
                level: CanonicalLevel::High,
            }),
            &mut t,
        );
        let c = t.config.unwrap();
        assert!(c.thinking_budget.is_none());
        assert_eq!(c.thinking_level, Some(NativeThinkingLevel::High));
    }

    #[test]
    fn level_g25_falls_back_to_budget() {
        let mut t = GeminiThinkingTarget::default();
        projector().apply(
            "gemini-2.5-flash",
            Some(&ThinkingRequest::Level {
                level: CanonicalLevel::Medium,
            }),
            &mut t,
        );
        let c = t.config.unwrap();
        assert_eq!(c.thinking_budget, Some(8_192));
        assert!(c.thinking_level.is_none());
    }

    #[test]
    fn level_xhigh_g3_collapses_to_high() {
        let mut t = GeminiThinkingTarget::default();
        projector().apply(
            "gemini-3-pro",
            Some(&ThinkingRequest::Level {
                level: CanonicalLevel::XHigh,
            }),
            &mut t,
        );
        let c = t.config.unwrap();
        assert_eq!(c.thinking_level, Some(NativeThinkingLevel::High));
    }

    #[test]
    fn custom_g3_matcher() {
        let p = projector().with_gemini_3_matcher(|m| m == "my-future-model");
        let mut t = GeminiThinkingTarget::default();
        p.apply(
            "my-future-model",
            Some(&ThinkingRequest::Level {
                level: CanonicalLevel::Low,
            }),
            &mut t,
        );
        let c = t.config.unwrap();
        assert_eq!(c.thinking_level, Some(NativeThinkingLevel::Low));
    }

    #[test]
    fn dyn_dispatch_compiles() {
        let _: Box<dyn ThinkingProjector<GeminiThinkingTarget>> =
            Box::new(GeminiThinkingProjector::default());
    }
}
