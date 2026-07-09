//! Gateway configuration, loaded from a TOML file.
//!
//! The sidecar has exactly one inbound protocol (Anthropic `/v1/messages`) and
//! one outbound upstream, so the config is deliberately small:
//!
//! ```toml
//! [upstream]
//! base_url = "https://api.openai.com/v1"
//! api_key  = "sk-..."
//! wire     = "openai-chat"        # or "openai-responses" (not yet supported)
//!
//! # optional: map inbound Anthropic model names onto upstream model names
//! [upstream.models]
//! "claude-sonnet-4-20250514" = "gpt-4.1"
//!
//! # optional: fall back to this upstream model when no mapping matches
//! # default_model = "gpt-4.1"
//! # timeout_seconds = 600
//! # [upstream.default_headers]
//! # x-custom = "value"
//! ```

use std::collections::BTreeMap;
use std::fmt::{self, Debug, Formatter};
use std::path::Path;

use secrecy::SecretString;
use serde::Deserialize;
use serde::de::Deserializer;

/// Top-level gateway configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct GatewayConfig {
    /// The single outbound upstream all inbound traffic is routed to.
    pub upstream: UpstreamConfig,
}

impl GatewayConfig {
    /// Parse a [`GatewayConfig`] from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`toml::de::Error`] if the string is not valid
    /// TOML or is missing required fields (`upstream.base_url`,
    /// `upstream.api_key`).
    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Load a [`GatewayConfig`] from a TOML file on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or does not parse.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading config {}: {e}", path.display()))?;
        Self::from_toml_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing config {}: {e}", path.display()))
    }
}

/// The outbound upstream the gateway forwards to.
#[derive(Clone, Deserialize)]
pub struct UpstreamConfig {
    /// Base URL of the OpenAI-compatible upstream (e.g.
    /// `"https://api.openai.com/v1"`). The `/chat/completions` path is
    /// appended by the translator.
    pub base_url: String,
    /// Upstream API key, injected as `Authorization: Bearer <key>`. Never
    /// logged (redacted in `Debug`).
    #[serde(deserialize_with = "deserialize_secret_string")]
    pub api_key: SecretString,
    /// Upstream wire protocol. Defaults to [`Wire::OpenaiChat`].
    #[serde(default)]
    pub wire: Wire,
    /// Request timeout in seconds (applied as an idle read timeout so long
    /// streams aren't cut). Defaults to 600.
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
    /// Extra headers sent on every upstream request.
    #[serde(default)]
    pub default_headers: BTreeMap<String, String>,
    /// Inbound (Anthropic) model name → upstream model name.
    #[serde(default)]
    pub models: BTreeMap<String, String>,
    /// Upstream model used when no `models` entry matches the request. When
    /// unset, the inbound model name is forwarded unchanged.
    #[serde(default)]
    pub default_model: Option<String>,
}

impl UpstreamConfig {
    /// Resolve an inbound (Anthropic) model name to the upstream model name,
    /// applying `models` mapping then `default_model`, else passing through.
    #[must_use]
    pub fn resolve_model(&self, requested: &str) -> String {
        self.models
            .get(requested)
            .or(self.default_model.as_ref())
            .map_or_else(|| requested.to_owned(), Clone::clone)
    }
}

impl Debug for UpstreamConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpstreamConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &"[REDACTED]")
            .field("wire", &self.wire)
            .field("timeout_seconds", &self.timeout_seconds)
            .field("default_headers", &self.default_headers)
            .field("models", &self.models)
            .field("default_model", &self.default_model)
            .finish()
    }
}

/// Upstream wire protocol.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Wire {
    /// OpenAI Chat Completions (`POST /chat/completions`).
    #[default]
    OpenaiChat,
    /// OpenAI Responses API (`POST /responses`). Not yet implemented.
    OpenaiResponses,
}

fn deserialize_secret_string<'de, D>(deserializer: D) -> Result<SecretString, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(SecretString::from(s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;

    #[test]
    fn minimal_config_defaults_to_openai_chat() {
        let cfg = GatewayConfig::from_toml_str(
            r#"
            [upstream]
            base_url = "https://api.openai.com/v1"
            api_key = "sk-test"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.upstream.base_url, "https://api.openai.com/v1");
        assert_eq!(cfg.upstream.api_key.expose_secret(), "sk-test");
        assert_eq!(cfg.upstream.wire, Wire::OpenaiChat);
        assert!(cfg.upstream.timeout_seconds.is_none());
    }

    #[test]
    fn wire_parses_kebab_case() {
        let cfg = GatewayConfig::from_toml_str(
            r#"
            [upstream]
            base_url = "http://localhost:8000/v1"
            api_key = "none"
            wire = "openai-responses"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.upstream.wire, Wire::OpenaiResponses);
    }

    #[test]
    fn model_mapping_and_fallback() {
        let cfg = GatewayConfig::from_toml_str(
            r#"
            [upstream]
            base_url = "http://localhost:8000/v1"
            api_key = "none"
            default_model = "fallback-model"

            [upstream.models]
            "claude-sonnet-4-20250514" = "gpt-4.1"
            "#,
        )
        .unwrap();
        assert_eq!(
            cfg.upstream.resolve_model("claude-sonnet-4-20250514"),
            "gpt-4.1"
        );
        assert_eq!(
            cfg.upstream.resolve_model("claude-opus-4-6"),
            "fallback-model"
        );
    }

    #[test]
    fn model_passthrough_without_mapping() {
        let cfg = GatewayConfig::from_toml_str(
            r#"
            [upstream]
            base_url = "http://localhost:8000/v1"
            api_key = "none"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.upstream.resolve_model("some-model"), "some-model");
    }

    #[test]
    fn debug_redacts_api_key() {
        let cfg = GatewayConfig::from_toml_str(
            r#"
            [upstream]
            base_url = "http://localhost:8000/v1"
            api_key = "super-secret"
            "#,
        )
        .unwrap();
        let debug = format!("{:?}", cfg.upstream);
        assert!(!debug.contains("super-secret"), "api_key leaked in Debug");
    }
}
