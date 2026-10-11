use std::collections::HashMap;
use std::fs;

use serde::Deserialize;
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Protocol {
    /// OpenAI-compatible POST {base}/chat/completions (pass-through).
    #[default]
    Chat,
    /// Anthropic POST {base}/messages (translated).
    Anthropic,
    /// OpenAI Responses POST {base}/responses (translated).
    Responses,
}

impl Protocol {
    pub fn as_str(&self) -> &'static str {
        match self {
            Protocol::Chat => "chat",
            Protocol::Anthropic => "anthropic",
            Protocol::Responses => "responses",
        }
    }
}

/// How the gateway renders `thinking` for Anthropic upstreams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingMode {
    /// Legacy `{"type":"enabled","budget_tokens":N}` (effort → budget).
    Enabled,
    /// `{"type":"adaptive"}` — upstream decides the budget.
    #[default]
    Adaptive,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelCfg {
    /// Public model id exposed to clients.
    pub name: String,
    /// Model id sent upstream. Defaults to `name`.
    #[serde(default)]
    pub upstream: Option<String>,
    pub protocol: Protocol,
    /// Base URL including version prefix, e.g. https://host/v1
    #[serde(default)]
    pub base_url: Option<String>,
    /// Extra headers merged for this model only.
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
    /// Anthropic upstream `thinking` shape: `adaptive` (default) or
    /// `enabled`. Only affects `protocol = "anthropic"` upstreams.
    #[serde(default)]
    pub thinking_type: Option<ThinkingMode>,
    /// Whether the model accepts image input (advertised via /v1/models).
    /// `None` (unset) lets models.dev inference fill it in; `Some(_)`
    /// always wins over inferred metadata.
    #[serde(default)]
    pub vision: Option<bool>,
    /// Explicit models.dev model id to match against. Defaults to matching
    /// by `upstream` (fallback `name`).
    #[serde(default)]
    pub models_dev_id: Option<String>,
    /// User overrides for models.dev-enriched metadata. Each `Some(_)`
    /// wins over the inferred value. Keys mirror models.dev model entries
    /// (except `id`, which is covered by `name`/`upstream`/`models_dev_id`).
    /// Display name (models.dev `name`; local `name` stays the routing key).
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    /// models.dev `type` (e.g. "embedding"). Rare; only a few entries set it.
    #[serde(default, rename = "type")]
    pub model_type: Option<String>,
    #[serde(default)]
    pub knowledge: Option<String>,
    #[serde(default)]
    pub release_date: Option<String>,
    #[serde(default)]
    pub last_updated: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub canonical_model_id: Option<String>,
    /// Capability flags.
    #[serde(default)]
    pub attachment: Option<bool>,
    #[serde(default)]
    pub reasoning: Option<bool>,
    /// Free-form list, e.g. `[{ type = "effort", values = ["none", "high"] }]`.
    #[serde(default)]
    pub reasoning_options: Option<JsonValue>,
    #[serde(default)]
    pub tool_call: Option<bool>,
    #[serde(default)]
    pub structured_output: Option<bool>,
    #[serde(default)]
    pub temperature: Option<bool>,
    #[serde(default)]
    pub open_weights: Option<bool>,
    /// Modality vocab: text/image/audio/video/pdf. Overrides drive `vision`
    /// inference when `vision` itself is unset.
    #[serde(default)]
    pub modalities_input: Option<Vec<String>>,
    #[serde(default)]
    pub modalities_output: Option<Vec<String>>,
    /// models.dev `limit` subkeys.
    #[serde(default)]
    pub context_limit: Option<u64>,
    #[serde(default)]
    pub input_limit: Option<u64>,
    #[serde(default)]
    pub output_limit: Option<u64>,
    /// models.dev `cost` subkeys (per-million-token prices).
    #[serde(default)]
    pub cost_input: Option<f64>,
    #[serde(default)]
    pub cost_output: Option<f64>,
    #[serde(default)]
    pub cost_cache_read: Option<f64>,
    #[serde(default)]
    pub cost_cache_write: Option<f64>,
    /// Tiered pricing list, e.g. `[{ input = 5.0, tier = { type = "context",
    /// size = 32000 } }]`.
    #[serde(default)]
    pub cost_tiers: Option<JsonValue>,
    /// Alternate price object for long context (same shape as `cost`).
    #[serde(default)]
    pub cost_context_over_200k: Option<JsonValue>,
    #[serde(default)]
    pub cost_reasoning: Option<f64>,
    #[serde(default)]
    pub cost_input_audio: Option<f64>,
    #[serde(default)]
    pub cost_output_audio: Option<f64>,
    /// Nested provider descriptors, e.g. `interleaved = { field = "..." }`.
    #[serde(default)]
    pub interleaved: Option<JsonValue>,
    /// models.dev per-model `provider` map. Renamed to avoid confusion with
    /// the top-level `provider` advertised by `/v1/models`.
    #[serde(default, rename = "provider")]
    pub model_provider: Option<JsonValue>,
    #[serde(default)]
    pub experimental: Option<JsonValue>,
}

impl ModelCfg {
    pub fn upstream_model(&self) -> &str {
        self.upstream.as_deref().unwrap_or(&self.name)
    }

    /// Backwards-compatible accessor: unset vision defaults to false.
    #[allow(dead_code)]
    pub fn vision_flag(&self) -> bool {
        self.vision.unwrap_or(false)
    }

    /// Thinking shape for Anthropic upstreams; unset defaults to `adaptive`.
    pub fn thinking_mode(&self) -> ThinkingMode {
        self.thinking_type.unwrap_or_default()
    }

    /// Id used for models.dev matching: explicit override, else upstream.
    pub fn match_id(&self) -> &str {
        self.models_dev_id
            .as_deref()
            .unwrap_or_else(|| self.upstream_model())
    }

    /// Constructor for auto-discovered models. Metadata stays unset so
    /// models.dev inference fills it in; explicit `[[models]]` entries
    /// always win over discovered ones at lookup time.
    pub fn discovered(name: String, protocol: Protocol, base_url: String) -> Self {
        Self {
            name,
            upstream: None,
            protocol,
            base_url: Some(base_url),
            extra_headers: HashMap::new(),
            thinking_type: None,
            vision: None,
            models_dev_id: None,
            display_name: None,
            description: None,
            family: None,
            model_type: None,
            knowledge: None,
            release_date: None,
            last_updated: None,
            status: None,
            canonical_model_id: None,
            attachment: None,
            reasoning: None,
            reasoning_options: None,
            tool_call: None,
            structured_output: None,
            temperature: None,
            open_weights: None,
            modalities_input: None,
            modalities_output: None,
            context_limit: None,
            input_limit: None,
            output_limit: None,
            cost_input: None,
            cost_output: None,
            cost_cache_read: None,
            cost_cache_write: None,
            cost_tiers: None,
            cost_context_over_200k: None,
            cost_reasoning: None,
            cost_input_audio: None,
            cost_output_audio: None,
            interleaved: None,
            model_provider: None,
            experimental: None,
        }
    }
}

fn default_models_dev_url() -> String {
    "https://models.dev/api.json".to_string()
}

fn default_refresh_interval() -> u64 {
    86400
}

fn default_watch_interval() -> u64 {
    30
}

fn default_discovery_path() -> String {
    "/models".to_string()
}

fn default_discovery_interval() -> u64 {
    3600
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelsDevCfg {
    /// Master switch. Default false so existing configs are unaffected.
    #[serde(default)]
    pub enable: bool,
    /// Full api.json URL.
    #[serde(default = "default_models_dev_url")]
    pub url: String,
    /// Restrict matching to one provider id (e.g. "opencode-go").
    /// Omit to search all providers.
    #[serde(default)]
    pub provider: Option<String>,
    /// Background refresh period. 0 disables periodic refresh.
    #[serde(default = "default_refresh_interval")]
    pub refresh_interval_secs: u64,
    /// Disk cache for offline startup (read on fetch failure, written on success).
    #[serde(default)]
    pub cache_path: Option<String>,
}

impl Default for ModelsDevCfg {
    fn default() -> Self {
        Self {
            enable: false,
            url: default_models_dev_url(),
            provider: None,
            refresh_interval_secs: default_refresh_interval(),
            cache_path: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReloadCfg {
    /// Master switch, read at startup. Enables SIGHUP handling (unix) and
    /// mtime file watching. Note: registering the SIGHUP handler replaces
    /// the default terminate-on-hangup behavior with a config reload.
    #[serde(default = "default_true")]
    pub enable: bool,
    /// mtime poll period. 0 disables polling (SIGHUP still works).
    #[serde(default = "default_watch_interval")]
    pub watch_interval_secs: u64,
}

impl Default for ReloadCfg {
    fn default() -> Self {
        Self {
            enable: true,
            watch_interval_secs: default_watch_interval(),
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiscoveryCfg {
    /// Master switch. Merges upstream `{base}{path}` ids into the model
    /// list. Explicit `[[models]]` entries always win over discovered ones.
    #[serde(default)]
    pub enable: bool,
    /// Base URL for the discovery endpoint. Defaults to `default_base_url`.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Path appended to the base URL. OpenAI-compatible `/models` shape.
    #[serde(default = "default_discovery_path")]
    pub path: String,
    /// Periodic refresh. Startup always fetches once when enabled;
    /// 0 disables periodic refresh.
    #[serde(default = "default_discovery_interval")]
    pub interval_secs: u64,
    /// Protocol assumed for discovered models. Pin special models with an
    /// explicit `[[models]]` entry instead.
    #[serde(default)]
    pub protocol: Protocol,
    /// Only discover ids starting with this prefix. Omit for all.
    #[serde(default)]
    pub prefix: Option<String>,
    /// Remove discovered models that vanish upstream. Explicit entries
    /// are never pruned. Default false (additive only).
    #[serde(default)]
    pub prune_missing: bool,
}

impl Default for DiscoveryCfg {
    fn default() -> Self {
        Self {
            enable: false,
            base_url: None,
            path: default_discovery_path(),
            interval_secs: default_discovery_interval(),
            protocol: Protocol::Chat,
            prefix: None,
            prune_missing: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_listen")]
    pub listen: String,
    /// Read the API key from this env var.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Or inline the key (not recommended).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Optional stable session id sent upstream via `session_header`.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Header name used to carry `session_id` upstream. Omit to send nothing.
    #[serde(default)]
    pub session_header: Option<String>,
    /// User-Agent advertised upstream.
    #[serde(default = "default_user_agent")]
    pub user_agent: String,
    /// Default base URL for models that do not set one.
    #[serde(default)]
    pub default_base_url: Option<String>,
    /// `owned_by` value advertised by GET /v1/models.
    #[serde(default)]
    pub owned_by: Option<String>,
    /// `provider` value advertised by GET /v1/models.
    #[serde(default)]
    pub provider: Option<String>,
    /// Extra headers applied to every upstream request.
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
    /// `max_tokens` the gateway sends to Anthropic upstreams when the client
    /// omitted it (Anthropic requires the field; Chat treats it as optional).
    /// A hardcoded default silently truncates long answers, so it is tunable.
    #[serde(default = "default_anthropic_max_tokens")]
    pub anthropic_max_tokens: i64,
    /// Reject requests carrying parameters the translation would silently
    /// drop (see PROTOCOL-AUDIT S-2). Off by default: warnings only.
    #[serde(default)]
    pub strict_params: bool,
    /// Largest client request body accepted, in bytes. Axum's built-in limit
    /// is 2 MiB, which rejects real workloads (base64 images, long agent
    /// histories) with a plain-text 413 that no SDK can parse. The gateway
    /// applies this limit itself so the rejection can carry the ingress
    /// protocol's error shape. 0 disables the limit.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// models.dev enrichment (modalities, price, limits) for /v1/models.
    #[serde(default)]
    pub models_dev: ModelsDevCfg,
    /// Hot-reload (SIGHUP + config file mtime watching).
    #[serde(default)]
    pub reload: ReloadCfg,
    /// Upstream model auto-discovery merged under explicit entries.
    #[serde(default)]
    pub discovery: DiscoveryCfg,
    pub models: Vec<ModelCfg>,
}

fn default_listen() -> String {
    "0.0.0.0:8787".to_string()
}
fn default_user_agent() -> String {
    format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
}
fn default_anthropic_max_tokens() -> i64 {
    8192
}
fn default_max_body_bytes() -> usize {
    // 32 MiB. Large enough for a few hundred turns of conversation or a
    // handful of base64 images, small enough that a runaway client cannot
    // exhaust memory before the read completes.
    32 * 1024 * 1024
}

impl Config {
    pub fn load(path: &str) -> Result<Config, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("read config {path}: {e}"))?;
        let cfg: Config = toml::from_str(&text).map_err(|e| format!("parse config: {e}"))?;
        if cfg.models.is_empty() {
            return Err("config has no [[models]]".to_string());
        }
        Ok(cfg)
    }

    pub fn api_key(&self) -> Result<String, String> {
        if let Some(env) = &self.api_key_env {
            if let Ok(v) = std::env::var(env) {
                if !v.is_empty() {
                    return Ok(v);
                }
            }
            return Err(format!("env var {env} is not set"));
        }
        if let Some(k) = &self.api_key {
            if !k.is_empty() {
                return Ok(k.clone());
            }
        }
        Err("no api key configured (api_key_env or api_key)".to_string())
    }

    pub fn base_url_for(&self, m: &ModelCfg) -> Result<String, String> {
        m.base_url
            .clone()
            .or_else(|| self.default_base_url.clone())
            .ok_or_else(|| format!("model {} has no base_url", m.name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_parity_keys_parse_from_toml() {
        let text = r#"
listen = "127.0.0.1:8787"
api_key = "x"
default_base_url = "https://api.example.com/v1"

[[models]]
name = "m"
protocol = "chat"
display_name = "M Display"
description = "d"
family = "f"
type = "embedding"
knowledge = "2024-12"
release_date = "2026-01-01"
last_updated = "2026-02-01"
status = "active"
canonical_model_id = "org/m"
attachment = true
reasoning = true
tool_call = true
structured_output = true
temperature = false
open_weights = true
modalities_input = ["text", "image"]
modalities_output = ["text"]
context_limit = 100
input_limit = 50
output_limit = 10
cost_input = 1.0
cost_output = 2.0
cost_cache_read = 0.1
cost_cache_write = 0.2
cost_reasoning = 0.6
cost_input_audio = 1.5
cost_output_audio = 3.0
reasoning_options = [{ type = "effort", values = ["none", "high"] }]
cost_tiers = [{ input = 5.0, tier = { type = "context", size = 32000 } }]
cost_context_over_200k = { input = 4.0, output = 8.0 }
interleaved = { field = "reasoning_content" }
provider = { npm = "@ai-sdk/openai-compatible" }
experimental = { modes = {} }
"#;
        let cfg: Config = toml::from_str(text).expect("parse toml");
        let m = &cfg.models[0];
        assert_eq!(m.display_name.as_deref(), Some("M Display"));
        assert_eq!(m.model_type.as_deref(), Some("embedding"));
        assert_eq!(m.status.as_deref(), Some("active"));
        assert_eq!(
            m.modalities_input,
            Some(vec!["text".to_string(), "image".to_string()])
        );
        assert_eq!(m.input_limit, Some(50));
        assert_eq!(m.cost_reasoning, Some(0.6));
        assert!(m
            .reasoning_options
            .as_ref()
            .and_then(|v| v.as_array())
            .is_some());
        assert!(m.cost_tiers.as_ref().and_then(|v| v.as_array()).is_some());
        assert!(m
            .cost_context_over_200k
            .as_ref()
            .and_then(|v| v.as_object())
            .is_some());
        assert!(m.interleaved.as_ref().and_then(|v| v.as_object()).is_some());
        assert!(m
            .model_provider
            .as_ref()
            .and_then(|v| v.as_object())
            .is_some());
        assert!(m
            .experimental
            .as_ref()
            .and_then(|v| v.as_object())
            .is_some());
    }

    #[test]
    fn thinking_type_parses_with_adaptive_default() {
        // Unset -> Adaptive.
        let text = r#"
[[models]]
name = "m"
protocol = "anthropic"
"#;
        let cfg: Config = toml::from_str(text).expect("parse toml");
        assert_eq!(cfg.models[0].thinking_mode(), ThinkingMode::Adaptive);

        // Explicit legacy override.
        let text = r#"
[[models]]
name = "m"
protocol = "anthropic"
thinking_type = "enabled"
"#;
        let cfg: Config = toml::from_str(text).expect("parse toml");
        assert_eq!(cfg.models[0].thinking_mode(), ThinkingMode::Enabled);

        // Explicit adaptive.
        let text = r#"
[[models]]
name = "m"
protocol = "anthropic"
thinking_type = "adaptive"
"#;
        let cfg: Config = toml::from_str(text).expect("parse toml");
        assert_eq!(cfg.models[0].thinking_mode(), ThinkingMode::Adaptive);
    }

    #[test]
    fn legacy_minimal_toml_still_parses() {
        let text = r#"
api_key = "x"
default_base_url = "https://api.example.com/v1"

[[models]]
name = "m"
protocol = "chat"
"#;
        let cfg: Config = toml::from_str(text).expect("parse toml");
        assert!(!cfg.models_dev.enable);
        assert_eq!(cfg.models[0].vision, None);
        assert_eq!(cfg.models[0].model_type, None);
    }
}
