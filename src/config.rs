use std::collections::HashMap;
use std::fs;

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Protocol {
    /// OpenAI-compatible POST {base}/chat/completions (pass-through).
    Chat,
    /// Anthropic POST {base}/messages (translated).
    Anthropic,
    /// OpenAI Responses POST {base}/responses (translated).
    Responses,
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
    /// Whether the model accepts image input (advertised via /v1/models).
    #[serde(default)]
    pub vision: bool,
}

impl ModelCfg {
    pub fn upstream_model(&self) -> &str {
        self.upstream.as_deref().unwrap_or(&self.name)
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
    pub models: Vec<ModelCfg>,
}

fn default_listen() -> String {
    "0.0.0.0:8787".to_string()
}
fn default_user_agent() -> String {
    format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
}

impl Config {
    pub fn load(path: &str) -> Result<Config, String> {
        let text = fs::read_to_string(path)
            .map_err(|e| format!("read config {path}: {e}"))?;
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
