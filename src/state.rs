//! Shared gateway state and model lookup.

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::SystemTime;
use crate::config::{Config, ModelCfg};

pub(crate) struct Inner {
    pub(crate) cfg: RwLock<Config>,
    pub(crate) api_key: RwLock<String>,
    pub(crate) client: reqwest::Client,
    /// Explicit models from TOML. Always wins over `discovered`.
    pub(crate) models: RwLock<HashMap<String, ModelCfg>>,
    /// Auto-discovered upstream models (see `[discovery]`).
    pub(crate) discovered: RwLock<HashMap<String, ModelCfg>>,
    pub(crate) enrichment: Arc<std::sync::RwLock<crate::models_dev::Store>>,
    pub(crate) config_path: String,
    pub(crate) config_mtime: RwLock<Option<SystemTime>>,
}

pub(crate) type St = Arc<Inner>;

/// Static lookup first, discovered fallback. Explicit `[[models]]`
/// entries always win over auto-discovered ones.
pub(crate) fn find_model(st: &St, name: &str) -> Option<ModelCfg> {
    let statics = st.models.read().unwrap();
    let discovered = st.discovered.read().unwrap();
    find_model_in(&statics, &discovered, name)
}

/// Static lookup first, discovered fallback. Explicit `[[models]]`
/// entries always win over auto-discovered ones. On an exact miss, one
/// separator-insensitive retry (`claude-haiku-5.5` matches upstream
/// `claude-haiku-5-5`) — only when unambiguous, otherwise None.
pub(crate) fn find_model_in(
    statics: &HashMap<String, ModelCfg>,
    discovered: &HashMap<String, ModelCfg>,
    name: &str,
) -> Option<ModelCfg> {
    if let Some(m) = statics.get(name) {
        return Some(m.clone());
    }
    if let Some(m) = discovered.get(name) {
        return Some(m.clone());
    }
    let want = crate::models_dev::normalize(name);
    let mut hit: Option<ModelCfg> = None;
    for m in statics.values().chain(discovered.values()) {
        if crate::models_dev::normalize(&m.name) == want {
            if hit.is_some() {
                return None; // ambiguous: refuse to guess
            }
            hit = Some(m.clone());
        }
    }
    hit
}

/// Minimal in-process state: no models, no upstream, no network.
#[cfg(test)]
pub(crate) fn test_state(cfg: Config) -> St {
    // The static model table is normally built from `cfg.models` in `main`;
    // tests reach the state through this constructor, so it has to happen
    // here too or every model reads as unknown.
    let models: HashMap<String, ModelCfg> = cfg
        .models
        .iter()
        .map(|m| (m.name.clone(), m.clone()))
        .collect();
    Arc::new(Inner {
        cfg: RwLock::new(cfg),
        api_key: RwLock::new("test-key".into()),
        client: reqwest::Client::new(),
        models: RwLock::new(models),
        discovered: RwLock::new(HashMap::new()),
        enrichment: Arc::new(RwLock::new(crate::models_dev::Store::empty())),
        config_path: String::new(),
        config_mtime: RwLock::new(None),
    })
}

/// A `Config` pointing every model at `base_url`.
#[cfg(test)]
pub(crate) fn config_for(base_url: &str, models: Vec<ModelCfg>) -> Config {
    Config {
        api_key: Some("test-key".into()),
        default_base_url: Some(base_url.to_string()),
        models,
        ..sample_config()
    }
}

#[cfg(test)]
pub(crate) fn sample_config() -> Config {
    Config {
        listen: "127.0.0.1:0".into(),
        api_key_env: None,
        api_key: Some("k".into()),
        session_id: None,
        session_header: None,
        user_agent: "test".into(),
        default_base_url: None,
        owned_by: None,
        provider: None,
        extra_headers: HashMap::new(),
        anthropic_max_tokens: 8192,
        strict_params: false,
        reasoning_output: Default::default(),
        max_body_bytes: 32 * 1024 * 1024,
        models_dev: Default::default(),
        reload: Default::default(),
        discovery: Default::default(),
        models: vec![],
    }
}
