//! models.dev enrichment: parse api.json, match local models, merge metadata.
//!
//! Design:
//! - Local TOML is the source of truth for routing (`protocol`, `base_url`,
//!   `upstream`). This module only enriches `/v1/models` metadata.
//! - Every user-set `Some(_)` field on `ModelCfg` wins over inferred data.
//! - Unset `vision` (`None`) lets inference fill it in; explicit
//!   `Some(true/false)` always wins.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use crate::config::ModelCfg;

#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct Modalities {
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub output: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct DevLimit {
    #[serde(default)]
    pub context: Option<u64>,
    #[serde(default)]
    pub input: Option<u64>,
    #[serde(default)]
    pub output: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct DevCost {
    #[serde(default)]
    pub input: Option<f64>,
    #[serde(default)]
    pub output: Option<f64>,
    #[serde(default)]
    pub cache_read: Option<f64>,
    #[serde(default)]
    pub cache_write: Option<f64>,
    /// Tiered pricing entries (provider-specific shapes preserved as-is).
    #[serde(default)]
    pub tiers: Option<Value>,
    #[serde(default)]
    pub context_over_200k: Option<Value>,
    #[serde(default)]
    pub reasoning: Option<f64>,
    #[serde(default)]
    pub input_audio: Option<f64>,
    #[serde(default)]
    pub output_audio: Option<f64>,
}

/// Subset of a models.dev model entry. Unknown fields are ignored so
/// upstream schema additions don't break us.
#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct DevModel {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub attachment: Option<bool>,
    #[serde(default)]
    pub reasoning: Option<bool>,
    #[serde(default)]
    pub reasoning_options: Option<Value>,
    #[serde(default)]
    pub tool_call: Option<bool>,
    #[serde(default)]
    pub structured_output: Option<bool>,
    #[serde(default)]
    pub temperature: Option<bool>,
    #[serde(default)]
    pub modalities: Option<Modalities>,
    #[serde(default)]
    pub limit: Option<DevLimit>,
    #[serde(default)]
    pub cost: Option<DevCost>,
    #[serde(default, rename = "type")]
    pub model_type: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub last_updated: Option<String>,
    #[serde(default)]
    pub release_date: Option<String>,
    #[serde(default)]
    pub open_weights: Option<bool>,
    #[serde(default)]
    pub knowledge: Option<String>,
    #[serde(default)]
    pub canonical_model_id: Option<String>,
    #[serde(default)]
    pub interleaved: Option<Value>,
    #[serde(default)]
    pub provider: Option<Value>,
    #[serde(default)]
    pub experimental: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct ProviderEntry {
    #[serde(default)]
    models: HashMap<String, DevModel>,
}

/// Lowercase alphanumeric only: `Muse-Spark-1.3` and `muse_spark1.3`
/// normalize identically. Cheap and good enough for alias matching.
pub fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Enrichment store: exact map plus normalized indexes for alias matching.
#[derive(Debug, Clone, Default)]
pub struct Store {
    /// Exact model id -> entry.
    pub by_id: HashMap<String, DevModel>,
    /// Normalized id (and canonical id) -> exact id.
    norm_index: HashMap<String, String>,
}

impl Store {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    fn insert(&mut self, key: String, m: DevModel) {
        let norm = normalize(&key);
        self.norm_index.entry(norm).or_insert_with(|| key.clone());
        if let Some(c) = m.canonical_model_id.clone() {
            let nc = normalize(&c);
            self.norm_index.entry(nc).or_insert_with(|| key.clone());
        }
        self.by_id.insert(key, m);
    }

    /// Lookup order: exact id -> exact canonical handled by index ->
    /// normalized id. Returns the matched entry.
    pub fn lookup(&self, id: &str) -> Option<&DevModel> {
        if let Some(m) = self.by_id.get(id) {
            return Some(m);
        }
        self.norm_index
            .get(&normalize(id))
            .and_then(|k| self.by_id.get(k))
    }
}

/// Parse raw api.json bytes. If `provider` is `Some`, only that provider's
/// models are indexed; otherwise all providers are merged (first wins).
pub fn parse_api_json(text: &str, provider: Option<&str>) -> Result<Store, String> {
    let root: HashMap<String, ProviderEntry> =
        serde_json::from_str(text).map_err(|e| format!("parse models.dev json: {e}"))?;
    let mut store = Store::empty();
    match provider {
        Some(p) => {
            let entry = root
                .get(p)
                .ok_or_else(|| format!("models.dev provider '{p}' not found"))?;
            for (id, mut m) in entry.models.clone() {
                if m.id.is_empty() {
                    m.id = id.clone();
                }
                store.insert(id, m);
            }
        }
        None => {
            for entry in root.values() {
                for (id, mut m) in entry.models.clone() {
                    if m.id.is_empty() {
                        m.id = id.clone();
                    }
                    if !store.by_id.contains_key(&id) {
                        store.insert(id, m);
                    }
                }
            }
        }
    }
    Ok(store)
}

/// Merged view for one local model: user override wins, else inferred.
#[derive(Debug, Clone)]
pub struct Merged {
    pub dev_id: Option<String>,
    pub vision: bool,
    pub vision_source: &'static str, // "config" | "models.dev" | "default"
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub family: Option<String>,
    pub model_type: Option<String>,
    pub knowledge: Option<String>,
    pub release_date: Option<String>,
    pub last_updated: Option<String>,
    pub status: Option<String>,
    pub canonical_model_id: Option<String>,
    pub attachment: Option<bool>,
    pub reasoning: Option<bool>,
    pub reasoning_options: Option<Value>,
    pub tool_call: Option<bool>,
    pub structured_output: Option<bool>,
    pub temperature: Option<bool>,
    pub open_weights: Option<bool>,
    pub modalities: Option<Modalities>,
    pub context_limit: Option<u64>,
    pub input_limit: Option<u64>,
    pub output_limit: Option<u64>,
    pub cost_input: Option<f64>,
    pub cost_output: Option<f64>,
    pub cost_cache_read: Option<f64>,
    pub cost_cache_write: Option<f64>,
    pub cost_tiers: Option<Value>,
    pub cost_context_over_200k: Option<Value>,
    pub cost_reasoning: Option<f64>,
    pub cost_input_audio: Option<f64>,
    pub cost_output_audio: Option<f64>,
    pub interleaved: Option<Value>,
    pub model_provider: Option<Value>,
    pub experimental: Option<Value>,
    pub matched: bool,
}

pub fn merge_model(cfg: &ModelCfg, store: &Store) -> Merged {
    let dev = store.lookup(cfg.match_id());
    let matched = dev.is_some();

    // Effective modalities: explicit override wins over inferred.
    let eff_input: Option<Vec<String>> = cfg.modalities_input.clone().or_else(|| {
        dev.and_then(|d| d.modalities.as_ref())
            .map(|m| m.input.clone())
    });
    let eff_output: Option<Vec<String>> = cfg.modalities_output.clone().or_else(|| {
        dev.and_then(|d| d.modalities.as_ref())
            .map(|m| m.output.clone())
    });
    let modalities = match (&eff_input, &eff_output) {
        (None, None) => None,
        _ => Some(Modalities {
            input: eff_input.clone().unwrap_or_default(),
            output: eff_output.clone().unwrap_or_default(),
        }),
    };
    // Vision = `image` in effective input modalities. `attachment` alone
    // is not enough (it can mean file attach only).
    let image_in = eff_input.as_ref().is_some_and(|i| {
        i.iter().any(|x| x.eq_ignore_ascii_case("image"))
    });

    // Vision: explicit config wins; modalities override counts as config;
    // else inferred from models.dev; else default false.
    let (vision, vision_source) = match cfg.vision {
        Some(v) => (v, "config"),
        None if cfg.modalities_input.is_some() => (image_in, "config"),
        None => match dev {
            // If models.dev knows the model, that's an informed value
            // rather than a blind default, even when text-only.
            Some(_) => (image_in, "models.dev"),
            None => (false, "default"),
        },
    };

    let pick_str = |cfg_v: &Option<String>, dev_v: Option<&String>| {
        cfg_v.clone().or_else(|| dev_v.cloned())
    };
    let pick_bool = |cfg_v: Option<bool>, dev_v: Option<bool>| cfg_v.or(dev_v);
    let pick_u64 = |cfg_v: Option<u64>, dev_v: Option<u64>| cfg_v.or(dev_v);
    let pick_f64 = |cfg_v: Option<f64>, dev_v: Option<f64>| cfg_v.or(dev_v);
    let pick_json = |cfg_v: &Option<Value>, dev_v: Option<&Value>| {
        cfg_v.clone().or_else(|| dev_v.cloned())
    };
    let cost = |f: fn(&DevCost) -> Option<f64>| dev.and_then(|d| d.cost.as_ref().and_then(f));
    let cost_json =
        |f: fn(&DevCost) -> Option<&Value>| dev.and_then(|d| d.cost.as_ref().and_then(f));

    Merged {
        dev_id: dev.map(|d| d.id.clone()),
        vision,
        vision_source,
        display_name: pick_str(&cfg.display_name, dev.and_then(|d| d.name.as_ref())),
        description: pick_str(&cfg.description, dev.and_then(|d| d.description.as_ref())),
        family: pick_str(&cfg.family, dev.and_then(|d| d.family.as_ref())),
        model_type: pick_str(&cfg.model_type, dev.and_then(|d| d.model_type.as_ref())),
        knowledge: pick_str(&cfg.knowledge, dev.and_then(|d| d.knowledge.as_ref())),
        release_date: pick_str(
            &cfg.release_date,
            dev.and_then(|d| d.release_date.as_ref()),
        ),
        last_updated: pick_str(
            &cfg.last_updated,
            dev.and_then(|d| d.last_updated.as_ref()),
        ),
        status: pick_str(&cfg.status, dev.and_then(|d| d.status.as_ref())),
        canonical_model_id: pick_str(
            &cfg.canonical_model_id,
            dev.and_then(|d| d.canonical_model_id.as_ref()),
        ),
        attachment: pick_bool(cfg.attachment, dev.and_then(|d| d.attachment)),
        reasoning: pick_bool(cfg.reasoning, dev.and_then(|d| d.reasoning)),
        reasoning_options: pick_json(
            &cfg.reasoning_options,
            dev.and_then(|d| d.reasoning_options.as_ref()),
        ),
        tool_call: pick_bool(cfg.tool_call, dev.and_then(|d| d.tool_call)),
        structured_output: pick_bool(
            cfg.structured_output,
            dev.and_then(|d| d.structured_output),
        ),
        temperature: pick_bool(cfg.temperature, dev.and_then(|d| d.temperature)),
        open_weights: pick_bool(cfg.open_weights, dev.and_then(|d| d.open_weights)),
        modalities,
        context_limit: pick_u64(
            cfg.context_limit,
            dev.and_then(|d| d.limit.as_ref().and_then(|l| l.context)),
        ),
        input_limit: pick_u64(
            cfg.input_limit,
            dev.and_then(|d| d.limit.as_ref().and_then(|l| l.input)),
        ),
        output_limit: pick_u64(
            cfg.output_limit,
            dev.and_then(|d| d.limit.as_ref().and_then(|l| l.output)),
        ),
        cost_input: pick_f64(cfg.cost_input, cost(|c| c.input)),
        cost_output: pick_f64(cfg.cost_output, cost(|c| c.output)),
        cost_cache_read: pick_f64(cfg.cost_cache_read, cost(|c| c.cache_read)),
        cost_cache_write: pick_f64(cfg.cost_cache_write, cost(|c| c.cache_write)),
        cost_tiers: pick_json(&cfg.cost_tiers, cost_json(|c| c.tiers.as_ref())),
        cost_context_over_200k: pick_json(
            &cfg.cost_context_over_200k,
            cost_json(|c| c.context_over_200k.as_ref()),
        ),
        cost_reasoning: pick_f64(cfg.cost_reasoning, cost(|c| c.reasoning)),
        cost_input_audio: pick_f64(cfg.cost_input_audio, cost(|c| c.input_audio)),
        cost_output_audio: pick_f64(cfg.cost_output_audio, cost(|c| c.output_audio)),
        interleaved: pick_json(&cfg.interleaved, dev.and_then(|d| d.interleaved.as_ref())),
        model_provider: pick_json(
            &cfg.model_provider,
            dev.and_then(|d| d.provider.as_ref()),
        ),
        experimental: pick_json(
            &cfg.experimental,
            dev.and_then(|d| d.experimental.as_ref()),
        ),
        matched,
    }
}

/// Render the `info.meta` object for `/v1/models`. Keeps the existing
/// `capabilities.vision` shape for client compatibility and nests the
/// rest under `models_dev` (null when unmatched and no overrides).
pub fn meta_json(_cfg: &ModelCfg, merged: &Merged) -> Value {
    let mut cost = serde_json::Map::new();
    let mut has_cost = false;
    for (k, v) in [
        ("input", merged.cost_input),
        ("output", merged.cost_output),
        ("cache_read", merged.cost_cache_read),
        ("cache_write", merged.cost_cache_write),
        ("reasoning", merged.cost_reasoning),
        ("input_audio", merged.cost_input_audio),
        ("output_audio", merged.cost_output_audio),
    ] {
        if let Some(x) = v {
            has_cost = true;
            cost.insert(k.to_string(), Value::from(x));
        }
    }
    for (k, v) in [
        ("tiers", merged.cost_tiers.as_ref()),
        ("context_over_200k", merged.cost_context_over_200k.as_ref()),
    ] {
        if let Some(x) = v {
            has_cost = true;
            cost.insert(k.to_string(), x.clone());
        }
    }
    let mut limit = serde_json::Map::new();
    let mut has_limit = false;
    for (k, v) in [
        ("context", merged.context_limit),
        ("input", merged.input_limit),
        ("output", merged.output_limit),
    ] {
        if let Some(x) = v {
            has_limit = true;
            limit.insert(k.to_string(), Value::from(x));
        }
    }

    let has_dev = merged.matched
        || merged.display_name.is_some()
        || merged.description.is_some()
        || merged.family.is_some()
        || merged.model_type.is_some()
        || merged.knowledge.is_some()
        || merged.release_date.is_some()
        || merged.last_updated.is_some()
        || merged.status.is_some()
        || merged.canonical_model_id.is_some()
        || merged.attachment.is_some()
        || merged.reasoning.is_some()
        || merged.reasoning_options.is_some()
        || merged.tool_call.is_some()
        || merged.structured_output.is_some()
        || merged.temperature.is_some()
        || merged.open_weights.is_some()
        || merged.modalities.is_some()
        || has_cost
        || has_limit
        || merged.interleaved.is_some()
        || merged.model_provider.is_some()
        || merged.experimental.is_some();

    let mut dev = serde_json::Map::new();
    if has_dev {
        // Only emit keys we actually know, so clients can distinguish
        // "unknown" (absent) from explicit values.
        if let Some(id) = merged.dev_id.as_ref() {
            dev.insert("id".into(), Value::String(id.clone()));
        }
        if let Some(v) = merged.display_name.as_ref() {
            dev.insert("name".into(), Value::String(v.clone()));
        }
        if let Some(v) = merged.description.as_ref() {
            dev.insert("description".into(), Value::String(v.clone()));
        }
        if let Some(v) = merged.family.as_ref() {
            dev.insert("family".into(), Value::String(v.clone()));
        }
        if let Some(v) = merged.model_type.as_ref() {
            dev.insert("type".into(), Value::String(v.clone()));
        }
        if let Some(v) = merged.knowledge.as_ref() {
            dev.insert("knowledge".into(), Value::String(v.clone()));
        }
        if let Some(v) = merged.attachment {
            dev.insert("attachment".into(), Value::Bool(v));
        }
        if let Some(v) = merged.reasoning {
            dev.insert("reasoning".into(), Value::Bool(v));
        }
        if let Some(v) = merged.reasoning_options.as_ref() {
            dev.insert("reasoning_options".into(), v.clone());
        }
        if let Some(v) = merged.tool_call {
            dev.insert("tool_call".into(), Value::Bool(v));
        }
        if let Some(v) = merged.structured_output {
            dev.insert("structured_output".into(), Value::Bool(v));
        }
        if let Some(v) = merged.temperature {
            dev.insert("temperature".into(), Value::Bool(v));
        }
        if let Some(m) = merged.modalities.as_ref() {
            dev.insert(
                "modalities".into(),
                serde_json::json!({"input": m.input, "output": m.output}),
            );
        }
        if has_limit {
            dev.insert("limit".into(), Value::Object(limit));
        }
        if has_cost {
            dev.insert("cost".into(), Value::Object(cost));
        }
        if let Some(v) = merged.status.as_ref() {
            dev.insert("status".into(), Value::String(v.clone()));
        }
        if let Some(v) = merged.last_updated.as_ref() {
            dev.insert("last_updated".into(), Value::String(v.clone()));
        }
        if let Some(v) = merged.release_date.as_ref() {
            dev.insert("release_date".into(), Value::String(v.clone()));
        }
        if let Some(v) = merged.open_weights {
            dev.insert("open_weights".into(), Value::Bool(v));
        }
        if let Some(v) = merged.canonical_model_id.as_ref() {
            dev.insert("canonical_model_id".into(), Value::String(v.clone()));
        }
        if let Some(v) = merged.interleaved.as_ref() {
            dev.insert("interleaved".into(), v.clone());
        }
        if let Some(v) = merged.model_provider.as_ref() {
            dev.insert("provider".into(), v.clone());
        }
        if let Some(v) = merged.experimental.as_ref() {
            dev.insert("experimental".into(), v.clone());
        }
        dev.insert("matched".into(), Value::Bool(merged.matched));
    }

    serde_json::json!({
        "capabilities": {"vision": merged.vision, "vision_source": merged.vision_source},
        "models_dev": if dev.is_empty() { Value::Null } else { Value::Object(dev) },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Protocol;

    fn test_cfg(name: &str) -> ModelCfg {
        ModelCfg {
            name: name.into(),
            upstream: None,
            protocol: Protocol::Chat,
            base_url: None,
            extra_headers: HashMap::new(),
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

    #[test]
    fn normalize_strips_case_and_separators() {
        assert_eq!(normalize("Muse-Spark-1.3"), "musespark13");
        assert_eq!(normalize("muse_spark1.3"), "musespark13");
        assert_eq!(normalize("kimi-k2.7-code"), "kimik27code");
    }

    #[test]
    fn parse_and_lookup_exact_and_normalized() {
        let raw = r#"{
            "opencode-go": {"id": "opencode-go", "models": {
                "mimo-v2.6-pro": {
                    "id": "mimo-v2.6-pro",
                    "modalities": {"input": ["text", "image"], "output": ["text"]},
                    "limit": {"context": 1048576, "output": 131072},
                    "cost": {"input": 0.435, "output": 0.87},
                    "canonical_model_id": "xiaomi/mimo-v2.6-pro"
                }
            }}
        }"#;
        let store = parse_api_json(raw, Some("opencode-go")).unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.lookup("mimo-v2.6-pro").is_some());
        // Alias with different separators still matches.
        assert!(store.lookup("MIMO_V2.6.PRO").is_some());
        // Canonical id matches too.
        assert!(store.lookup("xiaomi/mimo-v2.6-pro").is_some());
        assert!(store.lookup("nope").is_none());
    }

    #[test]
    fn merge_user_override_wins() {
        let raw = r#"{
            "p": {"id": "p", "models": {
                "m": {"id": "m",
                    "modalities": {"input": ["text", "image"], "output": ["text"]},
                    "cost": {"input": 1.0, "output": 2.0}}
            }}
        }"#;
        let store = parse_api_json(raw, Some("p")).unwrap();

        // Unset vision -> inferred true.
        let m = merge_model(&test_cfg("m"), &store);
        assert!(m.vision);
        assert_eq!(m.vision_source, "models.dev");
        assert_eq!(m.cost_input, Some(1.0));

        // Explicit vision=false wins over inference.
        let mut c = test_cfg("m");
        c.vision = Some(false);
        c.cost_input = Some(9.0);
        let m2 = merge_model(&c, &store);
        assert!(!m2.vision);
        assert_eq!(m2.vision_source, "config");
        assert_eq!(m2.cost_input, Some(9.0));

        // Unknown model -> defaults.
        let m3 = merge_model(&test_cfg("unknown"), &store);
        assert!(!m3.vision);
        assert_eq!(m3.vision_source, "default");
        assert!(!m3.matched);
    }

    #[test]
    fn unknown_provider_errors() {
        let raw = r#"{"a": {"id": "a", "models": {}}}"#;
        assert!(parse_api_json(raw, Some("missing")).is_err());
    }

    #[test]
    fn full_parity_keys_parse_and_emit() {
        let raw = r#"{
            "p": {"id": "p", "models": {
                "m": {"id": "m", "name": "M",
                    "description": "d", "family": "f", "type": "embedding",
                    "knowledge": "2024-12", "release_date": "2026-01-01",
                    "last_updated": "2026-02-01", "status": "deprecated",
                    "canonical_model_id": "org/m",
                    "attachment": true, "reasoning": true,
                    "reasoning_options": [{"type": "effort", "values": ["none", "high"]}],
                    "tool_call": true, "structured_output": true,
                    "temperature": false, "open_weights": true,
                    "modalities": {"input": ["text", "image"], "output": ["text"]},
                    "limit": {"context": 100, "input": 50, "output": 10},
                    "cost": {"input": 1.0, "output": 2.0, "cache_read": 0.1,
                        "cache_write": 0.2, "reasoning": 0.6, "input_audio": 1.5,
                        "output_audio": 3.0,
                        "tiers": [{"input": 5, "tier": {"type": "context", "size": 32000}}],
                        "context_over_200k": {"input": 4, "output": 8}},
                    "interleaved": {"field": "reasoning_content"},
                    "provider": {"npm": "@ai-sdk/openai-compatible"},
                    "experimental": {"modes": {}}}
            }}
        }"#;
        let store = parse_api_json(raw, Some("p")).unwrap();
        let m = merge_model(&test_cfg("m"), &store);
        assert!(m.matched);
        assert!(m.vision);
        assert_eq!(m.display_name.as_deref(), Some("M"));
        assert_eq!(m.model_type.as_deref(), Some("embedding"));
        assert_eq!(m.input_limit, Some(50));
        assert_eq!(m.cost_reasoning, Some(0.6));
        assert_eq!(m.cost_input_audio, Some(1.5));
        assert!(m.cost_tiers.is_some());
        assert!(m.cost_context_over_200k.is_some());
        assert!(m.reasoning_options.is_some());
        assert!(m.interleaved.is_some());
        assert!(m.model_provider.is_some());
        assert!(m.experimental.is_some());
        assert_eq!(m.temperature, Some(false));
        assert_eq!(m.structured_output, Some(true));

        let meta = meta_json(&test_cfg("m"), &m);
        let dev = meta.pointer("/models_dev").unwrap();
        for k in ["name", "type", "knowledge", "status", "attachment",
            "reasoning_options", "structured_output", "temperature",
            "open_weights", "canonical_model_id", "interleaved",
            "provider", "experimental"]
        {
            assert!(dev.get(k).is_some(), "meta missing {k}");
        }
        assert_eq!(
            dev.pointer("/limit/input").and_then(|v| v.as_u64()),
            Some(50)
        );
        assert_eq!(
            dev.pointer("/cost/reasoning").and_then(|v| v.as_f64()),
            Some(0.6)
        );
        assert!(dev.pointer("/cost/tiers").and_then(|v| v.as_array()).is_some());
    }

    #[test]
    fn modalities_override_drives_vision_as_config() {
        let raw = r#"{"p": {"id": "p", "models": {
            "m": {"id": "m", "modalities": {"input": ["text"], "output": ["text"]}}
        }}}"#;
        let store = parse_api_json(raw, Some("p")).unwrap();

        // Dev says text-only -> informed false.
        let m = merge_model(&test_cfg("m"), &store);
        assert!(!m.vision);
        assert_eq!(m.vision_source, "models.dev");

        // Modalities override adds image -> vision true, source config.
        let mut c = test_cfg("m");
        c.modalities_input = Some(vec!["text".into(), "image".into()]);
        let m2 = merge_model(&c, &store);
        assert!(m2.vision);
        assert_eq!(m2.vision_source, "config");

        // Overrides also work with no dev match at all.
        let mut c2 = test_cfg("unknown");
        c2.status = Some("active".into());
        c2.temperature = Some(true);
        let m3 = merge_model(&c2, &store);
        assert!(!m3.matched);
        assert_eq!(m3.status.as_deref(), Some("active"));
        let meta = meta_json(&c2, &m3);
        assert!(meta.pointer("/models_dev").is_some());
        assert_eq!(
            meta.pointer("/models_dev/status").and_then(|v| v.as_str()),
            Some("active")
        );
    }
}
