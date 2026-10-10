//! Upstream model auto-discovery.
//!
//! Merges ids from an OpenAI-compatible `{base}/models` endpoint into the
//! gateway's model list. Explicit `[[models]]` entries always win: the
//! discovered map keeps every upstream id, and lookups prefer the static
//! table (so a discovered id shadowed by an explicit entry is harmless).
//!
//! Refresh is never fatal: failures keep the previous discovered set.

use std::collections::HashMap;

use serde_json::Value;

use crate::config::ModelCfg;
use crate::St;

/// Parse a `/models` response body into model ids. Accepts the OpenAI list
/// shape (`{"data": [{"id": ...}]}`) and a bare array (of `{id}` objects
/// or plain strings).
pub fn parse_models_list(text: &str) -> Result<Vec<String>, String> {
    let v: Value =
        serde_json::from_str(text).map_err(|e| format!("parse /models json: {e}"))?;
    let items: &[Value] = match &v {
        Value::Array(a) => a,
        Value::Object(_) => v
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| "parse /models json: missing `data` array".to_string())?,
        _ => return Err("parse /models json: expected object or array".to_string()),
    };
    let mut ids = Vec::with_capacity(items.len());
    for item in items {
        let id = match item {
            Value::String(s) => Some(s.clone()),
            Value::Object(_) => item
                .get("id")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string()),
            _ => None,
        };
        match id {
            Some(id) if !id.is_empty() => ids.push(id),
            _ => return Err("parse /models json: entry without string `id`".to_string()),
        }
    }
    Ok(ids)
}

/// Fetch the discovery endpoint and merge ids into the discovered map.
/// Returns the discovered count. With `prune_missing`, the map is replaced;
/// otherwise it only grows (additive, never removes).
pub async fn refresh_discovery(st: &St) -> Result<usize, String> {
    let (base, path, protocol, prefix, prune) = {
        let guard = st.cfg.read().unwrap();
        let d = &guard.discovery;
        let base = d
            .base_url
            .clone()
            .or_else(|| guard.default_base_url.clone())
            .ok_or_else(|| "discovery has no base_url (and no default_base_url)".to_string())?;
        (
            base,
            d.path.clone(),
            d.protocol.clone(),
            d.prefix.clone(),
            d.prune_missing,
        )
    };
    // NOTE: some /models endpoints need no auth; an empty key just means
    // no Authorization header is sent.
    let api_key = st.api_key.read().unwrap().clone();
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    let mut req = st.client.get(&url);
    if !api_key.is_empty() {
        req = req.header(axum::http::header::AUTHORIZATION, format!("Bearer {api_key}"));
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("discovery fetch {url} failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("discovery http {} from {url}", resp.status()));
    }
    let text = resp
        .text()
        .await
        .map_err(|e| format!("discovery read body failed: {e}"))?;
    let ids = parse_models_list(&text)?;

    let mut fresh = HashMap::new();
    for id in ids {
        if let Some(p) = prefix.as_ref() {
            if !id.starts_with(p.as_str()) {
                continue;
            }
        }
        fresh.insert(
            id.clone(),
            ModelCfg::discovered(id, protocol.clone(), base.clone()),
        );
    }
    let n = {
        let mut disc = st.discovered.write().unwrap();
        if prune {
            *disc = fresh;
        } else {
            disc.extend(fresh);
        }
        disc.len()
    };
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_list_shape() {
        let ids = parse_models_list(
            r#"{"object":"list","data":[{"id":"a","object":"model"},{"id":"b"}]}"#,
        )
        .unwrap();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn parses_bare_array_shapes() {
        let ids = parse_models_list(r#"[{"id":"a"},{"id":"b"}]"#).unwrap();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
        let ids = parse_models_list(r#"["a","b"]"#).unwrap();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(parse_models_list(r#"{"object":"list"}"#).is_err());
        assert!(parse_models_list(r#"{"data":[{"no_id":1}]}"#).is_err());
        assert!(parse_models_list(r#"[1,2]"#).is_err());
        assert!(parse_models_list(r#"not json"#).is_err());
    }

    #[test]
    fn discovered_ctor_leaves_metadata_unset() {
        let m = ModelCfg::discovered(
            "x".into(),
            crate::config::Protocol::Responses,
            "https://h/v1".into(),
        );
        assert_eq!(m.base_url.as_deref(), Some("https://h/v1"));
        assert_eq!(m.vision, None);
        assert_eq!(m.description, None);
        assert_eq!(m.cost_input, None);
    }
}
