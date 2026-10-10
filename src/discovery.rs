//! Upstream model auto-discovery.
//!
//! Merges ids from an OpenAI-compatible `{base}/models` endpoint into the
//! gateway's model list. Explicit `[[models]]` entries always win: the
//! discovered map keeps every upstream id, and lookups prefer the static
//! table (so a discovered id shadowed by an explicit entry is harmless).
//!
//! Refresh is never fatal: failures keep the previous discovered set.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::config::{ModelCfg, Protocol};
use crate::St;

/// Parse a `/models` response body into model ids. Accepts the OpenAI list
/// shape (`{"data": [{"id": ...}]}`) and a bare array (of `{id}` objects
/// or plain strings).
pub fn parse_models_list(text: &str) -> Result<Vec<String>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("parse /models json: {e}"))?;
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

/// Smallest request each protocol accepts, used to test which one the
/// upstream speaks. `max_tokens: 1` keeps the probe from costing anything
/// meaningful — the response body is discarded either way.
fn probe_body(protocol: &Protocol, model: &str) -> Value {
    match protocol {
        Protocol::Chat => json!({
            "model": model, "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]
        }),
        Protocol::Anthropic => json!({
            "model": model, "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]
        }),
        Protocol::Responses => json!({
            "model": model, "max_output_tokens": 16, "input": "hi"
        }),
    }
}

/// Order to try. `chat` first: it is the most common upstream shape and the
/// configured default, so the common case costs a single request.
const PROBE_ORDER: [Protocol; 3] = [Protocol::Chat, Protocol::Anthropic, Protocol::Responses];

/// Which completion endpoint does this upstream actually serve?
///
/// Probes one model with a minimal request per protocol and keeps the first
/// whose endpoint exists. A `404`/`405` means the path is not an endpoint at
/// all; any other status (including 400 and 401) means it is one and the
/// problem lies elsewhere, so it is a positive identification.
///
/// Returns `None` when no candidate answered, which leaves the configured
/// protocol in place — a wrong guess and an unverified assumption fail the
/// same way, and guessing here would hide the real problem behind a config
/// error.
pub async fn detect_protocol(st: &St, base: &str, model: &str) -> Option<Protocol> {
    for protocol in PROBE_ORDER {
        let m = ModelCfg::discovered(model.to_string(), protocol, base.to_string());
        let url = crate::upstream_url(base, &m);
        let resp = st
            .client
            .post(&url)
            .headers(crate::upstream_headers(st, &m))
            .json(&probe_body(&protocol, model))
            .send()
            .await;
        let status = match resp {
            Ok(r) => r.status(),
            // Transport failure (DNS, refused, TLS): says nothing about which
            // protocol this host speaks, so try the next one.
            Err(_) => continue,
        };
        if endpoint_exists(status) {
            eprintln!(
                "[gw] discovery: upstream speaks `{p}` (probed {url})",
                p = protocol.as_str()
            );
            return Some(protocol);
        }
    }
    None
}

/// Does this status mean the path is a real endpoint?
fn endpoint_exists(status: axum::http::StatusCode) -> bool {
    !(status == axum::http::StatusCode::NOT_FOUND
        || status == axum::http::StatusCode::METHOD_NOT_ALLOWED)
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
            d.protocol,
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
        req = req.header(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {api_key}"),
        );
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("discovery fetch {url} failed: {e:#}"))?;
    if !resp.status().is_success() {
        return Err(format!("discovery http {} from {url}", resp.status()));
    }
    let text = resp
        .text()
        .await
        .map_err(|e| format!("discovery read body failed: {e:#}"))?;
    let ids = parse_models_list(&text)?;
    let ids: Vec<String> = ids
        .into_iter()
        .filter(|id| prefix.as_ref().is_none_or(|p| id.starts_with(p.as_str())))
        .collect();

    // A `/models` list says nothing about which completion endpoint the
    // upstream speaks. Guessing wrong makes every discovered model fail at
    // request time with a 404 on `POST {base}/messages` (or wherever), which
    // looks like a gateway bug rather than a config gap.
    let protocol = if ids.is_empty() {
        protocol
    } else {
        match detect_protocol(st, &base, &ids[0]).await {
            Some(p) => p,
            None => {
                eprintln!(
                    "[gw] discovery: could not determine the upstream protocol; \
                     assuming `{}`. If every discovered model returns 404, set \
                     `[discovery] protocol` explicitly.",
                    protocol.as_str()
                );
                protocol
            }
        }
    };

    let mut fresh = HashMap::new();
    for id in ids {
        fresh.insert(id.clone(), ModelCfg::discovered(id, protocol, base.clone()));
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

    // ---- protocol detection (M-10 / L-10) -------------------------------

    #[test]
    fn probe_bodies_are_minimal_but_valid_per_protocol() {
        // Anthropic rejects a request without `max_tokens`; Responses rejects
        // one without `input`. A malformed probe would report "not this
        // protocol" for the protocol that is actually in use.
        let chat = probe_body(&Protocol::Chat, "m");
        assert!(chat.get("messages").is_some());
        assert_eq!(chat["max_tokens"], 1);

        let anth = probe_body(&Protocol::Anthropic, "m");
        assert!(anth.get("messages").is_some());
        assert_eq!(anth["max_tokens"], 1);

        let resp = probe_body(&Protocol::Responses, "m");
        assert!(resp.get("input").is_some());
        assert!(resp.get("messages").is_none());
        assert!(resp.get("max_output_tokens").is_some());

        // Nothing larger than a single token, so a probe is not a billable
        // generation.
        for (p, b) in [
            (Protocol::Chat, &chat),
            (Protocol::Anthropic, &anth),
            (Protocol::Responses, &resp),
        ] {
            assert_eq!(b["model"], "m", "{p:?} lost the model id");
        }
    }

    #[test]
    fn a_404_means_the_path_is_not_an_endpoint() {
        use axum::http::StatusCode;
        // Only these two say "wrong path". Everything else — including 400
        // and 401 — means the endpoint is real and something else is wrong,
        // which still identifies the protocol.
        assert!(!endpoint_exists(StatusCode::NOT_FOUND));
        assert!(!endpoint_exists(StatusCode::METHOD_NOT_ALLOWED));
        for s in [
            StatusCode::OK,
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
        ] {
            assert!(
                endpoint_exists(s),
                "{s} should count as an existing endpoint"
            );
        }
    }

    /// Serve one protocol's path and 404 the others, standing in for an
    /// upstream that speaks exactly one wire format. The `/models` list it
    /// reports is the same either way — that is the whole problem: the list
    /// gives no hint about the completion endpoint.
    async fn mock_upstream(speaks: Protocol) -> String {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use axum::response::Response;
        use axum::routing::post;
        use axum::Router;

        let want: &'static str = match speaks {
            Protocol::Chat => "/v1/chat/completions",
            Protocol::Anthropic => "/v1/messages",
            Protocol::Responses => "/v1/responses",
        };
        let handler = move |req: Request<Body>| async move {
            if req.uri().path() == want {
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Body::from(r#"{"choices":[]}"#))
                    .unwrap()
            } else {
                Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(Body::from("not found"))
                    .unwrap()
            }
        };
        let models = axum::routing::get(|| async {
            axum::response::Response::builder()
                .status(axum::http::StatusCode::OK)
                .body(Body::from(r#"{"data":[{"id":"m1"},{"id":"m2"}]}"#))
                .unwrap()
        });
        let router = Router::new()
            .route("/v1/models", models)
            .route("/v1/chat/completions", post(handler))
            .route("/v1/messages", post(handler))
            .route("/v1/responses", post(handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}/v1")
    }

    /// Config wired at the base URL of `mock_upstream`, discovery enabled,
    /// and `protocol` left at its default — the M-10 setup.
    fn discovery_config(base: &str) -> crate::config::Config {
        crate::config::Config {
            api_key: Some("test".into()),
            discovery: crate::config::DiscoveryCfg {
                enable: true,
                base_url: Some(base.to_string()),
                ..Default::default()
            },
            ..crate::sample_config()
        }
    }

    #[tokio::test]
    async fn refresh_registers_discovered_models_with_the_detected_protocol() {
        // The end-to-end version of the M-10 fix: an Anthropic upstream whose
        // `/models` list looks identical to a chat one. Before detection these
        // two models would both be registered as `chat` and every request
        // would 404 on POST {base}/chat/completions.
        let base = mock_upstream(Protocol::Anthropic).await;
        let st = crate::test_state(discovery_config(&base));
        let n = refresh_discovery(&st).await.expect("discovery succeeds");
        assert_eq!(n, 2);
        let disc = st.discovered.read().unwrap();
        assert_eq!(disc.len(), 2);
        for m in disc.values() {
            assert_eq!(
                m.protocol,
                Protocol::Anthropic,
                "{} registered with the wrong protocol",
                m.name
            );
        }
    }

    #[tokio::test]
    async fn refresh_falls_back_to_the_configured_protocol_when_unreachable() {
        // Nothing listening: the list cannot be fetched at all, so discovery
        // fails rather than registering models of an unknown protocol.
        let st = crate::test_state(discovery_config("http://127.0.0.1:1/v1"));
        assert!(refresh_discovery(&st).await.is_err());
        assert!(st.discovered.read().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_prefix_filter_applies_before_the_probe_picks_a_model() {
        // The probe uses the first surviving id; with the filter applied
        // first, a non-matching leading entry cannot skew the detection.
        let base = mock_upstream(Protocol::Anthropic).await;
        let mut cfg = discovery_config(&base);
        cfg.discovery.prefix = Some("m".to_string());
        let st = crate::test_state(cfg);
        let n = refresh_discovery(&st).await.expect("discovery succeeds");
        assert_eq!(n, 2);
        assert!(st
            .discovered
            .read()
            .unwrap()
            .values()
            .all(|m| m.protocol == Protocol::Anthropic));
    }

    fn probe_state() -> St {
        crate::test_state(crate::config::Config {
            api_key: Some("test".into()),
            ..crate::sample_config()
        })
    }

    #[tokio::test]
    async fn detects_an_anthropic_upstream_that_defaults_to_chat() {
        // The exact M-10 failure: discovery defaults to `chat`, the upstream
        // speaks Anthropic, and without detection every discovered model 404s
        // at request time with nothing pointing at the real cause.
        let base = mock_upstream(Protocol::Anthropic).await;
        let found = detect_protocol(&probe_state(), &base, "some-model").await;
        assert_eq!(found, Some(Protocol::Anthropic));
    }

    #[tokio::test]
    async fn detects_chat_and_responses_upstreams() {
        for speaks in [Protocol::Chat, Protocol::Responses] {
            let base = mock_upstream(speaks).await;
            let found = detect_protocol(&probe_state(), &base, "m").await;
            assert_eq!(found, Some(speaks));
        }
    }

    #[tokio::test]
    async fn a_dead_upstream_leaves_the_protocol_undetermined() {
        // Nothing listening: detection must report "unknown" rather than
        // pick a protocol at random and blame it in the config.
        let found = detect_protocol(&probe_state(), "http://127.0.0.1:1/v1", "m").await;
        assert_eq!(found, None);
    }

    #[tokio::test]
    async fn detection_costs_one_request_in_the_common_case() {
        // The default `chat` upstream must be identified by its first probe;
        // every extra request is latency added to every discovery refresh.
        let base = mock_upstream(Protocol::Chat).await;
        assert_eq!(
            detect_protocol(&probe_state(), &base, "m").await,
            Some(Protocol::Chat)
        );
    }
}
