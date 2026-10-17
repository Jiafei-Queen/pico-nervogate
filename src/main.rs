mod config;
mod discovery;
#[cfg(test)]
mod e2e;
mod errors;
mod handlers;
mod models_dev;
mod service;
mod sse;
mod state;
mod translate;
mod upstream;

use crate::state::{Inner, St};
use crate::handlers::{chat, google_ingress, healthz, list_models, messages_ingress, responses_ingress};
use discovery::refresh_discovery;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::Instant;
use axum::extract::DefaultBodyLimit;
use axum::Router;
use axum::serve::ListenerExt;
use axum::routing::{get, post};
use crate::config::Config;

#[tokio::main]
async fn main() {
    let path = std::env::var("GATEWAY_CONFIG").unwrap_or_else(|_| "gateway.toml".to_string());
    let cfg = match Config::load(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config error: {e}");
            std::process::exit(1);
        }
    };
    let api_key = match cfg.api_key() {
        Ok(k) => k,
        Err(e) => {
            eprintln!("api key error: {e}");
            std::process::exit(1);
        }
    };

    let mut models = HashMap::new();
    for m in &cfg.models {
        models.insert(m.name.clone(), m.clone());
    }

    let client = reqwest::Client::builder()
        .build()
        .expect("build http client");

    // models.dev enrichment: fetch once at startup (cache fallback).
    // Periodic refresh runs in the maintenance loop below. Never fatal:
    // an empty store means plain (unenriched) behavior.
    let enrichment = Arc::new(std::sync::RwLock::new(crate::models_dev::Store::empty()));
    if cfg.models_dev.enable {
        let loaded = load_enrichment(
            &client,
            &cfg.models_dev.url,
            cfg.models_dev.provider.as_deref(),
            cfg.models_dev.cache_path.as_deref(),
        )
        .await;
        eprintln!(
            "[gw] models.dev: {} entries (provider={})",
            loaded.len(),
            cfg.models_dev.provider.as_deref().unwrap_or("all")
        );
        *enrichment.write().unwrap() = loaded;
    }

    let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
    let listen = cfg.listen.clone();
    let n = models.len();
    let reload_boot = cfg.reload.enable;
    let state: St = Arc::new(Inner {
        cfg: RwLock::new(cfg),
        api_key: RwLock::new(api_key),
        client,
        models: RwLock::new(models),
        discovered: RwLock::new(HashMap::new()),
        enrichment,
        config_path: path,
        config_mtime: RwLock::new(mtime),
    });

    if state.cfg.read().unwrap().discovery.enable {
        // Container network/DNS is often not ready in the first seconds
        // after boot (scratch image, daemon-attached network), so retry a
        // few times before serving with static models only.
        for attempt in 1..=4 {
            match refresh_discovery(&state).await {
                Ok(d) => {
                    eprintln!("[gw] discovery: {d} models");
                    break;
                }
                Err(e) if attempt < 4 => {
                    eprintln!(
                        "[gw] discovery failed at startup (attempt {attempt}/4): {e}; retrying in 5s"
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
                Err(e) => eprintln!("[gw] discovery failed at startup: {e}"),
            }
        }
    }

    // Maintenance loop: config mtime watch, models.dev refresh, discovery
    // refresh. Single task, 15s tick; each job re-reads live config and
    // fires only when its own interval has elapsed.
    {
        let st = Arc::clone(&state);
        tokio::spawn(async move {
            const TICK_SECS: u64 = 15;
            let mut last_watch = Instant::now();
            let mut last_enrich = Instant::now();
            let mut last_discover = Instant::now();
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(TICK_SECS)).await;

                let (r_enable, w_interval) = {
                    let c = st.cfg.read().unwrap();
                    (c.reload.enable, c.reload.watch_interval_secs)
                };
                if r_enable && w_interval > 0 && last_watch.elapsed().as_secs() >= w_interval
                {
                    last_watch = Instant::now();
                    match reload_config(&st, false).await {
                        Ok(msg) if msg != "unchanged" => eprintln!("[gw] reload: {msg}"),
                        Err(e) => eprintln!("[gw] reload failed: {e}"),
                        _ => {}
                    }
                }

                let (e_enable, e_interval) = {
                    let c = st.cfg.read().unwrap();
                    (c.models_dev.enable, c.models_dev.refresh_interval_secs)
                };
                if e_enable && e_interval > 0 && last_enrich.elapsed().as_secs() >= e_interval
                {
                    last_enrich = Instant::now();
                    let (url, provider, cache) = {
                        let c = st.cfg.read().unwrap();
                        (
                            c.models_dev.url.clone(),
                            c.models_dev.provider.clone(),
                            c.models_dev.cache_path.clone(),
                        )
                    };
                    let next =
                        load_enrichment(&st.client, &url, provider.as_deref(), cache.as_deref())
                            .await;
                    eprintln!("[gw] models.dev: refreshed ({} entries)", next.len());
                    // Keep the old store if refresh yields nothing and we
                    // already have data (avoids flapping on transient errors).
                    if next.is_empty() && !st.enrichment.read().unwrap().is_empty() {
                        eprintln!("[gw] models.dev: refresh empty, keeping old data");
                    } else {
                        *st.enrichment.write().unwrap() = next;
                    }
                }

                let (d_enable, d_interval) = {
                    let c = st.cfg.read().unwrap();
                    (c.discovery.enable, c.discovery.interval_secs)
                };
                // If startup discovery failed (empty map), retry every minute
                // instead of waiting out the full interval.
                let discovered_empty = st.discovered.read().unwrap().is_empty();
                let elapsed = last_discover.elapsed().as_secs();
                if d_enable
                    && d_interval > 0
                    && (elapsed >= d_interval || (discovered_empty && elapsed >= 60))
                {
                    last_discover = Instant::now();
                    match refresh_discovery(&st).await {
                        Ok(d) => eprintln!("[gw] discovery: refreshed ({d} models)"),
                        Err(e) => eprintln!("[gw] discovery refresh failed: {e}"),
                    }
                }
            }
        });
    }

    // SIGHUP reloads the config file. The master switch is read at startup:
    // when disabled, no handler is installed and SIGHUP keeps its default
    // (terminate) behavior.
    #[cfg(unix)]
    if reload_boot {
        let st = Arc::clone(&state);
        tokio::spawn(async move {
            let mut sig =
                match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup()) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("[gw] sighup watch unavailable: {e}");
                        return;
                    }
                };
            while sig.recv().await.is_some() {
                match reload_config(&st, true).await {
                    Ok(msg) => eprintln!("[gw] sighup reload: {msg}"),
                    Err(e) => eprintln!("[gw] sighup reload failed: {e}"),
                }
            }
        });
    }

    let app = app(state.clone());

    let listener = tokio::net::TcpListener::bind(&listen)
        .await
        .unwrap_or_else(|e| panic!("bind {listen}: {e}"));
    println!("pico-nervogate listening on {listen} ({n} models)");
    // Without TCP_NODELAY, Nagle holds multi-segment SSE responses until the
    // peer's delayed ACK lands, adding ~40ms per streamed response.
    let listener = listener.tap_io(|stream| {
        if let Err(e) = stream.set_nodelay(true) {
            eprintln!("[gw] set_nodelay failed: {e}");
        }
    });
    axum::serve(listener, app).await.unwrap();
}
/// Route table, kept out of `main` so tests can drive it without a socket.
fn app(state: St) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/responses", post(responses_ingress))
        .route("/v1/messages", post(messages_ingress))
        .route("/v1beta/models/{action}", post(google_ingress))
        // The gateway applies `max_body_bytes` itself so an oversized request
        // can be rejected in the ingress protocol's error shape. Axum's own
        // 2 MiB `Bytes` extractor limit would fire first, as plain text.
        .layer(DefaultBodyLimit::disable())
        .with_state(state)
}
/// Reload the config file into live state. Validates before swapping, so a
/// broken file keeps the old config serving. Returns a short summary.
async fn reload_config(st: &St, force: bool) -> Result<String, String> {
    let path = st.config_path.clone();
    let mtime = std::fs::metadata(&path)
        .and_then(|m| m.modified())
        .ok();
    if !force {
        let cur = *st.config_mtime.read().unwrap();
        if cur.is_some() && mtime == cur {
            return Ok("unchanged".to_string());
        }
    }
    let cfg = Config::load(&path)?;
    // Re-resolve the key (env): supports rotation without restart.
    let api_key = cfg.api_key()?;
    let old = st.cfg.read().unwrap().clone();

    let mut models = HashMap::new();
    for m in &cfg.models {
        models.insert(m.name.clone(), m.clone());
    }
    let n = models.len();
    let d = st.discovered.read().unwrap().len();

    if old.listen != cfg.listen {
        eprintln!(
            "[gw] reload: `listen` changed ({} -> {}); rebind requires restart, still on {}",
            old.listen, cfg.listen, old.listen
        );
    }
    let models_dev_changed = old.models_dev.enable != cfg.models_dev.enable
        || old.models_dev.url != cfg.models_dev.url
        || old.models_dev.provider != cfg.models_dev.provider;
    let discovery_changed = old.discovery.enable != cfg.discovery.enable
        || old.discovery.base_url != cfg.discovery.base_url
        || old.discovery.path != cfg.discovery.path
        || old.discovery.protocol != cfg.discovery.protocol
        || old.discovery.prefix != cfg.discovery.prefix
        || old.discovery.prune_missing != cfg.discovery.prune_missing;
    let discovery_now_off = old.discovery.enable && !cfg.discovery.enable;

    *st.cfg.write().unwrap() = cfg;
    *st.models.write().unwrap() = models;
    *st.api_key.write().unwrap() = api_key;
    *st.config_mtime.write().unwrap() = mtime;

    // React promptly to section changes instead of waiting for the next tick.
    if models_dev_changed {
        let (url, provider, cache, enabled) = {
            let c = st.cfg.read().unwrap();
            (
                c.models_dev.url.clone(),
                c.models_dev.provider.clone(),
                c.models_dev.cache_path.clone(),
                c.models_dev.enable,
            )
        };
        if enabled {
            let next =
                load_enrichment(&st.client, &url, provider.as_deref(), cache.as_deref()).await;
            eprintln!("[gw] models.dev: reloaded ({} entries)", next.len());
            *st.enrichment.write().unwrap() = next;
        } else {
            *st.enrichment.write().unwrap() = crate::models_dev::Store::empty();
        }
    }
    if discovery_now_off {
        st.discovered.write().unwrap().clear();
    } else if discovery_changed && st.cfg.read().unwrap().discovery.enable {
        match refresh_discovery(st).await {
            Ok(count) => eprintln!("[gw] discovery: reloaded ({count} models)"),
            Err(e) => eprintln!("[gw] discovery reload failed: {e}"),
        }
    }

    Ok(format!("{n} static + {d} discovered models"))
}
/// Fetch api.json, parse for the configured provider, fall back to disk
/// cache, else return an empty store (gateway keeps serving regardless).
async fn load_enrichment(
    client: &reqwest::Client,
    url: &str,
    provider: Option<&str>,
    cache_path: Option<&str>,
) -> crate::models_dev::Store {
    match client.get(url).send().await {
        Ok(resp) if resp.status().is_success() => match resp.text().await {
            Ok(text) => match crate::models_dev::parse_api_json(&text, provider) {
                Ok(store) => {
                    if let Some(p) = cache_path {
                        if let Err(e) = std::fs::write(p, &text) {
                            eprintln!("[gw] models.dev: cache write {p} failed: {e}");
                        }
                    }
                    return store;
                }
                Err(e) => eprintln!("[gw] models.dev: parse failed: {e}"),
            },
            Err(e) => eprintln!("[gw] models.dev: read body failed: {e}"),
        },
        Ok(resp) => eprintln!("[gw] models.dev: http {} from {url}", resp.status()),
        Err(e) => eprintln!("[gw] models.dev: fetch failed ({url}): {e:#}"),
    }
    // Fallback: disk cache.
    if let Some(p) = cache_path {
        match std::fs::read_to_string(p) {
            Ok(text) => match crate::models_dev::parse_api_json(&text, provider) {
                Ok(store) => {
                    eprintln!(
                        "[gw] models.dev: using disk cache {p} ({} entries)",
                        store.len()
                    );
                    return store;
                }
                Err(e) => eprintln!("[gw] models.dev: cache parse failed: {e}"),
            },
            Err(e) => eprintln!("[gw] models.dev: no cache at {p}: {e}"),
        }
    }
    crate::models_dev::Store::empty()
}

// ---------------------------------------------------------------------------
// Upstream request builders
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    // Route-table integration tests only. Unit tests for lookup, SSE,
    // error shapes, body limits and upstream bodies live next to the code
    // they cover (`state` / `sse` / `errors` / `service` / `upstream`).
    use super::*;
    use crate::errors::Ingress;
    use crate::state::{sample_config, test_state};
    use axum::body::Body;
    use axum::extract::Request;
    use axum::http::StatusCode;
    use serde_json::Value;

    /// Assert `body` is a well-formed error for `ingress`, and return the
    /// message.
    fn assert_error_shape(ingress: Ingress, body: &Value) -> String {
        match ingress {
            Ingress::Anthropic => assert_eq!(
                body["type"], "error",
                "Anthropic clients dispatch on the top-level `type`: {body}"
            ),
            _ => assert!(
                body.get("error").is_some(),
                "missing `error` object: {body}"
            ),
        }
        assert!(
            body.pointer("/error/message")
                .and_then(|m| m.as_str())
                .is_some(),
            "no readable message for {ingress:?}: {body}"
        );
        body.pointer("/error/message")
            .and_then(|m| m.as_str())
            .unwrap()
            .to_string()
    }

    /// Drive the real router with an oversized body and read the response
    /// back, so the test covers the limit and the rejection together rather
    /// than only the helper that formats it.
    #[tokio::test]
    async fn router_rejects_an_oversized_body_in_protocol_shape() {
        use tower::ServiceExt;

        let limit = 1024usize;
        let mut cfg = sample_config();
        cfg.max_body_bytes = limit;
        let state = test_state(cfg);
        let router = app(state);

        // Comfortably past the limit, with the secret in it: the response
        // must describe the failure without echoing the payload back.
        let secret = "sk-live-DO-NOT-ECHO";
        let big = format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"{secret}{}"}}]}}"#,
            "x".repeat(limit * 2)
        );

        for (path, ingress) in [
            ("/v1/chat/completions", Ingress::OpenAiCompatible),
            ("/v1/responses", Ingress::OpenAi),
            ("/v1/messages", Ingress::Anthropic),
            ("/v1beta/models/m:generateContent", Ingress::Google),
        ] {
            let req = Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .header("content-length", big.len())
                .body(Body::from(big.clone()))
                .unwrap();
            let resp = router.clone().oneshot(req).await.expect("router responded");
            assert_eq!(
                resp.status(),
                StatusCode::PAYLOAD_TOO_LARGE,
                "{path} must be rejected before the model lookup"
            );
            let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
                .await
                .unwrap();
            let body: Value = serde_json::from_slice(&bytes)
                .unwrap_or_else(|e| panic!("{path} body was not JSON ({e}): {bytes:?}"));
            let msg = assert_error_shape(ingress, &body);
            assert!(!msg.contains(secret), "payload echoed back: {body}");
            assert!(msg.contains("1024"), "limit not reported: {body}");
        }
    }

    #[tokio::test]
    async fn a_body_under_the_limit_is_parsed_not_rejected() {
        // The limit must not reject legitimate requests: an unknown model is
        // a 404, which proves the body was read and dispatched normally.
        use tower::ServiceExt;

        let mut cfg = sample_config();
        cfg.max_body_bytes = 64 * 1024;
        let router = app(test_state(cfg));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"nope","messages":[]}"#))
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
