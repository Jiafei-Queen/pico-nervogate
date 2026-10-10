mod config;
mod discovery;
mod models_dev;
mod translate;

use std::collections::HashMap;
use std::io;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use futures_util::StreamExt;
use serde_json::{json, Value};

use config::{Config, ModelCfg, Protocol};
use discovery::refresh_discovery;
use translate::{
    anthropic_to_chat_request, anthropic_to_openai, chat_to_anthropic_response,
    chat_to_responses_request, chat_to_responses_response, frame_data, frame_event,
    openai_to_anthropic, responses_stateful_error, responses_to_chat_request, responses_to_openai,
    AnthropicStream, ChatToAnthropicStream, ChatToResponsesStream, ResponsesStream,
};

struct Inner {
    cfg: RwLock<Config>,
    api_key: RwLock<String>,
    client: reqwest::Client,
    /// Explicit models from TOML. Always wins over `discovered`.
    models: RwLock<HashMap<String, ModelCfg>>,
    /// Auto-discovered upstream models (see `[discovery]`).
    discovered: RwLock<HashMap<String, ModelCfg>>,
    enrichment: Arc<std::sync::RwLock<models_dev::Store>>,
    config_path: String,
    config_mtime: RwLock<Option<SystemTime>>,
}

type St = Arc<Inner>;

/// Static lookup first, discovered fallback. Explicit `[[models]]`
/// entries always win over auto-discovered ones.
fn find_model(st: &St, name: &str) -> Option<ModelCfg> {
    let statics = st.models.read().unwrap();
    let discovered = st.discovered.read().unwrap();
    find_model_in(&statics, &discovered, name)
}

/// Static lookup first, discovered fallback. Explicit `[[models]]`
/// entries always win over auto-discovered ones. On an exact miss, one
/// separator-insensitive retry (`claude-haiku-5.5` matches upstream
/// `claude-haiku-5-5`) — only when unambiguous, otherwise None.
fn find_model_in(
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
    let want = models_dev::normalize(name);
    let mut hit: Option<ModelCfg> = None;
    for m in statics.values().chain(discovered.values()) {
        if models_dev::normalize(&m.name) == want {
            if hit.is_some() {
                return None; // ambiguous: refuse to guess
            }
            hit = Some(m.clone());
        }
    }
    hit
}

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
    let enrichment = Arc::new(std::sync::RwLock::new(models_dev::Store::empty()));
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

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/responses", post(responses_ingress))
        .route("/v1/messages", post(messages_ingress))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&listen)
        .await
        .unwrap_or_else(|e| panic!("bind {listen}: {e}"));
    println!("pico-nervogate listening on {listen} ({n} models)");
    axum::serve(listener, app).await.unwrap();
}

async fn healthz() -> Response {
    json_response(StatusCode::OK, json!({"status": "ok"}))
}

async fn list_models(State(st): State<St>) -> Response {
    let store = st.enrichment.read().unwrap();
    let statics = st.models.read().unwrap();
    let discovered = st.discovered.read().unwrap();
    let cfg = st.cfg.read().unwrap();
    let owned_by = cfg
        .owned_by
        .clone()
        .unwrap_or_else(|| env!("CARGO_PKG_NAME").to_string());
    let provider = cfg
        .provider
        .clone()
        .unwrap_or_else(|| env!("CARGO_PKG_NAME").to_string());
    let mut data: Vec<Value> = statics
        .values()
        // Discovered ids shadowed by explicit entries are skipped: the
        // explicit entry is already listed above.
        .chain(discovered.values().filter(|m| !statics.contains_key(&m.name)))
        .map(|m| {
            let merged = models_dev::merge_model(m, &store);
            let meta = models_dev::meta_json(m, &merged);
            json!({
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": owned_by,
                "provider": provider,
                "info": {"meta": meta}
            })
        })
        .collect();
    // Stable ordering for clients/diffs.
    data.sort_by(|a, b| {
        a.get("id")
            .and_then(|x| x.as_str())
            .cmp(&b.get("id").and_then(|x| x.as_str()))
    });
    json_response(StatusCode::OK, json!({"object": "list", "data": data}))
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
            *st.enrichment.write().unwrap() = models_dev::Store::empty();
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
) -> models_dev::Store {
    match client.get(url).send().await {
        Ok(resp) if resp.status().is_success() => match resp.text().await {
            Ok(text) => match models_dev::parse_api_json(&text, provider) {
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
            Ok(text) => match models_dev::parse_api_json(&text, provider) {
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
    models_dev::Store::empty()
}

// ---------------------------------------------------------------------------
// Upstream request builders
// ---------------------------------------------------------------------------

fn upstream_headers(st: &St, m: &ModelCfg) -> HeaderMap {
    let mut h = HeaderMap::new();
    let api_key = st.api_key.read().unwrap().clone();
    let cfg = st.cfg.read().unwrap();
    let bearer = format!("Bearer {api_key}");
    if let Ok(v) = HeaderValue::from_str(&bearer) {
        h.insert(header::AUTHORIZATION, v);
    }
    if let Ok(v) = HeaderValue::from_str(&cfg.user_agent) {
        h.insert(header::USER_AGENT, v);
    }
    if let (Some(session_id), Some(session_header)) = (&cfg.session_id, &cfg.session_header) {
        if let (Ok(v), Ok(name)) = (
            HeaderValue::from_str(session_id),
            header::HeaderName::from_bytes(session_header.as_bytes()),
        ) {
            h.insert(name, v);
        }
    }
    for (k, v) in cfg.extra_headers.iter().chain(m.extra_headers.iter()) {
        if let (Ok(name), Ok(val)) = (
            header::HeaderName::from_bytes(k.as_bytes()),
            HeaderValue::from_str(v),
        ) {
            h.insert(name, val);
        }
    }
    if m.protocol == Protocol::Anthropic {
        if let Ok(v) = HeaderValue::from_str(&api_key) {
            h.insert(header::HeaderName::from_static("x-api-key"), v);
        }
        h.insert(
            header::HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static("2023-06-01"),
        );
    }
    h
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn trunc(s: &str, n: usize) -> &str {
    if s.len() <= n {
        return s;
    }
    let mut end = n;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn upstream_url(base: &str, m: &ModelCfg) -> String {
    let base = base.trim_end_matches('/');
    match m.protocol {
        Protocol::Chat => format!("{base}/chat/completions"),
        Protocol::Anthropic => format!("{base}/messages"),
        Protocol::Responses => format!("{base}/responses"),
    }
}

/// Resolve model + build the upstream body for the given protocol.
fn build_body(m: &ModelCfg, req: &Value, stream: bool) -> Value {
    let upstream = m.upstream_model();
    let mut body = match m.protocol {
        Protocol::Chat => {
            let mut v = req.clone();
            // The reserved passthrough key never goes on the wire.
            v.as_object_mut().map(|o| o.remove("x_nervogate"));
            v
        }
        Protocol::Anthropic => openai_to_anthropic(req),
        Protocol::Responses => chat_to_responses_request(req),
    };
    body["model"] = json!(upstream);
    if stream {
        body["stream"] = json!(true);
    } else {
        // avoid asking upstream for a stream we will not consume
        body.as_object_mut().map(|o| o.remove("stream"));
    }
    body
}

enum UpstreamError {
    Status(StatusCode, String),
    Message(String),
}

/// Error response in the shape the ingress protocol expects.
fn ingress_error(ingress: Ingress, status: StatusCode, msg: &str) -> Response {
    match ingress {
        Ingress::Anthropic => json_response(
            status,
            json!({"type": "error", "error": {"type": "api_error", "message": msg}}),
        ),
        _ => error_json(status, msg),
    }
}

/// Upstream failure in the ingress protocol's error shape (the raw upstream
/// body would not make sense to a client speaking a different protocol).
fn upstream_error_response(ingress: Ingress, e: UpstreamError) -> Response {
    match e {
        UpstreamError::Message(msg) => ingress_error(ingress, StatusCode::BAD_GATEWAY, &msg),
        UpstreamError::Status(code, body) => {
            if ingress != Ingress::Anthropic {
                let mut resp = Response::new(Body::from(body));
                *resp.status_mut() = code;
                resp.headers_mut().insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );
                return resp;
            }
            let msg = serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|v| {
                    v.pointer("/error/message")
                        .or_else(|| v.get("message"))
                        .and_then(|x| x.as_str())
                        .map(String::from)
                })
                .unwrap_or(body);
            json_response(
                code,
                json!({"type": "error", "error": {"type": "api_error", "message": msg}}),
            )
        }
    }
}

async fn call_upstream(
    st: &St,
    m: &ModelCfg,
    req: &Value,
    stream: bool,
) -> Result<reqwest::Response, UpstreamError> {
    let base = st
        .cfg
        .read()
        .unwrap()
        .base_url_for(m)
        .map_err(UpstreamError::Message)?;
    let url = upstream_url(&base, m);
    let body = build_body(m, req, stream);
    let t0 = now_ms();
    let stream_tag = if stream { "stream" } else { "nonstream" };
    let resp = match st
        .client
        .post(&url)
        .headers(upstream_headers(st, m))
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "[gw] req model={} stream={} -> connect-error ms={}: {}",
                m.name,
                stream_tag,
                now_ms() - t0,
                e
            );
            return Err(UpstreamError::Message(format!("upstream request failed: {e}")));
        }
    };

    if resp.status().is_success() {
        eprintln!(
            "[gw] req model={} stream={} upstream={} ms={}",
            m.name,
            stream_tag,
            resp.status().as_u16(),
            now_ms() - t0
        );
        Ok(resp)
    } else {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        eprintln!(
            "[gw] req model={} stream={} upstream={} ms={} ERROR url={} body={}",
            m.name,
            stream_tag,
            status.as_u16(),
            now_ms() - t0,
            url,
            trunc(&text, 2000)
        );
        eprintln!(
            "[gw] outgoing model={} body={}",
            m.name,
            trunc(&serde_json::to_string(&body).unwrap_or_default(), 2000)
        );
        Err(UpstreamError::Status(
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            text,
        ))
    }
}

// ---------------------------------------------------------------------------
// Ingress handlers: POST /v1/chat/completions, /v1/responses, /v1/messages
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Ingress {
    Chat,
    Responses,
    Anthropic,
}

async fn chat(State(st): State<St>, body: Bytes) -> Response {
    handle_ingress(st, body, Ingress::Chat).await
}

async fn responses_ingress(State(st): State<St>, body: Bytes) -> Response {
    handle_ingress(st, body, Ingress::Responses).await
}

async fn messages_ingress(State(st): State<St>, body: Bytes) -> Response {
    let mut resp = handle_ingress(st, body, Ingress::Anthropic).await;
    resp.headers_mut().insert(
        header::HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static("2023-06-01"),
    );
    resp
}

/// One pipeline for all three ingress surfaces: normalize to the canonical
/// chat request, call upstream, translate back into the ingress protocol.
async fn handle_ingress(st: St, body: Bytes, ingress: Ingress) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return ingress_error(
                ingress,
                StatusCode::BAD_REQUEST,
                &format!("invalid JSON body: {e}"),
            )
        }
    };
    let name = match req.get("model").and_then(|x| x.as_str()) {
        Some(n) => n.to_string(),
        None => return ingress_error(ingress, StatusCode::BAD_REQUEST, "missing field: model"),
    };
    let m = match find_model(&st, &name) {
        Some(m) => m,
        None => {
            eprintln!("[gw] req unknown-model={name}");
            return ingress_error(
                ingress,
                StatusCode::NOT_FOUND,
                &format!("unknown model: {name}"),
            );
        }
    };
    if ingress == Ingress::Responses {
        if let Some(msg) = responses_stateful_error(&req) {
            return ingress_error(ingress, StatusCode::BAD_REQUEST, &msg);
        }
    }
    let stream = req.get("stream").and_then(|x| x.as_bool()).unwrap_or(false);

    // Normalize the ingress request to the canonical chat shape.
    let chat_req = match ingress {
        Ingress::Chat => req,
        Ingress::Responses => responses_to_chat_request(&req),
        Ingress::Anthropic => anthropic_to_chat_request(&req),
    };

    let resp = match call_upstream(&st, &m, &chat_req, stream).await {
        Ok(r) => r,
        Err(e) => return upstream_error_response(ingress, e),
    };

    if stream {
        let inbound = match m.protocol {
            Protocol::Chat => Inbound::Chat {
                name: name.clone(),
                held: None,
            },
            Protocol::Anthropic => Inbound::Anthropic(AnthropicStream::new(&name)),
            Protocol::Responses => Inbound::Responses(ResponsesStream::new(&name)),
        };
        let egress = match ingress {
            Ingress::Chat => Egress::Chat,
            Ingress::Responses => Egress::Responses(ChatToResponsesStream::new(&name)),
            Ingress::Anthropic => Egress::Anthropic(ChatToAnthropicStream::new(&name)),
        };
        sse_response(sse_body(resp, inbound, egress))
    } else {
        let text = match resp.text().await {
            Ok(t) => t,
            Err(e) => {
                return ingress_error(
                    ingress,
                    StatusCode::BAD_GATEWAY,
                    &format!("read upstream: {e}"),
                )
            }
        };
        let up: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => return raw_json_response(StatusCode::OK, text),
        };
        let chat = match m.protocol {
            Protocol::Chat => {
                let mut v = up;
                v["model"] = json!(name.clone());
                v
            }
            Protocol::Anthropic => anthropic_to_openai(&up, &name),
            Protocol::Responses => responses_to_openai(&up, &name),
        };
        let out = match ingress {
            Ingress::Chat => chat,
            Ingress::Responses => chat_to_responses_response(&chat, &name),
            Ingress::Anthropic => chat_to_anthropic_response(&chat, &name),
        };
        json_response(StatusCode::OK, out)
    }
}

// ---------------------------------------------------------------------------
// SSE pipeline: upstream -> canonical chat chunks -> client framing
// ---------------------------------------------------------------------------

/// Upstream SSE -> canonical OpenAI chat chunks.
enum Inbound {
    /// Chat Completions upstream: chunks pass through, model rewritten.
    Chat {
        name: String,
        held: Option<Value>,
    },
    Anthropic(AnthropicStream),
    Responses(ResponsesStream),
}

impl Inbound {
    /// Handle one upstream SSE `data:` payload (a JSON object).
    fn handle(&mut self, ev: &Value) -> Vec<Value> {
        match self {
            Inbound::Chat { name, held } => {
                // OpenAI sends usage in a trailing `choices: []` chunk; merge
                // it into the held finish chunk so translated egresses still
                // see usage.
                if ev.pointer("/choices/0").is_none() {
                    if let (Some(h), Some(u)) = (held.as_mut(), ev.get("usage")) {
                        h["usage"] = u.clone();
                        return held.take().into_iter().collect();
                    }
                    return vec![];
                }
                let mut out: Vec<Value> = held.take().into_iter().collect();
                let mut v = ev.clone();
                v["model"] = json!(name.clone());
                if v.pointer("/choices/0/finish_reason").is_some() && v.get("usage").is_none() {
                    *held = Some(v);
                } else {
                    out.push(v);
                }
                out
            }
            Inbound::Anthropic(s) => s.handle(ev),
            Inbound::Responses(s) => s.handle(ev),
        }
    }

    /// Flush a held finish chunk at end of stream.
    fn flush(&mut self) -> Vec<Value> {
        match self {
            Inbound::Chat { held, .. } => held.take().into_iter().collect(),
            _ => vec![],
        }
    }
}

/// Canonical OpenAI chat chunks -> client-protocol SSE framing.
enum Egress {
    Chat,
    Responses(ChatToResponsesStream),
    Anthropic(ChatToAnthropicStream),
}

impl Egress {
    fn handle(&mut self, chunk: &Value) -> Vec<String> {
        match self {
            Egress::Chat => vec![frame_data(chunk)],
            Egress::Responses(s) => s.handle(chunk).iter().map(|e| e.frame()).collect(),
            Egress::Anthropic(s) => s.handle(chunk).iter().map(|e| e.frame()).collect(),
        }
    }

    /// Trailer: Chat streams end with the `[DONE]` sentinel; the other two
    /// end with their terminal event (already emitted).
    fn finish(&mut self) -> Vec<String> {
        match self {
            Egress::Chat => vec!["data: [DONE]\n\n".to_string()],
            _ => vec![],
        }
    }

    /// Mid-stream upstream error, in the client protocol's framing.
    fn fail(&mut self, msg: &str) -> Vec<String> {
        match self {
            Egress::Chat => vec![],
            Egress::Responses(_) => vec![frame_data(&json!({"type": "error", "message": msg}))],
            Egress::Anthropic(_) => vec![frame_event(
                "error",
                &json!({"type": "error", "error": {"type": "api_error", "message": msg}}),
            )],
        }
    }
}

fn sse_body(resp: reqwest::Response, mut inbound: Inbound, mut egress: Egress) -> Body {
    Body::from_stream(async_stream::stream! {
        let mut buf: Vec<u8> = Vec::new();
        let mut bs = resp.bytes_stream();
        let mut failed = false;
        while let Some(chunk) = bs.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => break,
            };
            buf.extend_from_slice(&chunk);
            while let Some(pos) = find_subslice(&buf, b"\n\n") {
                let block: Vec<u8> = buf.drain(..pos).collect();
                buf.drain(..2.min(buf.len()));
                let text = String::from_utf8_lossy(&block);
                let mut data = String::new();
                for line in text.lines() {
                    if let Some(rest) = line.strip_prefix("data:") {
                        if !data.is_empty() {
                            data.push('\n');
                        }
                        data.push_str(rest.trim_start());
                    }
                }
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let ev: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                // An upstream error event ends the stream early.
                if ev.get("type").and_then(|t| t.as_str()) == Some("error")
                    || ev.get("error").is_some()
                {
                    let msg = trunc(&data, 500);
                    eprintln!("[gw] upstream stream error: {msg}");
                    for line in egress.fail(msg) {
                        yield Ok::<Bytes, io::Error>(Bytes::from(line));
                    }
                    failed = true;
                    break;
                }
                for c in inbound.handle(&ev) {
                    for line in egress.handle(&c) {
                        yield Ok::<Bytes, io::Error>(Bytes::from(line));
                    }
                }
            }
            if failed {
                break;
            }
        }
        for c in inbound.flush() {
            for line in egress.handle(&c) {
                yield Ok::<Bytes, io::Error>(Bytes::from(line));
            }
        }
        for line in egress.finish() {
            yield Ok::<Bytes, io::Error>(Bytes::from(line));
        }
    })
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

fn sse_response(body: Body) -> Response {
    let mut resp = Response::new(body);
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    resp.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    resp
}

fn json_response(status: StatusCode, v: Value) -> Response {
    let body = serde_json::to_vec(&v).unwrap_or_else(|_| b"{}".to_vec());
    let mut resp = Response::new(Body::from(body));
    *resp.status_mut() = status;
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    resp
}

fn raw_json_response(status: StatusCode, body: String) -> Response {
    let mut resp = Response::new(Body::from(body));
    *resp.status_mut() = status;
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    resp
}

fn error_json(status: StatusCode, msg: &str) -> Response {
    json_response(
        status,
        json!({"error": {"message": msg, "type": "gateway_error"}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use config::Protocol;

    fn tables() -> (HashMap<String, ModelCfg>, HashMap<String, ModelCfg>) {
        let mut statics = HashMap::new();
        statics.insert(
            "pinned".to_string(),
            ModelCfg::discovered(
                "pinned".to_string(),
                Protocol::Responses,
                "https://h/v1".to_string(),
            ),
        );
        let mut discovered = HashMap::new();
        discovered.insert(
            "claude-haiku-5-5".to_string(),
            ModelCfg::discovered(
                "claude-haiku-5-5".to_string(),
                Protocol::Chat,
                "https://h/v1".to_string(),
            ),
        );
        (statics, discovered)
    }

    #[test]
    fn lookup_prefers_exact_then_static() {
        let (s, d) = tables();
        assert_eq!(find_model_in(&s, &d, "pinned").unwrap().name, "pinned");
        assert_eq!(
            find_model_in(&s, &d, "claude-haiku-5-5").unwrap().name,
            "claude-haiku-5-5"
        );
        assert!(find_model_in(&s, &d, "nope").is_none());
    }

    #[test]
    fn lookup_falls_back_to_normalized_match() {
        let (s, d) = tables();
        // Dotted client spelling matches the dashed upstream id.
        let m = find_model_in(&s, &d, "claude-haiku-5.5").unwrap();
        assert_eq!(m.name, "claude-haiku-5-5");
    }

    #[test]
    fn lookup_refuses_ambiguous_normalized_match() {
        let (mut s, d) = tables();
        // Same normalized form as the discovered entry, but explicit.
        s.insert(
            "claude.haiku_5-5".to_string(),
            ModelCfg::discovered(
                "claude.haiku_5-5".to_string(),
                Protocol::Chat,
                "https://h/v1".to_string(),
            ),
        );
        assert!(find_model_in(&s, &d, "claude-haiku-5.5").is_none());
    }
}
