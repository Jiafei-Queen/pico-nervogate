mod config;
mod discovery;
mod models_dev;
mod translate;

use std::collections::HashMap;
use std::io;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use futures_util::StreamExt;
use serde_json::{json, Value};

use config::{Config, ModelCfg, Protocol};
use discovery::refresh_discovery;
use translate::{
    anthropic_to_chat_request, anthropic_to_openai, chat_failure, chat_to_anthropic_fatal,
    chat_to_anthropic_loss, chat_to_anthropic_response, chat_to_anthropic_thinking_conflict,
    chat_to_responses_fatal, chat_to_responses_loss, chat_to_responses_request,
    chat_to_responses_response, frame_data, frame_event, openai_to_anthropic_with_max,
    responses_stateful_error, responses_to_chat_request, responses_to_openai, AnthropicStream,
    ChatToAnthropicStream, ChatToResponsesStream, ResponsesStream, SseEvent,
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

    let app = app(state.clone());

    let listener = tokio::net::TcpListener::bind(&listen)
        .await
        .unwrap_or_else(|e| panic!("bind {listen}: {e}"));
    println!("pico-nervogate listening on {listen} ({n} models)");
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
        // The gateway applies `max_body_bytes` itself so an oversized request
        // can be rejected in the ingress protocol's error shape. Axum's own
        // 2 MiB `Bytes` extractor limit would fire first, as plain text.
        .layer(DefaultBodyLimit::disable())
        .with_state(state)
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

pub(crate) fn upstream_headers(st: &St, m: &ModelCfg) -> HeaderMap {
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

/// Top-level field names of an outbound request, sorted.
///
/// Enough to answer "did the translation produce the shape the upstream
/// expects?" without writing the prompt itself to the log.
fn body_summary(body: &Value) -> String {
    match body.as_object() {
        Some(o) => {
            let mut keys: Vec<&str> = o.keys().map(String::as_str).collect();
            keys.sort_unstable();
            keys.join(",")
        }
        None => kind_of(body).to_string(),
    }
}

/// JSON type name, for error messages about a malformed upstream body.
fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

pub(crate) fn upstream_url(base: &str, m: &ModelCfg) -> String {
    let base = base.trim_end_matches('/');
    match m.protocol {
        Protocol::Chat => format!("{base}/chat/completions"),
        Protocol::Anthropic => format!("{base}/messages"),
        Protocol::Responses => format!("{base}/responses"),
    }
}

/// Resolve model + build the upstream body for the given protocol.
///
/// When the client's protocol already matches the upstream's, `native` is
/// the client's own body and it is forwarded untouched (bar the model id
/// and `stream`). Routing an Anthropic request through the canonical chat
/// shape and back would silently drop block-level `cache_control` (killing
/// prompt caching), thinking signatures and `tool_result.is_error`.
fn build_body(
    m: &ModelCfg,
    req: &Value,
    stream: bool,
    native: Option<&Value>,
    anthropic_max_tokens: i64,
) -> Value {
    let upstream = m.upstream_model();
    let mut body = match native {
        Some(v) => {
            let mut v = v.clone();
            // The reserved passthrough key never goes on the wire, on any path.
            v.as_object_mut().map(|o| o.remove("x_nervogate"));
            v
        }
        None => match m.protocol {
            Protocol::Chat => {
                let mut v = req.clone();
                // The reserved passthrough key never goes on the wire.
                v.as_object_mut().map(|o| o.remove("x_nervogate"));
                v
            }
            Protocol::Anthropic => {
                openai_to_anthropic_with_max(req, m.thinking_mode(), anthropic_max_tokens)
            }
            Protocol::Responses => chat_to_responses_request(req),
        },
    };
    if let Some(o) = body.as_object_mut() {
        o.insert("model".into(), json!(upstream));
        if stream {
            o.insert("stream".into(), json!(true));
        } else {
            // avoid asking upstream for a stream we will not consume
            o.remove("stream");
        }
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
///
/// Only the *envelope* is rewritten; the upstream's own `error` object is
/// carried through, so `param`, `code`, `inner_error` and friends survive.
/// The upstream status code is preserved too — a 429 stays a 429, which is
/// what every SDK's retry logic keys on.
fn upstream_error_response(ingress: Ingress, e: UpstreamError) -> Response {
    match e {
        UpstreamError::Message(msg) => ingress_error(ingress, StatusCode::BAD_GATEWAY, &msg),
        UpstreamError::Status(status, body) => {
            json_response(status, upstream_error_body(ingress, &body))
        }
    }
}

/// Error classes Anthropic clients recognize in `error.type`. Anything else
/// becomes `api_error`: an unknown class makes SDK retry logic fall through
/// to its generic branch, which is the safe default for a gateway fault.
const ANTHROPIC_ERROR_CLASSES: &[&str] = &[
    "api_error",
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "request_too_large",
    "rate_limit_error",
    "api_timeout_error",
    "overloaded_error",
];

/// Rewrite an upstream error body into `ingress`'s error envelope.
///
/// An upstream can speak any of the three protocols, so forwarding its body
/// verbatim hands e.g. an OpenAI-shaped error to an Anthropic client whose
/// SDK then fails to parse it and surfaces `undefined` instead of the
/// message. Three envelopes are produced:
///
/// - OpenAI (Chat and Responses share it): `{"error": {...}}`
/// - Anthropic: `{"type": "error", "error": {...}}`
fn upstream_error_body(ingress: Ingress, body: &str) -> Value {
    let inner = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| match v.get("error") {
            Some(Value::Object(o)) => Some(Value::Object(o.clone())),
            _ => None,
        })
        .unwrap_or_else(|| {
            json!({
                "message": upstream_error_message(body),
                "type": "upstream_error",
            })
        });

    match ingress {
        Ingress::Anthropic => {
            let mut o = inner.as_object().cloned().unwrap_or_default();
            o.entry("message")
                .or_insert_with(|| json!(upstream_error_message(body)));
            // Anthropic puts the error *class* inside `error`, and its
            // clients branch on it. Only real Anthropic classes may pass
            // through: the literal "error" is the SSE framing marker, and an
            // OpenAI class such as "rate_limit_exceeded" is meaningless here.
            // The upstream's own value is not lost — it stays in `error.code`.
            let class = o
                .get("type")
                .and_then(|x| x.as_str())
                .filter(|s| ANTHROPIC_ERROR_CLASSES.contains(s))
                .unwrap_or("api_error")
                .to_string();
            o.insert("type".into(), json!(class));
            json!({"type": "error", "error": Value::Object(o)})
        }
        _ => {
            let mut o = inner.as_object().cloned().unwrap_or_default();
            o.entry("message")
                .or_insert_with(|| json!(upstream_error_message(body)));
            json!({"error": Value::Object(o)})
        }
    }
}

/// Best-effort `message` out of an upstream error body of any protocol.
fn upstream_error_message(body: &str) -> String {
    let text = body.trim();
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        // Not JSON at all: an HTML error page or a proxy banner. The raw
        // text is still the most useful thing we have, bounded so a stray
        // megabyte of markup cannot end up in a log line or a response.
        Err(_) => return fallback_error_message(text),
    };
    // `{"error": "rate limited"}` — some gateways send a bare string.
    if let Some(s) = v.get("error").and_then(|x| x.as_str()) {
        return s.to_string();
    }
    v.pointer("/error/message")
        .or_else(|| v.get("message"))
        .or_else(|| v.pointer("/error/detail"))
        .or_else(|| v.get("detail"))
        .and_then(|x| x.as_str())
        .filter(|s| !s.trim().is_empty())
        .map_or_else(|| fallback_error_message(text), |s| s.to_string())
}

fn fallback_error_message(text: &str) -> String {
    if text.is_empty() {
        "upstream returned an empty error body".to_string()
    } else {
        format!("upstream error: {}", trunc(text, 500))
    }
}

async fn call_upstream(
    st: &St,
    m: &ModelCfg,
    req: &Value,
    stream: bool,
    native: Option<&Value>,
) -> Result<reqwest::Response, UpstreamError> {
    let base = st
        .cfg
        .read()
        .unwrap()
        .base_url_for(m)
        .map_err(UpstreamError::Message)?;
    let url = upstream_url(&base, m);
    let body = build_body(
        m,
        req,
        stream,
        native,
        st.cfg.read().unwrap().anthropic_max_tokens,
    );
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
            return Err(UpstreamError::Message(format!(
                "upstream request failed: {e}"
            )));
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
        // The outgoing body holds the user's prompt verbatim plus whatever
        // they inlined (keys in headers aside, tool arguments and file
        // contents routinely carry secrets too). Logging it writes all of
        // that to stderr, which in practice is a container log shipped
        // somewhere durable. Shape only, unless explicitly opted in.
        eprintln!(
            "[gw] outgoing model={} keys={}",
            m.name,
            body_summary(&body)
        );
        if std::env::var("GATEWAY_DEBUG_BODY").is_ok_and(|v| v == "1" || v == "true") {
            eprintln!(
                "[gw] outgoing body model={} body={}",
                m.name,
                trunc(&serde_json::to_string(&body).unwrap_or_default(), 2000)
            );
        }
        Err(UpstreamError::Status(
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            text,
        ))
    }
}

// ---------------------------------------------------------------------------
// Ingress handlers: POST /v1/chat/completions, /v1/responses, /v1/messages
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Ingress {
    Chat,
    Responses,
    Anthropic,
}

impl Ingress {
    fn protocol(&self) -> Protocol {
        match self {
            Ingress::Chat => Protocol::Chat,
            Ingress::Responses => Protocol::Responses,
            Ingress::Anthropic => Protocol::Anthropic,
        }
    }
}

async fn chat(State(st): State<St>, req: Request) -> Response {
    handle_ingress(st, req, Ingress::Chat).await
}

async fn responses_ingress(State(st): State<St>, req: Request) -> Response {
    handle_ingress(st, req, Ingress::Responses).await
}

async fn messages_ingress(State(st): State<St>, req: Request) -> Response {
    let mut resp = handle_ingress(st, req, Ingress::Anthropic).await;
    resp.headers_mut().insert(
        header::HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static("2023-06-01"),
    );
    resp
}

/// Byte ceiling for a client body. 0 means unlimited, which `to_bytes` would
/// otherwise read as "reject everything".
fn body_limit(cfg: &Config) -> usize {
    if cfg.max_body_bytes == 0 {
        usize::MAX
    } else {
        cfg.max_body_bytes
    }
}

fn declared_len(req: &Request) -> Option<u64> {
    req.headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
}

/// Protocol-shaped 413.
///
/// The `Bytes` extractor would reject an oversized body on its own, but with
/// a plain-text body no SDK can parse, so the gateway reads the body itself
/// and reports the failure in the ingress protocol's envelope.
fn body_too_large(ingress: Ingress, limit: usize) -> Response {
    ingress_error(
        ingress,
        StatusCode::PAYLOAD_TOO_LARGE,
        &format!("request body exceeds the {limit} byte limit"),
    )
}

/// One pipeline for all three ingress surfaces: normalize to the canonical
/// chat request, call upstream, translate back into the ingress protocol.
async fn handle_ingress(st: St, req: Request, ingress: Ingress) -> Response {
    let limit = body_limit(&st.cfg.read().unwrap());
    // A declared Content-Length over the limit is refused before anything is
    // buffered. Chunked bodies declare none and are caught by the counting
    // read below instead.
    if let Some(len) = declared_len(&req) {
        if len > limit as u64 {
            return body_too_large(ingress, limit);
        }
    }
    let body = match axum::body::to_bytes(req.into_body(), limit).await {
        Ok(b) => b,
        // Under a limit the only failure `to_bytes` raises is exceeding it;
        // a truncated upload is indistinguishable and equally unusable.
        Err(_) => return body_too_large(ingress, limit),
    };
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
    let stream = match req.get("stream") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(other) => {
            // `"stream": "true"` would otherwise be read as `false` and the
            // client gets a JSON body where it expected SSE.
            return ingress_error(
                ingress,
                StatusCode::BAD_REQUEST,
                &format!("`stream` must be a boolean, got {}", kind_of(other)),
            );
        }
    };

    // Same protocol on both ends: forward the client's own body untouched.
    let native = ingress.protocol() == m.protocol;
    let strict = st.cfg.read().unwrap().strict_params;

    // Normalize the ingress request to the canonical chat shape. Skipped
    // on the native path, which keeps the original body intact.
    let chat_req = if native {
        Value::Null
    } else {
        match ingress {
            Ingress::Chat => req.clone(),
            Ingress::Responses => responses_to_chat_request(&req),
            Ingress::Anthropic => anthropic_to_chat_request(&req),
        }
    };

    if !native {
        let (lost, fatal) = match m.protocol {
            Protocol::Anthropic => (
                chat_to_anthropic_loss(&chat_req),
                chat_to_anthropic_fatal(&chat_req),
            ),
            Protocol::Responses => (
                chat_to_responses_loss(&chat_req),
                chat_to_responses_fatal(&chat_req),
            ),
            Protocol::Chat => (vec![], None),
        };
        if let Some(msg) = fatal {
            if strict {
                return ingress_error(ingress, StatusCode::BAD_REQUEST, &msg);
            }
            eprintln!("[gw] req model={name}: {msg} (dropping; strict_params off)");
        }
        if !lost.is_empty() {
            eprintln!(
                "[gw] req model={name} protocol={}: dropped unsupported params: {}",
                m.protocol.as_str(),
                lost.join(", ")
            );
        }
        // Forwarded, not dropped — so this warns but never rejects, even
        // under `strict_params`. Anthropic-protocol upstreams differ on
        // whether they enforce this, and a 400 here would break the ones
        // that don't.
        if m.protocol == Protocol::Anthropic {
            let conflicting = chat_to_anthropic_thinking_conflict(&chat_req);
            if !conflicting.is_empty() {
                eprintln!(
                    "[gw] req model={name} protocol=anthropic: thinking is on and \
                     {} was also set; Anthropic may reject it (thinking fixes \
                     sampling at the default of 1). Forwarding as-is — drop it \
                     from the client request to avoid a 400.",
                    conflicting.join(", ")
                );
            }
        }
    }

    let resp = match call_upstream(
        &st,
        &m,
        &chat_req,
        stream,
        if native { Some(&req) } else { None },
    )
    .await
    {
        Ok(r) => r,
        Err(e) => return upstream_error_response(ingress, e),
    };

    if stream {
        let inbound = match m.protocol {
            Protocol::Chat => Inbound::Chat {
                name: name.clone(),
                held: None,
                usage: None,
            },
            Protocol::Anthropic => Inbound::Anthropic(AnthropicStream::new(&name)),
            Protocol::Responses => Inbound::Responses(ResponsesStream::new(&name)),
        };
        let egress = match ingress {
            Ingress::Chat => Egress::Chat,
            Ingress::Responses => Egress::Responses(ChatToResponsesStream::new(&name)),
            Ingress::Anthropic => Egress::Anthropic(ChatToAnthropicStream::new(&name)),
        };
        sse_response(sse_body(
            resp,
            inbound,
            egress,
            native.then(|| name.clone()),
        ))
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
            // A 200 with a non-JSON body is an upstream fault (HTML error
            // page, proxy banner, ...). The raw bytes are useless to a client
            // speaking any of the three protocols, so return a protocol-shaped
            // 502 and keep the excerpt in the log.
            Err(_) => {
                eprintln!(
                    "[gw] non-JSON upstream body model={} len={} body={}",
                    m.name,
                    text.len(),
                    trunc(&text, 200)
                );
                return ingress_error(
                    ingress,
                    StatusCode::BAD_GATEWAY,
                    &format!("upstream returned a non-JSON body ({} bytes)", text.len()),
                );
            }
        };
        // Every translator below indexes into the body as an object; a
        // non-object (`[]`, `"OK"`, `null`) would panic on `Value` IndexMut
        // and, with panic=abort, kill the whole gateway.
        if !up.is_object() {
            eprintln!(
                "[gw] non-object upstream JSON model={} protocol={}",
                m.name,
                m.protocol.as_str()
            );
            return ingress_error(
                ingress,
                StatusCode::BAD_GATEWAY,
                &format!(
                    "upstream returned a non-object JSON body ({})",
                    kind_of(&up)
                ),
            );
        }
        if native {
            // Same protocol both ends: only the public model id needs
            // rewriting, everything else is the client's own response.
            let mut v = up;
            if let Some(o) = v.as_object_mut() {
                o.insert("model".into(), json!(name.clone()));
            }
            return json_response(StatusCode::OK, v);
        }
        let mut chat = match m.protocol {
            Protocol::Chat => {
                let mut v = up;
                if let Some(o) = v.as_object_mut() {
                    o.insert("model".into(), json!(name.clone()));
                }
                v
            }
            Protocol::Anthropic => anthropic_to_openai(&up, &name),
            Protocol::Responses => responses_to_openai(&up, &name),
        };
        // A `failed` Responses upstream must not reach the client as a
        // 200 with an empty answer: surface it as a gateway error.
        if let Some(msg) = chat_failure(&chat) {
            eprintln!("[gw] upstream response failed model={}: {msg}", m.name);
            return ingress_error(ingress, StatusCode::BAD_GATEWAY, &msg);
        }
        // `upstream_error` is an internal hand-off from `responses_to_openai`
        // to `chat_failure`; it is not part of any wire protocol.
        chat.as_object_mut().map(|o| o.remove("upstream_error"));
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
        /// Finish chunk parked until its usage shows up (or the stream ends).
        held: Option<Value>,
        /// Usage seen so far, held independently of `held` so that upstream
        /// sending usage *before* the finish chunk still counts.
        usage: Option<Value>,
    },
    Anthropic(AnthropicStream),
    Responses(ResponsesStream),
}

impl Inbound {
    /// Handle one upstream SSE `data:` payload (a JSON object).
    fn handle(&mut self, ev: &Value) -> Vec<Value> {
        match self {
            Inbound::Chat { name, held, usage } => {
                // OpenAI sends usage in a trailing `choices: []` chunk. Buffer
                // it separately from `held` so the order of the two chunks
                // does not decide whether usage survives.
                if let Some(u) = ev.get("usage").filter(|u| !u.is_null()) {
                    *usage = Some(u.clone());
                }
                // A usage-only chunk carries no choices and no content.
                if ev.pointer("/choices/0").is_none() {
                    return vec![];
                }
                let mut out: Vec<Value> = held.take().into_iter().collect();
                let mut v = ev.clone();
                if let Some(o) = v.as_object_mut() {
                    o.insert("model".into(), json!(name.clone()));
                }
                if let Some(u) = v.get("usage").cloned().filter(|u| !u.is_null()) {
                    *usage = Some(u);
                }
                let finished = v
                    .pointer("/choices/0/finish_reason")
                    .is_some_and(|f| !f.is_null());
                // Park the finish chunk only while usage is still missing:
                // translated egresses read usage off that very chunk.
                if finished && v.get("usage").is_none_or(|u| u.is_null()) && usage.is_none() {
                    *held = Some(v);
                } else if finished {
                    if let (Some(o), Some(u)) = (v.as_object_mut(), usage.as_ref()) {
                        o.insert("usage".into(), u.clone());
                    }
                    out.push(v);
                } else {
                    out.push(v);
                }
                out
            }
            Inbound::Anthropic(s) => s.handle(ev),
            Inbound::Responses(s) => s.handle(ev),
        }
    }

    /// Flush a held finish chunk at end of stream, backfilling any usage
    /// that arrived separately.
    fn flush(&mut self) -> Vec<Value> {
        match self {
            Inbound::Chat { held, usage, .. } => {
                let mut out = vec![];
                if let Some(mut h) = held.take() {
                    if let (Some(o), Some(u)) = (h.as_object_mut(), usage.as_ref()) {
                        o.insert("usage".into(), u.clone());
                    }
                    out.push(h);
                }
                out
            }
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

    /// Trailer for a stream that ended cleanly. Chat closes with the
    /// `[DONE]` sentinel; the other two emit their terminal event if the
    /// upstream never sent a finish chunk (otherwise the client waits
    /// forever on a closed connection).
    ///
    /// `passthrough` marks the same-protocol path, where the gateway never
    /// ran a single chunk through this state machine and the upstream has
    /// already emitted its own terminal event. Synthesizing one here would
    /// append a second `message_start` / `response.created` to a stream that
    /// is already correctly framed.
    fn finish(&mut self, passthrough: bool) -> Vec<String> {
        if passthrough {
            return match self {
                // The upstream's own `[DONE]` is swallowed by the parser and
                // has to be re-emitted; the other protocols terminate
                // themselves upstream.
                Egress::Chat => vec!["data: [DONE]\n\n".to_string()],
                _ => vec![],
            };
        }
        match self {
            Egress::Chat => vec!["data: [DONE]\n\n".to_string()],
            Egress::Responses(s) => match s.finish() {
                Some(events) => events.iter().map(|e| e.frame()).collect(),
                None => vec![],
            },
            Egress::Anthropic(s) => match s.finish() {
                Some(events) => events.iter().map(|e| e.frame()).collect(),
                None => vec![],
            },
        }
    }

    /// Mid-stream upstream failure, in the client protocol's framing. This
    /// is a terminal signal: Chat clients otherwise read a truncated answer
    /// terminated by `[DONE]` as a successful one.
    ///
    /// On the passthrough path there is no gateway-side block bookkeeping to
    /// close, so only the error frame itself is emitted.
    fn fail(&mut self, msg: &str, passthrough: bool) -> Vec<String> {
        match self {
            Egress::Chat => vec![
                frame_data(&json!({
                    "error": {"message": msg, "type": "gateway_error"}
                })),
                "data: [DONE]\n\n".to_string(),
            ],
            Egress::Responses(s) => {
                let mut out: Vec<SseEvent> = if passthrough { vec![] } else { s.abort() };
                out.push(SseEvent::data(json!({"type": "error", "message": msg})));
                out.iter().map(|e| e.frame()).collect()
            }
            Egress::Anthropic(s) => {
                let mut out: Vec<SseEvent> = if passthrough { vec![] } else { s.abort() };
                out.push(SseEvent::ev(
                    "error",
                    json!({"type": "error", "error": {"type": "api_error", "message": msg}}),
                ));
                out.iter().map(|e| e.frame()).collect()
            }
        }
    }
}

/// What one upstream SSE block means, once parsed and classified.
///
/// Lives outside the stream body so the main loop and the end-of-stream
/// flush share one implementation: the two paths used to duplicate the
/// error checks, and a fix applied to only one of them is a bug.
enum Block {
    /// Nothing to forward (blank, `[DONE]`, unparseable).
    Skip,
    /// The upstream signalled a failure; the stream ends here.
    Error(String),
    /// Native path: forward verbatim, keeping the upstream's `event:` name.
    Passthrough(Option<String>, Value),
    /// Translated path: client-facing SSE lines rendered by the egress.
    Chunks(Vec<String>),
}

fn classify_block(
    block: &[u8],
    model_rewrite: &Option<String>,
    inbound: &mut Inbound,
    egress: &mut Egress,
) -> Block {
    let f = sse_fields(block);
    if f.data.is_empty() || f.data == "[DONE]" {
        return Block::Skip;
    }
    let ev: Value = match serde_json::from_str(&f.data) {
        Ok(v) => v,
        Err(_) => return Block::Skip,
    };
    if is_stream_error(&ev) {
        eprintln!("[gw] upstream stream error: {}", trunc(&f.data, 500));
        return Block::Error(stream_error_msg(&ev, &f.data));
    }
    if let Some(m) = model_rewrite {
        // Native path: hand the client's own event straight back, event
        // name included — Anthropic clients dispatch on it and ignore an
        // event that arrives without one.
        let mut ev = ev;
        if let Some(o) = ev.as_object_mut() {
            o.insert("model".into(), json!(m));
        }
        return Block::Passthrough(f.name, ev);
    }
    let mut out = vec![];
    for c in inbound.handle(&ev) {
        // A canonical chunk flagged as failed (Responses `response.failed`)
        // must not be framed as a normal finish; it terminates the stream
        // with an error.
        if c.pointer("/choices/0/finish_reason")
            .and_then(|f| f.as_str())
            == Some("error")
        {
            eprintln!("[gw] upstream response failed");
            return Block::Error("upstream response failed".to_string());
        }
        out.extend(egress.handle(&c));
    }
    Block::Chunks(out)
}

/// Render one classified block as client-facing SSE lines. A passthrough
/// block that triggers an error is reported so the caller can end the stream.
fn render_block(block: Block) -> (Vec<String>, Option<String>) {
    match block {
        Block::Skip => (vec![], None),
        Block::Error(msg) => (vec![], Some(msg)),
        Block::Passthrough(name, ev) => (
            vec![match &name {
                Some(n) => frame_event(n, &ev),
                None => frame_data(&ev),
            }],
            None,
        ),
        Block::Chunks(lines) => (lines, None),
    }
}

/// `model_rewrite` is set on the native (same-protocol) path: events are
/// forwarded verbatim except for the public model id.
fn sse_body(
    resp: reqwest::Response,
    mut inbound: Inbound,
    mut egress: Egress,
    model_rewrite: Option<String>,
) -> Body {
    let passthrough = model_rewrite.is_some();
    Body::from_stream(async_stream::stream! {
        let mut parser = SseParser::new();
        let mut bs = resp.bytes_stream();
        let mut upstream_err: Option<String> = None;

        loop {
            let chunk = match bs.next().await {
                Some(Ok(c)) => c,
                Some(Err(e)) => {
                    // A transport failure is not a clean EOF: the answer is
                    // truncated and the client must not read it as success.
                    upstream_err = Some(format!("upstream stream error: {e}"));
                    break;
                }
                None => break,
            };
            parser.push(&chunk);

            while let Some(range) = parser.next_block() {
                let (lines, err) = render_block(classify_block(
                    parser.block(range), &model_rewrite, &mut inbound, &mut egress,
                ));
                for line in lines {
                    yield Ok::<Bytes, io::Error>(Bytes::from(line));
                }
                if let Some(msg) = err {
                    // An upstream error event ends the stream early.
                    upstream_err = Some(msg);
                    break;
                }
            }
            if upstream_err.is_some() {
                break;
            }
        }

        // Whatever is left in the buffer when the upstream closed without a
        // trailing blank line is still a real event; dropping it loses the
        // final chunk (often the one carrying finish_reason / usage).
        if upstream_err.is_none() {
            if let Some(range) = parser.finish() {
                let (lines, err) = render_block(classify_block(
                    parser.block(range), &model_rewrite, &mut inbound, &mut egress,
                ));
                for line in lines {
                    yield Ok::<Bytes, io::Error>(Bytes::from(line));
                }
                upstream_err = err;
            }
        }

        if !passthrough {
            for c in inbound.flush() {
                for line in egress.handle(&c) {
                    yield Ok::<Bytes, io::Error>(Bytes::from(line));
                }
            }
        }

        let trailer = match upstream_err {
            Some(msg) => egress.fail(&msg, passthrough),
            None => egress.finish(passthrough),
        };
        for line in trailer {
            yield Ok::<Bytes, io::Error>(Bytes::from(line));
        }
    })
}

// ---------------------------------------------------------------------------
// SSE event parsing
// ---------------------------------------------------------------------------

/// Incremental SSE block splitter.
///
/// Splits on a *blank line* per the SSE spec rather than on a literal
/// `\n\n`, so `\r\n\r\n` (nginx / CDN / Windows upstreams), `\r\r` and
/// mixed terminators all dispatch events. Scanning resumes where the
/// previous call stopped, keeping total work linear in stream size.
struct SseParser {
    buf: Vec<u8>,
    /// Bytes of `buf` already dispatched as events.
    pos: usize,
    /// Start of the line currently being scanned. Bytes before it belong to
    /// complete lines, so they are never rescanned.
    line_start: usize,
}

impl SseParser {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            pos: 0,
            line_start: 0,
        }
    }

    fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// Reclaim dispatched bytes. `drain` from the front is O(remaining), so
    /// doing it every event would be quadratic on a busy stream.
    fn compact(&mut self) {
        if self.pos >= 64 * 1024 {
            self.buf.drain(..self.pos);
            // `line_start` is always >= `pos`, but `saturating_sub` keeps a
            // future refactor from turning a bookkeeping slip into a panic —
            // and `panic = "abort"` would take the whole gateway with it.
            self.line_start = self.line_start.saturating_sub(self.pos);
            self.pos = 0;
        }
    }

    /// Next complete event block, as a range into `buf` (valid until the
    /// next call). The block holds the event's lines but not the blank line
    /// that terminated it.
    fn next_block(&mut self) -> Option<(usize, usize)> {
        self.compact();
        let mut i = self.line_start;
        loop {
            if i >= self.buf.len() {
                // No terminator yet: the tail is an incomplete line.
                self.line_start = i;
                return None;
            }
            let term = match self.buf[i] {
                b'\n' => 1,
                b'\r' => {
                    if i + 1 >= self.buf.len() {
                        // A lone trailing CR may still turn out to be the
                        // first half of a CRLF split across chunks.
                        self.line_start = i;
                        return None;
                    }
                    if self.buf[i + 1] == b'\n' {
                        2
                    } else {
                        1
                    }
                }
                _ => {
                    i += 1;
                    continue;
                }
            };
            let at = i;
            let blank = at == self.line_start;
            let start = self.pos;
            i += term;
            self.line_start = i;
            if blank {
                self.pos = i;
                return Some((start, at));
            }
        }
    }

    /// Trailing bytes left when the upstream closed without a final blank
    /// line. The last event must not be dropped: it is frequently the only
    /// chunk carrying `finish_reason` / usage.
    fn finish(&mut self) -> Option<(usize, usize)> {
        self.compact();
        let start = self.pos;
        let mut end = self.buf.len();
        // Drop a dangling line terminator, it carries no data.
        while end > start && matches!(self.buf[end - 1], b'\n' | b'\r') {
            end -= 1;
        }
        self.pos = self.buf.len();
        // Keep the two cursors consistent: `next_block()` resumes from
        // `line_start`, and `compact()` derives it from `pos`.
        self.line_start = self.pos;
        (end > start).then_some((start, end))
    }

    fn block(&self, range: (usize, usize)) -> &[u8] {
        &self.buf[range.0..range.1]
    }
}

/// Split an SSE block into lines, accepting LF, CRLF and bare CR.
fn sse_lines(block: &[u8]) -> Vec<&[u8]> {
    let mut out = vec![];
    let mut i = 0;
    while i < block.len() {
        let start = i;
        while i < block.len() && block[i] != b'\n' && block[i] != b'\r' {
            i += 1;
        }
        out.push(&block[start..i]);
        i = match block.get(i) {
            Some(b'\r') if block.get(i + 1) == Some(&b'\n') => i + 2,
            Some(_) => i + 1,
            None => block.len(),
        };
    }
    out
}

/// The two SSE fields the gateway cares about from one block.
///
/// `name` is the `event:` value. Anthropic clients dispatch on it and ignore
/// the event entirely when it is absent, so it has to survive passthrough —
/// dropping it silently turns a healthy stream into zero events.
struct SseFields {
    name: Option<String>,
    data: String,
}

/// Parse one SSE block. Multiple `data:` lines join with `\n` and exactly one
/// leading space is stripped (the spec's rule) — trimming all leading
/// whitespace would corrupt indented JSON payloads.
fn sse_fields(block: &[u8]) -> SseFields {
    let mut name = None;
    let mut data = String::new();
    for line in sse_lines(block) {
        // `:`-prefixed lines are comments (keep-alive padding, pings).
        if line.first() == Some(&b':') {
            continue;
        }
        let colon = match line.iter().position(|&c| c == b':') {
            Some(p) => p,
            // A bare field name with no colon carries an empty value, which
            // is a no-op for `data`; nothing to append.
            None => continue,
        };
        let mut val = &line[colon + 1..];
        if val.first() == Some(&b' ') {
            val = &val[1..];
        }
        match &line[..colon] {
            b"event" => name = Some(String::from_utf8_lossy(val).into_owned()),
            b"data" => {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(&String::from_utf8_lossy(val));
            }
            _ => {}
        }
    }
    SseFields { name, data }
}

/// Upstream event signalling a failure. `"error": null` (sent by some
/// OpenAI-compatible servers on healthy chunks) is *not* an error — that is
/// exactly what S-5 is about, and the `Value::Null` arm catches it. Any other
/// non-null error counts, including an empty `{}`: missing a real failure is
/// worse than aborting a stream that only looked healthy.
fn is_stream_error(ev: &Value) -> bool {
    if ev.get("type").and_then(|t| t.as_str()) == Some("error") {
        return true;
    }
    match ev.get("error") {
        Some(Value::Object(_)) => true,
        Some(Value::String(s)) => !s.is_empty(),
        _ => false,
    }
}

/// Best-effort human-readable message out of an upstream error event.
fn stream_error_msg(ev: &Value, raw: &str) -> String {
    ev.pointer("/error/message")
        .or_else(|| ev.pointer("/response/error/message"))
        .or_else(|| ev.get("message"))
        .and_then(|x| x.as_str())
        .map(String::from)
        .unwrap_or_else(|| trunc(raw, 500).to_string())
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

fn error_json(status: StatusCode, msg: &str) -> Response {
    json_response(
        status,
        json!({"error": {"message": msg, "type": "gateway_error"}}),
    )
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
        enrichment: Arc::new(RwLock::new(models_dev::Store::empty())),
        config_path: String::new(),
        config_mtime: RwLock::new(None),
    })
}

#[cfg(test)]
fn sample_config() -> Config {
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
        max_body_bytes: 32 * 1024 * 1024,
        models_dev: Default::default(),
        reload: Default::default(),
        discovery: Default::default(),
        models: vec![],
    }
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

    // ---- SSE parsing ------------------------------------------------------

    /// Feed raw bytes through the parser and collect every dispatched block.
    fn parse_all(chunks: &[&[u8]]) -> Vec<String> {
        let mut p = SseParser::new();
        let mut out = vec![];
        for c in chunks {
            p.push(c);
            while let Some(r) = p.next_block() {
                let d = sse_fields(p.block(r)).data;
                if !d.is_empty() {
                    out.push(d);
                }
            }
        }
        if let Some(r) = p.finish() {
            let d = sse_fields(p.block(r)).data;
            if !d.is_empty() {
                out.push(d);
            }
        }
        out
    }

    #[test]
    fn sse_splits_on_lf_crlf_and_cr_blank_lines() {
        for (label, sep) in [("LF", "\n"), ("CRLF", "\r\n"), ("CR", "\r")] {
            let raw = format!("data: one{sep}{sep}data: two{sep}{sep}");
            let got = parse_all(&[raw.as_bytes()]);
            assert_eq!(got, vec!["one", "two"], "separator {label}");
        }
    }

    #[test]
    fn sse_handles_mixed_terminators_and_split_chunks() {
        // CRLF/LF mixed, and an event split mid-payload across chunks.
        let got = parse_all(&[b"data: a\r\n\r\ndata: b\n\nda", b"ta: c\r\r"]);
        assert_eq!(got, vec!["a", "b", "c"]);
    }

    #[test]
    fn sse_does_not_split_a_crlf_across_chunks() {
        // The trailing CR of chunk 1 must not be read as a lone terminator,
        // which would split `data: x` into an early dispatch.
        let mut p = SseParser::new();
        p.push(b"data: x\r");
        assert!(p.next_block().is_none());
        p.push(b"\n\r\n");
        let (s, e) = p.next_block().expect("event after CRLF completes");
        assert_eq!(sse_fields(p.block((s, e))).data, "x");
    }

    #[test]
    fn sse_strips_only_one_leading_space_and_skips_comments() {
        let got = parse_all(&[b": keep-alive\ndata:  {\"a\":1}\n\n"]);
        assert_eq!(got, vec![" {\"a\":1}"]);
    }

    #[test]
    fn sse_multi_line_data_joins_with_newline() {
        let got = parse_all(&[b"data: {\"a\":\ndata: 1}\n\n"]);
        assert_eq!(got, vec!["{\"a\":\n1}"]);
    }

    #[test]
    fn sse_flushes_trailing_event_without_blank_line() {
        // B-3: upstream closed without a trailing blank line. The last event
        // (here the finish chunk) must still be delivered.
        let mut p = SseParser::new();
        p.push(b"data: {\"choices\":[]}\n\ndata: {\"finish\":1}");
        assert!(p.next_block().is_some());
        assert!(p.next_block().is_none());
        let r = p.finish().expect("trailing event");
        assert_eq!(sse_fields(p.block(r)).data, "{\"finish\":1}");
        // Draining must not repeat the same event.
        assert!(p.finish().is_none());
    }

    #[test]
    fn sse_reclaims_consumed_prefix_across_many_events() {
        // L-7: repeatedly dispatching events must not grow the buffer
        // without bound, and scanning must not rescan consumed bytes.
        let mut p = SseParser::new();
        let mut events = 0;
        for i in 0..2000 {
            p.push(format!("data: {{\"i\":{i},\"pad\":\"{}\"}}\n\n", "x".repeat(200)).as_bytes());
            while let Some(r) = p.next_block() {
                assert!(!sse_fields(p.block(r)).data.is_empty());
                events += 1;
            }
            // Compaction keeps the live window bounded regardless of how
            // many events have already gone out.
            assert!(
                p.buf.len() < 64 * 1024,
                "buffer grew to {} after {} events",
                p.buf.len(),
                events
            );
        }
        assert_eq!(events, 2000);
    }

    #[test]
    fn null_error_field_is_not_a_stream_error() {
        // S-5: some OpenAI-compatible servers send `"error": null` on
        // healthy chunks. Treating that as a failure kills a good stream.
        assert!(!is_stream_error(
            &json!({"choices": [{"delta": {}}], "error": null})
        ));
        assert!(is_stream_error(&json!({"error": {"message": "boom"}})));
        assert!(is_stream_error(
            &json!({"type": "error", "message": "boom"})
        ));
        assert!(!is_stream_error(
            &json!({"type": "response.output_text.delta"})
        ));
        // An empty object carries the same intent as `null` only when the
        // upstream meant it as a placeholder; as a standalone failure it is
        // the shape some gateways emit, and missing it is the worse error.
        assert!(is_stream_error(
            &json!({"choices": [{"delta": {}}], "error": {}})
        ));
        assert!(!is_stream_error(&json!({"error": ""})));
    }

    #[test]
    fn non_object_upstream_body_is_a_502_not_a_panic() {
        // B-1: `Value` IndexMut panics on arrays/strings/numbers and the
        // release profile is panic = "abort", so this killed the gateway.
        for bad in [json!([]), json!("OK"), json!(null), json!(7)] {
            assert!(!bad.is_object(), "{:?} must not pass the object guard", bad);
            assert_ne!(kind_of(&bad), "object");
        }
        assert_eq!(kind_of(&json!([])), "array");
        assert_eq!(kind_of(&json!("OK")), "string");
        assert_eq!(kind_of(&Value::Null), "null");
        assert_eq!(kind_of(&json!({})), "object");
    }

    #[test]
    fn stream_error_msg_prefers_nested_message() {
        assert_eq!(
            stream_error_msg(&json!({"error": {"message": "rate limited"}}), "{}"),
            "rate limited"
        );
        assert_eq!(
            stream_error_msg(&json!({"response": {"error": {"message": "nope"}}}), "{}"),
            "nope"
        );
        assert_eq!(stream_error_msg(&json!({"message": "flat"}), "{}"), "flat");
        // Falls back to the raw payload when there is no message field.
        assert_eq!(stream_error_msg(&json!({"code": "x"}), "raw"), "raw");
    }

    // ---- S-3: same-protocol passthrough -----------------------------------

    #[test]
    fn sse_keeps_the_event_name_anthropic_clients_dispatch_on() {
        // The official Anthropic SDK branches on `sse.event` and silently
        // drops an event that has none, so the name must survive parsing.
        let f = sse_fields(b"event: message_delta\ndata: {\"t\":1}");
        assert_eq!(f.name.as_deref(), Some("message_delta"));
        assert_eq!(f.data, "{\"t\":1}");

        // Field order is irrelevant to the spec.
        let f = sse_fields(b"data: {\"t\":1}\nevent: message_stop");
        assert_eq!(f.name.as_deref(), Some("message_stop"));
        assert_eq!(f.data, "{\"t\":1}");

        // Chat framing carries no name.
        assert_eq!(sse_fields(b"data: [DONE]").name, None);
        // Comments and unknown fields must not be mistaken for the name.
        assert_eq!(sse_fields(b": ping\nid: 7\ndata: x").name, None);
        assert_eq!(sse_fields(b": ping\nid: 7\ndata: x").data, "x");
    }

    #[test]
    fn passthrough_keeps_the_upstream_event_name() {
        let mut inbound = Inbound::Responses(ResponsesStream::new("m"));
        let mut egress = Egress::Responses(ChatToResponsesStream::new("m"));
        let name = Some("m".to_string());
        let block = b"event: response.output_text.delta\ndata: {\"type\":\"x\"}";
        let (lines, err) = render_block(classify_block(block, &name, &mut inbound, &mut egress));
        assert!(err.is_none());
        assert_eq!(lines.len(), 1);
        assert!(
            lines[0].starts_with("event: response.output_text.delta\ndata: "),
            "event name was dropped: {}",
            lines[0]
        );
        // ...and the public model id is still rewritten on the way out.
        assert!(lines[0].contains("\"model\":\"m\""));
    }

    #[test]
    fn passthrough_does_not_synthesize_a_second_terminal_event() {
        // The gateway never ran a chunk through these state machines on the
        // native path, so `terminated` is still false. Calling `finish()`
        // unconditionally appended a duplicate `message_start` /
        // `response.created` to an already-complete stream.
        let mut anthropic = Egress::Anthropic(ChatToAnthropicStream::new("m"));
        assert!(
            anthropic.finish(true).is_empty(),
            "anthropic passthrough must defer to the upstream's own terminal event"
        );
        let fail = anthropic.fail("boom", true);
        assert_eq!(fail.len(), 1, "only the error frame, no block bookkeeping");
        assert!(fail[0].starts_with("event: error\n"));

        let mut responses = Egress::Responses(ChatToResponsesStream::new("m"));
        assert!(responses.finish(true).is_empty());
        let fail = responses.fail("boom", true);
        assert_eq!(fail.len(), 1);
        assert!(fail[0].contains("\"type\":\"error\""));

        // Chat still needs its `[DONE]`: the upstream's own sentinel is
        // swallowed by the parser and must be re-emitted.
        let mut chat = Egress::Chat;
        assert_eq!(chat.finish(true), vec!["data: [DONE]\n\n".to_string()]);
        assert_eq!(chat.fail("boom", true).len(), 2);

        // The translated path is unchanged and still synthesizes.
        let mut responses = Egress::Responses(ChatToResponsesStream::new("m"));
        assert!(!responses.finish(false).is_empty());
    }

    #[test]
    fn x_nervogate_is_stripped_on_every_path() {
        // README promises the reserved key never reaches the wire, and the
        // native path forwards the client's body verbatim.
        let m = ModelCfg::discovered("m".to_string(), Protocol::Anthropic, "https://h/v1".into());
        let req = json!({"model": "m", "x_nervogate": {"top_k": 5}});
        let body = build_body(&m, &req, false, Some(&req), 4096);
        assert!(
            body.get("x_nervogate").is_none(),
            "reserved key leaked upstream: {body}"
        );
    }

    #[test]
    fn classify_block_treats_a_failed_chunk_as_a_stream_error() {
        // The main loop and the end-of-stream flush used to carry separate
        // copies of this check; both paths must agree.
        let mut inbound = Inbound::Responses(ResponsesStream::new("m"));
        let mut egress = Egress::Chat;
        let block =
            b"data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"boom\"}}}";
        let (_, err) = render_block(classify_block(block, &None, &mut inbound, &mut egress));
        assert_eq!(err.as_deref(), Some("upstream response failed"));

        // A healthy event is not an error.
        let block = b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}";
        let (lines, err) = render_block(classify_block(block, &None, &mut inbound, &mut egress));
        assert!(err.is_none());
        assert!(!lines.is_empty());
    }

    #[test]
    fn sse_parser_tolerates_use_after_finish() {
        // `finish()` resets the cursors; a later `next_block()` used to
        // underflow `line_start - pos`, which under `panic = "abort"` would
        // take the whole gateway down.
        let mut p = SseParser::new();
        p.push(b"data: {\"a\":1}\n\ndata: {\"b\":2}");
        while p.next_block().is_some() {}
        assert!(p.finish().is_some());
        assert!(p.next_block().is_none());
        assert!(p.finish().is_none());
    }

    // ---- error shape (L-2) ------------------------------------------------

    /// Assert `body` is a well-formed error for `ingress`, and return the
    /// message. A body missing that field is exactly the bug: the SDK
    /// surfaces `undefined` and the user sees nothing useful.
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

    #[test]
    fn upstream_errors_are_reshaped_for_every_ingress() {
        // A client must never receive an error in a protocol it does not
        // speak: an OpenAI-shaped 429 handed to an Anthropic SDK fails to
        // parse and the user sees `undefined` instead of the rate limit.
        let upstreams = [
            // OpenAI / Responses
            r#"{"error":{"message":"rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#,
            // Anthropic
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            // bare string
            r#"{"error":"nope"}"#,
            // bare message
            r#"{"message":"bad request"}"#,
            // not JSON at all
            "<html>502 Bad Gateway</html>",
            // empty
            "",
        ];
        for up in upstreams {
            for ingress in [Ingress::Chat, Ingress::Responses, Ingress::Anthropic] {
                let body = upstream_error_body(ingress, up);
                let msg = assert_error_shape(ingress, &body);
                assert!(!msg.is_empty(), "empty message from {up:?} -> {ingress:?}");
            }
        }
    }

    #[test]
    fn upstream_error_keeps_the_status_code_and_extra_fields() {
        // SDK retry logic keys on the status, and operators debug with the
        // upstream's own `code` / `param` fields; neither may be flattened.
        let up = r#"{"error":{"message":"bad","type":"invalid_request_error","param":"model","code":"model_not_found"}}"#;
        for ingress in [Ingress::Chat, Ingress::Responses, Ingress::Anthropic] {
            let resp = upstream_error_response(
                ingress,
                UpstreamError::Status(StatusCode::TOO_MANY_REQUESTS, up.to_string()),
            );
            assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        }
        let chat = upstream_error_body(Ingress::Chat, up);
        assert_eq!(chat["error"]["code"], "model_not_found");
        assert_eq!(chat["error"]["param"], "model");
        let anth = upstream_error_body(Ingress::Anthropic, up);
        assert_eq!(anth["error"]["code"], "model_not_found");
    }

    #[test]
    fn anthropic_error_class_is_always_a_class_the_sdk_knows() {
        // The literal "error" is Anthropic's SSE framing marker, not an
        // error class. An OpenAI class is meaningless to an Anthropic SDK and
        // makes its retry logic fall through to the generic branch.
        for (upstream, want) in [
            (r#"{"type":"error","message":"boom"}"#, "api_error"),
            (
                r#"{"error":{"message":"m","type":"rate_limit_exceeded"}}"#,
                "api_error",
            ),
            (r#"{"error":{"message":"m","type":""}}"#, "api_error"),
            (r#"{"message":"m"}"#, "api_error"),
            // Real Anthropic classes survive untouched.
            (
                r#"{"type":"error","error":{"type":"overloaded_error","message":"x"}}"#,
                "overloaded_error",
            ),
            (
                r#"{"error":{"message":"m","type":"invalid_request_error"}}"#,
                "invalid_request_error",
            ),
        ] {
            let out = upstream_error_body(Ingress::Anthropic, upstream);
            assert_eq!(out["type"], "error", "envelope: {out}");
            assert_eq!(out["error"]["type"], want, "class from {upstream}");
        }

        // The upstream's own value is not discarded, only relabelled.
        let out = upstream_error_body(
            Ingress::Anthropic,
            r#"{"error":{"message":"m","type":"rate_limit_exceeded","code":"rate_limit_exceeded"}}"#,
        );
        assert_eq!(out["error"]["code"], "rate_limit_exceeded");
    }

    #[test]
    fn non_json_upstream_error_stays_bounded() {
        // A proxy's HTML error page must not become a megabyte-long message
        // in both the log and the response body.
        let html = format!("<html>{}</html>", "x".repeat(5000));
        let out = upstream_error_body(Ingress::Chat, &html);
        let msg = &out["error"]["message"];
        assert!(msg.as_str().unwrap().starts_with("upstream error: <html>"));
        assert!(
            msg.as_str().unwrap().len() < 600,
            "unbounded: {}",
            msg.as_str().unwrap().len()
        );

        // An empty body still produces something a client can read.
        let out = upstream_error_body(Ingress::Chat, "");
        assert!(!out["error"]["message"].as_str().unwrap().is_empty());
    }

    // ---- log hygiene (L-8) ------------------------------------------------

    #[test]
    fn body_summary_reports_keys_never_values() {
        // The outbound body is the user's prompt. Whatever we log about it
        // must not contain any of the content.
        let secret = "sk-live-DO-NOT-LOG-THIS";
        let body = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": secret}],
            "temperature": 0.5,
        });
        let summary = body_summary(&body);
        assert_eq!(summary, "messages,model,temperature");
        assert!(!summary.contains(secret));
        assert!(!summary.contains("gpt-4"));

        // A non-object body degrades to its type rather than its content.
        assert_eq!(body_summary(&json!("hi")), "string");
        assert_eq!(body_summary(&json!([1, 2])), "array");
    }

    // ---- body limit (L-9) -------------------------------------------------

    #[test]
    fn zero_body_limit_means_unlimited() {
        // `to_bytes(body, 0)` rejects everything; 0 is the config's way of
        // saying "no limit" and must not reach it as an actual zero.
        assert_eq!(
            body_limit(&Config {
                max_body_bytes: 0,
                ..sample_config()
            }),
            usize::MAX
        );
        assert_eq!(
            body_limit(&Config {
                max_body_bytes: 4096,
                ..sample_config()
            }),
            4096
        );
    }

    #[test]
    fn oversized_body_is_rejected_in_the_ingress_protocol_shape() {
        // Axum's own limit produced a plain-text 413 that no SDK can parse;
        // the client could not tell a size rejection from a proxy error.
        for ingress in [Ingress::Chat, Ingress::Responses, Ingress::Anthropic] {
            let resp = body_too_large(ingress, 1024);
            assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
            let body: Value = read_body(resp);
            assert!(assert_error_shape(ingress, &body).contains("1024"));
        }
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
            ("/v1/chat/completions", Ingress::Chat),
            ("/v1/responses", Ingress::Responses),
            ("/v1/messages", Ingress::Anthropic),
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

    /// Drain a `Response` body into JSON.
    fn read_body(resp: Response) -> Value {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
                    .await
                    .expect("read body");
                serde_json::from_slice(&bytes).expect("body is JSON")
            })
    }
}
