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
    use super::*;
    use crate::config::Protocol;
    use crate::errors::*;
    use crate::service::*;
    use crate::sse::*;
    use crate::state::*;
    use crate::translate::*;
    use crate::upstream::*;
    use axum::body::Body;
    use axum::extract::Request;
    use axum::http::StatusCode;
    use axum::response::Response;
    use serde_json::{json, Value};
    use crate::config::ModelCfg;

    fn tables() -> (HashMap<String, ModelCfg>, HashMap<String, ModelCfg>) {
        let mut statics = HashMap::new();
        statics.insert(
            "pinned".to_string(),
            ModelCfg::discovered(
                "pinned".to_string(),
                Protocol::OpenAi,
                "https://h/v1".to_string(),
            ),
        );
        let mut discovered = HashMap::new();
        discovered.insert(
            "claude-haiku-5-5".to_string(),
            ModelCfg::discovered(
                "claude-haiku-5-5".to_string(),
                Protocol::OpenAiCompatible,
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
                Protocol::OpenAiCompatible,
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
        let mut chat = Egress::Chat(ChatReasoningEgress::new(
            crate::config::ReasoningOutput::ReasoningContent,
        ));
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
        let mut egress = Egress::Chat(ChatReasoningEgress::new(
            crate::config::ReasoningOutput::ReasoningContent,
        ));
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
            for ingress in [Ingress::OpenAiCompatible, Ingress::OpenAi, Ingress::Anthropic, Ingress::Google] {
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
        for ingress in [Ingress::OpenAiCompatible, Ingress::OpenAi, Ingress::Anthropic, Ingress::Google] {
            let resp = upstream_error_response(
                ingress,
                UpstreamError::Status(StatusCode::TOO_MANY_REQUESTS, up.to_string()),
            );
            assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        }
        let chat = upstream_error_body(Ingress::OpenAiCompatible, up);
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
        let out = upstream_error_body(Ingress::OpenAiCompatible, &html);
        let msg = &out["error"]["message"];
        assert!(msg.as_str().unwrap().starts_with("upstream error: <html>"));
        assert!(
            msg.as_str().unwrap().len() < 600,
            "unbounded: {}",
            msg.as_str().unwrap().len()
        );

        // An empty body still produces something a client can read.
        let out = upstream_error_body(Ingress::OpenAiCompatible, "");
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
        for ingress in [Ingress::OpenAiCompatible, Ingress::OpenAi, Ingress::Anthropic, Ingress::Google] {
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
