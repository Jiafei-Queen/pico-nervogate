mod config;
mod translate;

use std::collections::HashMap;
use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::Arc;

use config::{Config, ModelCfg, Protocol};
use translate::{
    anthropic_to_openai, chat_to_responses_request, openai_to_anthropic, responses_to_chat_request,
    responses_to_openai, AnthropicStream, ResponsesStream,
};

struct Inner {
    cfg: Config,
    api_key: String,
    client: reqwest::Client,
    models: HashMap<String, ModelCfg>,
}

type St = Arc<Inner>;

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

    let listen = cfg.listen.clone();
    let n = models.len();
    let state: St = Arc::new(Inner {
        cfg,
        api_key,
        client,
        models,
    });

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/responses", post(responses_ingress))
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
    let data: Vec<Value> = st
        .models
        .values()
        .map(|m| {
            json!({
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": st.cfg.owned_by.as_deref().unwrap_or(env!("CARGO_PKG_NAME")),
                "provider": st.cfg.provider.as_deref().unwrap_or(env!("CARGO_PKG_NAME")),
                "info": {"meta": {"capabilities": {"vision": m.vision}}}
            })
        })
        .collect();
    json_response(StatusCode::OK, json!({"object": "list", "data": data}))
}

// ---------------------------------------------------------------------------
// Upstream request builders
// ---------------------------------------------------------------------------

fn upstream_headers(st: &St, m: &ModelCfg) -> HeaderMap {
    let mut h = HeaderMap::new();
    let bearer = format!("Bearer {}", st.api_key);
    if let Ok(v) = HeaderValue::from_str(&bearer) {
        h.insert(header::AUTHORIZATION, v);
    }
    if let Ok(v) = HeaderValue::from_str(&st.cfg.user_agent) {
        h.insert(header::USER_AGENT, v);
    }
    if let (Some(session_id), Some(session_header)) = (&st.cfg.session_id, &st.cfg.session_header) {
        if let (Ok(v), Ok(name)) = (
            HeaderValue::from_str(session_id),
            header::HeaderName::from_bytes(session_header.as_bytes()),
        ) {
            h.insert(name, v);
        }
    }
    for (k, v) in st.cfg.extra_headers.iter().chain(m.extra_headers.iter()) {
        if let (Ok(name), Ok(val)) = (
            header::HeaderName::from_bytes(k.as_bytes()),
            HeaderValue::from_str(v),
        ) {
            h.insert(name, val);
        }
    }
    if m.protocol == Protocol::Anthropic {
        if let Ok(v) = HeaderValue::from_str(&st.api_key) {
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
        Protocol::Chat => req.clone(),
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

impl IntoResponse for UpstreamError {
    fn into_response(self) -> Response {
        match self {
            UpstreamError::Status(code, body) => {
                let mut resp = Response::new(Body::from(body));
                *resp.status_mut() = code;
                resp.headers_mut().insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );
                resp
            }
            UpstreamError::Message(msg) => error_json(StatusCode::BAD_GATEWAY, &msg),
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
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

async fn chat(State(st): State<St>, body: Bytes) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return error_json(StatusCode::BAD_REQUEST, &format!("invalid JSON body: {e}")),
    };
    let name = match req.get("model").and_then(|x| x.as_str()) {
        Some(n) => n.to_string(),
        None => return error_json(StatusCode::BAD_REQUEST, "missing field: model"),
    };
    let m = match st.models.get(&name) {
        Some(m) => m.clone(),
        None => {
            eprintln!("[gw] req unknown-model={name}");
            return error_json(StatusCode::NOT_FOUND, &format!("unknown model: {name}"));
        }
    };
    let stream = req.get("stream").and_then(|x| x.as_bool()).unwrap_or(false);

    let resp = match call_upstream(&st, &m, &req, stream).await {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };

    if stream {
        let b = match m.protocol {
            Protocol::Chat => passthrough_body(resp),
            Protocol::Anthropic => sse_body(resp, SseState::Anthropic(AnthropicStream::new(&name))),
            Protocol::Responses => sse_body(resp, SseState::Responses(ResponsesStream::new(&name))),
        };
        sse_response(b)
    } else {
        let text = match resp.text().await {
            Ok(t) => t,
            Err(e) => return error_json(StatusCode::BAD_GATEWAY, &format!("read upstream: {e}")),
        };
        let up: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => {
                return raw_json_response(StatusCode::OK, text);
            }
        };
        let out = match m.protocol {
            Protocol::Chat => {
                let mut v = up;
                v["model"] = json!(name);
                v
            }
            Protocol::Anthropic => anthropic_to_openai(&up, &name),
            Protocol::Responses => responses_to_openai(&up, &name),
        };
        json_response(StatusCode::OK, out)
    }
}

// ---------------------------------------------------------------------------
// POST /v1/responses (ingress -> chat canonical -> upstream)
// ---------------------------------------------------------------------------

async fn responses_ingress(State(st): State<St>, body: Bytes) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return error_json(StatusCode::BAD_REQUEST, &format!("invalid JSON body: {e}")),
    };
    let name = match req.get("model").and_then(|x| x.as_str()) {
        Some(n) => n.to_string(),
        None => return error_json(StatusCode::BAD_REQUEST, "missing field: model"),
    };
    let m = match st.models.get(&name) {
        Some(m) => m.clone(),
        None => {
            eprintln!("[gw] req unknown-model={name}");
            return error_json(StatusCode::NOT_FOUND, &format!("unknown model: {name}"));
        }
    };
    let stream = req.get("stream").and_then(|x| x.as_bool()).unwrap_or(false);

    // Responses ingress is normalized to a chat request, then routed normally.
    let chat_req = responses_to_chat_request(&req);

    let resp = match call_upstream(&st, &m, &chat_req, stream).await {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };

    // For simplicity the gateway always answers Responses ingress with the chat
    // completion translated back into a Responses-shaped object (non-stream).
    let text = match resp.text().await {
        Ok(t) => t,
        Err(e) => return error_json(StatusCode::BAD_GATEWAY, &format!("read upstream: {e}")),
    };
    let chat: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return raw_json_response(StatusCode::OK, text),
    };
    let openai = match m.protocol {
        Protocol::Chat => {
            let mut v = chat;
            v["model"] = json!(name);
            v
        }
        Protocol::Anthropic => anthropic_to_openai(&chat, &name),
        Protocol::Responses => responses_to_openai(&chat, &name),
    };
    json_response(StatusCode::OK, chat_to_responses_response(&openai, &name))
}

fn chat_to_responses_response(chat: &Value, model: &str) -> Value {
    let content = chat
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    json!({
        "id": translate::gen_id("resp_"),
        "object": "response",
        "created_at": translate::now_ts(),
        "model": model,
        "status": "completed",
        "output": [{
            "type": "message",
            "id": translate::gen_id("msg_"),
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": content, "annotations": []}]
        }],
        "usage": chat.get("usage").cloned().unwrap_or(json!({}))
    })
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

enum SseState {
    Anthropic(AnthropicStream),
    Responses(ResponsesStream),
}

fn passthrough_body(resp: reqwest::Response) -> Body {
    let s = resp
        .bytes_stream()
        .map(|r| r.map_err(io::Error::other));
    Body::from_stream(s)
}

fn sse_body(resp: reqwest::Response, mut state: SseState) -> Body {
    Body::from_stream(async_stream::stream! {
        let mut buf: Vec<u8> = Vec::new();
        let mut bs = resp.bytes_stream();
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
                let chunks = match &mut state {
                    SseState::Anthropic(s) => s.handle(&ev),
                    SseState::Responses(s) => s.handle(&ev),
                };
                for c in chunks {
                    let line = format!("data: {}\n\n", serde_json::to_string(&c).unwrap());
                    yield Ok::<Bytes, io::Error>(Bytes::from(line));
                }
            }
        }
        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
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
