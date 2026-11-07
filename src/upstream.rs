//! Upstream requests: URL, headers, body and dispatch.

use crate::errors::UpstreamError;
use crate::errors::{kind_of, trunc};
use crate::state::St;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use axum::http::{HeaderMap, HeaderValue};
use axum::http::header;
use axum::http::StatusCode;
use serde_json::{json, Value};
use crate::config::{Config, GoogleAuth, ModelCfg, Protocol};
use crate::translate::{
    chat_to_responses_request, openai_to_anthropic_with_max, reasoning_detail_to_openrouter,
};
use crate::translate::google::chat_to_google_request;

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
    if m.protocol == Protocol::Google && !google_uses_bearer(m, &cfg) {
        // AI Studio key auth: the Bearer token above does not apply.
        h.remove(header::AUTHORIZATION);
        if let Ok(v) = HeaderValue::from_str(&api_key) {
            h.insert(
                header::HeaderName::from_static("x-goog-api-key"),
                v,
            );
        }
    }
    h
}

/// Does this Google model authenticate with a Bearer token (Vertex)?
/// `google_auth = "bearer"` forces it, `"api_key"` forces key auth,
/// `"auto"` (default) sniffs the base URL for AI Platform.
pub(crate) fn google_uses_bearer(m: &ModelCfg, cfg: &Config) -> bool {
    match m.google_auth {
        GoogleAuth::Bearer => true,
        GoogleAuth::ApiKey => false,
        GoogleAuth::Auto => cfg
            .base_url_for(m)
            .map(|b| b.contains("aiplatform.googleapis.com"))
            .unwrap_or(false),
    }
}

pub(crate) fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Top-level field names of an outbound request, sorted.
///
/// Enough to answer "did the translation produce the shape the upstream
/// expects?" without writing the prompt itself to the log.
pub(crate) fn body_summary(body: &Value) -> String {
    match body.as_object() {
        Some(o) => {
            let mut keys: Vec<&str> = o.keys().map(String::as_str).collect();
            keys.sort_unstable();
            keys.join(",")
        }
        None => kind_of(body).to_string(),
    }
}

pub(crate) fn upstream_url(base: &str, m: &ModelCfg, stream: bool) -> String {
    let base = base.trim_end_matches('/');
    match m.protocol {
        Protocol::OpenAiCompatible => format!("{base}/chat/completions"),
        Protocol::Anthropic => format!("{base}/messages"),
        Protocol::OpenAi => format!("{base}/responses"),
        Protocol::Google => {
            let model = m.upstream_model();
            if stream {
                format!("{base}/models/{model}:streamGenerateContent?alt=sse")
            } else {
                format!("{base}/models/{model}:generateContent")
            }
        }
    }
}

/// Resolve model + build the upstream body for the given protocol.
///
/// When the client's protocol already matches the upstream's, `native` is
/// the client's own body and it is forwarded untouched (bar the model id
/// and `stream`). Routing an Anthropic request through the canonical chat
/// shape and back would silently drop block-level `cache_control` (killing
/// prompt caching), thinking signatures and `tool_result.is_error`.
pub(crate) fn build_body(
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
            Protocol::OpenAiCompatible => {
                let mut v = req.clone();
                // The reserved passthrough key never goes on the wire.
                if let Some(o) = v.as_object_mut() {
                    o.remove("x_nervogate");
                }
                if let Some(msgs) = v.get_mut("messages").and_then(|x| x.as_array_mut()) {
                    for msg in msgs {
                        let Some(details) = msg.get_mut("reasoning_details") else {
                            continue;
                        };
                        if !m.forward_reasoning_details {
                            if let Some(o) = msg.as_object_mut() {
                                o.remove("reasoning_details");
                            }
                        } else if let Some(arr) = details.as_array_mut() {
                            // OpenRouter understands the field, but only in
                            // its own schema; translate the canonical
                            // entries instead of leaking Anthropic-style
                            // member names upstream.
                            arr.retain_mut(|d| {
                                // Already in the upstream's schema: preserve
                                // the provider-specific `format` verbatim.
                                if d.get("type")
                                    .and_then(|t| t.as_str())
                                    .is_some_and(|t| t.starts_with("reasoning."))
                                {
                                    return true;
                                }
                                match reasoning_detail_to_openrouter(d) {
                                    Some(converted) => {
                                        *d = converted;
                                        true
                                    }
                                    None => false,
                                }
                            });
                        }
                    }
                }
                v
            }
            Protocol::Anthropic => {
                openai_to_anthropic_with_max(req, m.thinking_mode(), anthropic_max_tokens)
            }
            Protocol::OpenAi => chat_to_responses_request(req),
            Protocol::Google => chat_to_google_request(req),
        },
    };
    if let Some(o) = body.as_object_mut() {
        // Google carries the model in the URL path, not the body.
        if m.protocol != Protocol::Google {
            o.insert("model".into(), json!(upstream));
        }
        if stream {
            // Google switches endpoint instead of reading a `stream` flag.
            if m.protocol != Protocol::Google {
                o.insert("stream".into(), json!(true));
            }
        } else {
            // avoid asking upstream for a stream we will not consume
            o.remove("stream");
        }
    }
    body
}

/// Drop every canonical reasoning block. Used by the one-shot retry after a
/// reasoning 400: the documented safe degradation is to send no thinking
/// block at all (dropping the whole block is accepted, an empty signature or
/// a foreign blob is not).
fn strip_reasoning_details(req: &Value) -> Value {
    let mut v = req.clone();
    if let Some(msgs) = v.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for m in msgs {
            if let Some(o) = m.as_object_mut() {
                o.remove("reasoning_details");
            }
        }
    }
    v
}

/// Native-path counterpart: remove thinking blocks from an Anthropic body.
///
/// An assistant message that held *only* thinking would become
/// `content: []`, which Anthropic rejects. Drop such a message instead of
/// sending an empty turn.
fn strip_thinking_blocks(native: &Value) -> Value {
    let mut v = native.clone();
    if let Some(msgs) = v.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for m in msgs.iter_mut() {
            if let Some(blocks) = m.get_mut("content").and_then(|c| c.as_array_mut()) {
                blocks.retain(|b| {
                    !matches!(
                        b.get("type").and_then(|t| t.as_str()),
                        Some("thinking") | Some("redacted_thinking")
                    )
                });
            }
        }
        msgs.retain(|m| {
            if m.get("role").and_then(|r| r.as_str()) != Some("assistant") {
                return true;
            }
            match m.get("content") {
                Some(Value::Array(a)) => !a.is_empty(),
                _ => true,
            }
        });
    }
    v
}

/// Native-path counterpart: remove `reasoning` input items from a Responses
/// body.
fn strip_reasoning_items(native: &Value) -> Value {
    let mut v = native.clone();
    if let Some(items) = v.get_mut("input").and_then(|x| x.as_array_mut()) {
        items.retain(|it| it.get("type").and_then(|t| t.as_str()) != Some("reasoning"));
    }
    v
}

fn is_reasoning_error(text: &str, protocol: Protocol) -> bool {
    let lower = text.to_ascii_lowercase();
    match protocol {
        Protocol::Anthropic => lower.contains("signature") && lower.contains("thinking"),
        Protocol::OpenAi => {
            lower.contains("encrypted content") || lower.contains("invalid_encrypted_content")
        }
        _ => false,
    }
}

async fn post_json(
    st: &St,
    m: &ModelCfg,
    url: &str,
    body: &Value,
) -> Result<reqwest::Response, reqwest::Error> {
    st.client
        .post(url)
        .headers(upstream_headers(st, m))
        .json(body)
        .send()
        .await
}

pub(crate) async fn call_upstream(
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
    let url = upstream_url(&base, m, stream);
    let body = build_body(
        m,
        req,
        stream,
        native,
        st.cfg.read().unwrap().anthropic_max_tokens,
    );
    let t0 = now_ms();
    let stream_tag = if stream { "stream" } else { "nonstream" };
    let mut resp = match post_json(st, m, &url, &body).await {
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

    // A rejected reasoning blob is fatal upstream but recoverable here: the
    // documented degradation is to send the history without thinking /
    // reasoning blocks. Retry exactly once so a bad blob cannot amplify
    // traffic.
    if resp.status() == StatusCode::BAD_REQUEST
        && matches!(m.protocol, Protocol::Anthropic | Protocol::OpenAi)
    {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if is_reasoning_error(&text, m.protocol) {
            let max_tokens = st.cfg.read().unwrap().anthropic_max_tokens;
            let retry_body = match (m.protocol, native) {
                (Protocol::Anthropic, Some(n)) => {
                    let stripped = strip_thinking_blocks(n);
                    build_body(m, req, stream, Some(&stripped), max_tokens)
                }
                (Protocol::OpenAi, Some(n)) => {
                    let stripped = strip_reasoning_items(n);
                    build_body(m, req, stream, Some(&stripped), max_tokens)
                }
                (_, None) => {
                    let stripped = strip_reasoning_details(req);
                    build_body(m, &stripped, stream, None, max_tokens)
                }
                _ => body.clone(),
            };
            eprintln!(
                "[gw] retry model={} stream={} without reasoning blocks after an {} 400",
                m.name,
                stream_tag,
                m.protocol.as_str()
            );
            match post_json(st, m, &url, &retry_body).await {
                Ok(r) => resp = r,
                Err(e) => {
                    eprintln!(
                        "[gw] retry model={} stream={} -> connect-error: {}",
                        m.name, stream_tag, e
                    );
                    return Err(UpstreamError::Message(format!(
                        "upstream request failed: {e}"
                    )));
                }
            }
        } else {
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
                "[gw] outgoing model={} keys={}",
                m.name,
                body_summary(&body)
            );
            return Err(UpstreamError::Status(status, text));
        }
    }

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
// Ingress handlers: POST /v1/chat/completions, /v1/responses, /v1/messages,
// /v1beta/models/{action}
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
    fn stripping_thinking_never_leaves_an_empty_assistant_turn() {
        // Anthropic rejects `content: []`; an assistant turn that contained
        // only thinking must be dropped after the fallback strips it.
        let native = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig"}
                ]},
                {"role": "user", "content": "more"}
            ]
        });
        let stripped = strip_thinking_blocks(&native);
        let msgs = stripped["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2, "empty assistant turn survived: {stripped}");
        assert!(msgs
            .iter()
            .all(|m| m["role"] == "user" || !m["content"].as_array().unwrap().is_empty()));
    }

    #[test]
    fn stripping_reasoning_details_removes_the_field() {
        let req = json!({
            "model": "m",
            "messages": [{"role": "assistant", "content": "hi", "reasoning_details": [
                {"type": "thinking", "thinking": "x", "signature": "nv1:a:s"}
            ]}]
        });
        let stripped = strip_reasoning_details(&req);
        assert!(stripped["messages"][0].get("reasoning_details").is_none());
    }
}
