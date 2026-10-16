//! Client-protocol error envelopes.
//!
//! `Ingress` (the client surface) lives here because every
//! constructor below shapes failures for one ingress.

use axum::http::HeaderValue;
use axum::http::header;
use axum::http::StatusCode;
use axum::response::Response;
use axum::body::Body;
use serde_json::{json, Value};
use crate::config::Protocol;

pub(crate) enum UpstreamError {
    Status(StatusCode, String),
    Message(String),
}

/// Error response in the shape the ingress protocol expects.
pub(crate) fn ingress_error(ingress: Ingress, status: StatusCode, msg: &str) -> Response {
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
pub(crate) fn upstream_error_response(ingress: Ingress, e: UpstreamError) -> Response {
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
pub(crate) const ANTHROPIC_ERROR_CLASSES: &[&str] = &[
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
/// An upstream can speak any of the four protocols, so forwarding its body
/// verbatim hands e.g. an OpenAI-shaped error to an Anthropic client whose
/// SDK then fails to parse it and surfaces `undefined` instead of the
/// message. Three envelopes are produced:
///
/// - OpenAI (Chat, Responses and Google share it): `{"error": {...}}`
/// - Anthropic: `{"type": "error", "error": {...}}`
pub(crate) fn upstream_error_body(ingress: Ingress, body: &str) -> Value {
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
pub(crate) fn upstream_error_message(body: &str) -> String {
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

pub(crate) fn fallback_error_message(text: &str) -> String {
    if text.is_empty() {
        "upstream returned an empty error body".to_string()
    } else {
        format!("upstream error: {}", trunc(text, 500))
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Ingress {
    OpenAiCompatible,
    OpenAi,
    Anthropic,
    Google,
}

impl Ingress {
    pub(crate) fn protocol(&self) -> Protocol {
        match self {
            Ingress::OpenAiCompatible => Protocol::OpenAiCompatible,
            Ingress::OpenAi => Protocol::OpenAi,
            Ingress::Anthropic => Protocol::Anthropic,
            Ingress::Google => Protocol::Google,
        }
    }
}

pub(crate) fn json_response(status: StatusCode, v: Value) -> Response {
    let body = serde_json::to_vec(&v).unwrap_or_else(|_| b"{}".to_vec());
    let mut resp = Response::new(Body::from(body));
    *resp.status_mut() = status;
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    resp
}

pub(crate) fn error_json(status: StatusCode, msg: &str) -> Response {
    json_response(
        status,
        json!({"error": {"message": msg, "type": "gateway_error"}}),
    )
}

/// Truncate to `n` bytes at a character boundary, for log excerpts.
pub(crate) fn trunc(s: &str, n: usize) -> &str {
    if s.len() <= n {
        return s;
    }
    let mut end = n;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// JSON type name, for error messages about a malformed body.
pub(crate) fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
