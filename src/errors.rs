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
            json_response(status, upstream_error_body(ingress, status, &body))
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
pub(crate) fn upstream_error_body(ingress: Ingress, status: StatusCode, body: &str) -> Value {
    let inner = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| match v.get("error") {
            Some(Value::Object(o)) => Some(Value::Object(o.clone())),
            _ => None,
        })
        .unwrap_or_else(|| {
            json!({
                "message": upstream_error_message(status, body),
                "type": "upstream_error",
            })
        });

    match ingress {
        Ingress::Anthropic => {
            let mut o = inner.as_object().cloned().unwrap_or_default();
            o.entry("message")
                .or_insert_with(|| json!(upstream_error_message(status, body)));
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
                .or_insert_with(|| json!(upstream_error_message(status, body)));
            json!({"error": Value::Object(o)})
        }
    }
}

/// Best-effort `message` out of an upstream error body of any protocol.
pub(crate) fn upstream_error_message(status: StatusCode, body: &str) -> String {
    let text = body.trim();
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        // Not JSON at all: an HTML error page or a proxy banner. The raw
        // text is still the most useful thing we have, bounded so a stray
        // megabyte of markup cannot end up in a log line or a response.
        Err(_) => return fallback_error_message(status, text),
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
        .map_or_else(|| fallback_error_message(status, text), |s| s.to_string())
}

pub(crate) fn fallback_error_message(status: StatusCode, text: &str) -> String {
    if text.is_empty() {
        format!("upstream returned status {} with an empty error body", status.as_u16())
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
                let body = upstream_error_body(ingress, StatusCode::BAD_GATEWAY, up);
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
        let chat = upstream_error_body(Ingress::OpenAiCompatible, StatusCode::TOO_MANY_REQUESTS, up);
        assert_eq!(chat["error"]["code"], "model_not_found");
        assert_eq!(chat["error"]["param"], "model");
        let anth = upstream_error_body(Ingress::Anthropic, StatusCode::TOO_MANY_REQUESTS, up);
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
            let out = upstream_error_body(Ingress::Anthropic, StatusCode::BAD_GATEWAY, upstream);
            assert_eq!(out["type"], "error", "envelope: {out}");
            assert_eq!(out["error"]["type"], want, "class from {upstream}");
        }

        // The upstream's own value is not discarded, only relabelled.
        let out = upstream_error_body(
            Ingress::Anthropic,
            StatusCode::BAD_GATEWAY,
            r#"{"error":{"message":"m","type":"rate_limit_exceeded","code":"rate_limit_exceeded"}}"#,
        );
        assert_eq!(out["error"]["code"], "rate_limit_exceeded");
    }

    #[test]
    fn non_json_upstream_error_stays_bounded() {
        // A proxy's HTML error page must not become a megabyte-long message
        // in both the log and the response body.
        let html = format!("<html>{}</html>", "x".repeat(5000));
        let out = upstream_error_body(Ingress::OpenAiCompatible, StatusCode::BAD_GATEWAY, &html);
        let msg = &out["error"]["message"];
        assert!(msg.as_str().unwrap().starts_with("upstream error: <html>"));
        assert!(
            msg.as_str().unwrap().len() < 600,
            "unbounded: {}",
            msg.as_str().unwrap().len()
        );

        // An empty body still produces something a client can read.
        let out = upstream_error_body(Ingress::OpenAiCompatible, StatusCode::BAD_REQUEST, "");
        assert!(!out["error"]["message"].as_str().unwrap().is_empty());
        // The empty case carries the status so operators can tell it apart.
        assert!(
            out["error"]["message"].as_str().unwrap().contains("400"),
            "empty body lost the status: {}",
            out["error"]["message"]
        );
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
}
