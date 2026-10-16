//! Request pipeline: ingress normalization and dispatch.

use crate::errors::{Ingress, ingress_error, json_response, kind_of, trunc, upstream_error_response};
use crate::state::{St, find_model};
use crate::upstream::call_upstream;
use crate::sse::{Egress, Inbound, sse_body, sse_response};
use axum::http::header;
use axum::http::StatusCode;
use axum::extract::Request;
use axum::response::Response;
use serde_json::{json, Value};
use crate::config::{Config, ModelCfg, Protocol};
use crate::translate::{AnthropicStream, ChatReasoningEgress, ChatToAnthropicStream, ChatToResponsesStream, ResponsesStream, anthropic_to_chat_request, anthropic_to_openai, apply_reasoning_output_to_message, chat_failure, chat_to_anthropic_fatal, chat_to_anthropic_loss, chat_to_anthropic_response, chat_to_anthropic_thinking_conflict, chat_to_responses_fatal, chat_to_responses_loss, chat_to_responses_response, responses_stateful_error, responses_to_chat_request, responses_to_openai};
use crate::translate::google::{ChatToGoogleStream, GoogleStream, chat_to_google_fatal, chat_to_google_loss, chat_to_google_response, google_to_chat, google_to_chat_request};

/// Byte ceiling for a client body. 0 means unlimited, which `to_bytes` would
/// otherwise read as "reject everything".
pub(crate) fn body_limit(cfg: &Config) -> usize {
    if cfg.max_body_bytes == 0 {
        usize::MAX
    } else {
        cfg.max_body_bytes
    }
}

pub(crate) fn declared_len(req: &Request) -> Option<u64> {
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
pub(crate) fn body_too_large(ingress: Ingress, limit: usize) -> Response {
    ingress_error(
        ingress,
        StatusCode::PAYLOAD_TOO_LARGE,
        &format!("request body exceeds the {limit} byte limit"),
    )
}

/// One pipeline for all ingress surfaces: normalize to the canonical
/// chat request, call upstream, translate back into the ingress protocol.
pub(crate) async fn handle_ingress(st: St, req: Request, ingress: Ingress) -> Response {
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
    if ingress == Ingress::OpenAi {
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

    // Normalize the ingress request to the canonical chat shape. Skipped
    // on the native path, which keeps the original body intact.
    let chat_req = if native {
        Value::Null
    } else {
        match ingress {
            Ingress::OpenAiCompatible => req.clone(),
            Ingress::OpenAi => responses_to_chat_request(&req),
            Ingress::Anthropic => anthropic_to_chat_request(&req),
            Ingress::Google => google_to_chat_request(&req, &name),
        }
    };

    serve_normalized(st, ingress, name, m, req, chat_req, stream, native).await
}

/// Shared pipeline tail: param audit, upstream call, translate back into the
/// ingress protocol. `req` is the client's original body, `chat_req` its
/// canonical chat form (Null on the native path), `native` forwards `req`.
pub(crate) async fn serve_normalized(
    st: St,
    ingress: Ingress,
    name: String,
    m: ModelCfg,
    req: Value,
    chat_req: Value,
    stream: bool,
    native: bool,
) -> Response {
    let strict = st.cfg.read().unwrap().strict_params;

    if !native {
        let (lost, fatal) = match m.protocol {
            Protocol::Anthropic => (
                chat_to_anthropic_loss(&chat_req),
                chat_to_anthropic_fatal(&chat_req),
            ),
            Protocol::OpenAi => (
                chat_to_responses_loss(&chat_req),
                chat_to_responses_fatal(&chat_req),
            ),
            Protocol::Google => (
                chat_to_google_loss(&chat_req),
                chat_to_google_fatal(&chat_req),
            ),
            Protocol::OpenAiCompatible => (vec![], None),
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
            Protocol::OpenAiCompatible => Inbound::Chat {
                name: name.clone(),
                held: None,
                usage: None,
            },
            Protocol::Anthropic => Inbound::Anthropic(AnthropicStream::new(&name)),
            Protocol::OpenAi => Inbound::Responses(ResponsesStream::new(&name)),
            Protocol::Google => Inbound::Google(GoogleStream::new(&name)),
        };
        let egress = match ingress {
            Ingress::OpenAiCompatible => {
                Egress::Chat(ChatReasoningEgress::new(st.cfg.read().unwrap().reasoning_output))
            }
            Ingress::OpenAi => Egress::Responses(ChatToResponsesStream::new(&name)),
            Ingress::Anthropic => Egress::Anthropic(ChatToAnthropicStream::new(&name)),
            Ingress::Google => Egress::Google(ChatToGoogleStream::new(&name)),
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
            // speaking any of the four protocols, so return a protocol-shaped
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
            // Google carries it in `modelVersion`, not `model`.
            let mut v = up;
            if let Some(o) = v.as_object_mut() {
                if m.protocol == Protocol::Google {
                    o.insert("modelVersion".into(), json!(name.clone()));
                } else {
                    o.insert("model".into(), json!(name.clone()));
                }
            }
            return json_response(StatusCode::OK, v);
        }
        let mut chat = match m.protocol {
            Protocol::OpenAiCompatible => {
                let mut v = up;
                if let Some(o) = v.as_object_mut() {
                    o.insert("model".into(), json!(name.clone()));
                }
                v
            }
            Protocol::Anthropic => anthropic_to_openai(&up, &name),
            Protocol::OpenAi => responses_to_openai(&up, &name),
            Protocol::Google => google_to_chat(&up, &name),
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
            Ingress::OpenAiCompatible => {
                let mode = st.cfg.read().unwrap().reasoning_output;
                if let Some(msg) = chat.pointer_mut("/choices/0/message") {
                    apply_reasoning_output_to_message(msg, mode);
                }
                chat
            }
            Ingress::OpenAi => chat_to_responses_response(&chat, &name),
            Ingress::Anthropic => chat_to_anthropic_response(&chat, &name),
            Ingress::Google => chat_to_google_response(&chat, &name),
        };
        json_response(StatusCode::OK, out)
    }
}

// ---------------------------------------------------------------------------
// SSE pipeline: upstream -> canonical chat chunks -> client framing
// ---------------------------------------------------------------------------
