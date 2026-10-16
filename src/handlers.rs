//! HTTP handlers: one per client protocol surface.

use crate::errors::{Ingress, ingress_error, json_response};
use crate::service::{body_limit, body_too_large, declared_len, handle_ingress, serve_normalized};
use crate::state::{St, find_model};
use axum::http::HeaderValue;
use axum::http::header;
use axum::http::StatusCode;
use axum::extract::{Request, State};
use axum::response::Response;
use serde_json::{json, Value};
use crate::config::Protocol;
use crate::translate::google::{google_to_chat_request, parse_google_action};

pub(crate) async fn healthz() -> Response {
    json_response(StatusCode::OK, json!({"status": "ok"}))
}

pub(crate) async fn list_models(State(st): State<St>) -> Response {
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
            let merged = crate::models_dev::merge_model(m, &store);
            let meta = crate::models_dev::meta_json(m, &merged);
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

pub(crate) async fn chat(State(st): State<St>, req: Request) -> Response {
    handle_ingress(st, req, Ingress::OpenAiCompatible).await
}

pub(crate) async fn responses_ingress(State(st): State<St>, req: Request) -> Response {
    handle_ingress(st, req, Ingress::OpenAi).await
}

pub(crate) async fn messages_ingress(State(st): State<St>, req: Request) -> Response {
    let mut resp = handle_ingress(st, req, Ingress::Anthropic).await;
    resp.headers_mut().insert(
        header::HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static("2023-06-01"),
    );
    resp
}

/// Gemini ingress: `POST /v1beta/models/{model}:generateContent` and
/// `{model}:streamGenerateContent`. The model id lives in the path, not the
/// body, and the verb selects streaming.
pub(crate) async fn google_ingress(
    State(st): State<St>,
    axum::extract::Path(action): axum::extract::Path<String>,
    req: Request,
) -> Response {
    let (name, stream) = match parse_google_action(&action) {
        Some(v) => v,
        None => {
            return ingress_error(
                Ingress::Google,
                StatusCode::NOT_FOUND,
                &format!("unknown Google action: {action}"),
            )
        }
    };
    let limit = body_limit(&st.cfg.read().unwrap());
    if let Some(len) = declared_len(&req) {
        if len > limit as u64 {
            return body_too_large(Ingress::Google, limit);
        }
    }
    let body = match axum::body::to_bytes(req.into_body(), limit).await {
        Ok(b) => b,
        Err(_) => return body_too_large(Ingress::Google, limit),
    };
    let req_json: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return ingress_error(
                Ingress::Google,
                StatusCode::BAD_REQUEST,
                &format!("invalid JSON body: {e}"),
            )
        }
    };
    let m = match find_model(&st, &name) {
        Some(m) => m,
        None => {
            eprintln!("[gw] req unknown-model={name}");
            return ingress_error(
                Ingress::Google,
                StatusCode::NOT_FOUND,
                &format!("unknown model: {name}"),
            );
        }
    };
    let native = m.protocol == Protocol::Google;
    let chat_req = if native {
        Value::Null
    } else {
        google_to_chat_request(&req_json, &name)
    };
    serve_normalized(st, Ingress::Google, name, m, req_json, chat_req, stream, native).await
}
