//! End-to-end tests: a real mock upstream behind a real gateway router.
//!
//! The unit tests cover the translators and the SSE parser in isolation.
//! They cannot cover what actually broke twice in this codebase: whether a
//! real client can consume the bytes the gateway produces. Both blocking
//! bugs fixed so far — dropped `event:` names and a duplicated terminal
//! event — pass every unit test and still leave a client with an empty or
//! malformed stream.
//!
//! So these tests drive `app()` end to end: a client request goes in over a
//! real socket to a real router, which calls a real mock upstream over
//! another socket, and the response is parsed the way an SDK would parse it.
//!
//! Sections, in file order:
//!
//! - protocol matrix: every client surface x every upstream protocol,
//!   non-streaming and streaming;
//! - same-protocol passthrough: `event:` names kept, one terminal event;
//! - error paths: upstream 429 / 500 / non-JSON / mid-stream error /
//!   truncated stream;
//! - sampling params under thinking (M-6): forwarded, not stripped;
//! - gateway rejections: unknown model, malformed JSON;
//! - A. parameter passthrough matrix (`Capture` mock asserts the upstream
//!   body pointer by pointer);
//! - B. vision input across ingresses and upstreams;
//! - C. multi-round tool calls + Google regressions (issues 4/5/6).

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use axum::routing::post as route_post;
use axum::Router;
use serde_json::{json, Value};

use crate::config::{ModelCfg, Protocol};
use crate::{app, sse::parse_sse, state::{test_state, St}};

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

type Bytes = axum::body::Bytes;

const ANSWER: &str = "hello there world";

/// The three client-facing surfaces (Google's path varies by verb; see below).
const INGRESS: [&str; 3] = ["/v1/chat/completions", "/v1/responses", "/v1/messages"];

const GOOGLE_NONSTREAM: &str = "/v1beta/models/m:generateContent";
const GOOGLE_STREAM: &str = "/v1beta/models/m:streamGenerateContent";

/// All four client surfaces for the given streaming mode.
fn ingress_paths(stream: bool) -> [&'static str; 4] {
    if stream {
        [
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
            GOOGLE_STREAM,
        ]
    } else {
        [
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
            GOOGLE_NONSTREAM,
        ]
    }
}

fn is_google_path(path: &str) -> bool {
    path.starts_with("/v1beta/")
}

// ---------------------------------------------------------------------------
// Mock upstream
// ---------------------------------------------------------------------------

/// What the mock upstream answers with.
#[derive(Clone)]
enum Upstream {
    /// Speak this protocol's wire format: SSE when asked to stream,
    /// otherwise a plain response.
    Speak(Protocol),
    /// Return this status and body for every request.
    Error(StatusCode, String),
    /// Return a 200 with a non-JSON body.
    NonJson(String),
    /// Send a stream that stops early: a delta or two, then EOF with no
    /// terminal event. This is what a well-behaved server looks like when it
    /// is killed mid-generation.
    Truncated(Protocol),
    /// Send one delta, then an explicit error event, then EOF.
    MidStreamError(Protocol),
    /// Speak this protocol, but 400 unless every named field is present in
    /// the request body. This is how a test asserts what the gateway *sent*
    /// upstream — the response alone cannot show a parameter the gateway
    /// silently stripped.
    Requires(Protocol, Vec<&'static str>),
    /// Record the full upstream body for pointer-level assertions, then
    /// answer 200 in `proto`'s shape. Unlike `Requires` (top-level presence
    /// only), tests inspect `seen` afterwards for nested fields such as
    /// `generationConfig.thinkingConfig.thinkingBudget`.
    Capture {
        proto: Protocol,
        seen: Arc<Mutex<Value>>,
    },
    /// Mimic Gemini's strict rejections with empty-body 400s:
    /// - a `functionDeclaration` without a name,
    /// - `additionalProperties` at any depth,
    /// - a follow-up turn carrying `functionResponse` whose sibling
    ///   `functionCall` lost its `thoughtSignature`.
    /// Streaming failures use an empty body (what Gemini actually sends);
    /// non-streaming carries a JSON error so the gateway has a message.
    StrictGoogle,
    /// Return queued raw upstream bodies in order, recording each inbound
    /// request. Powers multi-round tool tests: first pop is the tool_call,
    /// second pop the final answer. Empty queue falls back to `ANSWER`.
    Scripted {
        proto: Protocol,
        queue: Arc<Mutex<VecDeque<Value>>>,
        seen: Arc<Mutex<Vec<Value>>>,
    },
}

/// A mock upstream listening on an ephemeral port.
struct Mock {
    base: String,
}

impl Mock {
    async fn start(kind: Upstream) -> Self {
        // The three routes share one handler, so it is a function rather
        // than a closure: a closure cannot be cloned into three routes.
        async fn handle(State(kind): State<Upstream>, body: Bytes) -> Response {
            let req: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            respond(&kind, &req, None)
        }
        // Google selects streaming by endpoint, not by a `stream` field.
        async fn handle_google(
            State(kind): State<Upstream>,
            axum::extract::Path(action): axum::extract::Path<String>,
            body: Bytes,
        ) -> Response {
            let req: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let stream = action.ends_with(":streamGenerateContent");
            respond(&kind, &req, Some(stream))
        }
        let router = Router::new()
            .route("/v1/chat/completions", route_post(handle))
            .route("/v1/messages", route_post(handle))
            .route("/v1/responses", route_post(handle))
            .route("/v1/models/{action}", route_post(handle_google))
            .with_state(kind);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        Self {
            base: format!("http://{addr}/v1"),
        }
    }

    /// A gateway state whose only model speaks `protocol` on this upstream.
    fn state_with(&self, protocol: Protocol) -> St {
        let m: ModelCfg = crate::config::test_model("m", protocol, &self.base);
        test_state(crate::state::config_for(&self.base, vec![m]))
    }
}

fn respond(kind: &Upstream, req: &Value, stream_override: Option<bool>) -> Response {
    let wants_stream = stream_override
        .unwrap_or_else(|| req.get("stream").and_then(|v| v.as_bool()) == Some(true));
    match kind {
        Upstream::Error(status, body) => Response::builder()
            .status(*status)
            .header("content-type", "application/json")
            .body(Body::from(body.clone()))
            .unwrap(),
        Upstream::NonJson(body) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/html")
            .body(Body::from(body.clone()))
            .unwrap(),
        Upstream::Speak(p) if wants_stream => sse(&stream_frames(*p, ANSWER)),
        Upstream::Speak(p) => json(StatusCode::OK, &completion(*p, ANSWER)),
        Upstream::Requires(p, fields) => {
            let missing: Vec<&str> = fields
                .iter()
                .copied()
                .filter(|k| req.get(*k).is_none_or(|v| v.is_null()))
                .collect();
            if !missing.is_empty() {
                return json(
                    StatusCode::BAD_REQUEST,
                    &json!({"error": {
                        "type": "invalid_request_error",
                        "message": format!("missing required field(s): {}", missing.join(", "))
                    }}),
                );
            }
            if wants_stream {
                sse(&stream_frames(*p, ANSWER))
            } else {
                json(StatusCode::OK, &completion(*p, ANSWER))
            }
        }
        Upstream::Capture { proto, seen } => {
            *seen.lock().unwrap() = req.clone();
            if wants_stream {
                sse(&stream_frames(*proto, ANSWER))
            } else {
                json(StatusCode::OK, &completion(*proto, ANSWER))
            }
        }
        Upstream::StrictGoogle => {
            if let Some(msg) = strict_google_rejection(req, wants_stream) {
                return msg;
            }
            if wants_stream {
                sse(&stream_frames(Protocol::Google, ANSWER))
            } else {
                json(StatusCode::OK, &completion(Protocol::Google, ANSWER))
            }
        }
        Upstream::Scripted { proto, queue, seen } => {
            seen.lock().unwrap().push(req.clone());
            let next = queue.lock().unwrap().pop_front();
            match next {
                Some(v) => json(StatusCode::OK, &v),
                None => {
                    if wants_stream {
                        sse(&stream_frames(*proto, ANSWER))
                    } else {
                        json(StatusCode::OK, &completion(*proto, ANSWER))
                    }
                }
            }
        }
        Upstream::Truncated(p) => {
            // Drop every terminal frame and end the body. The client sees a
            // stream that simply stops — no finish chunk, no protocol
            // terminator.
            let mut frames = stream_frames(*p, ANSWER);
            frames.retain(|f| !is_terminal_frame(f));
            if frames.len() > 1 {
                frames.truncate(1);
            }
            sse(&frames)
        }
        Upstream::MidStreamError(p) => {
            let mut frames = stream_frames(*p, ANSWER);
            frames.retain(|f| !is_terminal_frame(f));
            if frames.len() > 1 {
                frames.truncate(1);
            }
            frames.push(error_frame(*p, "upstream exploded"));
            sse(&frames)
        }
    }
}

fn sse(frames: &[String]) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .body(Body::from(frames.concat()))
        .unwrap()
}

fn json(status: StatusCode, v: &Value) -> Response {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(v).unwrap()))
        .unwrap()
}

/// One SSE frame with an `event:` name.
fn named(event: &str, data: Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

/// One SSE frame without an `event:` name.
fn plain(data: Value) -> String {
    format!("data: {data}\n\n")
}

/// The SSE frames an upstream speaking `protocol` emits for `text`.
fn stream_frames(protocol: Protocol, text: &str) -> Vec<String> {
    let pieces: Vec<&str> = text.split_inclusive(' ').collect();
    match protocol {
        Protocol::OpenAiCompatible => {
            let mut out: Vec<String> = pieces
                .iter()
                .map(|p| {
                    format!(
                        "data: {}\n\n",
                        crate::sse::chat_chunk(json!({"content": p}), None)
                    )
                })
                .collect();
            out.push(format!(
                "data: {}\n\n",
                crate::sse::chat_chunk(json!({}), Some("stop"))
            ));
            out.push("data: [DONE]\n\n".to_string());
            out
        }
        Protocol::Anthropic => {
            let mut out = vec![
                named(
                    "message_start",
                    json!({"type": "message_start", "message": {
                        "id": "msg_1", "type": "message", "role": "assistant",
                        "model": "mock", "content": [],
                        "usage": {"input_tokens": 3, "output_tokens": 0}}}),
                ),
                named(
                    "content_block_start",
                    json!({"type": "content_block_start", "index": 0,
                           "content_block": {"type": "text", "text": ""}}),
                ),
            ];
            for p in &pieces {
                out.push(named(
                    "content_block_delta",
                    json!({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta", "text": p}}),
                ));
            }
            out.push(named(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0}),
            ));
            out.push(named(
                "message_delta",
                json!({"type": "message_delta",
                       "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                       "usage": {"output_tokens": 2}}),
            ));
            out.push(named("message_stop", json!({"type": "message_stop"})));
            out
        }
        Protocol::OpenAi => {
            let item_done = json!({"id": "msg_1", "type": "message", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]});
            let skeleton = |status: &str| {
                json!({"id": "resp_1", "object": "response", "created_at": 1,
                       "model": "mock", "status": status, "output": [],
                       "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}})
            };
            let mut out = vec![
                plain(json!({"type": "response.created",
                    "response": skeleton("in_progress")})),
                plain(json!({"type": "response.in_progress",
                    "response": skeleton("in_progress")})),
                plain(
                    json!({"type": "response.output_item.added", "output_index": 0,
                    "item": {"id": "msg_1", "type": "message", "role": "assistant",
                             "status": "in_progress", "content": []}}),
                ),
                plain(
                    json!({"type": "response.content_part.added", "item_id": "msg_1",
                    "output_index": 0, "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []}}),
                ),
            ];
            for p in &pieces {
                out.push(plain(json!({"type": "response.output_text.delta",
                    "item_id": "msg_1", "output_index": 0, "content_index": 0,
                    "delta": p})));
            }
            out.push(plain(json!({"type": "response.output_text.done",
                "item_id": "msg_1", "output_index": 0, "content_index": 0,
                "text": text})));
            out.push(plain(json!({"type": "response.content_part.done",
                "item_id": "msg_1", "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []}})));
            out.push(plain(json!({"type": "response.output_item.done",
                "output_index": 0, "item": item_done})));
            let mut done = skeleton("completed");
            done["output"] = json!([item_done]);
            out.push(plain(
                json!({"type": "response.completed", "response": done}),
            ));
            out
        }
        Protocol::Google => {
            let mut out: Vec<String> = pieces
                .iter()
                .map(|p| {
                    plain(json!({"candidates": [{
                        "content": {"role": "model", "parts": [{"text": p}]},
                        "index": 0
                    }]}))
                })
                .collect();
            out.push(plain(json!({"candidates": [{
                "content": {"role": "model", "parts": []},
                "finishReason": "STOP", "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 3, "candidatesTokenCount": 2,
                "totalTokenCount": 5}})));
            out
        }
    }
}

/// Is this the frame that ends a stream in its own protocol?
fn is_terminal_frame(frame: &str) -> bool {
    frame.contains("message_stop")
        || frame.contains("response.completed")
        || frame.contains("[DONE]")
        || frame.contains("\"finish_reason\":\"stop\"")
        || frame.contains("\"finishReason\":\"STOP\"")
        || frame.contains("message_delta")
}

/// The mid-stream error frame each protocol uses to report a failure.
fn error_frame(protocol: Protocol, msg: &str) -> String {
    match protocol {
        Protocol::OpenAiCompatible => plain(json!({"error": {"message": msg, "type": "api_error"}})),
        Protocol::Anthropic => named(
            "error",
            json!({"type": "error", "error": {"type": "api_error", "message": msg}}),
        ),
        Protocol::OpenAi => plain(json!({"type": "error", "message": msg})),
        Protocol::Google => plain(json!({"error": {"message": msg, "code": 13}})),
    }
}

/// The non-streaming response an upstream speaking `protocol` returns.
fn completion(protocol: Protocol, text: &str) -> Value {
    match protocol {
        Protocol::OpenAiCompatible => crate::sse::chat_completion(text),
        Protocol::Anthropic => json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "mock",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 2}
        }),
        Protocol::OpenAi => json!({
            "id": "resp_1", "object": "response", "created_at": 1, "model": "mock",
            "status": "completed",
            "output": [{"id": "msg_1", "type": "message", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        }),
        Protocol::Google => json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": "STOP", "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 3, "candidatesTokenCount": 2,
                "totalTokenCount": 5}
        }),
    }
}

// ---------------------------------------------------------------------------
// StrictGoogle + Scripted helpers (A/B/C harness)
// ---------------------------------------------------------------------------

/// Recursively true when any object key is `additionalProperties`.
fn contains_additional_properties(v: &Value) -> bool {
    match v {
        Value::Object(m) => {
            m.contains_key("additionalProperties")
                || m.values().any(contains_additional_properties)
        }
        Value::Array(a) => a.iter().any(contains_additional_properties),
        _ => false,
    }
}

/// Collect every `functionDeclaration` name; `None` marks a missing/empty one.
fn google_decl_names(req: &Value) -> Vec<Option<String>> {
    let mut out = vec![];
    if let Some(tools) = req.get("tools").and_then(|t| t.as_array()) {
        for t in tools {
            if let Some(decls) = t
                .get("functionDeclarations")
                .and_then(|d| d.as_array())
            {
                for d in decls {
                    out.push(
                        d.get("name")
                            .and_then(|n| n.as_str())
                            .map(str::to_string),
                    );
                }
            }
        }
    }
    out
}

fn google_parts(req: &Value) -> Vec<Value> {
    let mut parts = vec![];
    if let Some(contents) = req.get("contents").and_then(|c| c.as_array()) {
        for c in contents {
            if let Some(ps) = c.get("parts").and_then(|p| p.as_array()) {
                parts.extend(ps.iter().cloned());
            }
        }
    }
    parts
}

/// Gemini-strict rejection, or `None` when the request would get a 200.
fn strict_google_rejection(req: &Value, wants_stream: bool) -> Option<Response> {
    // Empty-name declarations are rejected with an empty-body 400.
    let names = google_decl_names(req);
    if names.iter().any(|n| n.as_deref().is_none_or(str::is_empty)) {
        return Some(empty_400());
    }
    // `additionalProperties` anywhere is rejected the same way.
    if contains_additional_properties(req) {
        return Some(empty_400());
    }
    // A follow-up turn (it carries a `functionResponse`) must replay the
    // `thoughtSignature` next to every `functionCall`. A missing sibling
    // is a 400; streaming carries an empty body, like the real endpoint.
    let parts = google_parts(req);
    let has_response = parts.iter().any(|p| p.get("functionResponse").is_some());
    if has_response {
        let missing_sig = parts.iter().any(|p| {
            p.get("functionCall").is_some()
                && p.get("thoughtSignature").and_then(|s| s.as_str()).is_none_or(str::is_empty)
        });
        if missing_sig {
            if wants_stream {
                return Some(empty_400());
            }
            return Some(json(
                StatusCode::BAD_REQUEST,
                &json!({"error": {"message": "Function call is missing a thought_signature", "code": 400}}),
            ));
        }
    }
    None
}

fn empty_400() -> Response {
    Response::builder()
        .status(StatusCode::BAD_REQUEST)
        .header("content-type", "application/json")
        .body(Body::from(vec![]))
        .unwrap()
}

/// Upstream tool-call responses, one per wire protocol.
fn tool_completion(protocol: Protocol, name: &str, args: &Value) -> Value {
    let args_str = serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string());
    match protocol {
        Protocol::OpenAiCompatible => json!({
            "id": "chatcmpl-tool", "object": "chat.completion", "created": 1,
            "model": "mock",
            "choices": [{"index": 0,
                "message": {"role": "assistant", "content": Value::Null,
                    "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": name, "arguments": args_str}}]},
                "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        }),
        Protocol::Anthropic => json!({
            "id": "msg_tool", "type": "message", "role": "assistant", "model": "mock",
            "content": [{"type": "tool_use", "id": "tu_1", "name": name, "input": args}],
            "stop_reason": "tool_use", "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 2}
        }),
        Protocol::OpenAi => json!({
            "id": "resp_tool", "object": "response", "created_at": 1, "model": "mock",
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": name, "arguments": args_str, "status": "completed"}
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        }),
        Protocol::Google => json!({
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"functionCall": {"name": name, "args": args}}
                ]},
                "finishReason": "STOP", "index": 0
            }],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}
        }),
    }
}

/// Google tool-call response carrying a `thoughtSignature` sibling.
fn google_tool_with_signature(name: &str, args: &Value, sig: &str) -> Value {
    json!({
        "candidates": [{
            "content": {"role": "model", "parts": [
                {"functionCall": {"name": name, "args": args},
                 "thoughtSignature": sig}
            ]},
            "finishReason": "STOP", "index": 0
        }],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}
    })
}

fn new_capture(proto: Protocol) -> (Upstream, Arc<Mutex<Value>>) {
    let seen = Arc::new(Mutex::new(Value::Null));
    (Upstream::Capture { proto, seen: Arc::clone(&seen) }, seen)
}

fn new_scripted(proto: Protocol, steps: Vec<Value>) -> (Upstream, Arc<Mutex<Vec<Value>>>) {
    let queue = Arc::new(Mutex::new(VecDeque::from(steps)));
    let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(vec![]));
    (
        Upstream::Scripted { proto, queue: Arc::clone(&queue), seen: Arc::clone(&seen) },
        seen,
    )
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// A client request body for `path`.
fn client_body(path: &str, stream: bool) -> Value {
    if is_google_path(path) {
        return json!({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
        });
    }
    match path {
        "/v1/chat/completions" => json!({
            "model": "m", "stream": stream,
            "messages": [{"role": "user", "content": "hi"}]
        }),
        "/v1/responses" => json!({
            "model": "m", "stream": stream,
            "input": [{"role": "user", "content": "hi"}]
        }),
        _ => json!({
            "model": "m", "stream": stream, "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]
        }),
    }
}

/// POST `body` to `path` through the real router.
async fn post(router: Router, path: &str, body: &Value) -> Response {
    send(router, path, serde_json::to_vec(body).unwrap()).await
}

/// POST raw bytes to `path` through the real router.
async fn send(router: Router, path: &str, raw: Vec<u8>) -> Response {
    use tower::ServiceExt;
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(raw))
        .unwrap();
    router.oneshot(req).await.expect("router responded")
}

/// Drain a response body to a string.
async fn text(resp: Response) -> String {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The answer a client of `path` reads out of a non-streaming JSON body.
fn client_text(path: &str, body: &str) -> String {
    let v = parse_json(path, body);
    if is_google_path(path) {
        let parts = v["candidates"][0]["content"]["parts"]
            .as_array()
            .unwrap_or_else(|| panic!("{path} answer has no parts: {body}"));
        let mut out = String::new();
        for p in parts {
            if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                out.push_str(t);
            }
        }
        return out;
    }
    let got = match path {
        "/v1/chat/completions" => v["choices"][0]["message"]["content"].clone(),
        "/v1/responses" => v["output"][0]["content"][0]["text"].clone(),
        _ => v["content"][0]["text"].clone(),
    };
    got.as_str()
        .unwrap_or_else(|| panic!("{path} answer is not text: {body}"))
        .to_string()
}

fn parse_json(path: &str, body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{path} returned non-JSON ({e}): {body}"))
}

/// The streamed text a client of `path` reassembles from `events`.
fn streamed_text(path: &str, events: &[(Option<String>, String)]) -> String {
    if is_google_path(path) {
        return events
            .iter()
            .filter_map(|(_, d)| {
                let v: Value = serde_json::from_str(d).ok()?;
                v.pointer("/candidates/0/content/parts")
                    .and_then(|p| p.as_array())
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|q| q.get("text").and_then(|x| x.as_str()))
                            .collect::<String>()
                    })
            })
            .collect();
    }
    events
        .iter()
        .filter_map(|(_, d)| {
            let v: Value = serde_json::from_str(d).ok()?;
            match path {
                "/v1/chat/completions" => v["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(str::to_string),
                "/v1/responses" => (v["type"] == "response.output_text.delta")
                    .then(|| v["delta"].as_str().map(str::to_string))
                    .flatten(),
                _ => (v["type"] == "content_block_delta")
                    .then(|| v["delta"]["text"].as_str().map(str::to_string))
                    .flatten(),
            }
        })
        .collect()
}

/// The terminal event a client of `path` waits for, in the shape its SDK
/// recognises.
fn terminal_kind(path: &str, events: &[(Option<String>, String)]) -> &'static str {
    let last = events.last().expect("at least one event");
    if is_google_path(path) {
        assert_eq!(last.0, None, "google events carry no `event:` name");
        let fr = serde_json::from_str::<Value>(&last.1).unwrap()
            .pointer("/candidates/0/finishReason")
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string();
        assert_eq!(fr, "STOP", "google terminal finishReason");
        return "STOP";
    }
    match path {
        "/v1/chat/completions" => {
            assert_eq!(last.1, "[DONE]", "chat must close with [DONE]");
            "[DONE]"
        }
        "/v1/responses" => {
            assert_eq!(last.0, None, "responses events carry no `event:` name");
            let t = serde_json::from_str::<Value>(&last.1).unwrap()["type"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            assert_eq!(t, "response.completed", "responses terminal event");
            "response.completed"
        }
        _ => {
            assert_eq!(
                last.0.as_deref(),
                Some("message_stop"),
                "anthropic's last event must be named message_stop"
            );
            "message_stop"
        }
    }
}

fn count_containing(events: &[(Option<String>, String)], needle: &str) -> usize {
    events.iter().filter(|(_, d)| d.contains(needle)).count()
}

// ---------------------------------------------------------------------------
// The 16 protocol combinations
// ---------------------------------------------------------------------------

/// Every client surface reaches every upstream protocol and gets a usable
/// answer back in its own format.
///
/// This is the matrix the audit claimed to have covered but never
/// committed. A break in any one cell is a whole protocol pair unusable,
/// and no unit test covers the seam between them.
#[tokio::test]
async fn all_sixteen_protocol_combinations_answer() {
    for upstream in [
        Protocol::OpenAiCompatible,
        Protocol::Anthropic,
        Protocol::OpenAi,
        Protocol::Google,
    ] {
        for path in ingress_paths(false) {
            let mock = Mock::start(Upstream::Speak(upstream)).await;
            let resp = post(
                app(mock.state_with(upstream)),
                path,
                &client_body(path, false),
            )
            .await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "{path} from a {upstream:?} upstream failed"
            );
            let body = text(resp).await;
            assert_eq!(
                client_text(path, &body),
                ANSWER,
                "{path} from a {upstream:?} upstream lost the answer"
            );
        }
    }
}

/// The same matrix, streaming: the client gets its own framing, the whole
/// answer, and exactly one terminal event.
#[tokio::test]
async fn all_sixteen_protocol_combinations_stream() {
    for upstream in [
        Protocol::OpenAiCompatible,
        Protocol::Anthropic,
        Protocol::OpenAi,
        Protocol::Google,
    ] {
        for path in ingress_paths(true) {
            let mock = Mock::start(Upstream::Speak(upstream)).await;
            let resp = post(
                app(mock.state_with(upstream)),
                path,
                &client_body(path, true),
            )
            .await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "{path} from a {upstream:?} upstream failed to stream"
            );
            let body = text(resp).await;
            let events = parse_sse(&body);
            assert!(
                !events.is_empty(),
                "{path} from {upstream:?} produced no events: {body}"
            );
            assert_eq!(
                streamed_text(path, &events),
                ANSWER,
                "{path} from {upstream:?} lost stream content"
            );
            terminal_kind(path, &events);

            // The opening event appears exactly once. More than one means the
            // gateway restarted a stream the upstream had already begun,
            // which a stateful client treats as a different response.
            // Chat has no opening event at all.
            let openings = match path {
                "/v1/chat/completions" => None,
                "/v1/responses" => Some(count_containing(&events, "\"response.created\"")),
                p if is_google_path(p) => None,
                _ => Some(count_containing(&events, "\"message_start\"")),
            };
            if let Some(openings) = openings {
                assert_eq!(
                    openings, 1,
                    "{path} from {upstream:?} emitted {openings} opening events: {body}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Same-protocol passthrough
// ---------------------------------------------------------------------------

/// Passthrough keeps the upstream's `event:` names and appends no terminal
/// event of its own.
///
/// This is the regression test for the two blocking bugs. An Anthropic
/// client's SDK dispatches on `sse.event`, so a name lost in translation
/// yields a zero-event stream. And the gateway used to synthesize its own
/// `message_start` / `message_delta` / `message_stop` on top of an upstream
/// stream that had already terminated, producing a second `message_start`
/// that a strict client rejects.
#[tokio::test]
async fn anthropic_passthrough_keeps_event_names_and_one_terminal() {
    let mock = Mock::start(Upstream::Speak(Protocol::Anthropic)).await;
    let path = "/v1/messages";
    let resp = post(
        app(mock.state_with(Protocol::Anthropic)),
        path,
        &client_body(path, true),
    )
    .await;
    let body = text(resp).await;
    let events = parse_sse(&body);

    // Every event is named. The upstream's names are the client's only way
    // to tell them apart.
    for (name, data) in &events {
        assert!(
            name.is_some(),
            "passthrough dropped an `event:` name: {data}"
        );
    }
    for want in ["message_start", "message_stop"] {
        assert_eq!(
            count_containing(&events, &format!("\"{want}\"")),
            1,
            "{want} appeared more than once in the passthrough stream: {body}"
        );
    }
    assert_eq!(terminal_kind(path, &events), "message_stop");
    assert_eq!(streamed_text(path, &events), ANSWER);
}

/// Responses passthrough forwards the upstream's own terminal event and does
/// not add a second `response.completed`.
#[tokio::test]
async fn responses_passthrough_terminates_exactly_once() {
    let mock = Mock::start(Upstream::Speak(Protocol::OpenAi)).await;
    let path = "/v1/responses";
    let resp = post(
        app(mock.state_with(Protocol::OpenAi)),
        path,
        &client_body(path, true),
    )
    .await;
    let body = text(resp).await;
    let events = parse_sse(&body);
    assert_eq!(
        count_containing(&events, "\"response.completed\""),
        1,
        "response.completed sent more than once: {body}"
    );
    assert_eq!(
        count_containing(&events, "\"response.created\""),
        1,
        "response.created sent more than once: {body}"
    );
    assert_eq!(terminal_kind(path, &events), "response.completed");
    assert_eq!(streamed_text(path, &events), ANSWER);
}

/// Chat passthrough forwards one `[DONE]`, even though the upstream's own
/// sentinel is consumed by the parser and has to be re-emitted.
#[tokio::test]
async fn chat_passthrough_sends_done_exactly_once() {
    let mock = Mock::start(Upstream::Speak(Protocol::OpenAiCompatible)).await;
    let path = "/v1/chat/completions";
    let resp = post(
        app(mock.state_with(Protocol::OpenAiCompatible)),
        path,
        &client_body(path, true),
    )
    .await;
    let body = text(resp).await;
    let events = parse_sse(&body);
    assert_eq!(
        events.iter().filter(|(_, d)| d == "[DONE]").count(),
        1,
        "[DONE] sent more than once: {body}"
    );
    assert_eq!(terminal_kind(path, &events), "[DONE]");
    assert_eq!(streamed_text(path, &events), ANSWER);
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

/// An upstream failure reaches every client as a readable error in that
/// client's protocol, with the upstream's status preserved.
///
/// A 429 that arrives as an Anthropic-shaped body would be invisible to an
/// OpenAI SDK's rate-limit accounting, and its retry logic would not fire.
#[tokio::test]
async fn upstream_429_is_shaped_for_every_ingress() {
    let kind = Upstream::Error(
        StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":{"message":"rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#
            .into(),
    );
    for upstream in [
        Protocol::OpenAiCompatible,
        Protocol::Anthropic,
        Protocol::OpenAi,
        Protocol::Google,
    ] {
        for path in ingress_paths(false) {
            let mock = Mock::start(kind.clone()).await;
            let resp = post(
                app(mock.state_with(upstream)),
                path,
                &client_body(path, false),
            )
            .await;
            assert_eq!(
                resp.status(),
                StatusCode::TOO_MANY_REQUESTS,
                "{path} from {upstream:?} lost the status"
            );
            let body = text(resp).await;
            let v = parse_json(path, &body);
            assert_eq!(
                v.pointer("/error/message").and_then(|m| m.as_str()),
                Some("rate limited"),
                "{path} from {upstream:?} could not read the message"
            );
        }
    }
}

/// A 500 is an error, never a 200 with an empty answer.
#[tokio::test]
async fn upstream_500_is_not_dressed_up_as_success() {
    let kind = Upstream::Error(
        StatusCode::INTERNAL_SERVER_ERROR,
        r#"{"error":{"message":"boom"}}"#.into(),
    );
    for path in ingress_paths(false) {
        let mock = Mock::start(kind.clone()).await;
        let resp = post(
            app(mock.state_with(Protocol::OpenAiCompatible)),
            path,
            &client_body(path, false),
        )
        .await;
        assert_eq!(
            resp.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "{path} turned a 500 into a success"
        );
    }
}

/// A 200 with a non-JSON body is an upstream fault: a 502 in the client's
/// protocol, never an HTML page handed to an SDK.
#[tokio::test]
async fn non_json_upstream_body_becomes_a_502() {
    for path in ingress_paths(false) {
        let mock = Mock::start(Upstream::NonJson("<html>502 Bad Gateway</html>".into())).await;
        let resp = post(
            app(mock.state_with(Protocol::OpenAiCompatible)),
            path,
            &client_body(path, false),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY, "{path}");
        let v = parse_json(path, &text(resp).await);
        assert!(
            v.pointer("/error/message").is_some(),
            "{path} error body had no message"
        );
    }
}

/// An upstream error event mid-stream must end the client stream with a
/// failure, not with a success terminator.
///
/// This is the difference between a client retrying and a client storing a
/// truncated answer as if it were complete.
#[tokio::test]
async fn a_mid_stream_error_is_reported_as_an_error() {
    for upstream in [
        Protocol::OpenAiCompatible,
        Protocol::Anthropic,
        Protocol::OpenAi,
        Protocol::Google,
    ] {
        for path in ingress_paths(true) {
            let mock = Mock::start(Upstream::MidStreamError(upstream)).await;
            let resp = post(
                app(mock.state_with(upstream)),
                path,
                &client_body(path, true),
            )
            .await;
            let body = text(resp).await;
            let events = parse_sse(&body);

            let claims_success = events.iter().any(|(_, d)| {
                d.contains("\"finish_reason\":\"stop\"")
                    || d.contains("\"finishReason\":\"STOP\"")
                    || d.contains("response.completed")
                    || d.contains("\"message_stop\"")
            });
            assert!(
                !claims_success,
                "{path} from {upstream:?} reported a failed stream as complete: {body}"
            );

            let says_why = events.iter().any(|(_, d)| {
                d.contains("\"type\":\"error\"")
                    || d.contains("gateway_error")
                    || d.contains("api_error")
            });
            assert!(says_why, "{path} from {upstream:?} failed silently: {body}");
        }
    }
}

/// A stream that simply stops — upstream killed, no error event — still gets
/// its protocol's terminator, so the client is not left waiting on a closed
/// connection.
///
/// On the translated paths the gateway synthesizes the terminator from its
/// own state machine, so this holds for every ingress there.
#[tokio::test]
async fn a_stream_that_just_stops_still_terminates_when_translated() {
    for upstream in [
        Protocol::OpenAiCompatible,
        Protocol::Anthropic,
        Protocol::OpenAi,
        Protocol::Google,
    ] {
        for path in ingress_paths(true) {
            let ingress_protocol = if is_google_path(path) {
                Protocol::Google
            } else {
                match path {
                    "/v1/chat/completions" => Protocol::OpenAiCompatible,
                    "/v1/responses" => Protocol::OpenAi,
                    _ => Protocol::Anthropic,
                }
            };
            if ingress_protocol == upstream {
                continue; // passthrough; covered separately
            }
            let mock = Mock::start(Upstream::Truncated(upstream)).await;
            let resp = post(
                app(mock.state_with(upstream)),
                path,
                &client_body(path, true),
            )
            .await;
            let events = parse_sse(&text(resp).await);
            assert!(!events.is_empty(), "{path} emitted nothing at all");
            let last = events.last().unwrap();
            let closes = last.1 == "[DONE]"
                || last.0.as_deref() == Some("message_stop")
                || last.1.contains("response.completed")
                || last.1.contains("\"finishReason\"");
            assert!(
                closes,
                "{path} from {upstream:?} stream never closed: {last:?}"
            );
        }
    }
}

/// Known gap: on the same-protocol passthrough path an upstream that stops
/// without a terminal event leaves the client with an unterminated stream.
///
/// This is the deliberate other side of "the gateway never synthesizes a
/// terminal event on passthrough": the upstream owns the terminator, so if
/// it never sends one the gateway has nothing authoritative to add.
/// Synthesizing one would present a truncated answer as a finished one;
/// leaving it off means the client sees a short stream.
///
/// Recorded here so the behaviour is asserted rather than accidental, and so
/// changing it is a deliberate edit to this test.
#[tokio::test]
async fn passthrough_forwards_an_upstream_that_stops_without_a_terminator() {
    let mock = Mock::start(Upstream::Truncated(Protocol::Anthropic)).await;
    let path = "/v1/messages";
    let resp = post(
        app(mock.state_with(Protocol::Anthropic)),
        path,
        &client_body(path, true),
    )
    .await;
    let events = parse_sse(&text(resp).await);

    // The upstream's own frames, forwarded intact and still named.
    assert_eq!(events.first().unwrap().0.as_deref(), Some("message_start"));
    assert!(
        !events
            .iter()
            .any(|(n, _)| n.as_deref() == Some("message_stop")),
        "the gateway must not invent a message_stop: {events:?}"
    );
}

// ---------------------------------------------------------------------------
// Sampling params under thinking (M-6)
// ---------------------------------------------------------------------------

/// With thinking on, `temperature`/`top_p` are forwarded to the Anthropic
/// upstream, not stripped.
///
/// The decision (M-6) was warn-don't-strip: a caller sending
/// `temperature: 0` would otherwise believe the model is deterministic when
/// it is not. A unit test on the translator can't catch a strip introduced
/// anywhere between it and the wire — this one can, because the mock refuses
/// to answer unless the fields actually arrive.
#[tokio::test]
async fn sampling_params_survive_the_trip_to_an_anthropic_upstream() {
    let mock = Mock::start(Upstream::Requires(
        Protocol::Anthropic,
        vec!["temperature", "top_p", "thinking", "output_config"],
    ))
    .await;
    let mut body = client_body("/v1/chat/completions", false);
    body["reasoning_effort"] = json!("high");
    body["temperature"] = json!(0);
    body["top_p"] = json!(0.9);

    let resp = post(
        app(mock.state_with(Protocol::Anthropic)),
        "/v1/chat/completions",
        &body,
    )
    .await;
    let text = text(resp).await;
    assert_eq!(
        parse_json("/v1/chat/completions", &text)["choices"][0]["message"]["content"],
        json!(ANSWER),
        "the upstream never received temperature/top_p: {text}"
    );
}

/// Sampling, thinking and tools survive the trip to a Google upstream.
///
/// The mock refuses unless the translated top-level fields actually arrive;
/// the answer round-trips back through `google_to_chat`.
#[tokio::test]
async fn sampling_and_tools_survive_the_trip_to_a_google_upstream() {
    let mock = Mock::start(Upstream::Requires(
        Protocol::Google,
        vec!["contents", "tools", "toolConfig", "generationConfig"],
    ))
    .await;
    let mut body = client_body("/v1/chat/completions", false);
    body["reasoning_effort"] = json!("high");
    body["temperature"] = json!(0);
    body["top_p"] = json!(0.9);
    body["tools"] = json!([{"type": "function", "function": {
        "name": "get_weather", "description": "weather",
        "parameters": {"type": "object"}} }]);
    body["tool_choice"] = json!("auto");

    let resp = post(
        app(mock.state_with(Protocol::Google)),
        "/v1/chat/completions",
        &body,
    )
    .await;
    let text = text(resp).await;
    assert_eq!(
        parse_json("/v1/chat/completions", &text)["choices"][0]["message"]["content"],
        json!(ANSWER),
        "the google upstream never received the translated fields: {text}"
    );
}

/// The same request must survive the streaming path. Stream setup takes a
/// separate branch from the body translation, so "it works non-streaming" is
/// not evidence about streaming.
#[tokio::test]
async fn sampling_params_survive_the_trip_on_a_stream() {
    let mock = Mock::start(Upstream::Requires(
        Protocol::Anthropic,
        vec!["temperature", "top_p", "thinking"],
    ))
    .await;
    let mut body = client_body("/v1/chat/completions", true);
    body["reasoning_effort"] = json!("high");
    body["temperature"] = json!(0);
    body["top_p"] = json!(0.9);

    let resp = post(
        app(mock.state_with(Protocol::Anthropic)),
        "/v1/chat/completions",
        &body,
    )
    .await;
    let stream = text(resp).await;
    let events = parse_sse(&stream);
    assert!(
        streamed_text("/v1/chat/completions", &events) == ANSWER,
        "the upstream never received temperature/top_p: {stream}"
    );
}

// ---------------------------------------------------------------------------
// Requests the gateway rejects itself
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_unknown_model_is_a_404_in_protocol_shape() {
    let mock = Mock::start(Upstream::Speak(Protocol::OpenAiCompatible)).await;
    for path in INGRESS {
        let mut body = client_body(path, false);
        body["model"] = json!("does-not-exist");
        let resp = post(app(mock.state_with(Protocol::OpenAiCompatible)), path, &body).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND, "{path}");
        let v = parse_json(path, &text(resp).await);
        assert!(v.pointer("/error/message").is_some(), "{path}");
    }
    // Google takes the model from the URL path, not the body.
    for path in [
        "/v1beta/models/does-not-exist:generateContent",
        "/v1beta/models/does-not-exist:streamGenerateContent",
    ] {
        let body = client_body(path, false);
        let resp = post(app(mock.state_with(Protocol::OpenAiCompatible)), path, &body).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND, "{path}");
        let v = parse_json(path, &text(resp).await);
        assert!(v.pointer("/error/message").is_some(), "{path}");
    }
}

#[tokio::test]
async fn malformed_json_is_a_400_not_a_panic() {
    let mock = Mock::start(Upstream::Speak(Protocol::OpenAiCompatible)).await;
    for path in ingress_paths(false) {
        let resp = send(
            app(mock.state_with(Protocol::OpenAiCompatible)),
            path,
            b"{not json".to_vec(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{path}");
        let v = parse_json(path, &text(resp).await);
        assert!(v.pointer("/error/message").is_some(), "{path}");
    }
}

// ---------------------------------------------------------------------------
// A. Parameter passthrough matrix
// ---------------------------------------------------------------------------

/// Assert a JSON pointer in the captured upstream body.
fn assert_ptr(seen: &Value, ptr: &str, expected: &Value, ctx: &str) {
    let got = seen.pointer(ptr);
    assert_eq!(
        got,
        Some(expected),
        "{ctx}: upstream pointer {ptr} mismatch: seen={seen}"
    );
}

/// POST `body` to `path` through a `Capture` mock and return the seen body.
async fn captured_post(proto: Protocol, path: &str, body: &Value) -> (Response, Value) {
    let (kind, seen) = new_capture(proto);
    let mock = Mock::start(kind).await;
    let resp = post(app(mock.state_with(proto)), path, body).await;
    let v = seen.lock().unwrap().clone();
    (resp, v)
}

#[tokio::test]
async fn reasoning_effort_reaches_every_translated_upstream() {
    for stream in [false, true] {
        // Anthropic: adaptive thinking + effort in output_config.
        let mut body = client_body("/v1/chat/completions", stream);
        body["reasoning_effort"] = json!("high");
        let (resp, seen) =
            captured_post(Protocol::Anthropic, "/v1/chat/completions", &body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_ptr(&seen, "/thinking/type", &json!("adaptive"), "anthropic");
        assert_ptr(&seen, "/output_config/effort", &json!("high"), "anthropic");

        // Responses: reasoning.effort.
        let mut body = client_body("/v1/chat/completions", stream);
        body["reasoning_effort"] = json!("high");
        let (resp, seen) =
            captured_post(Protocol::OpenAi, "/v1/chat/completions", &body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_ptr(&seen, "/reasoning/effort", &json!("high"), "responses");

        // Google: thinking budget 16384 + includeThoughts.
        let mut body = client_body("/v1/chat/completions", stream);
        body["reasoning_effort"] = json!("high");
        let (resp, seen) =
            captured_post(Protocol::Google, "/v1/chat/completions", &body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_ptr(
            &seen,
            "/generationConfig/thinkingConfig/thinkingBudget",
            &json!(16384),
            "google",
        );
        assert_ptr(
            &seen,
            "/generationConfig/thinkingConfig/includeThoughts",
            &json!(true),
            "google",
        );

        // `none` disables thinking everywhere: no thinking keys upstream.
        let mut body = client_body("/v1/chat/completions", stream);
        body["reasoning_effort"] = json!("none");
        let (_, seen) =
            captured_post(Protocol::Anthropic, "/v1/chat/completions", &body).await;
        assert!(seen.get("thinking").is_none(), "anthropic none: {seen}");
        let (_, seen) =
            captured_post(Protocol::Google, "/v1/chat/completions", &body).await;
        assert!(
            seen.pointer("/generationConfig/thinkingConfig").is_none(),
            "google none: {seen}"
        );
    }
}

#[tokio::test]
async fn response_format_reaches_every_translated_upstream() {
    let schema = json!({"type": "object", "properties": {"n": {"type": "string"}}});
    for stream in [false, true] {
        let mut body = client_body("/v1/chat/completions", stream);
        body["response_format"] = json!({"type": "json_schema",
            "json_schema": {"name": "person", "schema": schema}});
        let (resp, seen) =
            captured_post(Protocol::Anthropic, "/v1/chat/completions", &body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_ptr(
            &seen,
            "/output_config/format/type",
            &json!("json_schema"),
            "anthropic",
        );
        assert_ptr(
            &seen,
            "/output_config/format/schema/properties/n/type",
            &json!("string"),
            "anthropic",
        );

        let (resp, seen) =
            captured_post(Protocol::OpenAi, "/v1/chat/completions", &body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_ptr(&seen, "/text/format/type", &json!("json_schema"), "responses");
        assert_ptr(
            &seen,
            "/text/format/schema/properties/n/type",
            &json!("string"),
            "responses",
        );

        let (resp, seen) =
            captured_post(Protocol::Google, "/v1/chat/completions", &body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_ptr(
            &seen,
            "/generationConfig/responseMimeType",
            &json!("application/json"),
            "google",
        );
        // Gemini uppercases schema types.
        assert_ptr(
            &seen,
            "/generationConfig/responseSchema/type",
            &json!("OBJECT"),
            "google",
        );

        // `json_object` downgrades to an unconstrained object schema.
        let mut body = client_body("/v1/chat/completions", stream);
        body["response_format"] = json!({"type": "json_object"});
        let (_, seen) =
            captured_post(Protocol::Anthropic, "/v1/chat/completions", &body).await;
        assert_ptr(
            &seen,
            "/output_config/format/schema",
            &json!({"type": "object"}),
            "anthropic json_object",
        );
    }
}

#[tokio::test]
async fn temperature_and_top_p_are_forwarded_verbatim() {
    for proto in [Protocol::Anthropic, Protocol::OpenAi, Protocol::Google] {
        for stream in [false, true] {
            let mut body = client_body("/v1/chat/completions", stream);
            body["temperature"] = json!(0.2);
            body["top_p"] = json!(0.9);
            let (resp, seen) =
                captured_post(proto, "/v1/chat/completions", &body).await;
            assert_eq!(resp.status(), StatusCode::OK, "{proto:?} stream={stream}");
            let (tptr, pptr) = match proto {
                Protocol::Google => ("/generationConfig/temperature", "/generationConfig/topP"),
                _ => ("/temperature", "/top_p"),
            };
            assert_ptr(&seen, tptr, &json!(0.2), &format!("{proto:?}"));
            assert_ptr(&seen, pptr, &json!(0.9), &format!("{proto:?}"));
        }
    }
}

#[tokio::test]
async fn tools_and_tool_choice_reach_every_translated_upstream() {
    fn tool_body(stream: bool) -> Value {
        let mut b = client_body("/v1/chat/completions", stream);
        b["tools"] = json!([{"type": "function", "function": {
            "name": "get_weather", "description": "weather",
            "parameters": {"type": "object",
                "properties": {"city": {"type": "string"}}}} }]);
        b
    }
    for stream in [false, true] {
        // `auto`.
        let mut body = tool_body(stream);
        body["tool_choice"] = json!("auto");
        let (_, seen) =
            captured_post(Protocol::Anthropic, "/v1/chat/completions", &body).await;
        assert_ptr(&seen, "/tools/0/name", &json!("get_weather"), "anthropic tools");
        assert_ptr(&seen, "/tool_choice/type", &json!("auto"), "anthropic choice");
        let (_, seen) =
            captured_post(Protocol::OpenAi, "/v1/chat/completions", &body).await;
        assert_ptr(&seen, "/tools/0/name", &json!("get_weather"), "responses tools");
        assert_ptr(&seen, "/tool_choice", &json!("auto"), "responses choice");
        let (_, seen) =
            captured_post(Protocol::Google, "/v1/chat/completions", &body).await;
        assert_ptr(
            &seen,
            "/tools/0/functionDeclarations/0/name",
            &json!("get_weather"),
            "google tools",
        );
        assert_ptr(
            &seen,
            "/toolConfig/functionCallingConfig/mode",
            &json!("AUTO"),
            "google choice",
        );

        // Named choice.
        let mut body = tool_body(stream);
        body["tool_choice"] =
            json!({"type": "function", "function": {"name": "get_weather"}});
        let (_, seen) =
            captured_post(Protocol::Anthropic, "/v1/chat/completions", &body).await;
        assert_ptr(&seen, "/tool_choice/name", &json!("get_weather"), "anthropic named");
        let (_, seen) =
            captured_post(Protocol::Google, "/v1/chat/completions", &body).await;
        assert_ptr(
            &seen,
            "/toolConfig/functionCallingConfig/mode",
            &json!("ANY"),
            "google named mode",
        );
        assert_ptr(
            &seen,
            "/toolConfig/functionCallingConfig/allowedFunctionNames/0",
            &json!("get_weather"),
            "google named fn",
        );

        // `none` removes tools upstream.
        let mut body = tool_body(stream);
        body["tool_choice"] = json!("none");
        let (_, seen) =
            captured_post(Protocol::Anthropic, "/v1/chat/completions", &body).await;
        assert!(seen.get("tools").is_none(), "anthropic none keeps tools: {seen}");
    }
}

#[tokio::test]
async fn max_tokens_maps_to_each_upstream_limit_field() {
    for stream in [false, true] {
        let mut body = client_body("/v1/chat/completions", stream);
        body["max_tokens"] = json!(55);
        let (_, seen) =
            captured_post(Protocol::Anthropic, "/v1/chat/completions", &body).await;
        assert_ptr(&seen, "/max_tokens", &json!(55), "anthropic");
        let (_, seen) =
            captured_post(Protocol::OpenAi, "/v1/chat/completions", &body).await;
        assert_ptr(&seen, "/max_output_tokens", &json!(55), "responses");
        let (_, seen) =
            captured_post(Protocol::Google, "/v1/chat/completions", &body).await;
        assert_ptr(
            &seen,
            "/generationConfig/maxOutputTokens",
            &json!(55),
            "google",
        );
    }
    // Anthropic requires the field: omitted client value falls back to 8192.
    let body = client_body("/v1/chat/completions", false);
    let (_, seen) =
        captured_post(Protocol::Anthropic, "/v1/chat/completions", &body).await;
    assert_ptr(&seen, "/max_tokens", &json!(8192), "anthropic default");
}

#[tokio::test]
async fn three_ingress_shapes_converge_on_one_upstream_shape() {
    // Responses `reasoning.effort` + `max_output_tokens` -> Anthropic thinking.
    let mut body = client_body("/v1/responses", false);
    body["reasoning"] = json!({"effort": "high"});
    body["max_output_tokens"] = json!(77);
    let (_, seen) =
        captured_post(Protocol::Anthropic, "/v1/responses", &body).await;
    assert_ptr(&seen, "/thinking/type", &json!("adaptive"), "resp->anthropic");
    assert_ptr(&seen, "/output_config/effort", &json!("high"), "resp->anthropic");
    assert_ptr(&seen, "/max_tokens", &json!(77), "resp->anthropic");

    // Anthropic `thinking` + `output_config.effort` -> Google budget.
    let mut body = client_body("/v1/messages", false);
    body["thinking"] = json!({"type": "adaptive"});
    body["output_config"] = json!({"effort": "low"});
    let (_, seen) =
        captured_post(Protocol::Google, "/v1/messages", &body).await;
    assert_ptr(
        &seen,
        "/generationConfig/thinkingConfig/thinkingBudget",
        &json!(4096),
        "anthropic->google",
    );

    // Google `thinkingBudget` -> Anthropic effort.
    let mut body = client_body(GOOGLE_NONSTREAM, false);
    body["generationConfig"] =
        json!({"thinkingConfig": {"thinkingBudget": 16384}, "maxOutputTokens": 99});
    let (_, seen) =
        captured_post(Protocol::Anthropic, GOOGLE_NONSTREAM, &body).await;
    assert_ptr(&seen, "/output_config/effort", &json!("high"), "google->anthropic");
    assert_ptr(&seen, "/max_tokens", &json!(99), "google->anthropic");
}

// ---------------------------------------------------------------------------
// B. Vision input
// ---------------------------------------------------------------------------

const VISION_B64: &str = "AA==";
const VISION_REMOTE: &str = "https://example.com/remote.png";

/// Chat request carrying one base64 image and one remote image.
fn chat_vision_body(stream: bool) -> Value {
    json!({
        "model": "m", "stream": stream,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{VISION_B64}")}},
            {"type": "image_url", "image_url": {"url": VISION_REMOTE}}
        ]}]
    })
}

fn body_contains(seen: &Value, needle: &str, ctx: &str) {
    let s = serde_json::to_string(seen).unwrap_or_default();
    assert!(
        s.contains(needle),
        "{ctx}: upstream body lost image payload {needle:?}: {s}"
    );
}

#[tokio::test]
async fn vision_images_survive_chat_to_every_upstream() {
    for stream in [false, true] {
        let body = chat_vision_body(stream);
        for proto in [
            Protocol::OpenAiCompatible,
            Protocol::Anthropic,
            Protocol::OpenAi,
            Protocol::Google,
        ] {
            let (kind, seen) = new_capture(proto);
            let mock = Mock::start(kind).await;
            let resp = post(app(mock.state_with(proto)), "/v1/chat/completions", &body).await;
            assert_eq!(resp.status(), StatusCode::OK, "{proto:?} stream={stream}");
            let seen_body = seen.lock().unwrap().clone();
            body_contains(&seen_body, VISION_B64, &format!("{proto:?} base64"));
            body_contains(&seen_body, VISION_REMOTE, &format!("{proto:?} remote"));
            // The client still gets a readable answer back.
            if !stream {
                let text = text(resp).await;
                assert_eq!(
                    parse_json("/v1/chat/completions", &text)["choices"][0]["message"]
                        ["content"],
                    json!(ANSWER),
                    "{proto:?} lost the answer"
                );
            } else {
                let events = parse_sse(&text(resp).await);
                assert_eq!(
                    streamed_text("/v1/chat/completions", &events),
                    ANSWER,
                    "{proto:?} lost the stream"
                );
            }
        }
    }
}

#[tokio::test]
async fn vision_google_splits_inline_and_file_data() {
    let body = chat_vision_body(false);
    let (_, seen) = captured_post(Protocol::Google, "/v1/chat/completions", &body).await;
    let parts: Vec<Value> = seen
        .pointer("/contents/0/parts")
        .and_then(|p| p.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(
        parts.iter().any(|p| {
            p.pointer("/inlineData/data").and_then(|d| d.as_str()) == Some(VISION_B64)
        }),
        "data: URL must become inlineData: {seen}"
    );
    assert!(
        parts.iter().any(|p| {
            p.pointer("/fileData/fileUri").and_then(|u| u.as_str()) == Some(VISION_REMOTE)
        }),
        "remote URL must become fileData: {seen}"
    );
}

#[tokio::test]
async fn vision_anthropic_splits_base64_and_url() {
    let body = chat_vision_body(false);
    let (_, seen) = captured_post(Protocol::Anthropic, "/v1/chat/completions", &body).await;
    let s = serde_json::to_string(&seen).unwrap_or_default();
    assert!(s.contains("\"type\":\"base64\""), "base64 source: {s}");
    assert!(s.contains("\"type\":\"url\""), "url source: {s}");
    body_contains(&seen, VISION_REMOTE, "anthropic remote");
}

#[tokio::test]
async fn vision_other_ingresses_reach_a_translated_upstream() {
    // Responses `input_image` -> Google `inlineData`/`fileData`.
    let body = json!({
        "model": "m",
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": format!("data:image/png;base64,{VISION_B64}")},
            {"type": "input_image", "image_url": VISION_REMOTE}
        ]}]
    });
    let (_, seen) = captured_post(Protocol::Google, "/v1/responses", &body).await;
    body_contains(&seen, VISION_B64, "responses->google base64");
    body_contains(&seen, VISION_REMOTE, "responses->google remote");

    // Anthropic `image` blocks -> Google parts.
    let body = json!({
        "model": "m", "max_tokens": 64,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": VISION_B64}},
            {"type": "image", "source": {"type": "url", "url": VISION_REMOTE}}
        ]}]
    });
    let (_, seen) = captured_post(Protocol::Google, "/v1/messages", &body).await;
    body_contains(&seen, VISION_B64, "anthropic->google base64");
    body_contains(&seen, VISION_REMOTE, "anthropic->google remote");

    // Google `inlineData`/`fileData` -> Anthropic images.
    let body = json!({
        "contents": [{"role": "user", "parts": [
            {"text": "look"},
            {"inlineData": {"mimeType": "image/png", "data": VISION_B64}},
            {"fileData": {"fileUri": VISION_REMOTE}}
        ]}]
    });
    let (_, seen) = captured_post(Protocol::Anthropic, GOOGLE_NONSTREAM, &body).await;
    body_contains(&seen, VISION_B64, "google->anthropic base64");
    body_contains(&seen, VISION_REMOTE, "google->anthropic remote");
}

// ---------------------------------------------------------------------------
// C. Multi-round tool calls + Google regressions (issues 4/5/6)
// ---------------------------------------------------------------------------

fn chat_tool_first_turn() -> Value {
    json!({
        "model": "m",
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "weather",
            "parameters": {"type": "object",
                "properties": {"city": {"type": "string"}}}}}],
        "tool_choice": "auto"
    })
}

/// Build the Chat follow-up turn from the gateway's own first answer:
/// copy the assistant `tool_calls` (including any `extra_content`
/// signature) and append the `tool` result, like a real client would.
fn chat_tool_follow_up(first_answer: &Value, result: &str) -> Value {
    let msg = first_answer
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or(json!({}));
    let tcs = msg.get("tool_calls").cloned().unwrap_or(json!([]));
    let call_id = tcs
        .as_array()
        .and_then(|a| a.first())
        .and_then(|c| c.get("id"))
        .and_then(|i| i.as_str())
        .unwrap_or("call_1")
        .to_string();
    json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": Value::Null, "tool_calls": tcs},
            {"role": "tool", "tool_call_id": call_id, "content": result}
        ],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "weather",
            "parameters": {"type": "object",
                "properties": {"city": {"type": "string"}}}}}],
        "tool_choice": "auto"
    })
}

#[tokio::test]
async fn multi_round_tool_calls_close_the_loop_on_every_upstream() {
    for proto in [
        Protocol::OpenAiCompatible,
        Protocol::Anthropic,
        Protocol::OpenAi,
        Protocol::Google,
    ] {
        let args = json!({"city": "Paris"});
        let (kind, seen) = new_scripted(
            proto,
            vec![tool_completion(proto, "get_weather", &args), completion(proto, ANSWER)],
        );
        let mock = Mock::start(kind).await;
        let router = app(mock.state_with(proto));

        // Turn 1: the model asks for a tool call.
        let resp = post(router.clone(), "/v1/chat/completions", &chat_tool_first_turn()).await;
        assert_eq!(resp.status(), StatusCode::OK, "{proto:?} turn 1");
        let first = parse_json("/v1/chat/completions", &text(resp).await);
        let tc = first.pointer("/choices/0/message/tool_calls/0").unwrap_or_else(|| {
            panic!("{proto:?} turn 1 has no tool_calls: {first}")
        });
        assert_eq!(tc.pointer("/function/name").unwrap(), "get_weather");
        assert_eq!(
            tc.pointer("/function/arguments").unwrap(),
            &serde_json::to_string(&args).unwrap()
        );

        // Turn 2: the client returns the tool result; the model answers.
        let follow = chat_tool_follow_up(&first, "sunny, 21C");
        let resp = post(router, "/v1/chat/completions", &follow).await;
        assert_eq!(resp.status(), StatusCode::OK, "{proto:?} turn 2");
        let second = parse_json("/v1/chat/completions", &text(resp).await);
        assert_eq!(
            second.pointer("/choices/0/message/content").unwrap(),
            &json!(ANSWER),
            "{proto:?} lost the final answer"
        );

        // The tool result actually reached the upstream on turn 2.
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 2, "{proto:?} expected exactly 2 upstream calls");
        let s = serde_json::to_string(&seen[1]).unwrap_or_default();
        assert!(
            s.contains("sunny"),
            "{proto:?} upstream never saw the tool result: {s}"
        );
    }
}

#[tokio::test]
async fn google_drops_nameless_tools_like_code_interpreter() {
    // Real OpenWebUI shape: a `code_interpreter` entry with no function name
    // next to a normal tool, plus an explicitly empty-named function.
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "function", "function": {
                "name": "get_weather", "description": "weather",
                "parameters": {"type": "object"}}},
            {"type": "code_interpreter"},
            {"type": "function", "function": {"name": "", "parameters": {"type": "object"}}}
        ],
        "tool_choice": "auto"
    });
    let mock = Mock::start(Upstream::StrictGoogle).await;
    let resp = post(app(mock.state_with(Protocol::Google)), "/v1/chat/completions", &body).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "nameless tools must be dropped, not sent to Gemini"
    );
    let v = parse_json("/v1/chat/completions", &text(resp).await);
    assert_eq!(
        v.pointer("/choices/0/message/content").unwrap(),
        &json!(ANSWER)
    );

    // Same request on the streaming endpoint: still a 200, not an empty 400.
    let mut stream_body = body.clone();
    stream_body["stream"] = json!(true);
    let mock = Mock::start(Upstream::StrictGoogle).await;
    let resp = post(
        app(mock.state_with(Protocol::Google)),
        "/v1/chat/completions",
        &stream_body,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let events = parse_sse(&text(resp).await);
    assert_eq!(
        streamed_text("/v1/chat/completions", &events),
        ANSWER,
        "streaming nameless-tool request lost the answer"
    );
}

#[tokio::test]
async fn google_strips_additional_properties_at_every_depth() {
    // Three poison schemas mirroring the OpenWebUI report: nested
    // `additionalProperties` keys that Gemini rejects with a bare 400.
    let poisoned = |name: &str| {
        json!({"type": "function", "function": {
            "name": name, "description": "d",
            "parameters": {"type": "object",
                "properties": {"payload": {
                    "type": "object",
                    "additionalProperties": true,
                    "properties": {"inner": {
                        "type": "object", "additionalProperties": {"type": "string"}}}}},
                "additionalProperties": false}}})
    };
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [poisoned("ask_user"), poisoned("update_memory"),
                  poisoned("replace_note_content")],
        "tool_choice": "auto"
    });
    // StrictGoogle 400s on any surviving key, so 200 proves the strip.
    let mock = Mock::start(Upstream::StrictGoogle).await;
    let resp = post(app(mock.state_with(Protocol::Google)), "/v1/chat/completions", &body).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "additionalProperties must be stripped before Gemini sees it"
    );

    // And the clean part of the schema survives the strip.
    let (resp, seen) = captured_post(Protocol::Google, "/v1/chat/completions", &body).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let s = serde_json::to_string(&seen).unwrap_or_default();
    assert!(!s.contains("additionalProperties"), "strip missed a nest: {s}");
    assert!(
        s.contains("ask_user") && s.contains("update_memory"),
        "clean schema was damaged: {s}"
    );
}

#[tokio::test]
async fn google_thought_signature_round_trips_across_turns() {
    let sig = "thought-sig-123";
    let args = json!({"city": "Paris"});
    let (kind, seen) = new_scripted(
        Protocol::Google,
        vec![
            google_tool_with_signature("get_weather", &args, sig),
            completion(Protocol::Google, ANSWER),
        ],
    );
    let mock = Mock::start(kind).await;
    let router = app(mock.state_with(Protocol::Google));

    // Turn 1: the signature arrives as `extra_content` for OpenAI-shaped clients.
    let resp = post(router.clone(), "/v1/chat/completions", &chat_tool_first_turn()).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let first = parse_json("/v1/chat/completions", &text(resp).await);
    let got = first
        .pointer("/choices/0/message/tool_calls/0/extra_content/google/thought_signature")
        .and_then(|s| s.as_str())
        .unwrap_or("");
    assert_eq!(got, sig, "signature missing on the way out: {first}");

    // Turn 2 replays it; the gateway must send it back as a sibling part.
    let follow = chat_tool_follow_up(&first, "sunny, 21C");
    let resp = post(router, "/v1/chat/completions", &follow).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let second = parse_json("/v1/chat/completions", &text(resp).await);
    assert_eq!(
        second.pointer("/choices/0/message/content").unwrap(),
        &json!(ANSWER)
    );
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    let s = serde_json::to_string(&seen[1]).unwrap_or_default();
    assert!(
        s.contains(sig),
        "follow-up never sent the thoughtSignature upstream: {s}"
    );
}

#[tokio::test]
async fn google_streaming_empty_400_becomes_a_readable_gateway_error() {
    // Force the empty-body 400 Gemini actually sends (nameless tool on a
    // raw path cannot happen through the gateway, so send a follow-up
    // without its signature by hand: assistant tool_call without
    // extra_content plus a tool result).
    let body = json!({
        "model": "m", "stream": true,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": Value::Null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
        ],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "parameters": {"type": "object"}}}]
    });
    // Sanity: a bare StrictGoogle mock really does answer empty on streams.
    // (The gateway must turn that into something an SDK can read.)
    let mock = Mock::start(Upstream::StrictGoogle).await;
    let resp = post(app(mock.state_with(Protocol::Google)), "/v1/chat/completions", &body).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "expected the strict 400");
    let text = text(resp).await;
    // Non-stream-shaped even for a stream request (fast failure): JSON with
    // a message, never an empty body handed to the SDK.
    assert!(!text.is_empty(), "gateway forwarded Gemini's empty body");
    let v: Value = serde_json::from_str(&text)
        .unwrap_or_else(|_| panic!("gateway error was not JSON: {text:?}"));
    assert!(
        v.pointer("/error/message").and_then(|m| m.as_str()).is_some_and(|m| !m.is_empty()),
        "no readable message in {v}"
    );
}
