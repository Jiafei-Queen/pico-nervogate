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

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use axum::routing::post as route_post;
use axum::Router;
use serde_json::{json, Value};

use crate::config::{ModelCfg, Protocol};
use crate::{app, parse_sse, test_state, St};

type Bytes = axum::body::Bytes;

const ANSWER: &str = "hello there world";

/// The three client-facing surfaces.
const INGRESS: [&str; 3] = ["/v1/chat/completions", "/v1/responses", "/v1/messages"];

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
            respond(&kind, &req)
        }
        let router = Router::new()
            .route("/v1/chat/completions", route_post(handle))
            .route("/v1/messages", route_post(handle))
            .route("/v1/responses", route_post(handle))
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
        test_state(crate::config_for(&self.base, vec![m]))
    }
}

fn respond(kind: &Upstream, req: &Value) -> Response {
    let wants_stream = req.get("stream").and_then(|v| v.as_bool()) == Some(true);
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
        Protocol::Chat => {
            let mut out: Vec<String> = pieces
                .iter()
                .map(|p| {
                    format!(
                        "data: {}\n\n",
                        crate::chat_chunk(json!({"content": p}), None)
                    )
                })
                .collect();
            out.push(format!(
                "data: {}\n\n",
                crate::chat_chunk(json!({}), Some("stop"))
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
        Protocol::Responses => {
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
    }
}

/// Is this the frame that ends a stream in its own protocol?
fn is_terminal_frame(frame: &str) -> bool {
    frame.contains("message_stop")
        || frame.contains("response.completed")
        || frame.contains("[DONE]")
        || frame.contains("\"finish_reason\":\"stop\"")
        || frame.contains("message_delta")
}

/// The mid-stream error frame each protocol uses to report a failure.
fn error_frame(protocol: Protocol, msg: &str) -> String {
    match protocol {
        Protocol::Chat => plain(json!({"error": {"message": msg, "type": "api_error"}})),
        Protocol::Anthropic => named(
            "error",
            json!({"type": "error", "error": {"type": "api_error", "message": msg}}),
        ),
        Protocol::Responses => plain(json!({"type": "error", "message": msg})),
    }
}

/// The non-streaming response an upstream speaking `protocol` returns.
fn completion(protocol: Protocol, text: &str) -> Value {
    match protocol {
        Protocol::Chat => crate::chat_completion(text),
        Protocol::Anthropic => json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "mock",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 2}
        }),
        Protocol::Responses => json!({
            "id": "resp_1", "object": "response", "created_at": 1, "model": "mock",
            "status": "completed",
            "output": [{"id": "msg_1", "type": "message", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        }),
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// A client request body for `path`.
fn client_body(path: &str, stream: bool) -> Value {
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
// The 9 protocol combinations
// ---------------------------------------------------------------------------

/// Every client surface reaches every upstream protocol and gets a usable
/// answer back in its own format.
///
/// This is the matrix the audit claimed to have covered but never
/// committed. A break in any one cell is a whole protocol pair unusable,
/// and no unit test covers the seam between them.
#[tokio::test]
async fn all_nine_protocol_combinations_answer() {
    for upstream in [Protocol::Chat, Protocol::Anthropic, Protocol::Responses] {
        for path in INGRESS {
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
async fn all_nine_protocol_combinations_stream() {
    for upstream in [Protocol::Chat, Protocol::Anthropic, Protocol::Responses] {
        for path in INGRESS {
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
    let mock = Mock::start(Upstream::Speak(Protocol::Responses)).await;
    let path = "/v1/responses";
    let resp = post(
        app(mock.state_with(Protocol::Responses)),
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
    let mock = Mock::start(Upstream::Speak(Protocol::Chat)).await;
    let path = "/v1/chat/completions";
    let resp = post(
        app(mock.state_with(Protocol::Chat)),
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
    for upstream in [Protocol::Chat, Protocol::Anthropic, Protocol::Responses] {
        for path in INGRESS {
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
    for path in INGRESS {
        let mock = Mock::start(kind.clone()).await;
        let resp = post(
            app(mock.state_with(Protocol::Chat)),
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
    for path in INGRESS {
        let mock = Mock::start(Upstream::NonJson("<html>502 Bad Gateway</html>".into())).await;
        let resp = post(
            app(mock.state_with(Protocol::Chat)),
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
    for upstream in [Protocol::Chat, Protocol::Anthropic, Protocol::Responses] {
        for path in INGRESS {
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
    for upstream in [Protocol::Chat, Protocol::Anthropic, Protocol::Responses] {
        for path in INGRESS {
            let ingress_protocol = match path {
                "/v1/chat/completions" => Protocol::Chat,
                "/v1/responses" => Protocol::Responses,
                _ => Protocol::Anthropic,
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
                || last.1.contains("response.completed");
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
    let mock = Mock::start(Upstream::Speak(Protocol::Chat)).await;
    for path in INGRESS {
        let mut body = client_body(path, false);
        body["model"] = json!("does-not-exist");
        let resp = post(app(mock.state_with(Protocol::Chat)), path, &body).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND, "{path}");
        let v = parse_json(path, &text(resp).await);
        assert!(v.pointer("/error/message").is_some(), "{path}");
    }
}

#[tokio::test]
async fn malformed_json_is_a_400_not_a_panic() {
    let mock = Mock::start(Upstream::Speak(Protocol::Chat)).await;
    for path in INGRESS {
        let resp = send(
            app(mock.state_with(Protocol::Chat)),
            path,
            b"{not json".to_vec(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{path}");
        let v = parse_json(path, &text(resp).await);
        assert!(v.pointer("/error/message").is_some(), "{path}");
    }
}
