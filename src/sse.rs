//! SSE pipeline: incremental parsing and protocol translation.

use std::io;

use futures_util::StreamExt;

use crate::errors::trunc;
use axum::http::HeaderValue;
use axum::http::header;
use axum::response::Response;
use axum::body::{Body, Bytes};
use serde_json::{json, Value};
use crate::translate::{AnthropicStream, ChatReasoningEgress, ChatToAnthropicStream, ChatToResponsesStream, ResponsesStream, SseEvent, frame_data, frame_event};
use crate::translate::google::{ChatToGoogleStream, GoogleStream};

/// Upstream SSE -> canonical OpenAI chat chunks.
pub(crate) enum Inbound {
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
    Google(GoogleStream),
}

impl Inbound {
    /// Handle one upstream SSE `data:` payload (a JSON object).
    pub(crate) fn handle(&mut self, ev: &Value) -> Vec<Value> {
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
            Inbound::Google(s) => s.handle(ev),
        }
    }

    /// Flush a held finish chunk at end of stream, backfilling any usage
    /// that arrived separately.
    pub(crate) fn flush(&mut self) -> Vec<Value> {
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
pub(crate) enum Egress {
    Chat(ChatReasoningEgress),
    Responses(ChatToResponsesStream),
    Anthropic(ChatToAnthropicStream),
    Google(ChatToGoogleStream),
}

impl Egress {
    pub(crate) fn handle(&mut self, chunk: &Value) -> Vec<String> {
        match self {
            Egress::Chat(e) => {
                let mut c = chunk.clone();
                e.transform(&mut c);
                vec![frame_data(&c)]
            }
            Egress::Responses(s) => s.handle(chunk).iter().map(|e| e.frame()).collect(),
            Egress::Anthropic(s) => s.handle(chunk).iter().map(|e| e.frame()).collect(),
            Egress::Google(s) => s.handle(chunk).iter().map(|e| e.frame()).collect(),
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
    pub(crate) fn finish(&mut self, passthrough: bool) -> Vec<String> {
        if passthrough {
            return match self {
                // The upstream's own `[DONE]` is swallowed by the parser and
                // has to be re-emitted; the other protocols terminate
                // themselves upstream.
                Egress::Chat(_) => vec!["data: [DONE]\n\n".to_string()],
                _ => vec![],
            };
        }
        match self {
            Egress::Chat(_) => vec!["data: [DONE]\n\n".to_string()],
            Egress::Responses(s) => match s.finish() {
                Some(events) => events.iter().map(|e| e.frame()).collect(),
                None => vec![],
            },
            Egress::Anthropic(s) => match s.finish() {
                Some(events) => events.iter().map(|e| e.frame()).collect(),
                None => vec![],
            },
            Egress::Google(s) => match s.finish() {
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
    pub(crate) fn fail(&mut self, msg: &str, passthrough: bool) -> Vec<String> {
        match self {
            Egress::Chat(_) => vec![
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
            Egress::Google(s) => {
                let mut out: Vec<SseEvent> = if passthrough { vec![] } else { s.abort() };
                out.push(SseEvent::data(
                    json!({"error": {"message": msg, "type": "gateway_error"}}),
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
pub(crate) enum Block {
    /// Nothing to forward (blank, `[DONE]`, unparseable).
    Skip,
    /// The upstream signalled a failure; the stream ends here.
    Error(String),
    /// Native path: forward verbatim, keeping the upstream's `event:` name.
    Passthrough(Option<String>, Value),
    /// Translated path: client-facing SSE lines rendered by the egress.
    Chunks(Vec<String>),
}

pub(crate) fn classify_block(
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
            // Google carries the public id in `modelVersion`, not `model`.
            if o.contains_key("candidates") || o.contains_key("modelVersion") {
                o.insert("modelVersion".into(), json!(m));
            }
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
pub(crate) fn render_block(block: Block) -> (Vec<String>, Option<String>) {
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
pub(crate) fn sse_body(
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
pub(crate) struct SseParser {
    pub(crate) buf: Vec<u8>,
    /// Bytes of `buf` already dispatched as events.
    pub(crate) pos: usize,
    /// Start of the line currently being scanned. Bytes before it belong to
    /// complete lines, so they are never rescanned.
    pub(crate) line_start: usize,
}

impl SseParser {
    pub(crate) fn new() -> Self {
        Self {
            buf: Vec::new(),
            pos: 0,
            line_start: 0,
        }
    }

    pub(crate) fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// Reclaim dispatched bytes. `drain` from the front is O(remaining), so
    /// doing it every event would be quadratic on a busy stream.
    pub(crate) fn compact(&mut self) {
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
    pub(crate) fn next_block(&mut self) -> Option<(usize, usize)> {
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
    pub(crate) fn finish(&mut self) -> Option<(usize, usize)> {
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

    pub(crate) fn block(&self, range: (usize, usize)) -> &[u8] {
        &self.buf[range.0..range.1]
    }
}

/// Split an SSE block into lines, accepting LF, CRLF and bare CR.
pub(crate) fn sse_lines(block: &[u8]) -> Vec<&[u8]> {
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
pub(crate) struct SseFields {
    pub(crate) name: Option<String>,
    pub(crate) data: String,
}

/// Parse one SSE block. Multiple `data:` lines join with `\n` and exactly one
/// leading space is stripped (the spec's rule) — trimming all leading
/// whitespace would corrupt indented JSON payloads.
pub(crate) fn sse_fields(block: &[u8]) -> SseFields {
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
pub(crate) fn is_stream_error(ev: &Value) -> bool {
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
pub(crate) fn stream_error_msg(ev: &Value, raw: &str) -> String {
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

pub(crate) fn sse_response(body: Body) -> Response {
    let mut resp = Response::new(body);
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    resp.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    resp
}

/// Baseline `Config` for tests, overridable field by field.
/// One chat-completion chunk, as a mock upstream would emit it.
#[cfg(test)]
pub(crate) fn chat_chunk(delta: Value, finish: Option<&str>) -> Value {
    let mut choice = json!({"index": 0, "delta": delta, "finish_reason": Value::Null});
    if let Some(f) = finish {
        choice["finish_reason"] = json!(f);
    }
    json!({"id": "c", "object": "chat.completion.chunk",
           "created": 1, "model": "mock", "choices": [choice]})
}

/// Mock non-streaming chat completion carrying `text`.
#[cfg(test)]
pub(crate) fn chat_completion(text: &str) -> Value {
    json!({"id": "c", "object": "chat.completion", "created": 1, "model": "mock",
           "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text}}],
           "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
}

/// Split an SSE body into `(event_name, data)` pairs, in wire order.
#[cfg(test)]
pub(crate) fn parse_sse(body: &str) -> Vec<(Option<String>, String)> {
    let mut out = Vec::new();
    for block in body.split("\n\n") {
        let mut name = None;
        let mut data = Vec::new();
        for line in block.split('\n') {
            if let Some(v) = line.strip_prefix("event:") {
                name = Some(v.strip_prefix(' ').unwrap_or(v).to_string());
            } else if let Some(v) = line.strip_prefix("data:") {
                data.push(v.strip_prefix(' ').unwrap_or(v));
            }
        }
        if !data.is_empty() {
            out.push((name, data.join("\n")));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReasoningOutput;
    use crate::translate::{ChatToAnthropicStream, ChatToResponsesStream, ResponsesStream};
    use serde_json::json;

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
            ReasoningOutput::ReasoningContent,
        ));
        assert_eq!(chat.finish(true), vec!["data: [DONE]\n\n".to_string()]);
        assert_eq!(chat.fail("boom", true).len(), 2);

        // The translated path is unchanged and still synthesizes.
        let mut responses = Egress::Responses(ChatToResponsesStream::new("m"));
        assert!(!responses.finish(false).is_empty());
    }

    #[test]
    fn classify_block_treats_a_failed_chunk_as_a_stream_error() {
        // The main loop and the end-of-stream flush used to carry separate
        // copies of this check; both paths must agree.
        let mut inbound = Inbound::Responses(ResponsesStream::new("m"));
        let mut egress = Egress::Chat(ChatReasoningEgress::new(
            ReasoningOutput::ReasoningContent,
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
}
