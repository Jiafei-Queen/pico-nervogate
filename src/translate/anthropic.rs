//! Anthropic Messages <-> canonical Chat translation.

use serde_json::{json, Value};

use crate::config::ThinkingMode;

use super::common::*;
use super::server_tool;
pub fn openai_image_to_anthropic(p: &Value) -> Option<Value> {
    let url = p.pointer("/image_url/url").and_then(|x| x.as_str())?;
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',')?;
        let media = meta.split(';').next().unwrap_or("image/png");
        Some(json!({"type":"image","source":{"type":"base64","media_type":media,"data":data}}))
    } else {
        Some(json!({"type":"image","source":{"type":"url","url":url}}))
    }
}

pub fn anthropic_image_to_openai(b: &Value) -> Option<Value> {
    match b.pointer("/source/type").and_then(|x| x.as_str()) {
        Some("base64") => {
            let media = b
                .pointer("/source/media_type")
                .and_then(|x| x.as_str())
                .unwrap_or("image/png");
            let data = b.pointer("/source/data").and_then(|x| x.as_str())?;
            Some(
                json!({"type":"image_url","image_url":{"url": format!("data:{media};base64,{data}")}}),
            )
        }
        Some("url") => {
            let url = b.pointer("/source/url").and_then(|x| x.as_str())?;
            Some(json!({"type":"image_url","image_url":{"url": url}}))
        }
        _ => None,
    }
}

/// OpenAI user content -> Anthropic content blocks.
///
/// Block types with no Anthropic equivalent (`input_audio`, `file`,
/// `document`, ...) are reported instead of silently dropped: the client
/// believes it sent the payload while the model never saw it.
pub fn user_blocks(c: &Value) -> Vec<Value> {
    match c {
        Value::String(s) => {
            if s.is_empty() {
                vec![]
            } else {
                vec![json!({"type": "text", "text": s})]
            }
        }
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                let ty = p.get("type").and_then(|x| x.as_str()).unwrap_or("text");
                match ty {
                    "text" | "input_text" => p
                        .get("text")
                        .filter(|t| !t.as_str().unwrap_or("").is_empty())
                        .map(|t| json!({"type": "text", "text": t.clone()})),
                    "image_url" => openai_image_to_anthropic(p),
                    other => {
                        warn_dropped_block("openai_to_anthropic", other);
                        None
                    }
                }
            })
            .collect(),
        _ => vec![],
    }
}

pub fn finish_from_anthropic(stop: &str) -> &'static str {
    match stop {
        "max_tokens" | "max_output_tokens" => "length",
        "tool_use" => "tool_calls",
        "refusal" => "content_filter",
        // end_turn, stop_sequence, pause_turn, ...
        _ => "stop",
    }
}

/// Chat `finish_reason` -> Anthropic `stop_reason`.
pub fn anthropic_stop(finish: &str) -> &'static str {
    match finish {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        // Closes the round trip with `finish_from_anthropic`, which already
        // maps `refusal` -> `content_filter`.
        "content_filter" => "refusal",
        _ => "end_turn",
    }
}

// ---------------------------------------------------------------------------
// OpenAI chat -> Anthropic Messages
// ---------------------------------------------------------------------------

/// Append blocks to the last message when it has the same role (Anthropic
/// wants alternating roles), else start a new message.
pub fn push_anthropic_blocks(msgs: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = msgs.last_mut() {
        if last.get("role").and_then(|r| r.as_str()) == Some(role) {
            if let Some(arr) = last.get_mut("content").and_then(|c| c.as_array_mut()) {
                arr.extend(blocks);
                return;
            }
        }
    }
    msgs.push(json!({"role": role, "content": blocks}));
}

/// Legacy `thinking.budget_tokens` for `ThinkingMode::Enabled`.
pub fn thinking_budget(effort: &str) -> i64 {
    match effort {
        "minimal" => 1024,
        "low" => 4096,
        "medium" => 8192,
        "high" => 16384,
        _ => 4096,
    }
}

/// Fallback `max_tokens` for Anthropic upstreams when the client omitted
/// it. Anthropic *requires* the field, Chat treats it as optional, so the
/// gateway must supply something; the value is configurable.
#[cfg(test)]
pub const ANTHROPIC_DEFAULT_MAX_TOKENS: i64 = 8192;

/// Translation with the built-in default. Production callers go through
/// `openai_to_anthropic_with_max` so the value comes from config.
#[cfg(test)]
pub fn openai_to_anthropic(req: &Value, thinking: ThinkingMode) -> Value {
    openai_to_anthropic_with_max(req, thinking, ANTHROPIC_DEFAULT_MAX_TOKENS)
}

/// Canonical reasoning detail -> Anthropic content block.
///
/// `wire` selects the signature policy: an outgoing response may pass an
/// `nv1:o:` blob through as an opaque marker for the client to echo, while an
/// outgoing request only replays an `nv1:a:` signature this gateway issued
/// for an Anthropic upstream.
fn reasoning_detail_to_anth_block(d: &Value, wire: bool) -> Option<Value> {
    let d = normalize_reasoning_detail(d)?;
    match reasoning_kind(&d)? {
        ReasoningKind::Thinking => {
            let sig = d.get("signature").and_then(|x| x.as_str())?;
            let sig = if wire {
                anthropic_signature_for_wire(sig)?
            } else {
                unwrap_anthropic_signature(sig)?
            };
            Some(json!({
                "type": "thinking",
                "thinking": reasoning_text(&d),
                "signature": sig
            }))
        }
        ReasoningKind::Redacted => Some(json!({
            "type": "redacted_thinking",
            "data": d.get("data").and_then(|x| x.as_str()).filter(|s| !s.is_empty())?
        })),
    }
}

/// A translated reasoning block plus the number of tool calls that preceded
/// it upstream (when the egress recorded that position).
fn reasoning_block_with_position(d: &Value, wire: bool) -> Option<(Option<usize>, Value)> {
    let pos = d
        .get("tool_calls_before")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize);
    reasoning_detail_to_anth_block(d, wire).map(|b| (pos, b))
}

/// Assemble an Anthropic assistant content array, honouring the recorded
/// interleaving of thinking blocks and tool calls. Without positions the
/// historical order (thinking, text, tools) is kept.
fn ordered_assistant_blocks(
    details: Vec<(Option<usize>, Value)>,
    text: &str,
    tool_blocks: Vec<Value>,
) -> Vec<Value> {
    let mut blocks = vec![];
    if details.iter().any(|(pos, _)| pos.is_some()) {
        let mut used = vec![false; details.len()];
        for (i, (pos, block)) in details.iter().enumerate() {
            if pos.is_none() || *pos == Some(0) {
                blocks.push(block.clone());
                used[i] = true;
            }
        }
        if !text.is_empty() {
            blocks.push(json!({"type":"text","text":text}));
        }
        for (idx, tool) in tool_blocks.iter().enumerate() {
            blocks.push(tool.clone());
            for (i, (pos, block)) in details.iter().enumerate() {
                if !used[i] && pos.is_some_and(|p| p <= idx + 1) {
                    blocks.push(block.clone());
                    used[i] = true;
                }
            }
        }
        for (i, (_, block)) in details.iter().enumerate() {
            if !used[i] {
                blocks.push(block.clone());
            }
        }
    } else {
        blocks.extend(details.into_iter().map(|(_, block)| block));
        if !text.is_empty() {
            blocks.push(json!({"type":"text","text":text}));
        }
        blocks.extend(tool_blocks);
    }
    blocks
}

pub fn openai_to_anthropic_with_max(
    req: &Value,
    thinking: ThinkingMode,
    default_max_tokens: i64,
) -> Value {
    let mut out = serde_json::Map::new();
    // `output_config` carries both `effort` (thinking) and `format`
    // (structured outputs); build it once so the two never clobber each other.
    let mut output_config: serde_json::Map<String, Value> = serde_json::Map::new();

    let max_tokens = req
        .get("max_tokens")
        .or_else(|| req.get("max_completion_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(default_max_tokens));
    out.insert("max_tokens".into(), max_tokens);

    for k in ["temperature", "top_p"] {
        if let Some(v) = req.get(k) {
            out.insert(k.into(), v.clone());
        }
    }
    // `top_k` has no Chat Completions equivalent; it rides through the
    // reserved passthrough key.
    if let Some(k) = req.pointer("/x_nervogate/top_k") {
        out.insert("top_k".into(), k.clone());
    }
    if let Some(stop) = req.get("stop") {
        let arr = match stop {
            Value::Array(a) => a.clone(),
            Value::String(s) => vec![Value::String(s.clone())],
            _ => vec![],
        };
        if !arr.is_empty() {
            out.insert("stop_sequences".into(), Value::Array(arr));
        }
    }
    if let Some(md) = req.get("metadata") {
        if md.get("user_id").is_some() {
            out.insert("metadata".into(), md.clone());
        }
    }

    let mut system = String::new();
    let mut msgs: Vec<Value> = vec![];

    if let Some(list) = req.get("messages").and_then(|m| m.as_array()) {
        for m in list {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
            match role {
                "system" | "developer" => {
                    let t = content_text(m.get("content").unwrap_or(&Value::Null));
                    if t.is_empty() {
                        continue;
                    }
                    if !system.is_empty() {
                        system.push('\n');
                    }
                    system.push_str(&t);
                }
                "tool" => {
                    let id = m.get("tool_call_id").and_then(|x| x.as_str()).unwrap_or("");
                    let content = m.get("content").cloned().unwrap_or(json!(""));
                    let content = content_text(&content);
                    // Anthropic wants all tool_results for one assistant turn
                    // in a single user message, leading the content.
                    push_anthropic_blocks(
                        &mut msgs,
                        "user",
                        vec![json!({"type":"tool_result","tool_use_id":id,"content":content})],
                    );
                }
                "assistant" => {
                    let mut blocks: Vec<Value> = vec![];
                    // Only a signature that this gateway issued for an
                    // Anthropic upstream can be replayed. `reasoning_content`
                    // alone has no signature, and Anthropic rejects a
                    // thinking block with an empty one (400).
                    let mut details: Vec<(Option<usize>, Value)> = vec![];
                    if let Some(ds) = m.get("reasoning_details").and_then(|x| x.as_array()) {
                        for d in ds {
                            if let Some(entry) = reasoning_block_with_position(d, false) {
                                details.push(entry);
                            }
                        }
                    }
                    // Falls back to the top-level `refusal` when `content`
                    // is empty, so a refusal-only turn is not turned into a
                    // blank assistant message.
                    let text = chat_message_text(m);
                    let mut tool_blocks: Vec<Value> = vec![];
                    if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()) {
                        for tc in tcs {
                            let id = tc.get("id").and_then(|x| x.as_str()).unwrap_or("");
                            let name = tc
                                .pointer("/function/name")
                                .and_then(|x| x.as_str())
                                .unwrap_or("");
                            let args = tc
                                .pointer("/function/arguments")
                                .and_then(|x| x.as_str())
                                .unwrap_or("{}");
                            let input: Value = serde_json::from_str(args).unwrap_or(json!({}));
                            tool_blocks
                                .push(json!({"type":"tool_use","id":id,"name":name,"input":input}));
                        }
                    }
                    blocks = ordered_assistant_blocks(details, &text, tool_blocks);
                    push_anthropic_blocks(&mut msgs, "assistant", blocks);
                }
                _ => {
                    let c = m.get("content").cloned().unwrap_or(json!(""));
                    push_anthropic_blocks(&mut msgs, "user", user_blocks(&c));
                }
            }
        }
    }

    if !system.is_empty() {
        out.insert("system".into(), json!(system));
    }
    out.insert("messages".into(), Value::Array(msgs));

    let mut tools = req
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|tools| {
            // Anthropic rejects a tool with an empty name (400), so a
            // nameless non-server tool (one the client cannot represent
            // here) is dropped rather than forwarded.
            tools
                .iter()
                .filter_map(|t| {
                    if let Some(kind) = server_tool::kind_of(t) {
                        return Some(match kind {
                            server_tool::WEB_SEARCH => {
                                let mut v = json!({
                                    "type": "web_search_20250305",
                                    "name": "web_search"
                                });
                                let domains = server_tool::allowed_domains(t);
                                if !domains.is_empty() {
                                    v["allowed_domains"] = Value::Array(domains);
                                }
                                v
                            }
                            _ => json!({
                                "type": "code_execution_20250522",
                                "name": "code_execution"
                            }),
                        });
                    }
                    let name = t
                        .pointer("/function/name")
                        .and_then(|x| x.as_str())
                        .filter(|s| !s.is_empty())?;
                    Some(json!({
                        "name": name,
                        "description": t.pointer("/function/description").cloned().unwrap_or(json!("")),
                        "input_schema": t.pointer("/function/parameters").cloned().unwrap_or(json!({"type":"object"}))
                    }))
                })
                .collect::<Vec<Value>>()
        })
        .unwrap_or_default();

    if !tools.is_empty() {
        // `tool_choice` constrains function tools; with only server tools
        // there is nothing to choose, and Anthropic may reject it.
        let has_function_tools = tools.iter().any(|t| t.get("type").is_none());
        if has_function_tools {
            if let Some(tc) = req.get("tool_choice") {
                match tc.as_str() {
                    Some("auto") => {
                        out.insert("tool_choice".into(), json!({"type":"auto"}));
                    }
                    Some("required") => {
                        out.insert("tool_choice".into(), json!({"type":"any"}));
                    }
                    Some("none") => tools.clear(),
                    _ => {
                        if let Some(name) = tc.pointer("/function/name").and_then(|x| x.as_str()) {
                            out.insert("tool_choice".into(), json!({"type":"tool","name":name}));
                        }
                    }
                }
            }
        }
    }
    if !tools.is_empty() {
        out.insert("tools".into(), Value::Array(tools));
    }

    // Structured outputs: Chat `response_format` -> Anthropic
    // `output_config.format`. Dropping this silently returns free-form text
    // where the caller expects schema-valid JSON.
    if let Some(rf) = req.get("response_format") {
        match rf.get("type").and_then(|x| x.as_str()) {
            Some("json_object") => {
                // Anthropic's `output_config.format` only accepts
                // `json_schema`; sending `json_object` gets a 400 back.
                // The closest faithful translation is an unconstrained
                // object schema — the caller asked for "valid JSON object",
                // and that is exactly what an empty object schema means.
                output_config.insert(
                    "format".into(),
                    json!({"type": "json_schema", "schema": {"type": "object"}}),
                );
            }
            Some("json_schema") => {
                let schema = rf
                    .pointer("/json_schema/schema")
                    .cloned()
                    .unwrap_or(json!({"type": "object"}));
                output_config.insert(
                    "format".into(),
                    json!({"type": "json_schema", "schema": schema}),
                );
            }
            // `text` is Anthropic's native (unconstrained) mode; nothing to do.
            Some("text") | None => {}
            _ => {}
        }
    }

    if let Some(e) = req.get("reasoning_effort").and_then(|x| x.as_str()) {
        if e != "none" {
            match thinking {
                // Newer upstream models reject `type: "enabled"` +
                // budget_tokens and require `type: "adaptive"` with
                // `output_config.effort`; the adaptive shape is accepted by
                // every Anthropic-protocol upstream tested so far.
                ThinkingMode::Adaptive => {
                    out.insert("thinking".into(), json!({"type":"adaptive"}));
                    output_config.insert("effort".into(), json!(e));
                }
                // Escape hatch for upstreams that only accept the legacy
                // `enabled` + budget_tokens shape.
                ThinkingMode::Enabled => {
                    out.insert(
                        "thinking".into(),
                        json!({"type":"enabled","budget_tokens": thinking_budget(e)}),
                    );
                }
            }
        }
    }
    if !output_config.is_empty() {
        out.insert("output_config".into(), Value::Object(output_config));
    }

    Value::Object(out)
}

/// Request parameters a Chat -> Anthropic translation cannot carry over.
/// Silently dropping them changes model behaviour, so the caller reports
/// them (stderr always, HTTP 400 when `strict_params` is on).
pub fn chat_to_anthropic_loss(req: &Value) -> Vec<&'static str> {
    const LOST: &[&str] = &[
        "n",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "seed",
        "logprobs",
        "top_logprobs",
        "user",
        // `stream_options` is deliberately absent: dropping it only means
        // "no OpenAI-style trailing usage chunk", which is the upstream
        // default anyway and `Inbound::Chat::usage` already covers the
        // ordering. Warning on it would fire on nearly every SDK request.
    ];
    LOST.iter()
        .copied()
        .filter(|k| req.get(*k).is_some_and(|v| !v.is_null()))
        .chain(has_nameless_tools(req).then_some("nameless tools (dropped: not function tools)"))
        .collect()
}

/// `n` changes the *shape* of the reply (one candidate vs many), so a
/// caller indexing `choices[1]` breaks silently. It gets a hard 400 under
/// `strict_params`. `parallel_tool_calls` is only warned about: OpenAI SDKs
/// send it by default, so rejecting it would break otherwise-working setups.
pub fn chat_to_anthropic_fatal(req: &Value) -> Option<String> {
    if req.get("n").and_then(|x| x.as_i64()).unwrap_or(1) > 1 {
        return Some(
            "`n > 1` is not supported for Anthropic upstreams: the translation \
             yields a single candidate and `choices[1]` would be undefined"
                .to_string(),
        );
    }
    None
}

pub fn anthropic_to_openai(resp: &Value, model: &str) -> Value {
    let id = gen_id("chatcmpl-");
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut reasoning_details: Vec<Value> = vec![];
    let mut tool_calls: Vec<Value> = vec![];
    let mut server_calls: Vec<Value> = vec![];
    let mut citations: Vec<Value> = vec![];

    let web_results = |b: &Value, server_calls: &mut Vec<Value>| {
        let call_id = b.get("tool_use_id").and_then(|x| x.as_str()).unwrap_or("");
        let results: Vec<Value> = b
            .get("content")
            .and_then(|c| c.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|r| {
                        let url = r.get("url").and_then(|x| x.as_str())?;
                        Some(json!({
                            "url": url,
                            "title": r.get("title").cloned().unwrap_or(json!(""))
                        }))
                    })
                    .collect()
            })
            .unwrap_or_default();
        match server_calls.iter_mut().rev().find(|c| {
            c["type"] == server_tool::WEB_SEARCH
                && (call_id.is_empty() || c["id"].as_str() == Some(call_id))
        }) {
            Some(call) => call["results"] = json!(results),
            None => server_calls.push(server_tool::web_search("", results)),
        }
    };

    if let Some(blocks) = resp.get("content").and_then(|c| c.as_array()) {
        for b in blocks {
            match b.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
                        text.push_str(t);
                    }
                    for c in b
                        .get("citations")
                        .and_then(|x| x.as_array())
                        .into_iter()
                        .flatten()
                    {
                        let url = c.get("url").and_then(|x| x.as_str()).unwrap_or("");
                        if url.is_empty() {
                            continue;
                        }
                        citations.push(server_tool::citation(
                            url,
                            c.get("title").and_then(|x| x.as_str()).unwrap_or(""),
                            c.get("cited_text").and_then(|x| x.as_str()).unwrap_or(""),
                        ));
                    }
                }
                Some("server_tool_use") => {
                    let name = b.get("name").and_then(|x| x.as_str()).unwrap_or("");
                    let call_id = b.get("id").and_then(|x| x.as_str()).unwrap_or("");
                    if name.starts_with("web_search") {
                        let query = b
                            .pointer("/input/query")
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        let mut call = server_tool::web_search(query, vec![]);
                        if !call_id.is_empty() {
                            call["id"] = json!(call_id);
                        }
                        server_calls.push(call);
                    } else if name.starts_with("code_execution") {
                        let code = b
                            .pointer("/input/code")
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        let mut call = server_tool::code_execution(code, "");
                        if !call_id.is_empty() {
                            call["id"] = json!(call_id);
                        }
                        server_calls.push(call);
                    }
                }
                Some("web_search_tool_result") => web_results(b, &mut server_calls),
                Some("code_execution_tool_result") => {
                    let call_id = b.get("tool_use_id").and_then(|x| x.as_str()).unwrap_or("");
                    let output = b
                        .pointer("/content/stdout")
                        .or_else(|| b.pointer("/content/output"))
                        .and_then(|x| x.as_str())
                        .unwrap_or("");
                    if let Some(call) = server_calls.iter_mut().rev().find(|c| {
                        c["type"] == server_tool::CODE_EXECUTION
                            && (call_id.is_empty() || c["id"].as_str() == Some(call_id))
                    }) {
                        call["output"] = json!(output);
                    }
                }
                Some("thinking") => {
                    let thinking = b.get("thinking").and_then(|x| x.as_str()).unwrap_or("");
                    if !thinking.is_empty() {
                        reasoning.push_str(thinking);
                    }
                    if let Some(sig) = b
                        .get("signature")
                        .and_then(|x| x.as_str())
                        .filter(|s| !s.is_empty())
                    {
                        reasoning_details.push(json!({
                            "type": "thinking",
                            "thinking": thinking,
                            "signature": wrap_anthropic_signature(sig),
                            // Position among the assistant's tool calls, so
                            // interleaved thinking can be rebuilt in order.
                            "tool_calls_before": tool_calls.len()
                        }));
                    }
                }
                Some("redacted_thinking") => {
                    if let Some(data) = b.get("data").and_then(|x| x.as_str()) {
                        if !data.is_empty() {
                            reasoning_details.push(json!({
                                "type": "redacted_thinking",
                                "data": data,
                                "tool_calls_before": tool_calls.len()
                            }));
                        }
                    }
                }
                Some("tool_use") => {
                    let name = b.get("name").and_then(|x| x.as_str()).unwrap_or("");
                    let input = b.get("input").cloned().unwrap_or(json!({}));
                    let args = serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());
                    let call_id = b
                        .get("id")
                        .and_then(|x| x.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| gen_id("call_"));
                    tool_calls.push(json!({
                        "id": call_id, "type": "function",
                        "function": {"name": name, "arguments": args}
                    }));
                }
                _ => {}
            }
        }
    }

    let stop = resp
        .get("stop_reason")
        .and_then(|s| s.as_str())
        .unwrap_or("end_turn");
    let finish = finish_from_anthropic(stop);
    let in_tok = resp
        .pointer("/usage/input_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let out_tok = resp
        .pointer("/usage/output_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    // Anthropic `input_tokens` excludes cache reads/writes, while OpenAI
    // `prompt_tokens` includes cached tokens. Adding them keeps
    // `prompt_tokens` comparable and `total_tokens` honest.
    let cached = resp
        .pointer("/usage/cache_read_input_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let cache_creation = resp
        .pointer("/usage/cache_creation_input_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let prompt_total = in_tok + cached + cache_creation;

    server_tool::locate_citations(&mut citations, &text);
    if !citations.is_empty() {
        let derived = server_tool::results_from_citations(&citations);
        for call in &mut server_calls {
            if call["type"] == server_tool::WEB_SEARCH
                && call["results"].as_array().is_some_and(|r| r.is_empty())
                && !derived.is_empty()
            {
                call["results"] = json!(derived);
            }
        }
    }

    let mut message = json!({"role": "assistant", "content": text});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !reasoning_details.is_empty() {
        message["reasoning_details"] = Value::Array(reasoning_details);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
        message["content"] = Value::Null;
    }
    if !server_calls.is_empty() {
        message["server_tool_calls"] = Value::Array(server_calls);
    }
    if !citations.is_empty() {
        message["citations"] = Value::Array(citations);
    }

    json!({
        "id": id,
        "object": "chat.completion",
        "created": now_ts(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": chat_usage(prompt_total, out_tok, cached, 0),
        "cache_creation_input_tokens": cache_creation
    })
}

/// Map an Anthropic citation object to the canonical one.
fn anthropic_citation(c: &Value) -> Option<Value> {
    let url = c.get("url").and_then(|x| x.as_str())?;
    if url.is_empty() {
        return None;
    }
    Some(server_tool::citation(
        url,
        c.get("title").and_then(|x| x.as_str()).unwrap_or(""),
        c.get("cited_text").and_then(|x| x.as_str()).unwrap_or(""),
    ))
}

/// Build a canonical server-tool call from an Anthropic `server_tool_use`.
fn anthropic_server_call(name: &str, id: &str, input: &Value) -> Option<Value> {
    let mut call = if name.starts_with("web_search") {
        server_tool::web_search(
            input.get("query").and_then(|x| x.as_str()).unwrap_or(""),
            vec![],
        )
    } else if name.starts_with("code_execution") {
        server_tool::code_execution(input.get("code").and_then(|x| x.as_str()).unwrap_or(""), "")
    } else {
        return None;
    };
    if !id.is_empty() {
        call["id"] = json!(id);
    }
    Some(call)
}

/// Merge a server-tool result block into the call it belongs to.
fn merge_server_result(call: &mut Value, block_ty: &str, content: &Value) {
    if block_ty == "web_search_tool_result" {
        let results: Vec<Value> = content
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|r| {
                        let url = r.get("url").and_then(|x| x.as_str())?;
                        Some(json!({
                            "url": url,
                            "title": r.get("title").cloned().unwrap_or(json!(""))
                        }))
                    })
                    .collect()
            })
            .unwrap_or_default();
        call["results"] = json!(results);
    } else {
        let output = content
            .get("stdout")
            .or_else(|| content.get("output"))
            .and_then(|x| x.as_str())
            .unwrap_or("");
        call["output"] = json!(output);
    }
}

/// Stateful translator for an Anthropic Messages SSE stream into OpenAI chunks.
pub struct AnthropicStream {
    id: String,
    model: String,
    role_sent: bool,
    tool_index: i64,
    block_ty: String,
    /// Signature fragments for the current thinking block. Anthropic sends
    /// them as `signature_delta` right before `content_block_stop`; they are
    /// attached to the block's `reasoning_details` entry emitted at stop.
    thinking_sig: String,
    /// Visible text of the current thinking block. The signature detail is
    /// emitted at stop with the whole block text, not just the signature, so
    /// a Chat client replaying `reasoning_details` has a complete block.
    thinking_text: String,
    /// `server_tool_use` block being assembled, as `(id, name)`.
    pending_server: Option<(String, String)>,
    /// `input_json_delta` fragments for that block.
    server_json: String,
    /// Complete `input` carried by `content_block_start`, when the upstream
    /// sends it whole instead of via deltas.
    server_input: Option<Value>,
    /// Completed server-tool call waiting for its result block.
    pending_server_call: Option<Value>,
    /// Result block content captured at `content_block_start`.
    pending_result: Option<(String, Value)>,
    in_tok: i64,
    out_tok: i64,
    cached: i64,
    cache_creation: i64,
    finish: Option<String>,
}

impl AnthropicStream {
    pub fn new(model: &str) -> Self {
        Self {
            id: gen_id("chatcmpl-"),
            model: model.to_string(),
            role_sent: false,
            tool_index: 0,
            block_ty: String::new(),
            thinking_sig: String::new(),
            thinking_text: String::new(),
            pending_server: None,
            server_json: String::new(),
            server_input: None,
            pending_server_call: None,
            pending_result: None,
            in_tok: 0,
            out_tok: 0,
            cached: 0,
            cache_creation: 0,
            finish: None,
        }
    }

    fn ensure_role(&mut self, out: &mut Vec<Value>) {
        if !self.role_sent {
            self.role_sent = true;
            out.push(chunk(
                &self.id,
                &self.model,
                json!({"role":"assistant"}),
                None,
            ));
        }
    }

    /// Handle one parsed Anthropic SSE `data:` payload. Returns OpenAI chunks.
    pub fn handle(&mut self, ev: &Value) -> Vec<Value> {
        let mut out = vec![];
        match ev.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "message_start" => {
                self.in_tok = ev
                    .pointer("/message/usage/input_tokens")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(0);
                self.cached = ev
                    .pointer("/message/usage/cache_read_input_tokens")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(0);
                self.cache_creation = ev
                    .pointer("/message/usage/cache_creation_input_tokens")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(0);
                self.ensure_role(&mut out);
            }
            "content_block_start" => {
                self.ensure_role(&mut out);
                let bt = ev
                    .pointer("/content_block/type")
                    .and_then(|x| x.as_str())
                    .unwrap_or("");
                self.block_ty = bt.to_string();
                self.thinking_sig.clear();
                self.thinking_text.clear();
                if bt == "thinking" {
                    if let Some(t) = ev
                        .pointer("/content_block/thinking")
                        .and_then(|x| x.as_str())
                    {
                        self.thinking_text.push_str(t);
                    }
                }
                if bt == "redacted_thinking" {
                    if let Some(data) = ev
                        .pointer("/content_block/data")
                        .and_then(|x| x.as_str())
                        .filter(|s| !s.is_empty())
                    {
                        out.push(chunk(
                            &self.id,
                            &self.model,
                            json!({"reasoning_details": [{
                                "type": "redacted_thinking",
                                "data": data
                            }]}),
                            None,
                        ));
                    }
                }
                if bt == "tool_use" {
                    let id = ev
                        .pointer("/content_block/id")
                        .and_then(|x| x.as_str())
                        .unwrap_or("");
                    let name = ev
                        .pointer("/content_block/name")
                        .and_then(|x| x.as_str())
                        .unwrap_or("");
                    out.push(chunk(
                        &self.id,
                        &self.model,
                        json!({"tool_calls":[{"index": self.tool_index, "id": id, "type": "function",
                            "function": {"name": name, "arguments": ""}}]}),
                        None,
                    ));
                }
                if bt == "server_tool_use" {
                    self.server_json.clear();
                    self.server_input = ev.pointer("/content_block/input").cloned();
                    let id = ev
                        .pointer("/content_block/id")
                        .and_then(|x| x.as_str())
                        .unwrap_or("");
                    let name = ev
                        .pointer("/content_block/name")
                        .and_then(|x| x.as_str())
                        .unwrap_or("");
                    self.pending_server = Some((id.to_string(), name.to_string()));
                }
                if bt == "web_search_tool_result" || bt == "code_execution_tool_result" {
                    let id = ev
                        .pointer("/content_block/tool_use_id")
                        .and_then(|x| x.as_str())
                        .unwrap_or("");
                    let content = ev
                        .pointer("/content_block/content")
                        .cloned()
                        .unwrap_or(json!([]));
                    self.pending_result = Some((id.to_string(), content));
                }
            }
            "content_block_delta" => {
                self.ensure_role(&mut out);
                let dt = ev
                    .pointer("/delta/type")
                    .and_then(|x| x.as_str())
                    .unwrap_or("");
                if dt == "text_delta" {
                    if let Some(t) = ev.pointer("/delta/text").and_then(|x| x.as_str()) {
                        if !t.is_empty() {
                            out.push(chunk(&self.id, &self.model, json!({"content": t}), None));
                        }
                    }
                } else if dt == "thinking_delta" {
                    if let Some(t) = ev.pointer("/delta/thinking").and_then(|x| x.as_str()) {
                        if !t.is_empty() {
                            self.thinking_text.push_str(t);
                            out.push(chunk(
                                &self.id,
                                &self.model,
                                json!({"reasoning_content": t}),
                                None,
                            ));
                        }
                    }
                } else if dt == "signature_delta" {
                    if let Some(sig) = ev.pointer("/delta/signature").and_then(|x| x.as_str()) {
                        self.thinking_sig.push_str(sig);
                    }
                } else if dt == "input_json_delta" {
                    if let Some(t) = ev.pointer("/delta/partial_json").and_then(|x| x.as_str()) {
                        if self.block_ty == "server_tool_use" {
                            self.server_json.push_str(t);
                        } else {
                            out.push(chunk(
                                &self.id,
                                &self.model,
                                json!({"tool_calls":[{"index": self.tool_index, "function": {"arguments": t}}]}),
                                None,
                            ));
                        }
                    }
                } else if dt == "citations_delta" {
                    if let Some(c) = ev.pointer("/delta/citation") {
                        if let Some(citation) = anthropic_citation(c) {
                            out.push(chunk(
                                &self.id,
                                &self.model,
                                json!({"citations": [citation]}),
                                None,
                            ));
                        }
                    }
                }
            }
            "content_block_stop" => {
                if self.block_ty == "thinking" && !self.thinking_sig.is_empty() {
                    let sig = wrap_anthropic_signature(&self.thinking_sig);
                    let text = std::mem::take(&mut self.thinking_text);
                    self.thinking_sig.clear();
                    out.push(chunk(
                        &self.id,
                        &self.model,
                        json!({"reasoning_details": [{
                            "type": "thinking",
                            "thinking": text,
                            "signature": sig
                        }]}),
                        None,
                    ));
                }
                // Only tool_use blocks consume a tool_calls index.
                if self.block_ty == "tool_use" {
                    self.tool_index += 1;
                }
                if self.block_ty == "server_tool_use" {
                    if let Some((id, name)) = self.pending_server.take() {
                        let input: Value = serde_json::from_str(&self.server_json)
                            .ok()
                            .filter(|v: &Value| v.as_object().is_some_and(|o| !o.is_empty()))
                            .or_else(|| self.server_input.take())
                            .unwrap_or(json!({}));
                        self.pending_server_call = anthropic_server_call(&name, &id, &input);
                    }
                    self.server_json.clear();
                    self.server_input = None;
                }
                if self.block_ty == "web_search_tool_result"
                    || self.block_ty == "code_execution_tool_result"
                {
                    let (id, content) = self
                        .pending_result
                        .take()
                        .unwrap_or_else(|| (String::new(), json!([])));
                    let mut call = self.pending_server_call.take().unwrap_or_else(|| {
                        if self.block_ty == "web_search_tool_result" {
                            server_tool::web_search("", vec![])
                        } else {
                            server_tool::code_execution("", "")
                        }
                    });
                    if !id.is_empty() {
                        call["id"] = json!(id);
                    }
                    merge_server_result(&mut call, &self.block_ty, &content);
                    out.push(chunk(
                        &self.id,
                        &self.model,
                        json!({"server_tool_calls": [call]}),
                        None,
                    ));
                }
                self.block_ty.clear();
                self.thinking_sig.clear();
                self.thinking_text.clear();
            }
            "message_delta" => {
                if let Some(r) = ev.pointer("/delta/stop_reason").and_then(|x| x.as_str()) {
                    self.finish = Some(finish_from_anthropic(r).to_string());
                }
                if let Some(o) = ev.pointer("/usage/output_tokens").and_then(|x| x.as_i64()) {
                    self.out_tok = o;
                }
            }
            "message_stop" => {
                self.ensure_role(&mut out);
                // A server tool without a result block still deserves a call
                // item; dropping it would hide that the provider ran it.
                if let Some(call) = self.pending_server_call.take() {
                    out.push(chunk(
                        &self.id,
                        &self.model,
                        json!({"server_tool_calls": [call]}),
                        None,
                    ));
                }
                let fr = self.finish.clone().unwrap_or_else(|| "stop".to_string());
                let mut c = chunk(&self.id, &self.model, json!({}), Some(&fr));
                // Anthropic's `input_tokens` excludes cache reads; OpenAI's
                // `prompt_tokens` includes them.
                c["usage"] = chat_usage(
                    self.in_tok + self.cached + self.cache_creation,
                    self.out_tok,
                    self.cached,
                    0,
                );
                c["cache_creation_input_tokens"] = json!(self.cache_creation);
                out.push(c);
            }
            _ => {}
        }
        out
    }
}

// ---------------------------------------------------------------------------
// OpenAI chat -> Responses API (egress)
// ---------------------------------------------------------------------------

/// Parameters an Anthropic upstream is likely to reject because thinking is
/// on.
///
/// Unlike [`chat_to_anthropic_loss`] these are **not** dropped — they are
/// forwarded. Removing `temperature: 0` would let the caller believe the
/// model is deterministic when it is not, which is a silent behaviour
/// change; making them an error would break the Anthropic-protocol proxies
/// that do accept them. So the gateway passes them through and says so.
///
/// Anthropic's docs state it as "you can't set a custom temperature while
/// thinking is on, so it uses the default of 1"; upstreams that enforce it
/// answer 400.
///
/// Only values differing from Anthropic's defaults (`temperature` 1,
/// `top_p` 1) are reported. Forwarding the default is accepted, and warning
/// on it would fire on essentially every request.
pub fn chat_to_anthropic_thinking_conflict(req: &Value) -> Vec<&'static str> {
    let thinking = req
        .get("reasoning_effort")
        .and_then(|x| x.as_str())
        .is_some_and(|e| e != "none");
    if !thinking {
        return vec![];
    }
    let mut out = vec![];
    for k in ["temperature", "top_p"] {
        if req
            .get(k)
            .and_then(|v| v.as_f64())
            .is_some_and(|v| v != 1.0)
        {
            out.push(k);
        }
    }
    out
}

#[derive(Default)]
struct AssistantSegment {
    parts: Vec<Value>,
    tools: Vec<Value>,
    reasoning: String,
    details: Vec<Value>,
}

impl AssistantSegment {
    fn push_message(self, messages: &mut Vec<Value>) {
        if self.parts.is_empty()
            && self.tools.is_empty()
            && self.reasoning.is_empty()
            && self.details.is_empty()
        {
            return;
        }
        let content = if self.parts.is_empty() {
            Value::Null
        } else if self
            .parts
            .iter()
            .all(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
        {
            // Text alongside tool calls stays a string for Chat compatibility.
            json!(content_text(&Value::Array(self.parts)))
        } else {
            Value::Array(self.parts)
        };
        let mut msg = json!({"role": "assistant", "content": content});
        if !self.reasoning.is_empty() {
            msg["reasoning_content"] = json!(self.reasoning);
        }
        if !self.details.is_empty() {
            msg["reasoning_details"] = Value::Array(self.details);
        }
        if !self.tools.is_empty() {
            msg["tool_calls"] = Value::Array(self.tools);
        }
        messages.push(msg);
    }
}

/// Split one Anthropic assistant message into contiguous Chat assistant
/// messages. Chat keeps text, tool calls and reasoning in separate fields, so
/// a single message would regroup the blocks (all thinking first, then all
/// tool calls). Splitting preserves the original order for the next
/// translation pass.
fn push_anthropic_assistant_segments(messages: &mut Vec<Value>, blocks: &[Value]) {
    let mut seg = AssistantSegment::default();
    for b in blocks {
        let ty = b.get("type").and_then(|t| t.as_str()).unwrap_or("text");
        match ty {
            "thinking" => {
                if !seg.parts.is_empty() || !seg.tools.is_empty() {
                    std::mem::take(&mut seg).push_message(messages);
                }
                let thinking = b.get("thinking").and_then(|x| x.as_str()).unwrap_or("");
                if !thinking.is_empty() {
                    seg.reasoning.push_str(thinking);
                }
                if let Some(sig) = b
                    .get("signature")
                    .and_then(|x| x.as_str())
                    .filter(|s| !s.is_empty())
                {
                    seg.details.push(json!({
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": wrap_anthropic_signature(sig)
                    }));
                }
            }
            "redacted_thinking" => {
                if !seg.parts.is_empty() || !seg.tools.is_empty() {
                    std::mem::take(&mut seg).push_message(messages);
                }
                if let Some(data) = b
                    .get("data")
                    .and_then(|x| x.as_str())
                    .filter(|s| !s.is_empty())
                {
                    seg.details
                        .push(json!({"type": "redacted_thinking", "data": data}));
                }
            }
            "text" => {
                if !seg.tools.is_empty() {
                    std::mem::take(&mut seg).push_message(messages);
                }
                if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
                    if !t.is_empty() {
                        seg.parts.push(json!({"type": "text", "text": t}));
                    }
                }
            }
            "image" => {
                if !seg.tools.is_empty() {
                    std::mem::take(&mut seg).push_message(messages);
                }
                if let Some(img) = anthropic_image_to_openai(b) {
                    seg.parts.push(img);
                }
            }
            "tool_use" => {
                let id = b.get("id").and_then(|x| x.as_str()).unwrap_or("");
                let name = b.get("name").and_then(|x| x.as_str()).unwrap_or("");
                let input = b.get("input").cloned().unwrap_or(json!({}));
                let args = serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());
                seg.tools.push(json!({
                    "id": id, "type": "function",
                    "function": {"name": name, "arguments": args}
                }));
            }
            "tool_result" => {
                // A tool_result inside an assistant turn is malformed but
                // still worth preserving as a Chat tool message.
                std::mem::take(&mut seg).push_message(messages);
                let id = b.get("tool_use_id").and_then(|x| x.as_str()).unwrap_or("");
                let content = b.get("content").cloned().unwrap_or(json!(""));
                messages.push(json!({
                    "role": "tool", "tool_call_id": id,
                    "content": content_text(&content)
                }));
            }
            other => warn_dropped_block("anthropic_to_chat_request", other),
        }
    }
    seg.push_message(messages);
}

pub fn anthropic_to_chat_request(req: &Value) -> Value {
    let mut messages: Vec<Value> = vec![];

    match req.get("system") {
        Some(Value::String(s)) => {
            if !s.is_empty() {
                messages.push(json!({"role": "system", "content": s}));
            }
        }
        Some(Value::Array(blocks)) => {
            let mut text = String::new();
            for b in blocks {
                if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                }
            }
            if !text.is_empty() {
                messages.push(json!({"role": "system", "content": text}));
            }
        }
        _ => {}
    }

    if let Some(list) = req.get("messages").and_then(|m| m.as_array()) {
        for m in list {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            match m.get("content") {
                Some(Value::String(s)) => {
                    messages.push(json!({"role": role, "content": s}));
                }
                Some(Value::Array(blocks)) => {
                    if role == "assistant" {
                        push_anthropic_assistant_segments(&mut messages, blocks);
                        continue;
                    }
                    let mut parts: Vec<Value> = vec![];
                    let mut tool_calls: Vec<Value> = vec![];
                    let mut reasoning = String::new();
                    let mut reasoning_details: Vec<Value> = vec![];
                    for b in blocks {
                        match b.get("type").and_then(|t| t.as_str()).unwrap_or("text") {
                            "text" => {
                                if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
                                    if !t.is_empty() {
                                        parts.push(json!({"type": "text", "text": t}));
                                    }
                                }
                            }
                            "image" => {
                                if let Some(img) = anthropic_image_to_openai(b) {
                                    parts.push(img);
                                }
                            }
                            "tool_use" => {
                                let id = b.get("id").and_then(|x| x.as_str()).unwrap_or("");
                                let name = b.get("name").and_then(|x| x.as_str()).unwrap_or("");
                                let input = b.get("input").cloned().unwrap_or(json!({}));
                                let args =
                                    serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());
                                tool_calls.push(json!({
                                    "id": id, "type": "function",
                                    "function": {"name": name, "arguments": args}
                                }));
                            }
                            "tool_result" => {
                                // tool_result answers an assistant tool_use and
                                // becomes a Chat `tool` message of its own.
                                let id =
                                    b.get("tool_use_id").and_then(|x| x.as_str()).unwrap_or("");
                                let content = b.get("content").cloned().unwrap_or(json!(""));
                                messages.push(json!({
                                    "role": "tool", "tool_call_id": id,
                                    "content": content_text(&content)
                                }));
                            }
                            "thinking" => {
                                let thinking =
                                    b.get("thinking").and_then(|x| x.as_str()).unwrap_or("");
                                if !thinking.is_empty() {
                                    reasoning.push_str(thinking);
                                }
                                if let Some(sig) = b
                                    .get("signature")
                                    .and_then(|x| x.as_str())
                                    .filter(|s| !s.is_empty())
                                {
                                    reasoning_details.push(json!({
                                        "type": "thinking",
                                        "thinking": thinking,
                                        "signature": wrap_anthropic_signature(sig)
                                    }));
                                }
                            }
                            "redacted_thinking" => {
                                if let Some(data) = b.get("data").and_then(|x| x.as_str()) {
                                    if !data.is_empty() {
                                        reasoning_details.push(
                                            json!({"type": "redacted_thinking", "data": data}),
                                        );
                                    }
                                }
                            }
                            other => warn_dropped_block("anthropic_to_chat_request", other),
                        }
                    }
                    if !tool_calls.is_empty() {
                        let mut msg = json!({"role": "assistant", "content": Value::Null});
                        if !reasoning.is_empty() {
                            msg["reasoning_content"] = json!(reasoning);
                        }
                        if !reasoning_details.is_empty() {
                            msg["reasoning_details"] = Value::Array(reasoning_details.clone());
                        }
                        if !parts.is_empty() {
                            // text alongside tool calls belongs in `content`
                            msg["content"] = json!(content_text(&Value::Array(parts.clone())));
                        }
                        msg["tool_calls"] = Value::Array(tool_calls);
                        messages.push(msg);
                    } else if !parts.is_empty()
                        || !reasoning.is_empty()
                        || !reasoning_details.is_empty()
                    {
                        // Plain-text stays a string (best Chat compatibility);
                        // structured parts (images) keep their array form.
                        let content = if parts
                            .iter()
                            .all(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
                        {
                            json!(content_text(&Value::Array(parts.clone())))
                        } else {
                            Value::Array(parts.clone())
                        };
                        let mut msg = json!({"role": role, "content": content});
                        if !reasoning.is_empty() {
                            msg["reasoning_content"] = json!(reasoning);
                        }
                        if !reasoning_details.is_empty() {
                            msg["reasoning_details"] = Value::Array(reasoning_details);
                        }
                        messages.push(msg);
                    }
                }
                _ => {}
            }
        }
    }

    let mut out = json!({
        "model": req.get("model").cloned().unwrap_or(json!("")),
        "messages": messages
    });
    if let Some(mt) = req.get("max_tokens") {
        out["max_tokens"] = mt.clone();
    }
    for k in ["temperature", "top_p"] {
        if let Some(v) = req.get(k) {
            out[k] = v.clone();
        }
    }
    if let Some(k) = req.get("top_k") {
        out["x_nervogate"] = json!({"top_k": k.clone()});
    }
    if let Some(md) = req.get("metadata") {
        out["metadata"] = md.clone();
    }
    if let Some(seq) = req.get("stop_sequences") {
        out["stop"] = seq.clone();
    }
    if let Some(tools) = req.get("tools").and_then(|t| t.as_array()) {
        if !tools.is_empty() {
            let ot: Vec<Value> = tools
                .iter()
                .map(|t| {
                    let ty = t.get("type").and_then(|x| x.as_str()).unwrap_or("");
                    if ty.starts_with("web_search") {
                        let domains = t
                            .get("allowed_domains")
                            .and_then(|x| x.as_array())
                            .map(|a| a.iter().filter(|d| d.is_string()).cloned().collect())
                            .unwrap_or_default();
                        return server_tool::request_entry(server_tool::WEB_SEARCH, domains);
                    }
                    if ty.starts_with("code_execution") {
                        return server_tool::request_entry(server_tool::CODE_EXECUTION, vec![]);
                    }
                    let name = t.get("name").and_then(|x| x.as_str()).unwrap_or("");
                    json!({"type":"function","function":{
                        "name": name,
                        "description": t.get("description").cloned().unwrap_or(json!("")),
                        "parameters": t.get("input_schema").cloned().unwrap_or(json!({"type":"object"}))
                    }})
                })
                .collect();
            out["tools"] = Value::Array(ot);
        }
    }
    if let Some(tc) = req.get("tool_choice") {
        let mapped = match tc.get("type").and_then(|x| x.as_str()).unwrap_or("auto") {
            "auto" => Some(json!("auto")),
            "any" => Some(json!("required")),
            "tool" => tc
                .get("name")
                .and_then(|x| x.as_str())
                .map(|name| json!({"type": "function", "function": {"name": name}})),
            "none" => Some(json!("none")),
            _ => None,
        };
        if let Some(m) = mapped {
            out["tool_choice"] = m;
        }
    }
    if let Some(e) = req
        .pointer("/output_config/effort")
        .and_then(|x| x.as_str())
    {
        // Adaptive round-trip: effort rides through `output_config`.
        out["reasoning_effort"] = json!(e);
    } else if let Some(budget) = req
        .pointer("/thinking/budget_tokens")
        .and_then(|x| x.as_i64())
    {
        let effort = match budget {
            0..=2047 => "low",
            2048..=8191 => "medium",
            _ => "high",
        };
        out["reasoning_effort"] = json!(effort);
    } else if req.pointer("/thinking/type").and_then(|x| x.as_str()) == Some("adaptive") {
        // Bare adaptive (no effort, no budget): keep reasoning on for
        // chat/responses upstreams at the middle level.
        out["reasoning_effort"] = json!("medium");
    }
    out
}

/// Chat completion -> Anthropic Messages response object.
pub fn chat_to_anthropic_response(chat: &Value, model: &str) -> Value {
    let msg = chat
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or(json!({}));
    let mut details: Vec<(Option<usize>, Value)> = vec![];
    if let Some(ds) = msg.get("reasoning_details").and_then(|x| x.as_array()) {
        for d in ds {
            if let Some(entry) = reasoning_block_with_position(d, true) {
                details.push(entry);
            }
        }
    }
    // `reasoning_content` without a signature is deliberately not turned
    // into a thinking block: Anthropic rejects `signature: ""` (400), and a
    // missing block is the documented safe degradation.
    let text = msg.get("content").and_then(|x| x.as_str()).unwrap_or("");
    let refusal = msg.get("refusal").and_then(|x| x.as_str()).unwrap_or("");
    let text = if !text.is_empty() {
        text
    } else if !refusal.is_empty() {
        // Anthropic reports a refusal through `stop_reason: "refusal"` with
        // the explanation in the text block; keeping the text preserves it.
        refusal
    } else {
        ""
    };
    let mut tool_blocks: Vec<Value> = vec![];
    if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tcs {
            let id = tc.get("id").and_then(|x| x.as_str()).unwrap_or("");
            let name = tc
                .pointer("/function/name")
                .and_then(|x| x.as_str())
                .unwrap_or("");
            let args = tc
                .pointer("/function/arguments")
                .and_then(|x| x.as_str())
                .unwrap_or("{}");
            let input: Value = serde_json::from_str(args).unwrap_or(json!({}));
            tool_blocks.push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
        }
    }
    let mut blocks: Vec<Value> = vec![];
    if let Some(calls) = msg.get("server_tool_calls").and_then(|x| x.as_array()) {
        for call in calls {
            match call.get("type").and_then(|x| x.as_str()) {
                Some(server_tool::WEB_SEARCH) => {
                    let id = call
                        .get("id")
                        .and_then(|x| x.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| gen_id("srvtoolu_"));
                    let query = call.get("query").cloned().unwrap_or(json!(""));
                    blocks.push(json!({
                        "type": "server_tool_use", "id": id,
                        "name": "web_search", "input": {"query": query}
                    }));
                    let results: Vec<Value> = call
                        .get("results")
                        .and_then(|x| x.as_array())
                        .map(|rs| {
                            rs.iter()
                                .map(|r| {
                                    json!({
                                        "type": "web_search_result",
                                        "url": r.get("url").cloned().unwrap_or(json!("")),
                                        "title": r.get("title").cloned().unwrap_or(json!(""))
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    blocks.push(json!({
                        "type": "web_search_tool_result", "tool_use_id": id,
                        "content": results
                    }));
                }
                Some(server_tool::CODE_EXECUTION) => {
                    let id = call
                        .get("id")
                        .and_then(|x| x.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| gen_id("srvtoolu_"));
                    let code = call.get("code").cloned().unwrap_or(json!(""));
                    blocks.push(json!({
                        "type": "server_tool_use", "id": id,
                        "name": "code_execution", "input": {"code": code}
                    }));
                    blocks.push(json!({
                        "type": "code_execution_tool_result", "tool_use_id": id,
                        "content": {
                            "type": "code_execution_result",
                            "stdout": call.get("output").cloned().unwrap_or(json!("")),
                            "stderr": ""
                        }
                    }));
                }
                _ => {}
            }
        }
    }
    blocks.extend(ordered_assistant_blocks(details, text, tool_blocks));

    // Web-search citations ride on the text block. Anthropic streams them
    // alongside the text, but a non-streaming reply carries one array.
    let mut citations = msg
        .get("citations")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    server_tool::locate_citations(&mut citations, text);
    if !citations.is_empty() {
        let citation_blocks: Vec<Value> = citations
            .iter()
            .map(|c| {
                json!({
                    "type": "web_search_result_location",
                    "url": c.get("url").cloned().unwrap_or(json!("")),
                    "title": c.get("title").cloned().unwrap_or(json!("")),
                    "cited_text": c.get("cited_text").cloned().unwrap_or(json!(""))
                })
            })
            .collect();
        for b in &mut blocks {
            if b.get("type").and_then(|x| x.as_str()) == Some("text") {
                b["citations"] = Value::Array(citation_blocks);
                break;
            }
        }
    }
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": ""}));
    }

    let finish = chat
        .pointer("/choices/0/finish_reason")
        .and_then(|x| x.as_str())
        .unwrap_or("stop");
    // Canonical `prompt_tokens` already includes cached tokens; Anthropic
    // reports cache reads and writes separately.
    let (p, c, cached, _) = chat_usage_parts(chat.get("usage"));
    let cache_creation = chat
        .get("cache_creation_input_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let input = (p - cached - cache_creation).max(0);

    json!({
        "id": gen_id("msg_"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": anthropic_stop(finish),
        "stop_sequence": Value::Null,
        "usage": anthropic_usage_full(input, c, cache_creation, cached)
    })
}

// ---------------------------------------------------------------------------
// Chat chunks -> Anthropic Messages SSE stream (egress)
// ---------------------------------------------------------------------------

/// Stateful translator for OpenAI chat chunks into Anthropic Messages SSE
/// events (`message_start` ... `message_stop`). Each returned [`SseEvent`]
/// carries an `event:` name, which is mandatory for Anthropic clients.
pub struct ChatToAnthropicStream {
    msg_id: String,
    model: String,
    started: bool,
    terminated: bool,
    index: usize,
    /// Index of the currently open text block.
    open: Option<usize>,
    /// Thinking text held back until its signature arrives. Anthropic rejects
    /// a `thinking` block with an empty `signature`, and the signature is only
    /// known at the end of the upstream block, so the block is emitted whole
    /// in one go.
    pending_thinking: String,
    /// Tool calls seen so far, in first-seen order so parallel calls keep
    /// their relative ordering in the Anthropic message.
    tools: Vec<(ToolKey, PendingTool)>,
    /// Full answer text so far, used to derive `cited_text` for citations
    /// that only carry offsets.
    text: String,
    /// Citations that arrived before the text block they annotate.
    pending_citations: Vec<Value>,
}

impl ChatToAnthropicStream {
    pub fn new(model: &str) -> Self {
        Self {
            msg_id: gen_id("msg_"),
            model: model.to_string(),
            started: false,
            terminated: false,
            index: 0,
            open: None,
            pending_thinking: String::new(),
            tools: vec![],
            text: String::new(),
            pending_citations: vec![],
        }
    }

    /// Emit every buffered tool call as a complete, contiguous Anthropic
    /// content block. Each call becomes exactly one `tool_use` block
    /// carrying its full argument JSON, which is what strict clients
    /// require (they reject interleaved and split blocks).
    ///
    /// Returns true when at least one block was emitted, so the caller can
    /// skip the "message needs a content block" fallback.
    fn flush_tools(&mut self, out: &mut Vec<SseEvent>) -> bool {
        let tools = std::mem::take(&mut self.tools);
        let emitted = !tools.is_empty();
        for (_, t) in tools {
            self.close_open(out);
            let ci = self.index;
            self.index += 1;
            let args = if t.args.trim().is_empty() {
                "{}".to_string()
            } else {
                t.args.clone()
            };
            out.push(SseEvent::ev(
                "content_block_start",
                json!({
                    "type": "content_block_start", "index": ci,
                    "content_block": {"type": "tool_use", "id": t.id, "name": t.name, "input": {}}
                }),
            ));
            out.push(SseEvent::ev(
                "content_block_delta",
                json!({
                    "type": "content_block_delta", "index": ci,
                    "delta": {"type": "input_json_delta", "partial_json": args}
                }),
            ));
            out.push(SseEvent::ev(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": ci}),
            ));
        }
        emitted
    }

    fn start(&mut self, out: &mut Vec<SseEvent>) {
        if !self.started {
            self.started = true;
            out.push(SseEvent::ev(
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": self.msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": {"input_tokens": 0, "output_tokens": 0}
                    }
                }),
            ));
        }
    }

    /// Open a text block if no block is currently open.
    fn ensure_block(&mut self, out: &mut Vec<SseEvent>) {
        if self.open.is_none() {
            let index = self.index;
            self.index += 1;
            out.push(SseEvent::ev(
                "content_block_start",
                json!({
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "text", "text": ""}
                }),
            ));
            self.open = Some(index);
            self.flush_citations(out);
        }
    }

    /// Render one canonical citation as an Anthropic location object.
    fn citation_block(&self, c: &Value) -> Value {
        let mut cited = c
            .get("cited_text")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        if cited.is_empty() {
            if let (Some(start), Some(end)) = (
                c.get("start_index").and_then(|x| x.as_u64()),
                c.get("end_index").and_then(|x| x.as_u64()),
            ) {
                cited = self
                    .text
                    .chars()
                    .skip(start as usize)
                    .take(end.saturating_sub(start) as usize)
                    .collect();
            }
        }
        json!({
            "type": "web_search_result_location",
            "url": c.get("url").cloned().unwrap_or(json!("")),
            "title": c.get("title").cloned().unwrap_or(json!("")),
            "cited_text": cited
        })
    }

    /// Emit buffered citations, or hold them until a text block opens.
    fn flush_citations(&mut self, out: &mut Vec<SseEvent>) {
        let Some(index) = self.open else {
            return;
        };
        for c in std::mem::take(&mut self.pending_citations) {
            out.push(SseEvent::ev(
                "content_block_delta",
                json!({
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "citations_delta", "citation": self.citation_block(&c)}
                }),
            ));
        }
    }

    fn queue_citations(&mut self, out: &mut Vec<SseEvent>, citations: &[Value]) {
        self.pending_citations.extend(citations.iter().cloned());
        self.flush_citations(out);
    }

    /// Emit a completed server-tool call as its `server_tool_use` block plus
    /// the matching result block, both fully deltad and stopped.
    fn emit_server(&mut self, out: &mut Vec<SseEvent>, call: &Value) {
        self.close_open(out);
        let id = call
            .get("id")
            .and_then(|x| x.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| gen_id("srvtoolu_"));
        let is_search = call.get("type").and_then(|x| x.as_str()) == Some(server_tool::WEB_SEARCH);
        let name = if is_search {
            "web_search"
        } else {
            "code_execution"
        };
        let input = if is_search {
            json!({"query": call.get("query").cloned().unwrap_or(json!(""))})
        } else {
            json!({"code": call.get("code").cloned().unwrap_or(json!(""))})
        };

        let index = self.index;
        self.index += 1;
        out.push(SseEvent::ev(
            "content_block_start",
            json!({
                "type": "content_block_start", "index": index,
                "content_block": {"type": "server_tool_use", "id": id, "name": name, "input": {}}
            }),
        ));
        out.push(SseEvent::ev(
            "content_block_delta",
            json!({
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": input.to_string()}
            }),
        ));
        out.push(SseEvent::ev(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        ));

        let index = self.index;
        self.index += 1;
        let result = if is_search {
            json!({
                "type": "web_search_tool_result", "tool_use_id": id,
                "content": call.get("results").and_then(|x| x.as_array()).map(|rs| {
                    rs.iter().map(|r| json!({
                        "type": "web_search_result",
                        "url": r.get("url").cloned().unwrap_or(json!("")),
                        "title": r.get("title").cloned().unwrap_or(json!(""))
                    })).collect::<Vec<_>>()
                }).unwrap_or_default()
            })
        } else {
            json!({
                "type": "code_execution_tool_result", "tool_use_id": id,
                "content": {
                    "type": "code_execution_result",
                    "stdout": call.get("output").cloned().unwrap_or(json!("")),
                    "stderr": ""
                }
            })
        };
        out.push(SseEvent::ev(
            "content_block_start",
            json!({"type": "content_block_start", "index": index, "content_block": result}),
        ));
        out.push(SseEvent::ev(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        ));
    }

    fn close_open(&mut self, out: &mut Vec<SseEvent>) {
        if let Some(index) = self.open.take() {
            out.push(SseEvent::ev(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": index}),
            ));
        }
    }

    /// Emit one complete thinking block. Called only once the upstream
    /// signature is known, because an empty signature is a hard 400.
    fn flush_thinking(&mut self, out: &mut Vec<SseEvent>, signature: &str) {
        if signature.is_empty() {
            self.pending_thinking.clear();
            return;
        }
        self.close_open(out);
        let index = self.index;
        self.index += 1;
        let text = std::mem::take(&mut self.pending_thinking);
        out.push(SseEvent::ev(
            "content_block_start",
            json!({
                "type": "content_block_start", "index": index,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""}
            }),
        ));
        if !text.is_empty() {
            out.push(SseEvent::ev(
                "content_block_delta",
                json!({
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "thinking_delta", "thinking": text}
                }),
            ));
        }
        out.push(SseEvent::ev(
            "content_block_delta",
            json!({
                "type": "content_block_delta", "index": index,
                "delta": {"type": "signature_delta", "signature": signature}
            }),
        ));
        out.push(SseEvent::ev(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        ));
    }

    fn emit_redacted(&mut self, out: &mut Vec<SseEvent>, data: &str) {
        if data.is_empty() {
            return;
        }
        self.pending_thinking.clear();
        self.close_open(out);
        let index = self.index;
        self.index += 1;
        out.push(SseEvent::ev(
            "content_block_start",
            json!({
                "type": "content_block_start", "index": index,
                "content_block": {"type": "redacted_thinking", "data": data}
            }),
        ));
        out.push(SseEvent::ev(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        ));
    }

    fn terminal(&mut self, stop_reason: &str, out: &mut Vec<SseEvent>) {
        // Unsigned thinking is not replayable; drop it rather than emitting a
        // block Anthropic will reject.
        self.pending_thinking.clear();
        // Tool blocks already satisfy Anthropic's "at least one content
        // block" rule, so only add an empty text block when there are none.
        let had_tools = self.flush_tools(out);
        if !had_tools {
            self.ensure_block(out);
        }
        self.close_open(out);
        self.terminated = true;
        out.push(SseEvent::ev(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": Value::Null},
                "usage": anthropic_usage(0, 0, 0)
            }),
        ));
        out.push(SseEvent::ev(
            "message_stop",
            json!({"type": "message_stop"}),
        ));
    }

    /// Handle one canonical OpenAI chat chunk. Returns Anthropic SSE events.
    pub fn handle(&mut self, ch: &Value) -> Vec<SseEvent> {
        let mut out = vec![];
        self.start(&mut out);

        let choice = ch.pointer("/choices/0").cloned().unwrap_or(json!({}));
        let delta = choice.get("delta").cloned().unwrap_or(json!({}));

        if let Some(t) = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(|x| x.as_str())
        {
            if !t.is_empty() {
                self.pending_thinking.push_str(t);
            }
        }

        // A `reasoning_details` delta carries the opaque signature. The
        // thinking block is flushed here, once the signature is available.
        if let Some(details) = delta.get("reasoning_details").and_then(|x| x.as_array()) {
            for d in details {
                let Some(d) = normalize_reasoning_detail(d) else {
                    continue;
                };
                match reasoning_kind(&d) {
                    Some(ReasoningKind::Thinking) => {
                        let text = reasoning_text(&d);
                        if !text.is_empty() && self.pending_thinking.is_empty() {
                            self.pending_thinking.push_str(text);
                        }
                        if let Some(sig) = d
                            .get("signature")
                            .and_then(|x| x.as_str())
                            .and_then(anthropic_signature_for_wire)
                        {
                            // A blob that only arrives in the terminal event
                            // after text has started cannot be placed before
                            // that text any more. Drop it rather than emit an
                            // out-of-order thinking block.
                            if self.open.is_some() {
                                self.pending_thinking.clear();
                            } else {
                                self.flush_thinking(&mut out, sig);
                            }
                        }
                    }
                    Some(ReasoningKind::Redacted) => {
                        if let Some(data) = d.get("data").and_then(|x| x.as_str()) {
                            if self.open.is_some() {
                                continue;
                            }
                            self.emit_redacted(&mut out, data);
                        }
                    }
                    None => {}
                }
            }
        }

        if let Some(t) = delta.get("content").and_then(|x| x.as_str()) {
            if !t.is_empty() {
                // Text cannot come before its thinking block: a buffered
                // block without a signature is not replayable.
                self.pending_thinking.clear();
                let ci = match self.open {
                    Some(ci) => ci,
                    None => {
                        self.close_open(&mut out);
                        let ci = self.index;
                        self.index += 1;
                        out.push(SseEvent::ev(
                            "content_block_start",
                            json!({
                                "type": "content_block_start", "index": ci,
                                "content_block": {"type": "text", "text": ""}
                            }),
                        ));
                        self.open = Some(ci);
                        self.flush_citations(&mut out);
                        ci
                    }
                };
                self.text.push_str(t);
                out.push(SseEvent::ev(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta", "index": ci,
                        "delta": {"type": "text_delta", "text": t}
                    }),
                ));
            }
        }

        // Citations annotate the text that precedes them, so they must be
        // flushed while that block is still open.
        if let Some(citations) = delta.get("citations").and_then(|x| x.as_array()) {
            self.queue_citations(&mut out, citations);
        }

        if let Some(calls) = delta.get("server_tool_calls").and_then(|x| x.as_array()) {
            for call in calls {
                self.emit_server(&mut out, call);
            }
        }

        // Anthropic's SSE requires each content block to be started, fully
        // deltad and stopped before the next one opens, but Chat streams may
        // interleave arguments across parallel tool calls. Tool blocks are
        // therefore accumulated per upstream `tool_calls[].index` and
        // emitted contiguously at the end of the stream; text and thinking
        // still stream through live.
        if let Some(tcs) = delta.get("tool_calls").and_then(|x| x.as_array()) {
            for tc in tcs {
                let pos = match tool_slot(tc, &self.tools) {
                    Some(key) => match self.tools.iter().position(|(k, _)| *k == key) {
                        Some(p) => p,
                        None => {
                            self.tools.push((key, PendingTool::default()));
                            self.tools.len() - 1
                        }
                    },
                    None => continue,
                };
                let entry = &mut self.tools[pos].1;
                // A later chunk may carry the field the first one omitted
                // (upstreams split `id` and `name` across chunks).
                if let Some(id) = tc
                    .get("id")
                    .and_then(|x| x.as_str())
                    .filter(|s| !s.is_empty())
                {
                    entry.id = id.to_string();
                }
                if let Some(n) = tc
                    .pointer("/function/name")
                    .and_then(|x| x.as_str())
                    .filter(|s| !s.is_empty())
                {
                    entry.name = n.to_string();
                }
                if let Some(a) = tc.pointer("/function/arguments").and_then(|x| x.as_str()) {
                    entry.args.push_str(a);
                }
            }
        }

        if let Some(fr) = choice.get("finish_reason").and_then(|x| x.as_str()) {
            let had_tools = self.flush_tools(&mut out);
            if !had_tools {
                self.ensure_block(&mut out);
            }
            self.close_open(&mut out);
            let (p, c, cached, _) = chat_usage_parts(ch.get("usage"));
            let cache_creation = ch
                .get("cache_creation_input_tokens")
                .and_then(|x| x.as_i64())
                .unwrap_or(0);
            // Canonical `prompt_tokens` includes cached tokens; Anthropic
            // reports cache reads and writes separately.
            let input = (p - cached - cache_creation).max(0);
            self.terminated = true;
            out.push(SseEvent::ev(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": anthropic_stop(fr), "stop_sequence": Value::Null},
                    "usage": anthropic_usage_full(input, c, cache_creation, cached)
                }),
            ));
            out.push(SseEvent::ev(
                "message_stop",
                json!({"type": "message_stop"}),
            ));
        }
        out
    }

    /// Terminal events for an upstream that closed without a finish chunk.
    /// Without them the client hangs on a connection that already closed.
    /// `None` when a finish chunk already emitted `message_stop`.
    pub fn finish(&mut self) -> Option<Vec<SseEvent>> {
        if self.terminated {
            return None;
        }
        let mut out = vec![];
        self.start(&mut out);
        self.terminal("end_turn", &mut out);
        Some(out)
    }

    /// Close the open content block before an error event so the client sees
    /// a well-formed stream instead of an abandoned block index.
    pub fn abort(&mut self) -> Vec<SseEvent> {
        if self.terminated {
            return vec![];
        }
        let mut out = vec![];
        self.start(&mut out);
        self.pending_thinking.clear();
        self.flush_tools(&mut out);
        self.close_open(&mut out);
        out
    }
}
