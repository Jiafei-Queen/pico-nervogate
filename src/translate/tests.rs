use serde_json::{json, Value};

use crate::config::ThinkingMode;
use super::*;

fn types(events: &[SseEvent]) -> Vec<String> {
    events
        .iter()
        .map(|e| {
            e.data
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("?")
                .to_string()
        })
        .collect()
}

// ---- framing ----------------------------------------------------------

#[test]
fn framing_data_only_vs_event_named() {
    assert_eq!(frame_data(&json!({"a": 1})), "data: {\"a\":1}\n\n");
    assert_eq!(
        frame_event("message_stop", &json!({"type": "message_stop"})),
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    );
    let ev = SseEvent::ev("x", json!({"type": "y"}));
    assert!(ev.frame().starts_with("event: x\n"));
}

// ---- chat -> anthropic ----------------------------------------------

#[test]
fn tool_results_merge_into_one_user_message() {
    let req = json!({
        "model": "m",
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "user", "content": "next"}
        ]
    });
    let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
    let msgs = out.get("messages").unwrap().as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "assistant");
    assert_eq!(msgs[0]["content"].as_array().unwrap().len(), 2);
    assert_eq!(msgs[1]["role"], "user");
    let blocks = msgs[1]["content"].as_array().unwrap();
    assert_eq!(blocks.len(), 3);
    assert_eq!(blocks[0]["type"], "tool_result");
    assert_eq!(blocks[0]["tool_use_id"], "c1");
    assert_eq!(blocks[1]["tool_use_id"], "c2");
    assert_eq!(blocks[2]["type"], "text");
}

#[test]
fn tool_choice_function_form_and_none() {
    let req = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": {"type": "function", "function": {"name": "f"}}
    });
    let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
    assert_eq!(out["tool_choice"], json!({"type": "tool", "name": "f"}));
    assert!(out.get("tools").is_some());

    let req = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": "none"
    });
    let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
    assert!(out.get("tools").is_none());
    assert!(out.get("tool_choice").is_none());
}

#[test]
fn reasoning_effort_maps_to_thinking() {
    let req = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high"
    });
    let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
    assert_eq!(out["thinking"], json!({"type": "adaptive"}));
    assert_eq!(out["output_config"], json!({"effort": "high"}));
}

#[test]
fn reasoning_effort_enabled_mode_keeps_legacy_shape() {
    let req = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high"
    });
    let out = openai_to_anthropic(&req, ThinkingMode::Enabled);
    assert_eq!(
        out["thinking"],
        json!({"type": "enabled", "budget_tokens": 16384})
    );
    assert!(out.get("output_config").is_none());

    // No effort -> no thinking key at all.
    let req = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    let out = openai_to_anthropic(&req, ThinkingMode::Enabled);
    assert!(out.get("thinking").is_none());
}

#[test]
fn anthropic_thinking_shapes_map_to_reasoning_effort() {
    // Adaptive round-trip: effort passes through output_config.
    let out = anthropic_to_chat_request(&json!({
        "model": "m",
        "max_tokens": 10,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": "hi"}]
    }));
    assert_eq!(out["reasoning_effort"], json!("low"));

    // Bare adaptive: keep reasoning on at the middle level.
    let out = anthropic_to_chat_request(&json!({
        "model": "m",
        "max_tokens": 10,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": "hi"}]
    }));
    assert_eq!(out["reasoning_effort"], json!("medium"));

    // Legacy enabled + budget maps by thresholds.
    let out = anthropic_to_chat_request(&json!({
        "model": "m",
        "max_tokens": 10,
        "thinking": {"type": "enabled", "budget_tokens": 20000},
        "messages": [{"role": "user", "content": "hi"}]
    }));
    assert_eq!(out["reasoning_effort"], json!("high"));

    // No thinking config -> no reasoning_effort.
    let out = anthropic_to_chat_request(&json!({
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}]
    }));
    assert!(out.get("reasoning_effort").is_none());
}

#[test]
fn anthropic_to_openai_maps_thinking_and_cache_usage() {
    let resp = json!({
        "content": [
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "text", "text": "hi"},
            {"type": "tool_use", "id": "tu1", "name": "f", "input": {"x": 1}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5,
            "cache_read_input_tokens": 7, "cache_creation_input_tokens": 2}
    });
    let out = anthropic_to_openai(&resp, "m");
    let msg = &out["choices"][0]["message"];
    assert_eq!(msg["content"], Value::Null);
    assert_eq!(msg["reasoning_content"], json!("hmm"));
    assert_eq!(msg["tool_calls"][0]["function"]["name"], json!("f"));
    assert_eq!(out["choices"][0]["finish_reason"], json!("tool_calls"));
    // prompt_tokens includes cached reads and writes (19 total);
    // Anthropic's bare input_tokens (10) excludes them.
    assert_eq!(out["usage"]["prompt_tokens"], json!(19));
    assert_eq!(
        out["usage"]["prompt_tokens_details"]["cached_tokens"],
        json!(7)
    );
    assert_eq!(out["cache_creation_input_tokens"], json!(2));
    assert_eq!(out["usage"]["total_tokens"], json!(24));
}

#[test]
fn anthropic_usage_round_trips_cache_semantics() {
    // chat -> anthropic must subtract cached tokens from input_tokens
    // and report the split the upstream expects.
    let chat = json!({
        "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "x"}}],
        "usage": {"prompt_tokens": 19, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 7}},
        "cache_creation_input_tokens": 2
    });
    let out = chat_to_anthropic_response(&chat, "m");
    assert_eq!(out["usage"]["input_tokens"], json!(10));
    assert_eq!(out["usage"]["cache_read_input_tokens"], json!(7));
    assert_eq!(out["usage"]["cache_creation_input_tokens"], json!(2));
    assert_eq!(out["usage"]["output_tokens"], json!(5));
}

#[test]
fn anthropic_stream_tool_index_only_counts_tool_use() {
    let mut s = AnthropicStream::new("m");
    let evs = [
        json!({"type": "message_start", "message": {"usage": {"input_tokens": 3}}}),
        json!({"type": "content_block_start", "content_block": {"type": "text"}}),
        json!({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "t"}}),
        json!({"type": "content_block_stop"}),
        json!({"type": "content_block_start", "content_block": {"type": "tool_use", "id": "c1", "name": "a"}}),
        json!({"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
        json!({"type": "content_block_stop"}),
        json!({"type": "content_block_start", "content_block": {"type": "tool_use", "id": "c2", "name": "b"}}),
        json!({"type": "content_block_stop"}),
        json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
        json!({"type": "message_stop"}),
    ];
    let mut chunks: Vec<Value> = vec![];
    for ev in &evs {
        chunks.extend(s.handle(ev));
    }
    // One start chunk per tool_use block (carries the call id).
    let starts: Vec<&Value> = chunks
        .iter()
        .filter(|c| c.pointer("/choices/0/delta/tool_calls/0/id").is_some())
        .collect();
    assert_eq!(starts.len(), 2);
    assert_eq!(
        starts[0]
            .pointer("/choices/0/delta/tool_calls/0/index")
            .unwrap(),
        0
    );
    assert_eq!(
        starts[1]
            .pointer("/choices/0/delta/tool_calls/0/index")
            .unwrap(),
        1
    );
    // thinking deltas surface as reasoning_content
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], json!("tool_calls"));
    assert_eq!(last["usage"]["prompt_tokens"], json!(3));
    assert_eq!(last["usage"]["completion_tokens"], json!(9));
}

// ---- responses ingress ------------------------------------------------

#[test]
fn responses_input_items_map_to_chat_messages() {
    let req = json!({
        "model": "m",
        "input": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "call"}]},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{\"x\":1}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "http://x/i.png"}
            ]}
        ]
    });
    let out = responses_to_chat_request(&req);
    let msgs = out["messages"].as_array().unwrap();
    // reasoning folds into the assistant message that follows it
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0]["role"], "assistant");
    assert_eq!(msgs[0]["content"], json!("call"));
    assert_eq!(msgs[0]["reasoning_content"], json!("think"));
    assert_eq!(msgs[0]["tool_calls"][0]["function"]["name"], json!("f"));
    assert_eq!(msgs[1]["role"], "tool");
    assert_eq!(msgs[1]["tool_call_id"], json!("c1"));
    assert_eq!(msgs[1]["content"], json!("ok"));
    assert_eq!(msgs[2]["role"], "user");
    let parts = msgs[2]["content"].as_array().unwrap();
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "image_url");
}

#[test]
fn responses_params_map_to_chat() {
    let req = json!({
        "model": "m",
        "input": "hi",
        "max_output_tokens": 55,
        "top_p": 0.9,
        "reasoning": {"effort": "high"},
        "text": {"format": {"type": "json_schema", "name": "s", "schema": {"type": "object"}, "strict": true}},
        "tool_choice": {"type": "function", "name": "f"},
        "parallel_tool_calls": false,
        "include": ["reasoning.encrypted_content"]
    });
    let out = responses_to_chat_request(&req);
    assert_eq!(out["max_tokens"], json!(55));
    assert_eq!(out["top_p"], json!(0.9));
    assert_eq!(out["reasoning_effort"], json!("high"));
    assert_eq!(out["response_format"]["type"], json!("json_schema"));
    assert_eq!(out["response_format"]["json_schema"]["name"], json!("s"));
    assert_eq!(
        out["tool_choice"],
        json!({"type": "function", "function": {"name": "f"}})
    );
    assert_eq!(out["parallel_tool_calls"], json!(false));
    assert_eq!(
        out["x_nervogate"]["include"],
        json!(["reasoning.encrypted_content"])
    );
}

#[test]
fn stateful_fields_policy() {
    assert!(responses_stateful_error(&json!({"previous_response_id": "resp_1"})).is_some());
    assert!(responses_stateful_error(&json!({"background": true})).is_some());
    // Stateless: `store: true` promises a retrievable id with no endpoint
    // behind it, and `item_reference` needs that same store.
    assert!(responses_stateful_error(&json!({"store": true})).is_some());
    assert!(responses_stateful_error(&json!({
        "input": [{"type": "item_reference", "id": "msg_1"}]
    }))
    .is_some());
    assert!(responses_stateful_error(&json!({"include": ["x"]})).is_none());
    assert!(responses_stateful_error(&json!({"store": false})).is_none());
    assert!(responses_stateful_error(&json!({"input": "hi"})).is_none());
}

#[test]
fn chat_to_responses_response_maps_tools_status_usage() {
    let chat = json!({
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": null,
            "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "f", "arguments": "{\"x\":1}"}}]
        }}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1}}
    });
    let out = chat_to_responses_response(&chat, "m");
    assert_eq!(out["object"], json!("response"));
    assert_eq!(out["status"], json!("completed"));
    assert_eq!(out["output"][0]["type"], json!("function_call"));
    assert_eq!(out["output"][0]["call_id"], json!("c1"));
    assert_eq!(out["output"][0]["arguments"], json!("{\"x\":1}"));
    assert_eq!(out["usage"]["input_tokens"], json!(4));
    assert_eq!(out["usage"]["output_tokens"], json!(6));
    assert_eq!(
        out["usage"]["input_tokens_details"]["cached_tokens"],
        json!(2)
    );
    assert_eq!(
        out["usage"]["output_tokens_details"]["reasoning_tokens"],
        json!(1)
    );

    let chat = json!({
        "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "x"}}]
    });
    let out = chat_to_responses_response(&chat, "m");
    assert_eq!(out["status"], json!("incomplete"));
    assert_eq!(
        out["incomplete_details"],
        json!({"reason": "max_output_tokens"})
    );
}

#[test]
fn chat_to_responses_stream_emits_full_event_sequence() {
    let mut s = ChatToResponsesStream::new("m");
    let mut events: Vec<SseEvent> = vec![];
    events.extend(s.handle(&chunk("i", "m", json!({"role": "assistant"}), None)));
    events.extend(s.handle(&chunk("i", "m", json!({"content": "Hel"}), None)));
    events.extend(s.handle(&chunk("i", "m", json!({"content": "lo"}), None)));
    events.extend(s.handle(&chunk(
        "i",
        "m",
        json!({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
            "function": {"name": "f", "arguments": ""}}]}),
        None,
    )));
    events.extend(s.handle(&chunk(
        "i",
        "m",
        json!({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
        None,
    )));
    let mut final_chunk = chunk("i", "m", json!({}), Some("tool_calls"));
    final_chunk["usage"] = chat_usage(1, 2, 0, 0);
    events.extend(s.handle(&final_chunk));

    let ts = types(&events);
    assert_eq!(ts[0], "response.created");
    assert_eq!(ts[1], "response.in_progress");
    assert_eq!(ts[2], "response.output_item.added"); // message
    assert_eq!(ts[3], "response.content_part.added");
    assert_eq!(ts[4], "response.output_text.delta");
    assert_eq!(ts[5], "response.output_text.delta");
    assert_eq!(ts[6], "response.output_text.done"); // message closes
    assert_eq!(ts[7], "response.content_part.done");
    assert_eq!(ts[8], "response.output_item.done");
    assert_eq!(ts[9], "response.output_item.added"); // function_call
    assert_eq!(ts[10], "response.function_call_arguments.delta");
    assert_eq!(ts[11], "response.function_call_arguments.done");
    assert_eq!(ts[12], "response.output_item.done");
    assert_eq!(ts[13], "response.completed");

    let last = events.last().unwrap();
    let output = last
        .data
        .pointer("/response/output")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(output.len(), 2);
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["content"][0]["text"], "Hello");
    assert_eq!(output[1]["type"], "function_call");
    assert_eq!(output[1]["arguments"], "{}");
    assert_eq!(last.data.pointer("/response/status").unwrap(), "completed");
    assert_eq!(
        last.data.pointer("/response/usage/input_tokens").unwrap(),
        1
    );
}

#[test]
fn chat_to_responses_stream_incomplete_status() {
    let mut s = ChatToResponsesStream::new("m");
    let mut events = s.handle(&chunk("i", "m", json!({"content": "x"}), None));
    events.extend(s.handle(&chunk("i", "m", json!({}), Some("length"))));
    let last = events.last().unwrap();
    assert_eq!(last.data["type"], "response.incomplete");
    assert_eq!(
        last.data.pointer("/response/incomplete_details"),
        Some(&json!({"reason": "max_output_tokens"}))
    );
}

#[test]
fn chat_to_anthropic_stream_emits_event_names() {
    let mut s = ChatToAnthropicStream::new("m");
    let mut events: Vec<SseEvent> = vec![];
    events.extend(s.handle(&chunk("i", "m", json!({"reasoning_content": "hmm"}), None)));
    events.extend(s.handle(&chunk("i", "m", json!({"content": "hi"}), None)));
    events.extend(s.handle(&chunk(
        "i",
        "m",
        json!({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
            "function": {"name": "f", "arguments": ""}}]}),
        None,
    )));
    events.extend(s.handle(&chunk(
        "i",
        "m",
        json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"a\":1}"}}]}),
        None,
    )));
    let mut final_chunk = chunk("i", "m", json!({}), Some("tool_calls"));
    final_chunk["usage"] = chat_usage(3, 4, 1, 0);
    events.extend(s.handle(&final_chunk));

    assert!(events.iter().all(|e| e.event.is_some()));
    let names: Vec<&str> = events.iter().map(|e| e.event.as_deref().unwrap()).collect();
    assert_eq!(names[0], "message_start");
    assert_eq!(names[1], "content_block_start"); // thinking
    assert_eq!(names[2], "content_block_delta"); // thinking_delta
    assert_eq!(names[3], "content_block_stop");
    assert_eq!(names[4], "content_block_start"); // text
    assert_eq!(names[5], "content_block_delta"); // text_delta
    assert_eq!(names[6], "content_block_stop");
    assert_eq!(names[7], "content_block_start"); // tool_use
    assert_eq!(names[8], "content_block_delta"); // input_json_delta
    assert_eq!(names[9], "content_block_stop");
    assert_eq!(names[10], "message_delta");
    assert_eq!(names[11], "message_stop");

    let delta = &events[8].data;
    assert_eq!(delta["delta"]["type"], "input_json_delta");
    assert_eq!(delta["delta"]["partial_json"], "{\"a\":1}");
    let msg_delta = &events[10].data;
    assert_eq!(msg_delta["delta"]["stop_reason"], "tool_use");
    // prompt_tokens 3 includes 1 cached read, so Anthropic's
    // input_tokens is 2 and the read is reported separately.
    assert_eq!(msg_delta["usage"]["input_tokens"], 2);
    assert_eq!(msg_delta["usage"]["cache_read_input_tokens"], 1);
    assert_eq!(msg_delta["usage"]["output_tokens"], 4);
}

// ---- anthropic ingress -----------------------------------------------

#[test]
fn anthropic_request_maps_to_chat_messages() {
    let req = json!({
        "model": "m",
        "max_tokens": 100,
        "system": [{"type": "text", "text": "be brief"}],
        "thinking": {"type": "enabled", "budget_tokens": 8192},
        "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "f"},
        "stop_sequences": ["stop!"],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
            ]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "s"},
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "tu1", "name": "f", "input": {"x": 1}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}
            ]}
        ]
    });
    let out = anthropic_to_chat_request(&req);
    assert_eq!(out["max_tokens"], json!(100));
    assert_eq!(out["stop"], json!(["stop!"]));
    assert_eq!(out["reasoning_effort"], json!("high"));
    assert_eq!(
        out["tool_choice"],
        json!({"type": "function", "function": {"name": "f"}})
    );
    assert_eq!(out["tools"][0]["function"]["name"], json!("f"));
    let msgs = out["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "be brief");
    assert_eq!(msgs[1]["role"], "user");
    let parts = msgs[1]["content"].as_array().unwrap();
    assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,AA==");
    assert_eq!(msgs[2]["role"], "assistant");
    assert_eq!(msgs[2]["reasoning_content"], "hmm");
    assert_eq!(msgs[2]["content"], json!("calling"));
    assert_eq!(msgs[2]["tool_calls"][0]["function"]["name"], "f");
    assert_eq!(
        msgs[2]["tool_calls"][0]["function"]["arguments"],
        "{\"x\":1}"
    );
    assert_eq!(msgs[3]["role"], "tool");
    assert_eq!(msgs[3]["tool_call_id"], "tu1");
    assert_eq!(msgs[3]["content"], "ok");
}

// ---- S-1: response_format survives the trip to Anthropic -------------

#[test]
fn response_format_maps_to_anthropic_output_config() {
    let req = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "person",
            "schema": {"type": "object", "properties": {"n": {"type": "string"}}}
        }}
    });
    let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
    assert_eq!(out["output_config"]["format"]["type"], json!("json_schema"));
    // Anthropic takes the bare JSON Schema, not OpenAI's
    // `{name, schema}` wrapper.
    assert_eq!(
        out["output_config"]["format"]["schema"]["properties"]["n"]["type"],
        json!("string")
    );
}

#[test]
fn output_config_holds_both_effort_and_format() {
    // Both features write `output_config`; neither may clobber the other.
    let req = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high",
        "response_format": {"type": "json_schema", "json_schema": {
            "schema": {"type": "object"}
        }}
    });
    let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
    assert_eq!(out["thinking"], json!({"type": "adaptive"}));
    assert_eq!(out["output_config"]["effort"], json!("high"));
    assert_eq!(out["output_config"]["format"]["type"], json!("json_schema"));
    assert_eq!(
        out["output_config"]["format"]["schema"]["type"],
        json!("object")
    );
}

#[test]
fn json_object_downgrades_to_an_unconstrained_object_schema() {
    // Anthropic's `output_config.format` only accepts `json_schema`; the
    // old mapping emitted `{"type":"json_object"}` and got a 400 back.
    let req = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"}
    });
    let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
    assert_eq!(out["output_config"]["format"]["type"], json!("json_schema"));
    assert_eq!(
        out["output_config"]["format"]["schema"],
        json!({"type": "object"})
    );
}

#[test]
fn anthropic_max_tokens_default_is_configurable() {
    let req = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    assert_eq!(
        openai_to_anthropic(&req, ThinkingMode::Adaptive)["max_tokens"],
        json!(8192)
    );
    assert_eq!(
        openai_to_anthropic_with_max(&req, ThinkingMode::Adaptive, 32000)["max_tokens"],
        json!(32000)
    );
    // An explicit value always wins over the default.
    let req = json!({"model": "m", "messages": [], "max_tokens": 50});
    assert_eq!(
        openai_to_anthropic_with_max(&req, ThinkingMode::Adaptive, 32000)["max_tokens"],
        json!(50)
    );
}

// ---- S-2: dropped parameters are reported ----------------------------

#[test]
fn unsupported_params_are_reported() {
    let req = json!({
        "model": "m", "messages": [],
        "n": 3, "seed": 7, "logit_bias": {"a": 1}, "top_logprobs": 2
    });
    let lost = chat_to_anthropic_loss(&req);
    assert!(lost.contains(&"n"));
    assert!(lost.contains(&"seed"));
    assert!(lost.contains(&"logit_bias"));
    assert!(chat_to_anthropic_fatal(&req).is_some());
    assert!(chat_to_anthropic_loss(&json!({"model": "m"})).is_empty());
    assert!(chat_to_anthropic_fatal(&json!({"n": 1})).is_none());

    let lost = chat_to_responses_loss(&req);
    assert!(lost.contains(&"seed"));
    assert!(chat_to_responses_fatal(&req).is_some());
    assert!(chat_to_responses_loss(&json!({"model": "m"})).is_empty());
}

// ---- M-3: refusal survives -------------------------------------------

#[test]
fn refusal_only_turn_is_not_dropped() {
    let chat = json!({
        "choices": [{"index": 0, "finish_reason": "content_filter", "message": {
            "role": "assistant", "content": Value::Null,
            "refusal": "I cannot help with that"
        }}]
    });
    // -> Anthropic
    let out = chat_to_anthropic_response(&chat, "m");
    assert_eq!(out["content"][0]["type"], "text");
    assert_eq!(out["content"][0]["text"], "I cannot help with that");
    // -> Responses
    let out = chat_to_responses_response(&chat, "m");
    assert_eq!(out["output"][0]["content"][0]["type"], "refusal");
    assert_eq!(
        out["output"][0]["content"][0]["refusal"],
        "I cannot help with that"
    );
    // -> Anthropic request (assistant turn)
    let out = openai_to_anthropic(
        &json!({"model": "m", "messages": [{
            "role": "assistant", "content": Value::Null, "refusal": "no"
        }]}),
        ThinkingMode::Adaptive,
    );
    assert_eq!(out["messages"][0]["content"][0]["text"], "no");
}

// ---- M-1: interleaved parallel tool calls ---------------------------

/// Feed canonical chat chunks through an Anthropic egress.
fn anth_stream(chunks: &[Value]) -> Vec<SseEvent> {
    let mut s = ChatToAnthropicStream::new("m");
    let mut events = vec![];
    for c in chunks {
        events.extend(s.handle(c));
    }
    events
}

/// Build one Chat `tool_calls[]` delta.
fn tc(index: i64, id: Option<&str>, name: Option<&str>, args: &str) -> Value {
    let mut t = tc_no_index(id, name, args);
    t.as_object_mut()
        .unwrap()
        .insert("index".into(), json!(index));
    t
}

/// Same, from an upstream that does not send `tool_calls[].index`.
fn tc_no_index(id: Option<&str>, name: Option<&str>, args: &str) -> Value {
    let mut f = serde_json::Map::new();
    if let Some(n) = name {
        f.insert("name".into(), json!(n));
    }
    f.insert("arguments".into(), json!(args));
    let mut t = serde_json::Map::new();
    t.insert("type".into(), json!("function"));
    if let Some(i) = id {
        t.insert("id".into(), json!(i));
    }
    t.insert("function".into(), Value::Object(f));
    Value::Object(t)
}

#[test]
fn interleaved_tool_calls_become_separate_contiguous_blocks() {
    // Anthropic requires start -> deltas -> stop per block index, so the
    // egress buffers each parallel call and emits whole blocks.
    let events = anth_stream(&[
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc(0, Some("c1"), Some("a"), "")]}),
            None,
        ),
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc(1, Some("c2"), Some("b"), "")]}),
            None,
        ),
        // Interleaved argument fragments: 0, 1, 0, 1
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc(0, None, None, "{\"x\":")]}),
            None,
        ),
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc(1, None, None, "{\"y\":")]}),
            None,
        ),
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc(0, None, None, "1}")]}),
            None,
        ),
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc(1, None, None, "2}")]}),
            None,
        ),
        chunk("i", "m", json!({}), Some("tool_calls")),
    ]);

    let names: Vec<&str> = events.iter().map(|e| e.event.as_deref().unwrap()).collect();
    assert_eq!(
        names,
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    // Each call is one complete block, in first-seen order.
    let blocks: Vec<&SseEvent> = events
        .iter()
        .filter(|e| e.data["type"] == "content_block_start")
        .collect();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].data["content_block"]["name"], "a");
    assert_eq!(blocks[0].data["content_block"]["id"], "c1");
    assert_eq!(blocks[1].data["content_block"]["name"], "b");
    assert_eq!(blocks[1].data["content_block"]["id"], "c2");
    // Fragments were reassembled in the right order, not interleaved.
    let deltas: Vec<&SseEvent> = events
        .iter()
        .filter(|e| e.data["type"] == "content_block_delta")
        .collect();
    assert_eq!(deltas[0].data["delta"]["partial_json"], "{\"x\":1}");
    assert_eq!(deltas[1].data["delta"]["partial_json"], "{\"y\":2}");
    assert_eq!(deltas[0].data["index"], 0);
    assert_eq!(deltas[1].data["index"], 1);
}

#[test]
fn split_id_and_name_chunks_stay_one_call() {
    // Upstreams may send `id` in one chunk and `name` in the next; both
    // must land on the same tool call, not two.
    let events = anth_stream(&[
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc(0, Some("c1"), None, "")]}),
            None,
        ),
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc(0, None, Some("f"), "{\"a\":1}")]}),
            None,
        ),
        chunk("i", "m", json!({}), Some("tool_calls")),
    ]);
    let blocks: Vec<&SseEvent> = events
        .iter()
        .filter(|e| e.data["type"] == "content_block_start")
        .collect();
    assert_eq!(blocks.len(), 1, "name-only continuation reuses the call");
    assert_eq!(blocks[0].data["content_block"]["name"], "f");
    assert_eq!(blocks[0].data["content_block"]["id"], "c1");
}

#[test]
fn responses_egress_also_keeps_parallel_calls_separate() {
    let mut s = ChatToResponsesStream::new("m");
    let mut events = vec![];
    events.extend(s.handle(&chunk(
        "i",
        "m",
        json!({"tool_calls": [
        tc(0, Some("c1"), Some("a"), "")]}),
        None,
    )));
    events.extend(s.handle(&chunk(
        "i",
        "m",
        json!({"tool_calls": [
        tc(1, Some("c2"), Some("b"), "")]}),
        None,
    )));
    events.extend(s.handle(&chunk(
        "i",
        "m",
        json!({"tool_calls": [
        tc(1, None, None, "{\"y\":2}")]}),
        None,
    )));
    events.extend(s.handle(&chunk(
        "i",
        "m",
        json!({"tool_calls": [
        tc(0, None, None, "{\"x\":1}")]}),
        None,
    )));
    events.extend(s.handle(&chunk("i", "m", json!({}), Some("tool_calls"))));

    let added: Vec<&SseEvent> = events
        .iter()
        .filter(|e| e.data["type"] == "response.output_item.added")
        .collect();
    assert_eq!(added.len(), 2);
    let args: Vec<String> = events
        .iter()
        .filter(|e| e.data["type"] == "response.function_call_arguments.delta")
        .map(|e| e.data["delta"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(args, vec!["{\"x\":1}", "{\"y\":2}"]);
    let last = events.last().unwrap();
    let output = last
        .data
        .pointer("/response/output")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(output.len(), 2);
    assert_eq!(output[0]["name"], "a");
    assert_eq!(output[0]["arguments"], "{\"x\":1}");
    assert_eq!(output[1]["name"], "b");
    assert_eq!(output[1]["arguments"], "{\"y\":2}");
}

// ---- B-4: terminal events on a truncated stream ---------------------

#[test]
fn truncated_stream_still_emits_terminal_events() {
    // Upstream closed with no finish chunk: the client must still get a
    // well-formed terminal event instead of hanging.
    let mut s = ChatToAnthropicStream::new("m");
    s.handle(&chunk("i", "m", json!({"content": "hi"}), None));
    let events = s.finish().expect("terminal events");
    let names: Vec<&str> = events.iter().map(|e| e.event.as_deref().unwrap()).collect();
    assert_eq!(
        names,
        vec!["content_block_stop", "message_delta", "message_stop"]
    );
    // Idempotent: a finish chunk already terminated the stream.
    assert!(s.finish().is_none());

    let mut s = ChatToResponsesStream::new("m");
    s.handle(&chunk("i", "m", json!({"content": "hi"}), None));
    let events = s.finish().expect("terminal event");
    assert_eq!(events.last().unwrap().data["type"], "response.completed");
    assert!(s.finish().is_none());
}

#[test]
fn abort_closes_open_block_before_an_error_event() {
    let mut s = ChatToAnthropicStream::new("m");
    s.handle(&chunk("i", "m", json!({"content": "hi"}), None));
    let out = s.abort();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].data["type"], "content_block_stop");
    assert!(s.abort().is_empty(), "abort is idempotent");
}

// ---- S-4: `failed` is an error, not a finish reason -----------------

#[test]
fn responses_failed_surfaces_as_an_error() {
    let resp = json!({
        "status": "failed",
        "error": {"message": "upstream exploded"},
        "output": []
    });
    assert_eq!(
        responses_failure(&resp),
        Some("upstream exploded".to_string())
    );
    assert!(chat_failure(&responses_to_openai(&resp, "m")).is_some());
    // A healthy response carries no failure.
    assert!(responses_failure(&json!({"status": "completed"})).is_none());
}

#[test]
fn incomplete_reason_drives_finish_reason() {
    let resp = json!({
        "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"},
        "output": []
    });
    let out = responses_to_openai(&resp, "m");
    assert_eq!(out["choices"][0]["finish_reason"], "content_filter");
    let resp = json!({
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": []
    });
    let out = responses_to_openai(&resp, "m");
    assert_eq!(out["choices"][0]["finish_reason"], "length");
}

#[test]
fn responses_stream_maps_failed_to_error_finish() {
    let mut s = ResponsesStream::new("m");
    s.handle(&json!({"type": "response.output_text.delta", "delta": "x"}));
    let out = s.handle(&json!({
        "type": "response.failed",
        "response": {"error": {"message": "boom"}}
    }));
    assert_eq!(out.last().unwrap()["choices"][0]["finish_reason"], "error");
}

#[test]
fn content_filter_survives_both_egresses() {
    // A content-filter stop is not a token limit. Reporting it as
    // `completed`/`end_turn` dresses a truncated answer up as finished.
    let chat = json!({
        "choices": [{"index": 0, "finish_reason": "content_filter",
            "message": {"role": "assistant", "content": "partial"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1}
    });

    // Non-streaming Responses egress.
    let out = chat_to_responses_response(&chat, "m");
    assert_eq!(out["status"], "incomplete");
    assert_eq!(out["incomplete_details"]["reason"], "content_filter");

    // Streaming Responses egress.
    let mut s = ChatToResponsesStream::new("m");
    let ev = s.handle(&chunk("i", "m", json!({}), Some("content_filter")));
    let last = ev.last().unwrap();
    assert_eq!(last.data["type"], "response.incomplete");
    assert_eq!(
        last.data["response"]["incomplete_details"]["reason"],
        "content_filter"
    );

    // Anthropic egress: closes the round trip with
    // `finish_from_anthropic`, which already maps `refusal`.
    let out = chat_to_anthropic_response(&chat, "m");
    assert_eq!(out["stop_reason"], "refusal");
    assert_eq!(
        anthropic_stop("content_filter"),
        "refusal",
        "streaming egress must agree with the non-streaming one"
    );
}

#[test]
fn tool_calls_without_index_are_not_merged_into_one() {
    // Compatibility upstreams that omit `tool_calls[].index` used to
    // collapse every parallel call onto key 0, concatenating their
    // arguments into a single unparseable blob.
    let events = anth_stream(&[
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc_no_index(Some("c1"), Some("a"), "")]}),
            None,
        ),
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc_no_index(Some("c2"), Some("b"), "")]}),
            None,
        ),
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc_no_index(None, None, "{\"y\":")] }),
            None,
        ),
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc_no_index(None, None, "2}")] }),
            None,
        ),
        // Continuation fragment with no id and no index: attaches to the
        // most recently opened slot, not to a merged blob.
        chunk(
            "i",
            "m",
            json!({"tool_calls": [tc_no_index(None, None, "")]}),
            None,
        ),
        chunk("i", "m", json!({}), Some("tool_calls")),
    ]);

    let blocks: Vec<&SseEvent> = events
        .iter()
        .filter(|e| e.data["type"] == "content_block_start")
        .collect();
    assert_eq!(blocks.len(), 2, "parallel calls must stay separate");
    assert_eq!(blocks[0].data["content_block"]["id"], "c1");
    assert_eq!(blocks[1].data["content_block"]["id"], "c2");
}

#[test]
fn stream_options_is_not_reported_as_a_loss() {
    // Every OpenAI SDK sends `stream_options`; warning on it would fire
    // on effectively every request and train operators to ignore the log.
    let req = json!({"model": "m", "messages": [],
        "stream_options": {"include_usage": true}});
    assert!(chat_to_anthropic_loss(&req).is_empty());
    assert!(chat_to_responses_loss(&req).is_empty());
}

// ---- M-6: thinking vs. sampling params -------------------------------

#[test]
fn thinking_with_sampling_params_is_reported_but_still_forwarded() {
    // The decision (M-6): warn, do not strip. Stripping `temperature: 0`
    // would make the caller believe the model is deterministic when it
    // is not — a silent behaviour change, worse than the 400 it prevents.
    let req = json!({
        "model": "m", "messages": [],
        "reasoning_effort": "high",
        "temperature": 0, "top_p": 0.9
    });
    let conflicting = chat_to_anthropic_thinking_conflict(&req);
    assert_eq!(conflicting, vec!["temperature", "top_p"]);

    // Forwarded unchanged: the warning is not a euphemism for a strip.
    let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
    assert_eq!(out["temperature"], 0);
    assert_eq!(out["top_p"], 0.9);
    assert_eq!(out["thinking"], json!({"type": "adaptive"}));
}

#[test]
fn sampling_params_without_thinking_are_not_reported() {
    // The common case: no thinking, no warning. A log line that fires on
    // every request is a log line operators learn to skip.
    let req = json!({"model": "m", "messages": [], "temperature": 0, "top_p": 0.9});
    assert!(chat_to_anthropic_thinking_conflict(&req).is_empty());
}

#[test]
fn thinking_at_the_default_sampling_value_is_not_reported() {
    // `temperature: 1` / `top_p: 1` are Anthropic's defaults and are
    // accepted with thinking. Warning there would be pure noise — and
    // some SDKs send them unconditionally.
    let req = json!({
        "model": "m", "messages": [],
        "reasoning_effort": "high", "temperature": 1, "top_p": 1.0
    });
    assert!(chat_to_anthropic_thinking_conflict(&req).is_empty());
}

#[test]
fn thinking_effort_none_is_not_thinking() {
    // `"none"` is the explicit off switch: it must not enable thinking,
    // and so must not arm the warning either.
    let req = json!({
        "model": "m", "messages": [],
        "reasoning_effort": "none", "temperature": 0
    });
    assert!(chat_to_anthropic_thinking_conflict(&req).is_empty());
    // And it really is off: no `thinking` goes upstream.
    let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
    assert!(out.get("thinking").is_none());
    assert!(out.get("output_config").is_none());
}

#[test]
fn thinking_conflict_is_not_reported_as_a_dropped_param() {
    // The two mechanisms must stay distinct: `loss` means "silently
    // removed", this one means "forwarded but may be rejected". Merging
    // them would make the log say temperature was dropped when it wasn't.
    let req = json!({
        "model": "m", "messages": [],
        "reasoning_effort": "high", "temperature": 0
    });
    assert!(!chat_to_anthropic_loss(&req).contains(&"temperature"));
    assert!(chat_to_anthropic_thinking_conflict(&req).contains(&"temperature"));
}

#[test]
fn chat_to_anthropic_response_with_thinking_and_tools() {
    let chat = json!({
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant",
            "content": "calling",
            "reasoning_content": "hmm",
            "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "f", "arguments": "{\"x\":1}"}}]
        }}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 1}}
    });
    let out = chat_to_anthropic_response(&chat, "m");
    assert_eq!(out["type"], "message");
    assert_eq!(out["role"], "assistant");
    assert_eq!(out["stop_reason"], "tool_use");
    let blocks = out["content"].as_array().unwrap();
    assert_eq!(blocks[0]["type"], "thinking");
    assert_eq!(blocks[0]["thinking"], "hmm");
    assert_eq!(blocks[1]["type"], "text");
    assert_eq!(blocks[2]["type"], "tool_use");
    assert_eq!(blocks[2]["input"], json!({"x": 1}));
    assert_eq!(out["usage"]["cache_read_input_tokens"], json!(1));
}

// ---- responses upstream -> chat --------------------------------------

#[test]
fn responses_to_openai_maps_details_refusal_reasoning() {
    let resp = json!({
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "hm"}]},
            {"type": "message", "content": [
                {"type": "output_text", "text": "hi"},
                {"type": "refusal", "refusal": "no"}
            ]},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}
        ],
        "usage": {"input_tokens": 5, "output_tokens": 7,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2}}
    });
    let out = responses_to_openai(&resp, "m");
    let msg = &out["choices"][0]["message"];
    assert_eq!(msg["reasoning_content"], json!("hm"));
    assert_eq!(msg["content"], Value::Null);
    assert_eq!(msg["tool_calls"][0]["id"], json!("c1"));
    assert_eq!(out["choices"][0]["finish_reason"], json!("tool_calls"));
    assert_eq!(
        out["usage"]["prompt_tokens_details"]["cached_tokens"],
        json!(3)
    );
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"],
        json!(2)
    );
}

// ---- chat -> responses upstream --------------------------------------

#[test]
fn chat_to_responses_request_maps_params_and_passthrough() {
    let req = json!({
        "model": "m",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "calling", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}
        ],
        "max_tokens": 9,
        "top_p": 0.8,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
        "tool_choice": {"type": "function", "function": {"name": "f"}},
        "x_nervogate": {"include": ["reasoning.encrypted_content"]}
    });
    let out = chat_to_responses_request(&req);
    assert_eq!(out["instructions"], json!("sys"));
    assert_eq!(out["max_output_tokens"], json!(9));
    assert_eq!(out["top_p"], json!(0.8));
    assert_eq!(out["reasoning"], json!({"effort": "low"}));
    assert_eq!(out["text"], json!({"format": {"type": "json_object"}}));
    assert_eq!(out["tool_choice"], json!({"type": "function", "name": "f"}));
    assert_eq!(out["store"], json!(false));
    assert_eq!(out["include"], json!(["reasoning.encrypted_content"]));
    let input = out["input"].as_array().unwrap();
    assert_eq!(input.len(), 4);
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[1]["role"], "assistant");
    assert_eq!(input[2]["type"], "function_call");
    assert_eq!(input[3]["type"], "function_call_output");
}

// ---- reasoning_output ------------------------------------------------

#[test]
fn reasoning_output_message_modes() {
    use crate::config::ReasoningOutput;
    let mut msg = json!({"role": "assistant", "content": "hi", "reasoning_content": "hmm"});
    apply_reasoning_output_to_message(&mut msg, ReasoningOutput::ReasoningContent);
    assert_eq!(msg["content"], json!("hi"));
    assert_eq!(msg["reasoning_content"], json!("hmm"));

    let mut msg = json!({"role": "assistant", "content": "hi", "reasoning_content": "hmm"});
    apply_reasoning_output_to_message(&mut msg, ReasoningOutput::ThinkTag);
    assert_eq!(msg["content"], json!("<think>\nhmm\n</think>\nhi"));
    assert!(msg.get("reasoning_content").is_none());

    let mut msg = json!({"role": "assistant", "content": "hi", "reasoning_content": "hmm"});
    apply_reasoning_output_to_message(&mut msg, ReasoningOutput::Both);
    assert_eq!(msg["content"], json!("<think>\nhmm\n</think>\nhi"));
    assert_eq!(msg["reasoning_content"], json!("hmm"));

    // No reasoning: untouched, even in think_tag mode.
    let mut msg = json!({"role": "assistant", "content": "hi"});
    apply_reasoning_output_to_message(&mut msg, ReasoningOutput::ThinkTag);
    assert_eq!(msg["content"], json!("hi"));
}

#[test]
fn reasoning_output_stream_wraps_one_block() {
    use crate::config::ReasoningOutput;
    let mut e = ChatReasoningEgress::new(ReasoningOutput::ThinkTag);
    let mut c1 = chunk("i", "m", json!({"reasoning_content": "r1"}), None);
    e.transform(&mut c1);
    assert_eq!(
        c1.pointer("/choices/0/delta/content").unwrap(),
        "<think>\nr1"
    );
    assert!(c1.pointer("/choices/0/delta/reasoning_content").is_none());
    let mut c2 = chunk("i", "m", json!({"reasoning_content": "r2"}), None);
    e.transform(&mut c2);
    assert_eq!(c2.pointer("/choices/0/delta/content").unwrap(), "r2");
    let mut c3 = chunk("i", "m", json!({"content": "hi"}), None);
    e.transform(&mut c3);
    assert_eq!(
        c3.pointer("/choices/0/delta/content").unwrap(),
        "\n</think>\nhi"
    );
    let mut c4 = chunk("i", "m", json!({}), Some("stop"));
    e.transform(&mut c4);
    assert!(c4.pointer("/choices/0/delta/content").is_none());
}

#[test]
fn reasoning_output_stream_closes_on_finish_without_content() {
    use crate::config::ReasoningOutput;
    let mut e = ChatReasoningEgress::new(ReasoningOutput::ThinkTag);
    let mut c1 = chunk("i", "m", json!({"reasoning_content": "r"}), None);
    e.transform(&mut c1);
    let mut c2 = chunk("i", "m", json!({}), Some("stop"));
    e.transform(&mut c2);
    assert_eq!(
        c2.pointer("/choices/0/delta/content").unwrap(),
        "\n</think>\n"
    );
}
