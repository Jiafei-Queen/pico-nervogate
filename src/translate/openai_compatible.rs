//! OpenAI Chat Completions egress rendering (canonical hub).

use serde_json::{json, Value};

use crate::config::ReasoningOutput;
/// `<think>` framing for Chat egress. The trailing newline in each marker
/// matches what tag-parsing clients expect (`<think>\n...`).
pub const THINK_OPEN: &str = "<think>\n";

pub const THINK_CLOSE: &str = "\n</think>\n";

/// Render reasoning into a Chat completion message per `mode`.
/// `ReasoningContent` is a no-op; the other modes prepend a `<think>`
/// block to `content` (`ThinkTag` also drops `reasoning_content`).
pub fn apply_reasoning_output_to_message(msg: &mut Value, mode: ReasoningOutput) {
    if mode == ReasoningOutput::ReasoningContent {
        return;
    }
    let reasoning = msg
        .get("reasoning_content")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    if reasoning.is_empty() {
        return;
    }
    let orig = msg
        .get("content")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    msg["content"] = json!(format!("{THINK_OPEN}{reasoning}{THINK_CLOSE}{orig}"));
    if mode == ReasoningOutput::ThinkTag {
        if let Some(o) = msg.as_object_mut() {
            o.remove("reasoning_content");
        }
    }
}

/// Stateful Chat chunk rewriter for `reasoning_output`.
/// Reasoning deltas open one `<think>` block; the first following content
/// (or the finish chunk) closes it, so multi-chunk thinking stays a single
/// block instead of one tag pair per chunk.
pub struct ChatReasoningEgress {
    mode: ReasoningOutput,
    in_think: bool,
}

impl ChatReasoningEgress {
    pub fn new(mode: ReasoningOutput) -> Self {
        Self {
            mode,
            in_think: false,
        }
    }

    pub fn transform(&mut self, chunk: &mut Value) {
        if self.mode == ReasoningOutput::ReasoningContent {
            return;
        }
        let keep = self.mode == ReasoningOutput::Both;
        let finish = chunk
            .pointer("/choices/0/finish_reason")
            .is_some_and(|f| !f.is_null());
        let delta = match chunk.pointer_mut("/choices/0/delta") {
            Some(Value::Object(m)) => m,
            _ => {
                if finish && self.in_think {
                    if let Some(o) = chunk
                        .pointer_mut("/choices/0")
                        .and_then(|c| c.as_object_mut())
                    {
                        o.insert("delta".into(), json!({"content": THINK_CLOSE}));
                    }
                    self.in_think = false;
                }
                return;
            }
        };
        let reasoning = delta
            .get("reasoning_content")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let content = delta
            .get("content")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        if !reasoning.is_empty() && !content.is_empty() {
            let wrapped = if self.in_think {
                format!("{reasoning}{THINK_CLOSE}{content}")
            } else {
                format!("{THINK_OPEN}{reasoning}{THINK_CLOSE}{content}")
            };
            self.in_think = false;
            delta.insert("content".into(), json!(wrapped));
            if !keep {
                delta.remove("reasoning_content");
            }
            return;
        }
        if !reasoning.is_empty() {
            let mut out = String::new();
            if !self.in_think {
                out.push_str(THINK_OPEN);
            }
            out.push_str(&reasoning);
            if finish {
                out.push_str(THINK_CLOSE);
                self.in_think = false;
            } else {
                self.in_think = true;
            }
            delta.insert("content".into(), json!(out));
            if !keep {
                delta.remove("reasoning_content");
            }
            return;
        }
        if !content.is_empty() && self.in_think {
            delta.insert("content".into(), json!(format!("{THINK_CLOSE}{content}")));
            self.in_think = false;
            return;
        }
        if finish && self.in_think {
            delta.insert("content".into(), json!(THINK_CLOSE));
            self.in_think = false;
        }
    }
}
