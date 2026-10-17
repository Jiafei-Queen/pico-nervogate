# Design notes

The upstream service is expected to own protocol correctness. This gateway does
not invent endpoints: it forwards to whichever endpoint each model declares and
only translates the request/response bodies (and re-frames SSE) so a client can
use a protocol the upstream does not natively expose.

Translation details worth knowing:

- **Tool calls** round-trip in all directions (`function_calls` ↔ `tool_use` ↔
  `function_call` items ↔ Gemini `functionCall`/`functionResponse` parts),
  including multi-call turns and tool results. Gemini has no call ids, so the
  gateway synthesizes `call_*` ids on the way to Chat and matches
  `functionResponse` back by function name; tool output text that is not a
  JSON object is wrapped as `{"result": ...}`.
- **Reasoning/thinking** maps to `message.reasoning_content` in the canonical
  Chat shape. Parameter mapping: `reasoning_effort` ↔ `reasoning.effort` ↔
  `thinking` ↔ Gemini `thinkingConfig.thinkingBudget` (+`includeThoughts`).
  For `anthropic` upstreams the per-model `thinking_type` picks
  the shape: `adaptive` (default) sends `thinking: {"type":"adaptive"}` +
  `output_config.effort`; `enabled` sends `thinking.budget_tokens`. For
  `google` upstreams the effort renders as a budget (low 4k / medium 8k /
  high 16k) and thought parts (`{"thought": true, ...}`) round-trip as
  `reasoning_content`. Across a
  _protocol change_, Anthropic thinking block _signatures_ arrive as `""`, so
  multi-turn thinking replay against a strict Anthropic upstream may be
  rejected. Same-protocol requests skip translation and keep their signatures.
  On Chat egress the global `reasoning_output` picks the render:
  `reasoning_content` (default), `think_tag` (`<think>` inside `content`,
  for tag-parsing clients like OpenWebUI), or `both`. Translated paths only;
  same-protocol passthrough stays verbatim.
- **Structured output**: `response_format` maps to `output_config.format` for
  `anthropic` upstreams, to `text.format` for `openai` upstreams, and to
  `generationConfig.responseMimeType` + `responseSchema` for `google`
  upstreams, so a `json_schema` constraint is not silently dropped. Anthropic
  only accepts `json_schema` there, so OpenAI's `json_object` is downgraded
  to an unconstrained `{"type": "object"}` schema (same for Google).
- **Google specifics**: the model travels in the URL path and streaming
  switches the endpoint (`:streamGenerateContent?alt=sse`) instead of a body
  flag; `data:` URLs become `inlineData`, remote `https://` images become
  `fileData` (Gemini may not fetch arbitrary https — prefer base64);
  `MALFORMED_FUNCTION_CALL` surfaces as a gateway error, never a normal tool
  turn; `usageMetadata` maps onto the canonical usage including
  `thoughtsTokenCount` → `reasoning_tokens`; tools without a function name
  (e.g. OpenAI's `{"type": "code_interpreter"}`) are dropped and
  `additionalProperties` is stripped from tool schemas at every depth —
  Gemini rejects both with a 400.
- **Responses statefulness**: the gateway is stateless. `previous_response_id`,
  `background`, `store: true` and `item_reference` input items are rejected
  with HTTP 400 — they need server-side response storage and polling endpoints
  that do not exist here; upstream requests always carry `store: false`;
  `include` is passed through to `openai` upstreams.
- **Usage** is remapped per protocol (`prompt_tokens` ↔ `input_tokens` ↔
  `usage`, including `cached_tokens` / `reasoning_tokens` details). Anthropic
  reports cache reads and writes outside `input_tokens` while OpenAI folds
  them into `prompt_tokens`, so the gateway adds them on the way to Chat and
  subtracts them on the way back.
- **Unsupported request parameters** (`n > 1`, `seed`, `logprobs`,
  `logit_bias`, `frequency_penalty`, …) cannot be represented in every
  upstream protocol. They are logged to stderr on every request; set
  `strict_params = true` to reject the ones that change the response _shape_
  (`n > 1`) with HTTP 400 instead. Unsupported content block types
  (`input_audio`, `file`, `document`, …) are reported for the same reason.
- **Streaming failures**: a mid-stream transport error is never dressed up as
  a clean finish. Chat clients receive an error chunk followed by `[DONE]`;
  Responses / Anthropic clients receive their terminal error event. A stream
  that ends without a finish chunk still gets its protocol's terminal event —
  except on the same-protocol passthrough path, where the upstream owns the
  terminator and the gateway does not invent one.
- **Parallel tool calls**: streamed tool calls are buffered per upstream
  `tool_calls[].index` and emitted as complete, contiguous blocks at the end
  of the stream, because Anthropic and Responses both require
  start → delta\* → stop per index and reject interleaving. The trade-off is
  that tool-call arguments reach the client only when the stream ends, so a
  long tool call has a later first byte than its content would.
- **Sampling params under thinking**: Anthropic fixes sampling at the default
  of 1 once thinking is on, and upstreams that enforce it answer 400. The
  gateway still forwards `temperature` / `top_p` and logs a warning, because
  stripping `temperature: 0` would leave the caller believing the model is
  deterministic when it is not. Values already at the default (1) do not warn.
  This warns even under `strict_params`, since nothing was dropped.
- **Refusals** are not dropped: a refusal reaches Anthropic as text with
  `stop_reason: refusal` and Responses as a `refusal` content part;
  `content_filter` closes the round trip in both directions.
- The request key `x_nervogate` is reserved for gateway-internal passthrough
  (e.g. `include`, `top_k`) and is never sent upstream — including on the
  same-protocol passthrough path.

## Logging

Request bodies are never written to the logs. On an upstream error only the
top-level field names are recorded — enough to answer "did the translation
produce the expected shape?" without recording any content.

`GATEWAY_DEBUG_BODY=1` also logs the body, for local debugging only. **Do not
enable it in production**: stderr is usually collected into persistent,
widely-readable storage, so this writes your users' prompts and any
credentials embedded in them to a place they are not deleted from. Turn it on
to debug, turn it off and rotate anything sensitive before deploying.
