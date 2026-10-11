# pico-nervogate

A tiny multi-protocol LLM gateway. It sits between your client and an
OpenAI-compatible / Anthropic-compatible upstream and:

- **passes configured HTTP headers upstream** (and lets you attach custom ones), and
- **translates between protocols** so a client that only speaks one wire format
  can talk to an upstream that expects another.

Built for people who want the routing/conversion behavior of a full proxy
(LiteLLM & friends) without the footprint. The release binary runs in well
under 1 MB of resident memory.

> `pico` (small) + `Nerv` (a nod to NERV) + `o` (nice to say) + `gate` (gateway).
> No relation to any of those.

## Features

- **Three protocol surfaces**, one upstream, all translated both ways:
  - `POST /v1/chat/completions` — OpenAI Chat Completions (pass-through)
  - `POST /v1/responses` — OpenAI Responses API (translated to/from Chat)
  - `POST /v1/messages` — Anthropic Messages (translated to/from Chat)
    Each model's `protocol` picks the upstream wire format (`chat` /
    `anthropic` / `responses`), so any client surface can reach any upstream.
    **When a client's protocol already matches its model's `protocol`, the
    request and response are forwarded verbatim** — no round trip through the
    canonical chat shape, so Anthropic `cache_control`, thinking `signature`,
    `redacted_thinking` and `tool_result.is_error` all survive. On streams this
    also preserves the upstream's `event:` names, so Anthropic clients (which
    dispatch on them) work unchanged.
- **Error shape**: an upstream failure is rewrapped in the _client's_ protocol
  regardless of what the upstream speaks, so every SDK reads `error.message`
  instead of failing to parse. The upstream's status code (a 429 stays a 429)
  and the rest of its `error` object survive. Gateway-originated failures —
  including an oversized body — use the same shape.
- **No prompt logging**: request bodies are never written to the logs. On an
  upstream error only the top-level field names are recorded; set
  `GATEWAY_DEBUG_BODY=1` to also log the body, for local debugging only.
  **Do not enable it in production**: stderr is usually collected into
  persistent, widely-readable storage, so this writes your users' prompts
  and any credentials embedded in them to a place they are not deleted from.
  The flag is off by default for that reason — turn it on to debug, turn it
  off and rotate anything sensitive before deploying.
- **Upstream headers**: configurable via `[extra_headers]` (global) and
  per-model `extra_headers`; `Authorization` / `x-api-key` / `anthropic-version`
  are derived from the configured key. Client headers are **not** forwarded.
- **Response headers**: the gateway returns its own `Content-Type` and, for
  streams, `Cache-Control`. Upstream `x-ratelimit-*`, `request-id`,
  `retry-after`, … are not relayed.
- **Streaming**: server-sent events are parsed and re-emitted in the client's
  own framing — Chat (`data:` chunks + `[DONE]`), Responses (`response.*`
  events), and Anthropic (`event:` + `data:` lines, `message_start` …
  `message_stop`).
- **Config-driven models**: each model declares a `protocol`, optional
  `base_url`, `upstream` id and `vision` flag.
- **models.dev enrichment** (opt-in `[models_dev]`): vision/modality, prices,
  and context limits are inferred into `/v1/models`; any explicitly
  configured model field wins over inferred data.
- **Hot-reload** (`[reload]`, on by default): SIGHUP or config mtime change
  re-reads the file without restart; broken files are rejected.
- **Auto-discovery** (opt-in `[discovery]`): upstream `/models` ids merge
  under explicit entries for providers that rotate models.
- **Tiny**: `opt-level = "z"`, LTO, `panic = "abort"`, static musl builds.

## Endpoints

| Method | Path                   | Description                                      |
| ------ | ---------------------- | ------------------------------------------------ |
| GET    | `/healthz`             | liveness check                                   |
| GET    | `/v1/models`           | list configured models (OpenAI model list shape) |
| POST   | `/v1/chat/completions` | OpenAI Chat Completions in/out                   |
| POST   | `/v1/responses`        | OpenAI Responses in → Chat up → Responses out    |
| POST   | `/v1/messages`         | Anthropic Messages in → Chat up → Anthropic out  |

## Quick start

```sh
# 1. Get a config and an API key
cp examples/gateway.min.toml gateway.local.toml
cp .env.example .env
$EDITOR gateway.local.toml   # set models + base_url
$EDITOR .env                 # set MY_API_KEY=...

# 2. Run (reads GATEWAY_CONFIG, defaulting to ./gateway.toml)
GATEWAY_CONFIG=gateway.local.toml cargo run --release

# 3. Try it
curl localhost:8787/healthz
curl localhost:8787/v1/models
```

Then point any OpenAI-compatible client at `http://localhost:8787/v1`.

## Configuration

`GATEWAY_CONFIG` selects the TOML file (default `gateway.toml`). See
`examples/gateway.min.toml` for a minimal config and
`examples/opencode-go.toml` for a real-world example.

Top-level keys:

| Key                             | Default                    | Description                                                                                                                    |
| ------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `listen`                        | `0.0.0.0:8787`             | bind address                                                                                                                   |
| `api_key_env`                   | —                          | env var holding the upstream Bearer key                                                                                        |
| `api_key`                       | —                          | inline key (prefer the env var)                                                                                                |
| `session_id` / `session_header` | —                          | optional stable session header sent upstream                                                                                   |
| `user_agent`                    | `pico-nervogate/<version>` | User-Agent sent upstream                                                                                                       |
| `default_base_url`              | —                          | base URL for models without their own                                                                                          |
| `owned_by` / `provider`         | binary name                | fields advertised by `/v1/models`                                                                                              |
| `extra_headers`                 | —                          | headers added to every upstream request                                                                                        |
| `anthropic_max_tokens`          | `8192`                     | `max_tokens` sent to Anthropic upstreams when the client omitted it (Anthropic requires the field; Chat treats it as optional) |
| `strict_params`                 | `false`                    | reject requests whose parameters the translation cannot carry, instead of warning on stderr                                    |
| `max_body_bytes`                | `33554432` (32 MiB)        | largest client request body accepted; `0` disables the limit                                                                   |
| `[models_dev]`                  | disabled                   | models.dev enrichment for `/v1/models`                                                                                         |
| `[reload]`                      | enabled                    | SIGHUP + mtime config hot-reload                                                                                               |
| `[discovery]`                   | disabled                   | upstream `/models` auto-discovery                                                                                              |
| `[[models]]`                    | —                          | one or more model definitions                                                                                                  |

`[models_dev]`: `enable`, `url` (default `https://models.dev/api.json`),
`provider` (e.g. `opencode-go`; omit to search all), `refresh_interval_secs`
(default `86400`, `0` disables background refresh), `cache_path` (disk
fallback used when the fetch fails). The gateway fetches once at startup,
then refreshes in the background; failures never break serving. Matching is
exact id first, then normalized id / `canonical_model_id` (so
`muse-spark-1.3-contributor` can match `muse-spark-1-3`).

`[reload]`: `enable` (default `true`, read at startup), `watch_interval_secs`
(default `30`, `0` = SIGHUP only). Reload validates before swapping: broken
files are rejected and the old config keeps serving. `listen` changes are
reported but need a restart (re-bind); everything else — models, headers,
`models_dev`/`discovery` sections, and the API key (re-read from env, so
rotation works) — applies live.

`[discovery]`: `enable`, `base_url` (default `default_base_url`), `path`
(default `/models`), `interval_secs` (default `3600`, `0` = startup fetch
only), `protocol` (assumed for discovered models, default `chat`),
`prefix` (only discover matching ids), `prune_missing` (default `false`:
additive only; `true` drops discovered models that vanish upstream).
Explicit `[[models]]` entries always win; discovered models get models.dev
enrichment automatically when their ids match.

A `/models` list says nothing about which completion endpoint the upstream
speaks, so a wrong `protocol` makes every discovered model fail with a 404
at request time. The gateway probes one discovered model with a `max_tokens:
1` request per refresh — one request in the common case, since `chat` is
tried first — and logs the result (`discovery: upstream speaks
\`anthropic\``). A `404`/`405`means "not this endpoint"; any other status
means the endpoint is real. Set`protocol` explicitly to skip probing.

Each `[[models]]`: `name` (public id), `protocol` (`chat` | `anthropic` |
`responses`), optional `upstream`, `base_url`, `vision`, `extra_headers`,
`thinking_type` (`adaptive` | `enabled`, default `adaptive`). `thinking_type`
controls how `reasoning_effort` renders as `thinking` for `anthropic`
upstreams: `adaptive` sends `thinking: {"type":"adaptive"}` +
`output_config: {"effort": ...}` (newer models reject the legacy shape);
`enabled` sends the legacy `{"type":"enabled","budget_tokens":N}`.
Optional models.dev overrides (each set value wins over inferred data;
keys mirror models.dev model entries, minus `id`):
`models_dev_id`, `display_name`, `description`, `family`, `type`,
`knowledge`, `release_date`, `last_updated`, `status`,
`canonical_model_id`, `attachment`, `reasoning`, `reasoning_options`,
`tool_call`, `structured_output`, `temperature`, `open_weights`,
`modalities_input`, `modalities_output`, `context_limit`, `input_limit`,
`output_limit`, `cost_input`, `cost_output`, `cost_cache_read`,
`cost_cache_write`, `cost_tiers`, `cost_context_over_200k`,
`cost_reasoning`, `cost_input_audio`, `cost_output_audio`, `interleaved`,
`provider`, `experimental`. Unset `vision` lets models.dev infer
it from the effective `image` input modality (an explicit
`modalities_input` counts as config); explicit `vision = true/false`
always wins.

## Docker

```sh
docker build -t pico-nervogate:arm64 .
cp examples/gateway.min.toml gateway.local.toml
cp .env.example .env
docker compose up -d
```

`docker-compose.yml` mounts `gateway.local.toml` (git-ignored) and reads keys
from `.env`.

## Design notes

The upstream service is expected to own protocol correctness. This gateway does
not invent endpoints: it forwards to whichever endpoint each model declares and
only translates the request/response bodies (and re-frames SSE) so a client can
use a protocol the upstream does not natively expose.

Translation details worth knowing:

- **Tool calls** round-trip in all directions (`function_calls` ↔ `tool_use` ↔
  `function_call` items), including multi-call turns and tool results.
- **Reasoning/thinking** maps to `message.reasoning_content` in the canonical
  Chat shape. Parameter mapping: `reasoning_effort` ↔ `reasoning.effort` ↔
  `thinking` — for `anthropic` upstreams the per-model `thinking_type` picks
  the shape: `adaptive` (default) sends `thinking: {"type":"adaptive"}` +
  `output_config.effort`; `enabled` sends `thinking.budget_tokens`. Across a
  _protocol change_, Anthropic thinking block _signatures_ arrive as `""`, so
  multi-turn thinking replay against a strict Anthropic upstream may be
  rejected. Same-protocol requests skip translation and keep their signatures.
- **Structured output**: `response_format` maps to `output_config.format` for
  `anthropic` upstreams and to `text.format` for `responses` upstreams, so a
  `json_schema` constraint is not silently dropped. Anthropic only accepts
  `json_schema` there, so OpenAI's `json_object` is downgraded to an
  unconstrained `{"type": "object"}` schema.
- **Responses statefulness**: the gateway is stateless. `previous_response_id`,
  `background`, `store: true` and `item_reference` input items are rejected
  with HTTP 400 — they need server-side response storage and polling endpoints
  that do not exist here; upstream requests always carry `store: false`;
  `include` is passed through to `responses` upstreams.
- **Usage** is remapped per protocol (`prompt_tokens` ↔ `input_tokens` ↔
  `usage`, including `cached_tokens` / `reasoning_tokens` details). Anthropic
  reports cache reads and writes outside `input_tokens` while OpenAI folds
  them into `prompt_tokens`, so the gateway adds them on the way to Chat and
  subtracts them on the way back.
- **Unsupported request parameters** (`n > 1`, `seed`, `logprobs`,
  `logit_bias`, `frequency_penalty`, …) cannot be represented in every
  upstream protocol. They are logged to stderr on every request; set
  `strict_params = true` to reject the ones that change the response _shape_
  (`n > 1`) with HTTP 400 instead.
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
- The request key `x_nervogate` is reserved for gateway-internal passthrough
  (e.g. `include`, `top_k`) and is never sent upstream — including on the
  same-protocol passthrough path.
- **Sampling params under thinking**: Anthropic fixes sampling at the default
  of 1 once thinking is on, and upstreams that enforce it answer 400. The
  gateway still forwards `temperature` / `top_p` and logs a warning, because
  stripping `temperature: 0` would leave the caller believing the model is
  deterministic when it is not. Values already at the default (1) do not warn.
  This warns even under `strict_params`, since nothing was dropped.

## Tests

```sh
cargo test
```

Unit tests cover the translators and the SSE parser in isolation. `src/e2e.rs`
covers what unit tests structurally cannot: a real mock upstream behind a
real gateway router, over real sockets, asserting that the bytes a client
would actually consume are correct — all nine client-surface × upstream
protocol combinations, the same-protocol passthrough framing, and the error
paths. Both blocking bugs fixed in this codebase (a dropped `event:` name, a
duplicated terminal event) passed every unit test at the time.

## Disclaimer

Unofficial and unaffiliated. `pico-nervogate` is a generic client-side gateway;
example configs may reference third-party services, which are the property of
their respective owners. You are responsible for complying with the terms of any
service you point it at.

## License

MIT. See `LICENSE`.
