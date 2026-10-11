# pico-nervogate

A tiny multi-protocol LLM gateway. It sits between your client and an
OpenAI-compatible / Anthropic-compatible upstream, **translates between
protocols** so a client that only speaks one wire format can reach an upstream
that expects another, and **passes configured HTTP headers upstream**.

Built for people who want the routing/conversion behavior of a full proxy
(LiteLLM & friends) without the footprint. The release binary runs in well
under 1 MB of resident memory.

## Features

- **Three protocol surfaces**, one upstream, translated both ways:
  `POST /v1/chat/completions` (OpenAI Chat),
  `POST /v1/responses` (OpenAI Responses),
  `POST /v1/messages` (Anthropic Messages).
  Each model's `protocol` picks the upstream wire format (`chat` /
  `anthropic` / `responses`), so any client surface can reach any upstream.
- **Same-protocol passthrough**: when the client's protocol already matches
  the model's, the request and response are forwarded verbatim — no round trip
  through the canonical chat shape, so `cache_control`, thinking signatures and
  `tool_result.is_error` survive, and the upstream's SSE `event:` names are
  preserved.
- **Error shape**: an upstream failure is rewrapped in the _client's_ protocol
  regardless of what the upstream speaks, keeping its status code (a 429 stays
  a 429) and error object. Gateway-originated failures use the same shape.
- **No prompt logging**: request bodies are never written to the logs.
  `GATEWAY_DEBUG_BODY=1` restores them for local debugging only — see
  [docs/design.md](docs/design.md#logging).
- **Headers**: upstream headers configurable globally (`[extra_headers]`) and
  per model; `Authorization` / `x-api-key` / `anthropic-version` are derived
  from the configured key. Client headers are **not** forwarded, and upstream
  `x-ratelimit-*` / `request-id` / `retry-after` are not relayed.
- **Streaming**: SSE parsed and re-emitted in the client's own framing for all
  three protocols, with errors surfaced in the client's error shape.
- **Config-driven models**: each model declares a `protocol`, optional
  `base_url`, `upstream` id and `vision` flag.
- **models.dev enrichment** (opt-in `[models_dev]`): vision, prices and context
  limits inferred into `/v1/models`; explicit config always wins.
- **Hot-reload** (`[reload]`, on by default): SIGHUP or config mtime change
  re-reads the file without restart; broken files are rejected.
- **Auto-discovery** (opt-in `[discovery]`): upstream `/models` ids merge under
  explicit entries, with a protocol probe so a wrong `protocol` is caught at
  refresh time instead of failing every request.
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

| Key                             | Default                    | Description                                        |
| ------------------------------- | -------------------------- | -------------------------------------------------- |
| `listen`                        | `0.0.0.0:8787`             | bind address                                       |
| `api_key_env`                   | —                          | env var holding the upstream Bearer key            |
| `api_key`                       | —                          | inline key (prefer the env var)                    |
| `session_id` / `session_header` | —                          | optional stable session header sent upstream       |
| `user_agent`                    | `pico-nervogate/<version>` | User-Agent sent upstream                           |
| `default_base_url`              | —                          | base URL for models without their own              |
| `owned_by` / `provider`         | binary name                | fields advertised by `/v1/models`                  |
| `extra_headers`                 | —                          | headers added to every upstream request            |
| `anthropic_max_tokens`          | `8192`                     | `max_tokens` sent to Anthropic upstreams on omission |
| `strict_params`                 | `false`                    | reject untranslatable parameters instead of warning |
| `max_body_bytes`                | `33554432` (32 MiB)        | largest client request body; `0` disables the limit |
| `[models_dev]`                  | disabled                   | models.dev enrichment for `/v1/models`             |
| `[reload]`                      | enabled                    | SIGHUP + mtime config hot-reload                   |
| `[discovery]`                   | disabled                   | upstream `/models` auto-discovery                  |
| `[[models]]`                    | —                          | one or more model definitions                      |

Full reference — the `[models_dev]` / `[reload]` / `[discovery]` sections,
per-model fields, `thinking_type` and the models.dev override keys — is in
[docs/configuration.md](docs/configuration.md).

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

Protocol translation semantics — tool calls, thinking, structured output, the
stateless Responses policy, usage mapping, parameter handling, streaming
failures, parallel tool calls — are documented in
[docs/design.md](docs/design.md).

## Tests

```sh
cargo test
```

Unit tests cover the translators and the SSE parser in isolation; `src/e2e.rs`
drives a real mock upstream behind the real router over real sockets, across
all nine client-surface × upstream protocol combinations, and asserts the
bytes a client would actually consume.

## Disclaimer

Unofficial and unaffiliated. `pico-nervogate` is a generic client-side gateway;
example configs may reference third-party services, which are the property of
their respective owners. You are responsible for complying with the terms of any
service you point it at.

## License

MIT. See `LICENSE`.
