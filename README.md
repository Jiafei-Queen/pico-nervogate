# pico-nervogate

A tiny multi-protocol LLM gateway. It sits between your client and an
OpenAI-compatible / Anthropic-compatible upstream and:

- **passes HTTP headers through** (and lets you attach custom headers), and
- **translates between protocols** so a client that only speaks one wire format
  can talk to an upstream that expects another.

Built for people who want the routing/conversion behavior of a full proxy
(LiteLLM & friends) without the footprint. The release binary runs in well
under 1 MB of resident memory.

> `pico` (small) + `Nerv` (a nod to NERV) + `o` (nice to say) + `gate` (gateway).
> No relation to any of those.

## Features

- **Three protocol surfaces**, one upstream:
  - `POST /v1/chat/completions` — OpenAI Chat Completions (pass-through)
  - `POST /v1/responses` — OpenAI Responses API (translated to/from Chat)
  - Anthropic Messages (`/v1/messages` upstream, translated to/from Chat via `/v1/chat/completions`)
- **Full header pass-through** plus global (`[extra_headers]`) and per-model headers.
- **Streaming**: server-sent events are parsed and re-emitted so streamed
  responses also come back in the client's protocol.
- **Config-driven models**: each model declares a `protocol`, optional
  `base_url`, `upstream` id and `vision` flag.
- **Tiny**: `opt-level = "z"`, LTO, `panic = "abort"`, static musl builds.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | liveness check |
| GET | `/v1/models` | list configured models (OpenAI model list shape) |
| POST | `/v1/chat/completions` | OpenAI Chat Completions in/out |
| POST | `/v1/responses` | OpenAI Responses in → Chat up → Responses out |

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

| Key | Default | Description |
|---|---|---|
| `listen` | `0.0.0.0:8787` | bind address |
| `api_key_env` | — | env var holding the upstream Bearer key |
| `api_key` | — | inline key (prefer the env var) |
| `session_id` / `session_header` | — | optional stable session header sent upstream |
| `user_agent` | `pico-nervogate/<version>` | User-Agent sent upstream |
| `default_base_url` | — | base URL for models without their own |
| `owned_by` / `provider` | binary name | fields advertised by `/v1/models` |
| `extra_headers` | — | headers added to every upstream request |
| `[[models]]` | — | one or more model definitions |

Each `[[models]]`: `name` (public id), `protocol` (`chat` | `anthropic` |
`responses`), optional `upstream`, `base_url`, `vision`, `extra_headers`.

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

## Disclaimer

Unofficial and unaffiliated. `pico-nervogate` is a generic client-side gateway;
example configs may reference third-party services, which are the property of
their respective owners. You are responsible for complying with the terms of any
service you point it at.

## License

MIT. See `LICENSE`.
