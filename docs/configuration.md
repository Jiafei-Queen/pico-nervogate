# Configuration

`GATEWAY_CONFIG` selects the TOML file (default `gateway.toml`). See
`examples/gateway.min.toml` for a minimal config and
`examples/opencode-go.toml` for a real-world example.

## Top-level keys

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

## `[models_dev]`

`enable`, `url` (default `https://models.dev/api.json`), `provider`
(e.g. `opencode-go`; omit to search all), `refresh_interval_secs`
(default `86400`, `0` disables background refresh), `cache_path` (disk
fallback used when the fetch fails). The gateway fetches once at startup,
then refreshes in the background; failures never break serving. Matching is
exact id first, then normalized id / `canonical_model_id` (so
`muse-spark-1.3-contributor` can match `muse-spark-1-3`).

## `[reload]`

`enable` (default `true`, read at startup), `watch_interval_secs`
(default `30`, `0` = SIGHUP only). Reload validates before swapping: broken
files are rejected and the old config keeps serving. `listen` changes are
reported but need a restart (re-bind); everything else — models, headers,
`models_dev`/`discovery` sections, and the API key (re-read from env, so
rotation works) — applies live.

## `[discovery]`

`enable`, `base_url` (default `default_base_url`), `path`
(default `/models`), `interval_secs` (default `3600`, `0` = startup fetch
only), `protocol` (assumed for discovered models, default `chat`),
`prefix` (only discover matching ids), `prune_missing` (default `false`:
additive only; `true` drops discovered models that vanish upstream).
Explicit `[[models]]` entries always win; discovered models get models.dev
enrichment automatically when their ids match.

A `/models` list says nothing about which completion endpoint the upstream
speaks, so a wrong `protocol` makes every discovered model fail with a 404
at request time. The gateway probes one discovered model with a
`max_tokens: 1` request per refresh — one request in the common case, since
`chat` is tried first — and logs the result (`discovery: upstream speaks
anthropic`). A `404`/`405` means "not this endpoint"; any other status
means the endpoint is real. Set `protocol` explicitly to skip probing.

## `[[models]]`

`name` (public id), `protocol` (`chat` | `anthropic` | `responses`), optional
`upstream`, `base_url`, `vision`, `extra_headers`, `thinking_type`
(`adaptive` | `enabled`, default `adaptive`).

`thinking_type` controls how `reasoning_effort` renders as `thinking` for
`anthropic` upstreams: `adaptive` sends `thinking: {"type":"adaptive"}` +
`output_config: {"effort": ...}` (newer models reject the legacy shape);
`enabled` sends the legacy `{"type":"enabled","budget_tokens":N}`. See
[design.md](design.md) for the reverse mapping.

Optional models.dev overrides (each set value wins over inferred data; keys
mirror models.dev model entries, minus `id`):

`models_dev_id`, `display_name`, `description`, `family`, `type`,
`knowledge`, `release_date`, `last_updated`, `status`,
`canonical_model_id`, `attachment`, `reasoning`, `reasoning_options`,
`tool_call`, `structured_output`, `temperature`, `open_weights`,
`modalities_input`, `modalities_output`, `context_limit`, `input_limit`,
`output_limit`, `cost_input`, `cost_output`, `cost_cache_read`,
`cost_cache_write`, `cost_tiers`, `cost_context_over_200k`, `cost_reasoning`,
`cost_input_audio`, `cost_output_audio`, `interleaved`, `provider`,
`experimental`.

Unset `vision` lets models.dev infer it from the effective `image` input
modality (an explicit `modalities_input` counts as config); explicit
`vision = true/false` always wins.
