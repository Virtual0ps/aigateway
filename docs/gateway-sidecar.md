# Gateway sidecar

The `aigateway` binary runs as a **loopback sidecar**: it exposes an inbound
Anthropic Messages endpoint (`POST /v1/messages`), translates each request to a
configured OpenAI-compatible upstream, and translates the response — including
streaming SSE, tool calls, and thinking blocks — back into Anthropic wire
format.

This lets a bare OpenAI-compatible or local-model API key back a **Claude Code**
session offline, with no hosted gateway in the middle. A supervising daemon
(e.g. LinkCode) spawns the binary, reads the bound address from stdout, and
points Claude Code's `ANTHROPIC_BASE_URL` at it.

## CLI / spawn contract

```
aigateway serve --host 127.0.0.1 --port 0 --config <path>
```

| Flag       | Default     | Meaning                                                    |
| ---------- | ----------- | ---------------------------------------------------------- |
| `--host`   | `127.0.0.1` | Bind address. Keep on loopback.                            |
| `--port`   | `0`         | Bind port. `0` lets the OS assign a free port.             |
| `--config` | *(required)*| Path to the TOML config file.                             |

On successful bind — **before** serving — the process prints exactly one line to
**stdout** and flushes it:

```
listening on http://127.0.0.1:49157
```

A spawning daemon should read stdout until it sees this line, parse the
`http://<host>:<port>` address, and use it as the Anthropic base URL. With
`--port 0` this is the only way to learn the actual port (mirrors the
`linkcode-pty/runtime.json` pattern). Errors (bad config, bind failure) are
reported on stderr with a non-zero exit code.

Shutdown is graceful on `SIGTERM` or `Ctrl-C` (`SIGINT`).

## Configuration (TOML)

Minimal:

```toml
[upstream]
base_url = "https://api.openai.com/v1"
api_key  = "sk-..."
```

Full:

```toml
[upstream]
base_url        = "https://api.openai.com/v1"
api_key         = "sk-..."
wire            = "openai-chat"   # "openai-chat" (default) | "openai-responses" (not yet supported)
timeout_seconds = 600             # idle read timeout; long streams are never cut

# Map inbound (Anthropic) model names → upstream model names.
[upstream.models]
"claude-sonnet-4-20250514" = "gpt-4.1"
"claude-3-5-haiku-20241022" = "gpt-4.1-mini"

# Optional: fall back to this upstream model when no mapping matches.
# When unset, the inbound model name is forwarded unchanged.
default_model = "gpt-4.1"

# Optional: extra headers sent on every upstream request.
[upstream.default_headers]
x-custom-header = "value"
```

| Field                    | Required | Notes                                                             |
| ------------------------ | -------- | ---------------------------------------------------------------- |
| `upstream.base_url`      | yes      | OpenAI-compatible base; `/chat/completions` is appended.         |
| `upstream.api_key`       | yes      | Injected as `Authorization: Bearer <key>`. Redacted in logs.    |
| `upstream.wire`          | no       | Only `openai-chat` is implemented today.                         |
| `upstream.timeout_seconds` | no     | Idle read timeout (default 600s), not a total-request timeout.   |
| `upstream.models`        | no       | Inbound → upstream model name map.                               |
| `upstream.default_model` | no       | Fallback upstream model; passthrough if unset.                   |
| `upstream.default_headers` | no     | Extra upstream request headers.                                  |

## Endpoints

### `POST /v1/messages`

Accepts an Anthropic [Messages API](https://docs.anthropic.com/en/api/messages)
request body. `max_tokens` is required (as upstream Anthropic requires). Honors
`stream: true` for SSE.

- **Streaming** returns `Content-Type: text/event-stream` with the Anthropic
  event lifecycle: `message_start` → (`content_block_start` →
  `content_block_delta`* → `content_block_stop`)* → `message_delta` →
  `message_stop`. Text, `tool_use` (with streamed `input_json_delta`), and
  `thinking` blocks (with `signature_delta`) are all emitted.
- **Unary** returns a JSON Anthropic `message` object.

### `GET /health`

Returns `200 {"status":"ok"}`.

## Authentication

Inbound authentication is **ignored** — Claude Code sends a placeholder token,
which the gateway discards. The real upstream key is injected from config. Only
bind loopback; do not expose this endpoint to a network.

## Behavior notes

- **Networking** — the outbound HTTP client bypasses the system proxy
  (`no_proxy`), so a loopback sidecar always talks directly to its configured
  upstream (important for `localhost` model servers). Explicit proxy support can
  be added as a config knob if needed.
- **Token accounting** — OpenAI-style upstreams report usage only at the end of
  a stream, after `message_start` has already been sent. So streaming
  `message_start.usage.input_tokens` is `0`; the real input/output token counts
  arrive in the terminal `message_delta.usage`. Unary responses carry full
  usage.
- **Response `model`** — the response echoes the upstream model name (what
  actually served the request), not the requested Anthropic model name.

## Example

Start the gateway:

```bash
aigateway serve --host 127.0.0.1 --port 0 --config gateway.toml
# → listening on http://127.0.0.1:49157
```

Point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:49157"
export ANTHROPIC_API_KEY="placeholder"   # ignored by the gateway
```

Or drive it directly:

```bash
curl -N http://127.0.0.1:49157/v1/messages \
  -H 'content-type: application/json' \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 256,
    "stream": true,
    "messages": [{ "role": "user", "content": "Hello!" }]
  }'
```
