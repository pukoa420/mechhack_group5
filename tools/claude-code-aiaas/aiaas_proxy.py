"""Localhost proxy that lets Claude Code talk to EPFL AIaaS.

Two transformations on chat-style request bodies:

1. Drop the Anthropic-format `thinking` field — LiteLLM (which fronts AIaaS) rejects
   it with HTTP 500 for non-Anthropic backends.
2. Inject `reasoning.enabled=false` for thinking-capable models that would otherwise
   burn the response budget on chain-of-thought. MiniMax is allow-listed: its tool-use
   depends on reasoning being on.

Run:
    python3 aiaas_proxy.py --port 8765 [--upstream https://inference.rcp.epfl.ch]

Then point Claude Code at it:
    export ANTHROPIC_BASE_URL=http://127.0.0.1:8765
    export ANTHROPIC_AUTH_TOKEN=$AIAAS_KEY
"""
from __future__ import annotations
import argparse, asyncio, json, sys, time
from aiohttp import ClientSession, ClientTimeout, web

# Models for which we should NOT inject reasoning.enabled=false.
# Their tool-use / quality degrades meaningfully without reasoning.
KEEP_REASONING_PREFIXES = (
    "MiniMaxAI/",
    "MiniMax",
    "anthropic/",   # passthrough if you ever route Sonnet/Opus through here
)


def should_keep_reasoning(model: str) -> bool:
    return any(model.startswith(p) for p in KEEP_REASONING_PREFIXES)


def transform_body(body_bytes: bytes) -> bytes:
    if not body_bytes:
        return body_bytes
    try:
        body = json.loads(body_bytes)
    except Exception:
        return body_bytes
    if not isinstance(body, dict) or "messages" not in body:
        return body_bytes

    # 1. Always strip Anthropic-format `thinking` field.
    body.pop("thinking", None)

    # 2. Inject reasoning.enabled=false unless the model is allow-listed or already
    #    set by the caller.
    model = body.get("model", "")
    if "reasoning" not in body and not should_keep_reasoning(model):
        body["reasoning"] = {"enabled": False}

    return json.dumps(body).encode()


async def proxy_handler(request: web.Request) -> web.StreamResponse:
    upstream = request.app["upstream"]
    # Build upstream URL — preserve the path (so /v1/messages or /v1/chat/completions
    # both forward identically; AIaaS supports both).
    target = f"{upstream}{request.rel_url}"

    body = await request.read()
    transformed = transform_body(body)

    in_keys = []
    out_keys = []
    try:
        in_keys = sorted(list(json.loads(body).keys())) if body else []
        out_keys = sorted(list(json.loads(transformed).keys())) if transformed else []
    except Exception:
        pass

    # Strip Host so aiohttp sets it correctly for the upstream.
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}

    print(f"-> {request.method} {request.path} in_keys={in_keys} out_keys={out_keys}",
          file=sys.stderr, flush=True)

    timeout = ClientTimeout(total=600.0, connect=30.0)
    async with ClientSession(timeout=timeout) as sess:
        async with sess.request(
            request.method, target,
            data=transformed,
            headers=headers,
            allow_redirects=False,
        ) as resp:
            data = await resp.read()
            print(f"<- {resp.status} {len(data)}B", file=sys.stderr, flush=True)
            out_headers = {k: v for k, v in resp.headers.items()
                           if k.lower() not in ("transfer-encoding", "content-encoding",
                                                "content-length", "connection")}
            return web.Response(body=data, status=resp.status, headers=out_headers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--upstream", default="https://inference.rcp.epfl.ch")
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["upstream"] = args.upstream.rstrip("/")
    app.router.add_route("*", "/{tail:.*}", proxy_handler)

    print(f"aiaas_proxy: listening on {args.host}:{args.port} -> {args.upstream}",
          file=sys.stderr, flush=True)
    web.run_app(app, host=args.host, port=args.port, access_log=None,
                handle_signals=True)


if __name__ == "__main__":
    main()
