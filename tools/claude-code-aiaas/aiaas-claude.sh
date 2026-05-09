#!/bin/bash
# Launch Claude Code routed through EPFL AIaaS via local reasoning-stripping proxy.
# Asks for key + model, starts proxy on localhost:8765, runs `claude`.
#
# Default model: MiniMaxAI/MiniMax-M2.7 (best tool-use, reasoning kept on).

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 1. Key ----------------------------------------------------------------
KEY="${AIAAS_KEY:-}"
if [ -z "$KEY" ]; then
  echo "Get your AIaaS key at: https://portal.rcp.epfl.ch/aiaas/keys"
  read -r -s -p "AIaaS API key (sk--...): " KEY
  echo
fi
[ -z "$KEY" ] && { echo "no key — aborting"; exit 1; }

# ---- 2. Model selection (constrained to the two we recommend for coding) ---
echo
echo "Pick a model:"
echo "  1) MiniMaxAI/MiniMax-M2.7    (RECOMMENDED — reasoning ON, ~8-10s)"
echo "  2) moonshotai/Kimi-K2.6      (THINKING — proxy disables, ~slower)"
read -r -p "Choice [1]: " CHOICE
CHOICE="${CHOICE:-1}"

case "$CHOICE" in
  1) MODEL="MiniMaxAI/MiniMax-M2.7" ;;
  2) MODEL="moonshotai/Kimi-K2.6" ;;
  *) echo "invalid choice: $CHOICE — pick 1 or 2"; exit 1 ;;
esac
echo "Using model: $MODEL"

# ---- 3. Verify model in catalog --------------------------------------------
HTTP=$(curl -s -o /tmp/_aiaas_models.json -w "%{http_code}" \
  -H "Authorization: Bearer $KEY" https://inference.rcp.epfl.ch/v1/models)
if [ "$HTTP" = "200" ]; then
  if python3 -c "
import json, sys
d=json.load(open('/tmp/_aiaas_models.json'))
ids=[m['id'] for m in d.get('data',[])]
sys.exit(0 if '$MODEL' in ids else 1)
" 2>/dev/null; then
    echo "✓ model verified in AIaaS catalog"
  else
    echo "WARN: '$MODEL' not in AIaaS catalog (proceeding — run check_model.sh to triage)"
  fi
else
  echo "WARN: AIaaS auth probe HTTP $HTTP (proceeding)"
fi

# ---- 4. Ensure aiohttp + start proxy in background -------------------------
PROXY_PORT="${PROXY_PORT:-8765}"
if ! python3 -c "import aiohttp" 2>/dev/null; then
  echo "installing aiohttp..."
  pip install --user --quiet aiohttp 2>&1 | tail -2
fi

# Kill any prior proxy instance on this port
pkill -f "aiaas_proxy.py.*--port $PROXY_PORT" 2>/dev/null || true
sleep 0.3

PROXY_LOG="/tmp/aiaas_proxy.log"
PROXY_SCRIPT="${PROXY_SCRIPT:-$HERE/aiaas_proxy.py}"
[ -f "$PROXY_SCRIPT" ] || { echo "proxy script not found at $PROXY_SCRIPT"; exit 1; }
nohup python3 "$PROXY_SCRIPT" --port "$PROXY_PORT" > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!
sleep 0.6

if ! kill -0 "$PROXY_PID" 2>/dev/null; then
  echo "proxy failed to start; log:"
  tail -20 "$PROXY_LOG"
  exit 1
fi
echo "✓ proxy listening on 127.0.0.1:$PROXY_PORT (pid $PROXY_PID, log $PROXY_LOG)"

cleanup() {
  echo
  echo "stopping proxy..."
  kill "$PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---- 5. Configure Claude Code env ------------------------------------------
export ANTHROPIC_BASE_URL="http://127.0.0.1:$PROXY_PORT"
export ANTHROPIC_AUTH_TOKEN="$KEY"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"

mkdir -p "$HOME/.claude"
SETTINGS="$HOME/.claude/settings.json"
[ -f "$SETTINGS" ] || echo '{"thinking": false, "permissions":{"defaultMode":"bypassPermissions"}}' > "$SETTINGS"

cat <<EOF

Launching Claude Code:
  proxy:    http://127.0.0.1:$PROXY_PORT  ->  https://inference.rcp.epfl.ch
  model:    $MODEL
  reasoning: handled by proxy (kept ON for MiniMax, OFF for Kimi/DeepSeek/Qwen-Thinking)

Press Ctrl-C in claude to exit; proxy will be stopped automatically.
If claude hangs >30s on the first prompt, run:  $HERE/check_model.sh $MODEL
EOF

# Resolve claude binary: prefer PATH, fall back to ./vendor/node/bin/claude
CLAUDE="$(command -v claude 2>/dev/null || true)"
[ -z "$CLAUDE" ] && CLAUDE="$HERE/vendor/node/bin/claude"
[ -x "$CLAUDE" ] || { echo "claude not found — run ./setup.sh first"; exit 1; }

# Tell Claude Code we're inside a sandbox. Cluster pods run as root by
# design (k8s isolation, NFS flat-permissions remap), and Claude Code
# refuses --dangerously-skip-permissions as root unless this is set.
# IS_SANDBOX=1 is the documented bypass for container environments.
export IS_SANDBOX=1
exec "$CLAUDE" --dangerously-skip-permissions
