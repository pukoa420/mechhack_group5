#!/bin/bash
# Mock-request health check for AIaaS-hosted models.
# Run when Claude Code is hanging on the first prompt or returning empty replies.
#
# Usage:
#   ./check_model.sh                                # default: MiniMaxAI/MiniMax-M2.7
#   ./check_model.sh Qwen/Qwen3-30B-A3B-Instruct-2507
#
# Exits 0 if all three checks pass (auth, catalog, chat-completion), 1 otherwise.
# Prints which step failed so you can tell "401 Unauthorized" from "503 overloaded".

set -e

MODEL="${1:-MiniMaxAI/MiniMax-M2.7}"
BASE="${AIAAS_BASE:-https://inference.rcp.epfl.ch}"
KEY="${AIAAS_KEY:-${ANTHROPIC_AUTH_TOKEN:-}}"

if [ -z "$KEY" ]; then
  echo "Get your AIaaS key at: https://portal.rcp.epfl.ch/aiaas/keys"
  read -r -s -p "AIaaS API key (sk--...): " KEY
  echo
fi
[ -z "$KEY" ] && { echo "no key — aborting"; exit 1; }

echo "→ checking $BASE for model: $MODEL"
echo

# ---- 1. auth + catalog ------------------------------------------------------
HTTP=$(curl -sS -o /tmp/_models.json -w "%{http_code}" \
       --max-time 30 \
       -H "Authorization: Bearer $KEY" "$BASE/v1/models" || echo "000")
if [ "$HTTP" != "200" ]; then
  echo "✗ FAIL [auth]: HTTP $HTTP"
  echo "  body:"
  head -c 400 /tmp/_models.json
  echo
  if [ "$HTTP" = "401" ] || [ "$HTTP" = "403" ]; then
    echo "  → re-issue your key from the RCP portal"
  elif [ "$HTTP" = "000" ]; then
    echo "  → network unreachable; check VPN / DNS"
  fi
  exit 1
fi
N=$(python3 -c "import json; print(len(json.load(open('/tmp/_models.json')).get('data',[])))" 2>/dev/null || echo "?")
echo "✓ auth OK ($N models in catalog)"

if ! python3 -c "
import json, sys
d=json.load(open('/tmp/_models.json'))
ids=[m['id'] for m in d.get('data',[])]
sys.exit(0 if '$MODEL' in ids else 1)
" 2>/dev/null; then
  echo "✗ FAIL [catalog]: '$MODEL' is NOT in the AIaaS catalog"
  echo "  available models matching same family:"
  python3 -c "
import json
d=json.load(open('/tmp/_models.json'))
prefix = '$MODEL'.split('/')[0]
for m in d.get('data',[]):
    if m['id'].split('/')[0] == prefix:
        print('   ', m['id'])
" 2>/dev/null || true
  echo "  → fall back: ./aiaas-claude.sh and pick another model"
  exit 1
fi
echo "✓ '$MODEL' is in catalog"
echo

# ---- 2. mock chat completion ------------------------------------------------
echo "→ sending mock chat completion (max_tokens=10, max-time 60s)..."
T0=$(python3 -c "import time; print(time.time())")
HTTP=$(curl -sS -o /tmp/_chat.json -w "%{http_code}" \
       --max-time 60 \
       -H "Authorization: Bearer $KEY" \
       -H "Content-Type: application/json" \
       -d "{
         \"model\": \"$MODEL\",
         \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word: OK\"}],
         \"max_tokens\": 10,
         \"reasoning\": {\"enabled\": false}
       }" \
       "$BASE/v1/chat/completions" || echo "000")
T1=$(python3 -c "import time; print(time.time())")
ELAPSED=$(python3 -c "print(f'{$T1 - $T0:.1f}')")

if [ "$HTTP" != "200" ]; then
  echo "✗ FAIL [chat]: HTTP $HTTP after ${ELAPSED}s"
  echo "  body:"
  head -c 500 /tmp/_chat.json
  echo
  case "$HTTP" in
    503|504|524) echo "  → model is overloaded / GPU-bound; try a different model" ;;
    400)         echo "  → request rejected (params mismatch). Show this output to organizers." ;;
    000)         echo "  → curl timed out (>60s). Model is wedged; fall back to another." ;;
  esac
  exit 1
fi

CONTENT=$(python3 -c "
import json
d=json.load(open('/tmp/_chat.json'))
c = d.get('choices',[{}])[0].get('message',{}).get('content') or ''
print(c.strip()[:200])
" 2>/dev/null)

if [ -z "$CONTENT" ]; then
  echo "✗ FAIL [empty response]: model returned no content after ${ELAPSED}s"
  echo "  full payload:"
  head -c 500 /tmp/_chat.json
  echo
  echo "  → this often means the model is stuck in <think>...</think> with no completion."
  echo "  → fall back: ./aiaas-claude.sh and pick a non-thinking model (option 2/3/4)."
  exit 1
fi

echo "✓ chat completion succeeded in ${ELAPSED}s"
echo "  reply: $CONTENT"
echo
echo "All checks passed — model is healthy. Launch with:"
echo "  ./aiaas-claude.sh"
