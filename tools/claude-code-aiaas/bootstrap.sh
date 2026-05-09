#!/bin/bash
# Bootstrap the aiaas tools after a pod resume (when $HOME is wiped).
# Source this once on each fresh shell; idempotent.
#
# Usage (assuming you put this directory on a persistent volume):
#   source /path/to/persistent/aiaas-tools/bootstrap.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add vendored node + user-pip bin to PATH
export PATH="$HERE/vendor/node/bin:$HOME/.local/bin:$PATH"

# Convenience symlinks in $HOME so `~/aiaas-claude.sh` works
ln -sf "$HERE/aiaas-claude.sh" "$HOME/aiaas-claude.sh" 2>/dev/null
ln -sf "$HERE/check_model.sh"  "$HOME/check_model.sh"  2>/dev/null

# (Optional) re-register an MCP server with Claude Code if one exists alongside.
# Drop your MCP server scripts in this directory; they're picked up here.
if [ -x "$HERE/vendor/node/bin/claude" ] && [ -f "$HERE/ddg_search_mcp.py" ]; then
  if ! "$HERE/vendor/node/bin/claude" mcp list 2>/dev/null | grep -q ddg-search; then
    "$HERE/vendor/node/bin/claude" mcp add ddg-search python3 "$HERE/ddg_search_mcp.py" 2>/dev/null || true
  fi
fi

echo "[aiaas-bootstrap] tools at $HERE"
echo "  ~/aiaas-claude.sh → $HERE/aiaas-claude.sh"
echo "  ~/check_model.sh  → $HERE/check_model.sh"
echo "  claude:            $(command -v claude 2>/dev/null || echo 'not in PATH')"
echo "Run: ~/aiaas-claude.sh"
