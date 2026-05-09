#!/bin/bash
# One-shot installer: Node v22 + Claude Code + aiohttp into ./vendor/.
# Idempotent — safe to re-run.

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$HERE/vendor"
NODE_VERSION="v22.11.0"

mkdir -p "$VENDOR"

# ---- 0. Silence git's "dubious ownership" warning -------------------------
# The cluster's NFS-backed /scratch uses flat-permissions remapping: every
# write lands owned by the group's mapped uid:gid (e.g., 20088699:88699),
# which doesn't match `whoami` inside the container. Git refuses to operate
# on repos whose owner doesn't match the caller — register the repo root as
# safe so `git status` / `git pull` work without arguing.
#
# Don't use `git rev-parse` here: that command itself trips the same
# dubious-ownership refusal, so we'd never resolve the root. Compute it
# from the script location (setup.sh lives at <repo>/tools/claude-code-aiaas/).
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
if [ -d "$REPO_ROOT/.git" ]; then
  git config --global --add safe.directory "$REPO_ROOT" 2>/dev/null || true
fi

# ---- 1. Node ---------------------------------------------------------------
if [ -x "$VENDOR/node/bin/node" ]; then
  echo "✓ node already installed: $($VENDOR/node/bin/node --version)"
else
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64)  NODE_ARCH="x64" ;;
    aarch64) NODE_ARCH="arm64" ;;
    *) echo "unsupported arch: $ARCH"; exit 1 ;;
  esac
  TARBALL="node-${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz"
  URL="https://nodejs.org/dist/${NODE_VERSION}/${TARBALL}"
  echo "→ downloading $URL"
  curl -fsSL "$URL" -o "$VENDOR/$TARBALL"
  # --no-same-owner: the tarball ships uid/gid 1000, but on the cluster's
  # NFS-backed /scratch root-squashing strips CAP_CHOWN and tar bombs on
  # every entry. Drop ownership preservation; let extracted files inherit
  # the writing user.
  tar --no-same-owner -xJf "$VENDOR/$TARBALL" -C "$VENDOR/"
  rm "$VENDOR/$TARBALL"
  mv "$VENDOR/node-${NODE_VERSION}-linux-${NODE_ARCH}" "$VENDOR/node"
  echo "✓ node installed: $($VENDOR/node/bin/node --version)"
fi

export PATH="$VENDOR/node/bin:$PATH"

# ---- 2. Claude Code --------------------------------------------------------
if [ -x "$VENDOR/node/bin/claude" ]; then
  echo "✓ claude already installed: $(claude --version 2>/dev/null || echo '?')"
else
  echo "→ installing @anthropic-ai/claude-code (npm global into ./vendor/node/...)"
  "$VENDOR/node/bin/npm" install -g @anthropic-ai/claude-code 2>&1 | tail -5
  echo "✓ claude installed: $(claude --version 2>/dev/null || echo '?')"
fi

# ---- 3. aiohttp (for proxy) ------------------------------------------------
if python3 -c "import aiohttp" 2>/dev/null; then
  echo "✓ aiohttp already importable"
else
  echo "→ installing aiohttp (--user)"
  pip install --user --quiet aiohttp 2>&1 | tail -2
  python3 -c "import aiohttp; print('✓ aiohttp', aiohttp.__version__)"
fi

# ---- 4. Make scripts executable --------------------------------------------
chmod +x "$HERE/aiaas-claude.sh" "$HERE/check_model.sh" "$HERE/bootstrap.sh" 2>/dev/null || true

cat <<EOF

Setup complete. Tools installed under: $VENDOR/

Next:
  export AIAAS_KEY=sk--...        # get yours at https://portal.rcp.epfl.ch/aiaas/keys
  $HERE/aiaas-claude.sh           # menu-pick a model, launches claude

To add ./vendor/node/bin to your PATH for this shell:
  export PATH="$VENDOR/node/bin:\$PATH"
EOF
