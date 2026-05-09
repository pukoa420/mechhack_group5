"""LLM clients for the edit-agent track.

Supports both OpenRouter (any frontier model) and EPFL AIaaS (Qwen/Kimi/GLM).
Both honor `response_format={"type": "json_schema", ...}` for guaranteed valid output.

Disable thinking on Anthropic-format proxies (vLLM-served Kimi, Qwen-Thinking,
DeepSeek-V4-Pro, etc.) using `reasoning.enabled=False` (OpenRouter) or
`reasoning.effort=none` (AIaaS / OpenRouter both).
"""
from __future__ import annotations
import os, json, time
from typing import Any
import httpx


class LLMError(Exception):
    pass


class _BaseClient:
    base_url: str
    model: str
    extra: dict

    def __init__(self, model: str, *, reasoning: dict | None = None, max_tokens: int = 1500):
        self.model = model
        self.max_tokens = max_tokens
        self.extra = {}
        if reasoning is not None:
            self.extra["reasoning"] = reasoning

    def call(self, system: str, user: str, *, schema: dict | None = None,
             max_tokens: int | None = None, timeout: float = 180.0) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens or self.max_tokens,
            **self.extra,
        }
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": schema}
        t0 = time.time()
        for attempt in range(3):
            try:
                with httpx.Client(timeout=timeout) as c:
                    r = c.post(f"{self.base_url}/v1/chat/completions",
                               headers={"Authorization": f"Bearer {self.api_key}",
                                        "Content-Type": "application/json"},
                               json=body)
                if r.status_code == 429:
                    time.sleep(15 * (2 ** attempt)); continue
                if r.status_code != 200:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                d = r.json()
                if "error" in d: raise LLMError(f"API err: {d['error']}")
                msg = d["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                if not content:
                    raise LLMError(f"empty content (usage={d.get('usage')})")
                if schema:
                    try: parsed = json.loads(content)
                    except Exception as e:
                        raise LLMError(f"json parse: {e}; head={content[:200]!r}")
                    return {"parsed": parsed, "content": content,
                            "usage": d.get("usage", {}),
                            "elapsed_s": time.time() - t0}
                return {"content": content, "usage": d.get("usage", {}),
                        "elapsed_s": time.time() - t0}
            except (httpx.HTTPError, LLMError) as e:
                if attempt == 2: raise
                time.sleep(3 * (2 ** attempt))


class OpenRouterClient(_BaseClient):
    base_url = "https://openrouter.ai/api"

    def __init__(self, model: str, *, reasoning: dict | None = None, **kwargs):
        super().__init__(model=model, reasoning=reasoning, **kwargs)
        self.api_key = os.environ.get("OPENROUTER_KEY")
        if not self.api_key:
            raise RuntimeError("set OPENROUTER_KEY")


class AIaaSClient(_BaseClient):
    base_url = "https://inference.rcp.epfl.ch"

    def __init__(self, model: str, *, reasoning: dict | None = None, **kwargs):
        super().__init__(model=model, reasoning=reasoning, **kwargs)
        self.api_key = os.environ.get("AIAAS_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if not self.api_key:
            raise RuntimeError("set AIAAS_KEY")


# Schema for the standard edit format
EDITS_SCHEMA = {
    "name": "edits_response", "strict": True,
    "schema": {
        "type": "object", "required": ["edits"],
        "properties": {"edits": {"type": "array", "items": {
            "type": "object", "required": ["start_pos", "original_text", "replacement"],
            "properties": {
                "start_pos":     {"type": "integer"},
                "original_text": {"type": "string"},
                "replacement":   {"type": "string"},
            }, "additionalProperties": False,
        }}}, "additionalProperties": False,
    },
}


JUDGE_SCHEMA = {
    "name": "intent_judge_response", "strict": True,
    "schema": {
        "type": "object",
        "required": ["score", "intent_preserved", "reason"],
        "properties": {
            "score":            {"type": "integer"},
            "intent_preserved": {"type": "boolean"},
            "reason":           {"type": "string"},
        },
        "additionalProperties": False,
    },
}


# --- Reasoning-disable presets per model class ---
def reasoning_off_for(model: str) -> dict | None:
    """Return the right reasoning-disable kwarg for OR-routed models that need it."""
    m = model.lower()
    # Models that emit content cleanly with reasoning disabled
    if "deepseek-v4-pro" in m or "kimi" in m or "thinking" in m:
        return {"enabled": False}
    # Qwen3.5/3.6 thinking sometimes needs effort=none instead
    if "qwen3.6" in m or "qwen3.5" in m:
        return {"effort": "none"}
    return None


# --- Convenience constructors ---
def make_editor(name: str = "minimax-m2.7") -> _BaseClient:
    """Pick a sensible editor.

    Default is MiniMax-M2.7 (AIaaS, ~8-10 s/call) — same model the
    hackathon uses across edit agents and judges, so results stay
    comparable. Reasoning is kept ON for MiniMax (it refuses or no-ops
    when disabled). Other choices below are kept for flexibility.

    name: 'minimax-m2.7' (default, recommended), 'kimi-k2.6' (thinking,
          OpenRouter), 'qwen3-30b' (fast/cheap), 'qwen3-235b' (smarter),
          'deepseek-v4-pro' (slow but best multi-token rewrites).
    """
    if name == "minimax-m2.7":
        return AIaaSClient("MiniMaxAI/MiniMax-M2.7")
    if name == "kimi-k2.6":
        return OpenRouterClient("moonshotai/kimi-k2.6", reasoning={"enabled": False})
    if name == "qwen3-30b":
        return AIaaSClient("Qwen/Qwen3-30B-A3B-Instruct-2507")
    if name == "qwen3-235b":
        return AIaaSClient("Qwen/Qwen3-235B-A22B-Instruct-2507-fp8")
    if name == "deepseek-v4-pro":
        return OpenRouterClient("deepseek/deepseek-v4-pro", reasoning={"enabled": False})
    raise ValueError(f"unknown editor: {name}")


def make_judge(name: str = "minimax-m2.7") -> _BaseClient:
    """Default judge for refusal/compliance + intent-preservation tasks.

    Pinned to MiniMax-M2.7 by default so every submission is graded by
    the same model. Any of `make_editor`'s options is also valid here —
    MiniMax is preferred because the same model also drives the edit
    agent path, which keeps the rubric consistent across the pipeline.
    """
    return make_editor(name)
