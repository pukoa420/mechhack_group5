"""Memory-efficient SDPA for Gemma-4-31B-it on H100.

Provides a chunked-Q replacement for torch.nn.functional.scaled_dot_product_attention
that handles head_dim=512 (Gemma-4 global layers) without OOM, plus a context
manager so the patch is scoped to the mech-interp forward and never leaks into
generate() or other code paths.

Why this exists
---------------
- Gemma-4-31B has interleaved sliding-window (head_dim=256) + global (head_dim=512)
  attention layers.
- flash-attn 2.x and 3.x stable cap head_dim at 256 -> reject Gemma-4 global layers.
- PyTorch sdpa "efficient"/"cudnn" backends are also unavailable for head_dim=512
  in this PyTorch build.
- Only sdpa "math" accepts -> O(N^2) memory -> OOMs above ~3-4k tokens on H100 80GB.
- expandable_segments doesn't help (it's real memory pressure, not fragmentation).
- FlexAttention's Triton autotuner can't find a config for head_dim=512 on H100
  ("out of resource: shared memory").

Solution: chunk along the Q dimension. Each call's score tensor drops from
H * N^2 * 4 bytes (~8.6 GB at N=8192) to H * CHUNK_Q * N * 4 bytes (~1 GB),
and the math kernel handles each chunk fine.

Correctness notes
-----------------
- enable_gqa=True (Gemma-4 32:4 GQA) flows in via **kwargs. The wrapper forwards
  **kwargs to the orig sdpa or you get "tensor a (32) must match tensor b (4) at
  dim 1".
- Causal alignment is preserved with an explicit boolean mask per chunk: True
  positions are allowed (matches PyTorch's documented sdpa boolean semantics).
  This is the upper-left causal alignment, correct for Nq==Nk prefill. Do NOT
  use this for KV-cached multi-token decoding where bottom-right alignment is
  expected.
- Broadcast attn_masks (e.g. [B,1,1,S] padding masks) are preserved via
  _slice_attn_mask: only sliced when the mask has an explicit Q axis equal to Nq.
"""
import os
from contextlib import contextmanager
import torch
import torch.nn.functional as F


CHUNK_Q = int(os.environ.get("CHUNK_Q", "1024"))
_orig_sdpa = F.scaled_dot_product_attention


def _slice_attn_mask(attn_mask, start, end, q_len):
    """Slice a mask along the Q dimension only when it has an explicit Q axis.

    Preserves broadcast masks like [B, 1, 1, S] or [1, S]. Slicing those
    naively (attn_mask[..., i:end, :]) would give an empty tensor on chunk #2+
    because shape[-2] == 1 != q_len.
    """
    if attn_mask is None:
        return None
    if attn_mask.ndim >= 2 and attn_mask.shape[-2] == q_len:
        return attn_mask[..., start:end, :]
    return attn_mask


def chunked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, **kwargs):
    """Drop-in replacement for F.scaled_dot_product_attention with chunked Q.

    Forward path equivalent to a single sdpa call when q_len <= CHUNK_Q
    (early-return). Otherwise slice Q into CHUNK_Q-sized blocks and call
    orig sdpa per chunk; concatenate outputs along the Q axis.
    """
    q_len = query.shape[-2]
    k_len = key.shape[-2]

    if q_len <= CHUNK_Q:
        return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal,
                          scale=scale, **kwargs)

    out_chunks = []
    for start in range(0, q_len, CHUNK_Q):
        end = min(start + CHUNK_Q, q_len)
        q_chunk = query[..., start:end, :]

        if is_causal and attn_mask is None:
            # Reconstruct upper-left causal mask for this Q slice.
            # Boolean sdpa mask semantics: True = participates in attention.
            row = torch.arange(start, end, device=query.device)
            col = torch.arange(k_len, device=query.device)
            chunk_mask = col[None, :] <= row[:, None]
            chunk_is_causal = False
        else:
            chunk_mask = _slice_attn_mask(attn_mask, start, end, q_len)
            chunk_is_causal = is_causal

        out_chunks.append(_orig_sdpa(q_chunk, key, value,
                                      attn_mask=chunk_mask,
                                      dropout_p=dropout_p,
                                      is_causal=chunk_is_causal,
                                      scale=scale, **kwargs))
    return torch.cat(out_chunks, dim=-2)


@contextmanager
def chunked_sdpa_scope():
    """Patch F.scaled_dot_product_attention for the duration of the with-block.

    Restored on exit (including exceptions). Anything called outside this
    scope - including model.generate() - sees the original sdpa.
    """
    F.scaled_dot_product_attention = chunked_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _orig_sdpa
