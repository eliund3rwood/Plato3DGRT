"""
src/models/conditioning.py — Conditioning modules for PlatoControlNet.

Implements:
  - Resampler:            CLIP patch tokens (B×257×1024) → N appearance tokens (IP-Adapter+)
  - IPAttnProcessor2_0:  decoupled cross-attention for IP-Adapter injection
  - ReferenceAttnProcessor: decoder self-attention with reference K/V injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── PerceiverResampler for IP-Adapter+ ────────────────────────────────────────

class _ResamplerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q    = nn.Linear(dim, inner, bias=False)
        self.to_k    = nn.Linear(dim, inner, bias=False)
        self.to_v    = nn.Linear(dim, inner, bias=False)
        self.to_out  = nn.Linear(inner, dim, bias=False)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False),
        )

    def forward(self, q: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, Nq, _ = q.shape
        _, Nc, _ = ctx.shape
        Q = self.to_q(self.norm_q(q))
        K = self.to_k(self.norm_kv(ctx))
        V = self.to_v(self.norm_kv(ctx))
        Q = Q.view(B, Nq, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Nc, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Nc, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, Nq, self.heads * self.head_dim)
        q = q + self.to_out(out)
        q = q + self.ff(self.ff_norm(q))
        return q


class Resampler(nn.Module):
    """
    PerceiverResampler: CLIP patch tokens (B × N_clip × clip_dim) →
    appearance tokens (B × num_queries × output_dim).

    Uses learnable queries that cross-attend to all CLIP patch tokens,
    retaining spatial texture detail lost by global pooling.
    """

    def __init__(
        self,
        clip_dim:    int = 1024,
        depth:       int = 4,
        heads:       int = 16,
        head_dim:    int = 64,
        num_queries: int = 16,
        output_dim:  int = 768,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, clip_dim) * clip_dim ** -0.5)
        self.layers  = nn.ModuleList([
            _ResamplerLayer(clip_dim, heads, head_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(clip_dim)
        self.proj = nn.Linear(clip_dim, output_dim, bias=False)

    def forward(self, clip_tokens: torch.Tensor) -> torch.Tensor:
        """clip_tokens: B × N × clip_dim  →  B × num_queries × output_dim"""
        B = clip_tokens.shape[0]
        q = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, clip_tokens)
        return self.proj(self.norm(q))


# ── IP-Adapter cross-attention processor ─────────────────────────────────────

class IPAttnProcessor2_0(nn.Module):
    """
    Drop-in for diffusers AttnProcessor2_0 that adds a decoupled IP-Adapter
    image cross-attention stream.

    Text cross-attention runs normally. Image cross-attention uses separate
    K_ip / V_ip projections and its output is added scaled by ip_scale.
    ip_hidden_states passed via cross_attention_kwargs["ip_hidden_states"].
    """

    def __init__(self, hidden_size: int, cross_attention_dim: int, ip_scale: float = 1.0):
        super().__init__()
        self.ip_scale = ip_scale
        self.to_k_ip  = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip  = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states:         torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask:        torch.Tensor | None = None,
        temb:                  torch.Tensor | None = None,
        ip_hidden_states:      torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        B, seq_len, _ = hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, B)
            attention_mask = attention_mask.view(B, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        Q = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        K = attn.to_k(encoder_hidden_states)
        V = attn.to_v(encoder_hidden_states)

        inner_dim = K.shape[-1]
        head_dim  = inner_dim // attn.heads

        Q = Q.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        K = K.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        V = V.view(B, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(B, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(Q.dtype)

        # ── IP-Adapter image stream ───────────────────────────────────────────
        if ip_hidden_states is not None:
            ip_k = self.to_k_ip(ip_hidden_states).view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ip_v = self.to_v_ip(ip_hidden_states).view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ip_out = F.scaled_dot_product_attention(Q, ip_k, ip_v, dropout_p=0.0, is_causal=False)
            ip_out = ip_out.transpose(1, 2).reshape(B, -1, attn.heads * head_dim).to(Q.dtype)
            hidden_states = hidden_states + self.ip_scale * ip_out

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ── Reference attention processor ────────────────────────────────────────────

class ReferenceAttnProcessor(nn.Module):
    """
    Decoder self-attention processor that can capture and re-inject K/V from a
    clean reference encoding of I_A.

    Two modes (controlled by the caller via set_reference_capture /
    set_reference_inject helpers in build.py):

      capture mode  — runs normal self-attention and stores K, V in a bank.
      inject mode   — runs self-attention but concatenates the banked K, V so
                      Q attends to both the current noisy features and the
                      clean reference features from I_A.
    """

    def __init__(self):
        super().__init__()
        self.do_capture: bool = False
        self.do_inject:  bool = False
        self._bank_k: torch.Tensor | None = None
        self._bank_v: torch.Tensor | None = None

    def __call__(
        self,
        attn,
        hidden_states:         torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask:        torch.Tensor | None = None,
        temb:                  torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        B, seq_len, _ = hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, B)
            attention_mask = attention_mask.view(B, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Self-attention: K/V come from hidden_states (encoder_hidden_states is None)
        Q = attn.to_q(hidden_states)
        K = attn.to_k(hidden_states)
        V = attn.to_v(hidden_states)

        if self.do_capture:
            self._bank_k = K.detach()
            self._bank_v = V.detach()

        if self.do_inject and self._bank_k is not None:
            K = torch.cat([K, self._bank_k.to(K.dtype)], dim=1)
            V = torch.cat([V, self._bank_v.to(V.dtype)], dim=1)

        inner_dim = K.shape[-1]
        head_dim  = inner_dim // attn.heads
        Nq = Q.shape[1]
        Nk = K.shape[1]

        Q = Q.view(B, Nq, attn.heads, head_dim).transpose(1, 2)
        K = K.view(B, Nk, attn.heads, head_dim).transpose(1, 2)
        V = V.view(B, Nk, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(B, Nq, attn.heads * head_dim)
        hidden_states = hidden_states.to(Q.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states
