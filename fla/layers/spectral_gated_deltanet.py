# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class SpectralGatedDeltaNet(nn.Module):
    """
    Spectral Gated DeltaNet layer:
    - A bank of damped-oscillatory fast-weight memories (modes).
    - Delta-rule write correction per mode.
    - Optional output gate.

    This mirrors the high-level API of GatedDeltaNet so it can be used as a drop-in
    linear-attention backend in downstream projects.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 256,
        num_heads: int = 8,
        num_v_heads: int | None = None,
        num_modes: int = 32,
        mode: str = "chunk",
        use_gate: bool = True,
        use_qk_l2norm_in_kernel: bool = True,
        layer_idx: int | None = None,
        **kwargs,
    ) -> SpectralGatedDeltaNet:
        super().__init__()
        del kwargs
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.num_modes = int(max(1, num_modes))
        self.layer_idx = layer_idx
        self.use_gate = use_gate
        self.mode = mode
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        if self.key_dim <= 0 or self.value_dim <= 0:
            raise ValueError("Invalid dimensions for SpectralGatedDeltaNet")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.beta_proj = nn.Linear(hidden_size, self.num_v_heads * self.num_modes, bias=False)
        self.trans_gate_proj = nn.Linear(hidden_size, self.num_v_heads * self.num_modes, bias=False)
        self.mode_logits = nn.Parameter(torch.zeros(self.num_v_heads, self.num_modes))
        self.log_decay = nn.Parameter(torch.zeros(self.num_v_heads, self.num_modes))
        # RoPE-like spectral phase: omega depends on key feature-pair index.
        # omega[h, r, m] = inv_freq[m] * exp(omega_log_scale[h, r]).
        self.omega_log_scale = nn.Parameter(torch.zeros(self.num_v_heads, self.num_modes))
        pair_idx = torch.arange(0, self.head_k_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (pair_idx / self.head_k_dim))
        self.register_buffer("omega_inv_freq", inv_freq, persistent=False)
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        else:
            self.g_proj = None
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Runtime diagnostics
        self._last_beta_mean = None
        self._last_trans_gate_mean = None
        self._last_output_gate_mean = None

    @staticmethod
    def _apply_rotary_emb(x, cos, sin):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], -1)

    @staticmethod
    def _positions_from_cu(cu_seqlens: torch.Tensor, device, dtype):
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if not lens:
            return torch.zeros(0, device=device, dtype=dtype)
        pos = [torch.arange(l, device=device, dtype=dtype) for l in lens]
        return torch.cat(pos, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        value_residual: torch.Tensor | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        del output_attentions, kwargs
        input_dtype = hidden_states.dtype
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] for padding."
            )

        batch_size, q_len, _ = hidden_states.shape
        indices = None
        cu_seqlens = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        if value_residual is not None:
            if value_residual.dtype != v.dtype:
                value_residual = value_residual.to(v.dtype)
            v = v + value_residual

        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads, d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", h=self.num_heads, d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", h=self.num_v_heads, d=self.head_v_dim)

        # If using grouped value heads, repeat q/k to align with value heads.
        if self.num_v_heads > self.num_heads:
            repeat_factor = self.num_v_heads // self.num_heads
            q = repeat(q, "... h d -> ... (h g) d", g=repeat_factor)
            k = repeat(k, "... h d -> ... (h g) d", g=repeat_factor)

        B, T, H, Dk = k.shape
        Dv = v.shape[-1]
        dtype = hidden_states.dtype
        kernel_dtype = v.dtype

        beta = torch.sigmoid(self.beta_proj(hidden_states).view(B, T, H, self.num_modes))  # (B,T,H,R)
        trans_gate = torch.sigmoid(self.trans_gate_proj(hidden_states).view(B, T, H, self.num_modes))  # (B,T,H,R)
        self._last_beta_mean = beta.detach().mean()
        self._last_trans_gate_mean = trans_gate.detach().mean()

        # RoPE-style spectral rotation applied to q/k, then integrate state with fused kernels.
        omega_scale = torch.exp(self.omega_log_scale).to(device=hidden_states.device, dtype=dtype)  # (H,R)
        inv_freq = self.omega_inv_freq.to(device=hidden_states.device, dtype=dtype)  # (Dk/2,)
        if attention_mask is not None and cu_seqlens is not None:
            pos = self._positions_from_cu(cu_seqlens, device=hidden_states.device, dtype=dtype)  # (T,)
        else:
            pos = torch.arange(T, device=hidden_states.device, dtype=dtype)
        angles = pos[:, None, None, None] * omega_scale[None, :, :, None] * inv_freq[None, None, None, :]  # (T,H,R,Dk/2)
        cos = torch.cos(angles).to(dtype=q.dtype).unsqueeze(0)  # (1,T,H,R,Dk/2)
        sin = torch.sin(angles).to(dtype=q.dtype).unsqueeze(0)

        # Keep shared k/v tensors (no H*R expansion); run one kernel per mode and mix outputs.
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            mode = "chunk"
        mode_w = F.softmax(self.mode_logits.to(device=hidden_states.device, dtype=dtype), dim=-1)  # (H,R)
        rho = torch.exp(-F.softplus(self.log_decay)).to(device=hidden_states.device, dtype=dtype)  # (H,R)
        o = torch.zeros(B, T, H, Dv, device=hidden_states.device, dtype=kernel_dtype)
        for r in range(self.num_modes):
            cos_r = cos[:, :, :, r, :]  # (1,T,H,Dk/2)
            sin_r = sin[:, :, :, r, :]
            q_r = self._apply_rotary_emb(q, cos_r, sin_r).to(kernel_dtype)
            k_r = self._apply_rotary_emb(k, cos_r, sin_r).to(kernel_dtype)
            v_r = v if v.dtype == kernel_dtype else v.to(kernel_dtype)
            decay_r = (trans_gate[..., r] * rho[:, r].unsqueeze(0).unsqueeze(0)).clamp_min(1e-6)  # (B,T,H)
            g_r = decay_r.log()
            beta_r = beta[..., r].to(kernel_dtype)
            if mode == "chunk":
                o_r, _ = chunk_gated_delta_rule(
                    q=q_r, k=k_r, v=v_r, g=g_r, beta=beta_r,
                    initial_state=None, output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                )
            elif mode == "fused_recurrent":
                o_r, _ = fused_recurrent_gated_delta_rule(
                    q=q_r, k=k_r, v=v_r, g=g_r, beta=beta_r,
                    initial_state=None, output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                )
            else:
                raise NotImplementedError(f"Not supported mode `{mode}`.")
            o = o + o_r * mode_w[:, r].to(o_r.dtype).view(1, 1, H, 1)
        if self.g_proj is not None:
            gate = torch.sigmoid(rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", h=H, d=Dv))
            self._last_output_gate_mean = gate.detach().mean()
            o = o * gate
        else:
            self._last_output_gate_mean = None
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if o.dtype != input_dtype:
            o = o.to(input_dtype)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        # This implementation currently does not maintain recurrent cache across calls.
        return o, None, past_key_values
