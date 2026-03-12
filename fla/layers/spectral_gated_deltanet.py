# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input

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
        use_gate: bool = True,
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
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] for padding."
            )

        batch_size, q_len, _ = hidden_states.shape
        indices = None
        if attention_mask is not None:
            indices, _, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        if value_residual is not None:
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

        beta = torch.sigmoid(self.beta_proj(hidden_states).view(B, T, H, self.num_modes))
        trans_gate = torch.sigmoid(self.trans_gate_proj(hidden_states).view(B, T, H, self.num_modes))
        self._last_beta_mean = beta.detach().mean()
        self._last_trans_gate_mean = trans_gate.detach().mean()

        omega_scale = torch.exp(self.omega_log_scale).to(device=hidden_states.device, dtype=dtype)  # (H, R)
        omega_pairs = omega_scale.unsqueeze(-1) * self.omega_inv_freq.to(device=hidden_states.device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # (H, R, Dk/2)
        omega_full = torch.repeat_interleave(omega_pairs, 2, dim=-1)  # (H, R, Dk), pairwise frequency sharing
        cos_w = torch.cos(omega_full)
        sin_w = torch.sin(omega_full)
        rho = torch.exp(-F.softplus(self.log_decay)).to(device=hidden_states.device, dtype=dtype)
        mode_w = F.softmax(self.mode_logits.to(device=hidden_states.device, dtype=dtype), dim=-1)

        state_r = torch.zeros(B, H, self.num_modes, Dv, Dk, device=hidden_states.device, dtype=dtype)
        state_i = torch.zeros_like(state_r)
        y_steps = []

        for t in range(T):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]

            decay = (trans_gate[:, t] * rho.unsqueeze(0)).unsqueeze(-1).unsqueeze(-1)
            c = cos_w.unsqueeze(0).unsqueeze(-2)  # (1, H, R, 1, Dk)
            s = sin_w.unsqueeze(0).unsqueeze(-2)
            rot_r = decay * (c * state_r - s * state_i)
            rot_i = decay * (s * state_r + c * state_i)

            pred = torch.einsum("bhrvk,bhk->bhrv", rot_r, k_t)
            err = v_t.unsqueeze(2) - pred
            write = torch.einsum("bhrv,bhk->bhrvk", err, k_t)
            b = beta[:, t].unsqueeze(-1).unsqueeze(-1)
            state_r = rot_r + b * write
            state_i = rot_i

            read = torch.einsum("bhrvk,bhk->bhrv", state_r, q_t)
            y_t = torch.einsum("bhrv,hr->bhv", read, mode_w)
            y_steps.append(y_t)

        o = torch.stack(y_steps, dim=1)  # (B, T, H, Dv)
        if self.g_proj is not None:
            gate = torch.sigmoid(rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", h=H, d=Dv))
            self._last_output_gate_mean = gate.detach().mean()
            o = o * gate
        else:
            self._last_output_gate_mean = None
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        # This implementation currently does not maintain recurrent cache across calls.
        return o, None, past_key_values
