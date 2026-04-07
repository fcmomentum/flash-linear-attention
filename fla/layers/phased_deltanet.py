# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule.phase_utils import build_fixed_rope_phase

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class PhasedDeltaNet(nn.Module):
    """
    Rotary-packed phased recurrent layer inspired by the Phased Delta GatedNet recurrence:

        h_t = (alpha_t * exp(i * phi_t)) * h_{t-1} + (beta_t * exp(i * psi_t)) * v_t

    The phased slice of each head is represented in RoPE-style interleaved pairs
    within a single real-valued state tensor. The remaining channels are stored as
    ordinary real channels.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        num_phase_channels: int | None = None,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        phase_base: float = 10000.0,
        **kwargs,
    ) -> PhasedDeltaNet:
        super().__init__()

        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.num_phase_channels = num_phase_channels if num_phase_channels is not None else self.head_v_dim
        self.num_nonphase_channels = self.head_v_dim - self.num_phase_channels
        self.layer_idx = layer_idx
        self.phase_base = phase_base

        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by "
                f"num_v_heads * head_dim={self.num_v_heads * self.head_dim}.",
            )
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}.",
            )
        if not (0 <= self.num_phase_channels <= self.head_v_dim):
            raise ValueError(
                f"num_phase_channels={self.num_phase_channels} must be between 0 and head_v_dim={self.head_v_dim}.",
            )
        if self.num_phase_channels % 2 != 0:
            raise ValueError(
                f"num_phase_channels={self.num_phase_channels} must be even for RoPE-style paired rotations.",
            )

        self.q_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        phase_inv_freq = torch.empty(self.num_phase_channels // 2, dtype=torch.float32)
        if self.num_phase_channels > 0:
            phase_inv_freq = 1.0 / (
                phase_base ** (torch.arange(0, self.num_phase_channels, 2, dtype=torch.float32) / self.num_phase_channels)
            )
        self.register_buffer("phase_inv_freq", phase_inv_freq, persistent=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _apply_nonphase_activation(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, '... (h d) -> ... h d', h=self.num_v_heads, d=self.head_v_dim)
        if self.num_phase_channels < self.head_v_dim:
            x_nonphase = x[..., self.num_phase_channels:]
            x = torch.cat((x[..., :self.num_phase_channels], F.silu(x_nonphase)), dim=-1)
        return rearrange(x, '... h d -> ... (h d)')

    def _restrict_phase_channels(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_phase_channels == self.head_v_dim:
            return x
        x = x.clone()
        x[..., self.num_phase_channels:] = 0
        return x

    def _rotate_phase_channels(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if self.num_phase_channels == 0:
            return x
        x_phase = x[..., :self.num_phase_channels]
        phase_pairs = phase[..., :self.num_phase_channels:2]
        cos = phase_pairs.float().cos().to(x.dtype).repeat_interleave(2, dim=-1)
        sin = phase_pairs.float().sin().to(x.dtype).repeat_interleave(2, dim=-1)
        x_even = x_phase[..., ::2]
        x_odd = x_phase[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
        x_phase = x_phase * cos + x_rot * sin
        if self.num_phase_channels == self.head_v_dim:
            return x_phase
        return torch.cat((x_phase, x[..., self.num_phase_channels:]), dim=-1)

    def _scan_sequences(
        self,
        theta: torch.Tensor,
        psi: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        phi: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cu_seqlens is not None:
            if theta.shape[0] != 1:
                raise ValueError("`cu_seqlens` requires batch size 1.")
            return self._scan_varlen(theta, psi, value, alpha, beta, phi, initial_state, cu_seqlens)
        return self._scan_padded(theta, psi, value, alpha, beta, phi, initial_state, attention_mask)

    def _scan_padded(
        self,
        theta: torch.Tensor,
        psi: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        phi: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_heads, head_dim = value.shape
        if initial_state is None:
            state = value.new_zeros(batch_size, num_heads, head_dim)
        else:
            state = initial_state
        output = value.new_zeros(batch_size, seq_len, num_heads, head_dim)

        valid_mask = None
        if attention_mask is not None:
            valid_mask = attention_mask[:, -seq_len:].to(torch.bool)

        for t in range(seq_len):
            retained = self._rotate_phase_channels(state, phi[:, t])
            retained = retained * alpha[:, t].unsqueeze(-1)
            write = self._rotate_phase_channels(value[:, t], psi[:, t])
            updated = retained + beta[:, t].unsqueeze(-1) * write
            step_output = self._rotate_phase_channels(updated, -theta[:, t])

            if valid_mask is not None:
                valid = valid_mask[:, t].view(batch_size, 1, 1)
                state = torch.where(valid, updated, state)
                output[:, t] = torch.where(valid, step_output, torch.zeros_like(step_output))
            else:
                state = updated
                output[:, t] = step_output

        return output, state

    def _scan_varlen(
        self,
        theta: torch.Tensor,
        psi: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        phi: torch.Tensor,
        initial_state: torch.Tensor | None,
        cu_seqlens: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_tokens = value.shape[1]
        num_sequences = len(cu_seqlens) - 1
        _, _, num_heads, head_dim = value.shape
        output = value.new_zeros(1, total_tokens, num_heads, head_dim)
        final_state = value.new_zeros(num_sequences, num_heads, head_dim)

        for seq_idx in range(num_sequences):
            start = cu_seqlens[seq_idx].item()
            end = cu_seqlens[seq_idx + 1].item()
            if initial_state is None:
                state = value.new_zeros(num_heads, head_dim)
            else:
                state = initial_state[seq_idx]

            for token_idx in range(start, end):
                retained = self._rotate_phase_channels(state, phi[0, token_idx])
                retained = retained * alpha[0, token_idx].unsqueeze(-1)
                write = self._rotate_phase_channels(value[0, token_idx], psi[0, token_idx])
                state = retained + beta[0, token_idx].unsqueeze(-1) * write
                output[0, token_idx] = self._rotate_phase_channels(state, -theta[0, token_idx])

            final_state[seq_idx] = state

        return output, final_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(self, past_key_values)
        cu_seqlens = kwargs.get('cu_seqlens')
        conv_mask = attention_mask[:, -q_len:] if attention_mask is not None else None

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = (None, None, None)
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            theta, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            psi, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            value, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            theta = self._apply_nonphase_activation(theta)
            psi = self._apply_nonphase_activation(psi)
            value = self._apply_nonphase_activation(value)
        else:
            theta = self._apply_nonphase_activation(self.q_proj(hidden_states))
            psi = self._apply_nonphase_activation(self.k_proj(hidden_states))
            value = self._apply_nonphase_activation(self.v_proj(hidden_states))
            if conv_mask is not None:
                value = value * conv_mask.unsqueeze(-1)
                theta = theta * conv_mask.unsqueeze(-1)
                psi = psi * conv_mask.unsqueeze(-1)
            conv_state_q, conv_state_k, conv_state_v = (None, None, None)

        theta = rearrange(theta, '... (h d) -> ... h d', d=self.head_v_dim)
        psi = rearrange(psi, '... (h d) -> ... h d', d=self.head_v_dim)
        value = rearrange(value, '... (h d) -> ... h d', d=self.head_v_dim)
        phase_offset = 0
        cache_position = kwargs.get('cache_position')
        if cache_position is not None:
            if torch.is_tensor(cache_position):
                phase_offset = int(cache_position.reshape(-1)[0].item())
            else:
                phase_offset = int(cache_position)
        elif past_key_values is not None and attention_mask is None:
            phase_offset = int(past_key_values.get_seq_length())
        phi = build_fixed_rope_phase(
            inv_freq=self.phase_inv_freq,
            batch_size=value.shape[0],
            seq_len=value.shape[1],
            num_heads=self.num_v_heads,
            head_v_dim=self.head_v_dim,
            num_phase_channels=self.num_phase_channels,
            device=value.device,
            dtype=value.dtype,
            cu_seqlens=cu_seqlens,
            offset=phase_offset,
        )
        theta = self._restrict_phase_channels(theta)
        psi = self._restrict_phase_channels(psi)
        alpha = torch.exp(-torch.nn.functional.softplus(self.a_proj(hidden_states).float())).to(value.dtype)
        beta = self.b_proj(hidden_states).float().sigmoid().to(value.dtype)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        output, recurrent_state = self._scan_sequences(
            theta=theta,
            psi=psi,
            value=value,
            alpha=alpha,
            beta=beta,
            phi=phi,
            initial_state=recurrent_state,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
        )

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        if self.use_gate:
            gate = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            output = self.o_norm(output, gate)
        else:
            output = self.o_norm(output)
        output = rearrange(output, 'b t h d -> b t (h d)')
        output = self.o_proj(output)
        return output, None, past_key_values
