# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import torch.nn.functional as F
from einops import rearrange

from fla.ops.gated_delta_rule.phase_utils import rotate_phase_channels


def _run_recurrent_gated_delta_rule_single_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    h: torch.Tensor,
):
    H, T, K = q.shape
    V = v.shape[-1]
    o = torch.zeros(H, T, V, device=v.device, dtype=torch.float32)

    for i in range(T):
        b_q = q[:, i]
        b_k = k[:, i]
        b_v = v[:, i]
        h = h * g[:, i].exp()[:, None, None]
        delta = beta[:, i, None] * (b_v - (h * b_k[..., None]).sum(-2))
        h = h + b_k.unsqueeze(-1) * delta.unsqueeze(-2)
        o[:, i] = torch.einsum('hk,hkv->hv', b_q * scale, h)
    return o, h


def naive_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    """
    Reference PyTorch implementation of recurrent gated delta rule.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]
        scale: float, optional
        initial_state: [B, H, K, V], optional
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    orig_dtype = v.dtype
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    if scale is None:
        scale = q.shape[-1] ** -0.5

    o = torch.zeros(B, H, T, V, device=v.device, dtype=torch.float32)
    h = torch.zeros(B, H, K, V, device=v.device, dtype=torch.float32)
    if initial_state is not None:
        h = initial_state.to(torch.float32)

    for b in range(B):
        o[b], h[b] = _run_recurrent_gated_delta_rule_single_sequence(
            q=q[b],
            k=k[b],
            v=v[b],
            beta=beta[b],
            g=g[b],
            scale=scale,
            h=h[b],
        )

    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous().to(orig_dtype)
    return o, h


def _run_rank1_dc_single_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    scale: float,
    h: torch.Tensor,
    b: torch.Tensor,
    phase: torch.Tensor | None = None,
    num_phase_channels: int = 0,
):
    H, T, K = q.shape
    V = v.shape[-1]
    o = torch.zeros(H, T, V, device=v.device, dtype=torch.float32)

    for i in range(T):
        phase_i = None if phase is None else phase[i]
        b_q = q[:, i]
        b_k = k[:, i]
        b_v = v[:, i]
        h = rotate_phase_channels(h, phase_i, num_phase_channels=num_phase_channels)
        b = rotate_phase_channels(b, phase_i, num_phase_channels=num_phase_channels)
        alpha = g[:, i].exp()
        h = h * alpha[:, None, None]
        b = b * alpha[:, None]
        v_rot = rotate_phase_channels(b_v, phase_i, num_phase_channels=num_phase_channels)
        pred = (h * b_k[..., None]).sum(-2) - lambda_k[:, i, None] * b
        delta = beta[:, i, None] * (v_rot - pred)
        h = h + b_k.unsqueeze(-1) * delta.unsqueeze(-2)
        b = b + delta
        h_out = rotate_phase_channels(h, None if phase_i is None else -phase_i, num_phase_channels=num_phase_channels)
        b_out = rotate_phase_channels(b, None if phase_i is None else -phase_i, num_phase_channels=num_phase_channels)
        o[:, i] = torch.einsum('hk,hkv->hv', b_q * scale, h_out) - lambda_q[:, i, None] * b_out
    return o, h, b


def naive_recurrent_gated_delta_rule_rank1_dc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    scale: float = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    phase: torch.Tensor | None = None,
    num_phase_channels: int = 0,
):
    """
    Reference PyTorch implementation of gated delta rule with rank-1 DC removal.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]
        lambda_q: [B, T, H]
        lambda_k: [B, T, H]
        scale: float, optional
        initial_state: tuple(state, bias_state), where
            state has shape [N, H, K, V] and bias_state has shape [N, H, V]
        output_final_state: bool
        cu_seqlens: optional cumulative sequence lengths for flattened varlen inputs

    Returns:
        o: [B, T, H, V]
        final_state: tuple(final_state, final_bias_state) if output_final_state else None
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    orig_dtype = v.dtype
    q, k, v, beta, g, lambda_q, lambda_k = map(
        lambda x: x.to(torch.float32),
        [q, k, v, beta, g, lambda_q, lambda_k],
    )
    if phase is not None:
        phase = phase.to(torch.float32)

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError("Expected batch size 1 when `cu_seqlens` is provided.")
        total = q.shape[1]
        H, K, V = q.shape[2], q.shape[3], v.shape[-1]
        num_seq = len(cu_seqlens) - 1
        init_h = torch.zeros(num_seq, H, K, V, device=v.device, dtype=torch.float32)
        init_b = torch.zeros(num_seq, H, V, device=v.device, dtype=torch.float32)
        if initial_state is not None:
            init_h = initial_state[0].to(torch.float32).clone()
            init_b = initial_state[1].to(torch.float32).clone()
        o_segments = []
        h_segments = []
        b_segments = []
        for n in range(num_seq):
            bos = int(cu_seqlens[n].item())
            eos = int(cu_seqlens[n + 1].item())
            q_n = q[0, bos:eos].transpose(0, 1).contiguous()
            k_n = k[0, bos:eos].transpose(0, 1).contiguous()
            v_n = v[0, bos:eos].transpose(0, 1).contiguous()
            beta_n = beta[0, bos:eos].transpose(0, 1).contiguous()
            g_n = g[0, bos:eos].transpose(0, 1).contiguous()
            lambda_q_n = lambda_q[0, bos:eos].transpose(0, 1).contiguous()
            lambda_k_n = lambda_k[0, bos:eos].transpose(0, 1).contiguous()
            phase_n = None if phase is None else phase[0, bos:eos]
            o_n, h_n, b_n = _run_rank1_dc_single_sequence(
                q=q_n,
                k=k_n,
                v=v_n,
                beta=beta_n,
                g=g_n,
                lambda_q=lambda_q_n,
                lambda_k=lambda_k_n,
                scale=scale,
                h=init_h[n],
                b=init_b[n],
                phase=phase_n,
                num_phase_channels=num_phase_channels,
            )
            o_segments.append(o_n.transpose(0, 1))
            h_segments.append(h_n)
            b_segments.append(b_n)
        o = torch.zeros(1, total, H, V, device=v.device, dtype=torch.float32)
        for n in range(num_seq):
            bos = int(cu_seqlens[n].item())
            eos = int(cu_seqlens[n + 1].item())
            o = o.clone()
            o[0, bos:eos] = o_segments[n]
        h = torch.stack(h_segments, dim=0)
        b = torch.stack(b_segments, dim=0)
    else:
        q, k, v, beta, g, lambda_q, lambda_k = map(
            lambda x: x.transpose(1, 2).contiguous(),
            [q, k, v, beta, g, lambda_q, lambda_k],
        )
        B, H, T, K, V = *k.shape, v.shape[-1]
        init_h = torch.zeros(B, H, K, V, device=v.device, dtype=torch.float32)
        init_b = torch.zeros(B, H, V, device=v.device, dtype=torch.float32)
        if initial_state is not None:
            init_h = initial_state[0].to(torch.float32).clone()
            init_b = initial_state[1].to(torch.float32).clone()
        o_list = []
        h_list = []
        b_list = []
        for batch_idx in range(B):
            phase_b = None if phase is None else phase[batch_idx]
            o_i, h_i, b_i = _run_rank1_dc_single_sequence(
                q=q[batch_idx],
                k=k[batch_idx],
                v=v[batch_idx],
                beta=beta[batch_idx],
                g=g[batch_idx],
                lambda_q=lambda_q[batch_idx],
                lambda_k=lambda_k[batch_idx],
                scale=scale,
                h=init_h[batch_idx],
                b=init_b[batch_idx],
                phase=phase_b,
                num_phase_channels=num_phase_channels,
            )
            o_list.append(o_i)
            h_list.append(h_i)
            b_list.append(b_i)
        o = torch.stack(o_list, dim=0).transpose(1, 2).contiguous()
        h = torch.stack(h_list, dim=0)
        b = torch.stack(b_list, dim=0)

    if not output_final_state:
        return o.to(orig_dtype), None
    return o.to(orig_dtype), (h, b)


def naive_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    """
    Reference PyTorch implementation of chunk gated delta rule.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H]
        beta: [B, T, H]
        chunk_size: int
        scale: float, optional
        initial_state: [B, H, K, V], optional
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)

    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))

    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert l % chunk_size == 0

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta, decay = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size),
        [q, k, v, k_beta, decay.unsqueeze(-1)],
    )
    decay = decay.squeeze(-1).cumsum(-1)
    decay_exp = decay.exp()[..., None]
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    k_cumsum = attn @ v
    k_cumdecay = attn @ (k_beta * decay_exp)
    v = k_cumsum

    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state.to(torch.float32)

    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_new
        S = S * decay[:, :, i, -1, None, None].exp() + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()
                                                        [..., None]).transpose(-1, -2) @ v_new
    if not output_final_state:
        S = None

    # unpad
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S
