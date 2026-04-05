# Copyright (c) 2026, Liang Ge

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule_rank1_dc,
    fused_recurrent_gated_delta_rule_rank1_dc,
)
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule_rank1_dc
from fla.utils import assert_close, device


def _clone_inputs(inputs):
    cloned = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            cloned[k] = v.clone()
        elif isinstance(v, tuple):
            cloned[k] = tuple(x.clone() if torch.is_tensor(x) else x for x in v)
        else:
            cloned[k] = v
    return cloned


def _make_rank1_dc_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = F.normalize(torch.randn(batch_size, seq_len, num_heads, head_k_dim, dtype=torch.float32), dim=-1)
    k = F.normalize(torch.randn(batch_size, seq_len, num_heads, head_k_dim, dtype=torch.float32), dim=-1)
    v = torch.randn(batch_size, seq_len, num_heads, head_v_dim, dtype=dtype)
    beta = torch.rand(batch_size, seq_len, num_heads, dtype=torch.float32).sigmoid()
    g = F.logsigmoid(torch.randn(batch_size, seq_len, num_heads, dtype=torch.float32))
    state0 = torch.randn(batch_size, num_heads, head_k_dim, head_v_dim, dtype=torch.float32)
    bias0 = torch.randn(batch_size, num_heads, head_v_dim, dtype=torch.float32)

    nu = F.softplus(torch.randn(num_heads, head_k_dim, dtype=torch.float32)) + 1e-6
    nu = F.normalize(nu, dim=-1)
    nu_sq = nu.square().sum(-1).clamp_min(1e-6)
    scale = head_k_dim ** -0.5
    lambda_k = (k * nu.unsqueeze(0).unsqueeze(0)).sum(-1) / nu_sq.unsqueeze(0).unsqueeze(0)
    lambda_q = ((q * scale) * nu.unsqueeze(0).unsqueeze(0)).sum(-1) / nu_sq.unsqueeze(0).unsqueeze(0)

    q, k, v, beta, g, lambda_q, lambda_k, state0, bias0 = [
        x.to(device) for x in (q, k, v, beta, g, lambda_q, lambda_k, state0, bias0)
    ]
    return dict(
        q=q.to(dtype),
        k=k.to(dtype),
        v=v,
        beta=beta,
        g=g,
        lambda_q=lambda_q.to(dtype),
        lambda_k=lambda_k.to(dtype),
        scale=scale,
        initial_state=(state0, bias0),
    )


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'dtype'),
    [
        (1, 63, 2, 64, 64, torch.float32),
        (2, 128, 4, 64, 64, torch.float16),
    ],
)
def test_rank1_dc_fused_recurrent_matches_naive(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
):
    inputs = _make_rank1_dc_inputs(B, T, H, K, V, dtype)
    ref_o, ref_state = naive_recurrent_gated_delta_rule_rank1_dc(output_final_state=True, **_clone_inputs(inputs))
    tri_o, tri_state = fused_recurrent_gated_delta_rule_rank1_dc(
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        **_clone_inputs(inputs),
    )
    ref_h, ref_b = ref_state
    tri_h, tri_b = tri_state
    assert_close('o', ref_o, tri_o, 0.002)
    assert_close('ht', ref_h, tri_h, 0.002)
    assert_close('bt', ref_b, tri_b, 0.002)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'dtype'),
    [
        (1, 128, 4, 64, 64, torch.float32),
        (1, 256, 4, 64, 64, torch.float16),
    ],
)
def test_rank1_dc_chunk_exact_chunk_size_1_matches_naive(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
):
    inputs = _make_rank1_dc_inputs(B, T, H, K, V, dtype)
    ref_o, ref_state = naive_recurrent_gated_delta_rule_rank1_dc(output_final_state=True, **_clone_inputs(inputs))
    tri_o, tri_state = chunk_gated_delta_rule_rank1_dc(
        output_final_state=True,
        chunk_size=1,
        **_clone_inputs(inputs),
    )
    ref_h, ref_b = ref_state
    tri_h, tri_b = tri_state
    assert_close('o', ref_o, tri_o, 0.002)
    assert_close('ht', ref_h, tri_h, 0.002)
    assert_close('bt', ref_b, tri_b, 0.002)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'dtype'),
    [
        (1, 512, 8, 64, 64, torch.float16),
    ],
)
def test_rank1_dc_chunk_triton_matches_naive(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
):
    inputs = _make_rank1_dc_inputs(B, T, H, K, V, dtype)
    ref_o, ref_state = naive_recurrent_gated_delta_rule_rank1_dc(output_final_state=True, **_clone_inputs(inputs))
    with torch.no_grad():
        tri_o, tri_state = chunk_gated_delta_rule_rank1_dc(
            output_final_state=True,
            chunk_size=64,
            **_clone_inputs(inputs),
        )
    ref_h, ref_b = ref_state
    tri_h, tri_b = tri_state
    assert_close('o', ref_o, tri_o, 0.002)
    assert_close('ht', ref_h, tri_h, 0.002)
    assert_close('bt', ref_b, tri_b, 0.002)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'dtype'),
    [
        (1, 512, 8, 64, 64, torch.float32),
    ],
)
def test_rank1_dc_chunk_exact_chunk64_reasonable_error(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
):
    inputs = _make_rank1_dc_inputs(B, T, H, K, V, dtype)
    ref_o, ref_state = naive_recurrent_gated_delta_rule_rank1_dc(output_final_state=True, **_clone_inputs(inputs))
    tri_o, tri_state = chunk_gated_delta_rule_rank1_dc(
        output_final_state=True,
        chunk_size=64,
        **_clone_inputs(inputs),
    )
    ref_h, ref_b = ref_state
    tri_h, tri_b = tri_state
    assert_close('o', ref_o, tri_o, 0.03)
    assert_close('ht', ref_h, tri_h, 0.08)
    assert_close('bt', ref_b, tri_b, 0.03)
