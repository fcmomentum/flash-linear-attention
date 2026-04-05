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


def _make_rank1_dc_inputs_requires_grad(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
):
    inputs = _make_rank1_dc_inputs(batch_size, seq_len, num_heads, head_k_dim, head_v_dim, dtype)
    grad_inputs = {}
    for k, v in inputs.items():
        if k == 'initial_state':
            state0, bias0 = v
            grad_inputs[k] = (state0.clone().requires_grad_(True), bias0.clone().requires_grad_(True))
        elif torch.is_tensor(v):
            grad_inputs[k] = v.clone().requires_grad_(True)
        else:
            grad_inputs[k] = v
    return grad_inputs


def _collect_grads(named_inputs):
    grads = {}
    for k, v in named_inputs.items():
        if k == 'initial_state':
            state0, bias0 = v
            grads['state0'] = state0.grad.clone()
            grads['bias0'] = bias0.grad.clone()
        elif torch.is_tensor(v) and v.grad is not None:
            grads[k] = v.grad.clone()
    return grads


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


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'dtype'),
    [
        (1, 64, 2, 32, 32, torch.float32),
    ],
)
def test_rank1_dc_chunk_exact_backward_chunk1_matches_naive(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
):
    ref_inputs = _make_rank1_dc_inputs_requires_grad(B, T, H, K, V, dtype)
    tri_inputs = _make_rank1_dc_inputs_requires_grad(B, T, H, K, V, dtype)

    ref_o, ref_state = naive_recurrent_gated_delta_rule_rank1_dc(
        output_final_state=True,
        **ref_inputs,
    )
    tri_o, tri_state = chunk_gated_delta_rule_rank1_dc(
        output_final_state=True,
        chunk_size=1,
        **tri_inputs,
    )

    do = torch.randn_like(ref_o)
    dht = torch.randn_like(ref_state[0])
    dbt = torch.randn_like(ref_state[1])

    ((ref_o * do).sum() + (ref_state[0] * dht).sum() + (ref_state[1] * dbt).sum()).backward()
    ((tri_o * do).sum() + (tri_state[0] * dht).sum() + (tri_state[1] * dbt).sum()).backward()

    ref_grads = _collect_grads(ref_inputs)
    tri_grads = _collect_grads(tri_inputs)
    for name, tol in [
        ('q', 0.003),
        ('k', 0.003),
        ('v', 0.003),
        ('beta', 0.003),
        ('g', 0.003),
        ('lambda_q', 0.003),
        ('lambda_k', 0.003),
        ('state0', 0.003),
        ('bias0', 0.003),
    ]:
        assert_close(name, ref_grads[name], tri_grads[name], tol)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'dtype'),
    [
        (1, 32, 2, 32, 32, torch.float32),
    ],
)
def test_rank1_dc_fused_recurrent_backward_not_implemented(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
):
    inputs = _make_rank1_dc_inputs_requires_grad(B, T, H, K, V, dtype)
    with pytest.raises(NotImplementedError):
        fused_recurrent_gated_delta_rule_rank1_dc(
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            **inputs,
        )


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'dtype'),
    [
        (1, 256, 4, 32, 32, torch.float32),
    ],
)
def test_rank1_dc_chunk_exact_backward_chunk64_reasonable_error(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
):
    ref_inputs = _make_rank1_dc_inputs_requires_grad(B, T, H, K, V, dtype)
    tri_inputs = _make_rank1_dc_inputs_requires_grad(B, T, H, K, V, dtype)

    ref_o, ref_state = naive_recurrent_gated_delta_rule_rank1_dc(
        output_final_state=True,
        **ref_inputs,
    )
    tri_o, tri_state = chunk_gated_delta_rule_rank1_dc(
        output_final_state=True,
        chunk_size=64,
        **tri_inputs,
    )

    do = torch.randn_like(ref_o)
    dht = torch.randn_like(ref_state[0])
    dbt = torch.randn_like(ref_state[1])

    ((ref_o * do).sum() + (ref_state[0] * dht).sum() + (ref_state[1] * dbt).sum()).backward()
    ((tri_o * do).sum() + (tri_state[0] * dht).sum() + (tri_state[1] * dbt).sum()).backward()

    ref_grads = _collect_grads(ref_inputs)
    tri_grads = _collect_grads(tri_inputs)
    for name, tol in [
        ('q', 0.05),
        ('k', 0.08),
        ('v', 0.05),
        ('beta', 0.08),
        ('g', 0.08),
        ('lambda_q', 0.05),
        ('lambda_k', 0.08),
        ('state0', 0.12),
        ('bias0', 0.08),
    ]:
        assert_close(name, ref_grads[name], tri_grads[name], tol)


@pytest.mark.parametrize(
    ('lengths', 'H', 'K', 'V', 'dtype'),
    [
        ([23, 41], 2, 32, 32, torch.float32),
    ],
)
def test_rank1_dc_chunk_exact_backward_varlen_chunk1_matches_naive(
    lengths,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
):
    batch_size = len(lengths)
    total = sum(lengths)
    inputs = _make_rank1_dc_inputs(batch_size, max(lengths), H, K, V, dtype)

    def _flatten_varlen(tensor):
        pieces = []
        for b_idx, length in enumerate(lengths):
            pieces.append(tensor[b_idx, :length])
        return torch.cat(pieces, dim=0).unsqueeze(0)

    q = _flatten_varlen(inputs['q']).clone().requires_grad_(True)
    k = _flatten_varlen(inputs['k']).clone().requires_grad_(True)
    v = _flatten_varlen(inputs['v']).clone().requires_grad_(True)
    g = _flatten_varlen(inputs['g']).clone().requires_grad_(True)
    beta = _flatten_varlen(inputs['beta']).clone().requires_grad_(True)
    lambda_q = _flatten_varlen(inputs['lambda_q']).clone().requires_grad_(True)
    lambda_k = _flatten_varlen(inputs['lambda_k']).clone().requires_grad_(True)
    state0 = inputs['initial_state'][0][:batch_size].clone().requires_grad_(True)
    bias0 = inputs['initial_state'][1][:batch_size].clone().requires_grad_(True)
    cu_seqlens = torch.tensor([0] + [sum(lengths[:i + 1]) for i in range(len(lengths))], device=q.device, dtype=torch.long)

    ref_inputs = {
        'q': q,
        'k': k,
        'v': v,
        'g': g,
        'beta': beta,
        'lambda_q': lambda_q,
        'lambda_k': lambda_k,
        'scale': inputs['scale'],
        'initial_state': (state0, bias0),
        'cu_seqlens': cu_seqlens,
    }
    tri_inputs = {
        'q': q.detach().clone().requires_grad_(True),
        'k': k.detach().clone().requires_grad_(True),
        'v': v.detach().clone().requires_grad_(True),
        'g': g.detach().clone().requires_grad_(True),
        'beta': beta.detach().clone().requires_grad_(True),
        'lambda_q': lambda_q.detach().clone().requires_grad_(True),
        'lambda_k': lambda_k.detach().clone().requires_grad_(True),
        'scale': inputs['scale'],
        'initial_state': (
            state0.detach().clone().requires_grad_(True),
            bias0.detach().clone().requires_grad_(True),
        ),
        'cu_seqlens': cu_seqlens,
    }

    ref_o, ref_state = naive_recurrent_gated_delta_rule_rank1_dc(output_final_state=True, **ref_inputs)
    tri_o, tri_state = chunk_gated_delta_rule_rank1_dc(
        output_final_state=True,
        chunk_size=1,
        **tri_inputs,
    )

    do = torch.randn_like(ref_o)
    dht = torch.randn_like(ref_state[0])
    dbt = torch.randn_like(ref_state[1])

    ((ref_o * do).sum() + (ref_state[0] * dht).sum() + (ref_state[1] * dbt).sum()).backward()
    ((tri_o * do).sum() + (tri_state[0] * dht).sum() + (tri_state[1] * dbt).sum()).backward()

    ref_named = {
        'q': ref_inputs['q'],
        'k': ref_inputs['k'],
        'v': ref_inputs['v'],
        'g': ref_inputs['g'],
        'beta': ref_inputs['beta'],
        'lambda_q': ref_inputs['lambda_q'],
        'lambda_k': ref_inputs['lambda_k'],
        'initial_state': ref_inputs['initial_state'],
    }
    tri_named = {
        'q': tri_inputs['q'],
        'k': tri_inputs['k'],
        'v': tri_inputs['v'],
        'g': tri_inputs['g'],
        'beta': tri_inputs['beta'],
        'lambda_q': tri_inputs['lambda_q'],
        'lambda_k': tri_inputs['lambda_k'],
        'initial_state': tri_inputs['initial_state'],
    }
    ref_grads = _collect_grads(ref_named)
    tri_grads = _collect_grads(tri_named)
    for name, tol in [
        ('q', 0.003),
        ('k', 0.003),
        ('v', 0.003),
        ('beta', 0.003),
        ('g', 0.003),
        ('lambda_q', 0.003),
        ('lambda_k', 0.003),
        ('state0', 0.003),
        ('bias0', 0.003),
    ]:
        assert_close(name, ref_grads[name], tri_grads[name], tol)
