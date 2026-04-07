# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp, exp2
from fla.utils import input_guard


_USE_TRITON_RANK1_DC_BACKWARD = os.environ.get("FLA_USE_TRITON_RANK1_DC_BWD", "0") == "1"


@triton.jit
def _rotate_phase_vec(x, phase, offsets, num_phase_channels):
    pair_offsets = tl.where((offsets & 1) == 0, offsets + 1, offsets - 1)
    pair_x = x[pair_offsets]
    base_offsets = offsets - (offsets & 1)
    pair_phase = phase[base_offsets]
    cos = tl.cos(pair_phase)
    sin = tl.sin(pair_phase)
    rot = tl.where((offsets & 1) == 0, -pair_x, pair_x)
    use_phase = offsets < num_phase_channels
    return tl.where(use_phase, x * cos + rot * sin, x)


@triton.jit
def _rotate_phase_state(b_h, phase, offsets, num_phase_channels, transpose_state: tl.constexpr):
    rotated = _rotate_phase_vec(b_h if transpose_state else b_h.T, phase, offsets, num_phase_channels)
    return rotated if transpose_state else rotated.T


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_PHASE': lambda args: args['phase'] is not None and args['num_phase_channels'] > 0,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    phase,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    num_phase_channels,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_PHASE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + bos * HV + i_hv
    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v
    if USE_PHASE:
        p_phase = phase + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    if TRANSPOSE_STATE:
        mask_h = mask_v[:, None] & mask_k[None, :]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]

    if TRANSPOSE_STATE:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0 = h0 + i_nh * K*V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_PHASE:
            b_phase = tl.load(p_phase, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            if USE_EXP2:
                b_h *= exp2(b_g)
            else:
                b_h *= exp(b_g)

        if USE_GK:
            b_gk = tl.load(p_gk).to(tl.float32)
            if USE_EXP2:
                if TRANSPOSE_STATE:
                    b_h *= exp2(b_gk[None, :])
                else:
                    b_h *= exp2(b_gk[:, None])
            else:
                if TRANSPOSE_STATE:
                    b_h *= exp(b_gk[None, :])
                else:
                    b_h *= exp(b_gk[:, None])

        if USE_GV:
            b_gv = tl.load(p_gv).to(tl.float32)
            if USE_EXP2:
                if TRANSPOSE_STATE:
                    b_h *= exp2(b_gv[:, None])
                else:
                    b_h *= exp2(b_gv[None, :])
            else:
                if TRANSPOSE_STATE:
                    b_h *= exp(b_gv[:, None])
                else:
                    b_h *= exp(b_gv[None, :])
        if USE_PHASE:
            b_h = _rotate_phase_state(b_h, b_phase, o_v, num_phase_channels, TRANSPOSE_STATE)
            b_v = _rotate_phase_vec(b_v, b_phase, o_v, num_phase_channels)

        if TRANSPOSE_STATE:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
        else:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_v
            b_o = tl.sum(b_h * b_q[:, None], 0)
        if USE_PHASE:
            b_o = _rotate_phase_vec(b_o, -b_phase, o_v, num_phase_channels)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H*K
        p_k += H*K
        p_v += HV*V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV*K
        if USE_GV:
            p_gv += HV*V
        if USE_PHASE:
            p_phase += HV*V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV*V

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = ht + i_nh * K*V + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_PHASE': lambda args: args['phase'] is not None and args['num_phase_channels'] > 0,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_INITIAL_BIAS_STATE': lambda args: args['b0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'STORE_FINAL_BIAS_STATE': lambda args: args['bt'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_delta_rule_rank1_dc_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    phase,
    lambda_q,
    lambda_k,
    o,
    h0,
    b0,
    ht,
    bt,
    cu_seqlens,
    scale,
    num_phase_channels,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_PHASE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_BIAS_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_FINAL_BIAS_STATE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_PHASE:
        p_phase = phase + (bos * HV + i_hv) * V + o_v
    p_lambda_q = lambda_q + bos * HV + i_hv
    p_lambda_k = lambda_k + bos * HV + i_hv
    if USE_G:
        p_g = g + bos * HV + i_hv
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    if TRANSPOSE_STATE:
        mask_h = mask_v[:, None] & mask_k[None, :]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]

    if TRANSPOSE_STATE:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    b_b = tl.zeros([BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0 = h0 + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    if USE_INITIAL_BIAS_STATE:
        p_b0 = b0 + i_nh * V + o_v
        b_b += tl.load(p_b0, mask=mask_v, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_PHASE:
            b_phase = tl.load(p_phase, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_lambda_q = tl.load(p_lambda_q).to(tl.float32)
        b_lambda_k = tl.load(p_lambda_k).to(tl.float32)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            if USE_EXP2:
                decay = exp2(b_g)
            else:
                decay = exp(b_g)
            b_b *= decay
            b_h *= decay
        if USE_PHASE:
            b_h = _rotate_phase_state(b_h, b_phase, o_v, num_phase_channels, TRANSPOSE_STATE)
            b_b = _rotate_phase_vec(b_b, b_phase, o_v, num_phase_channels)
            b_v = _rotate_phase_vec(b_v, b_phase, o_v, num_phase_channels)

        if TRANSPOSE_STATE:
            pred = tl.sum(b_h * b_k[None, :], 1) - b_lambda_k * b_b
            b_delta = b_beta * (b_v - pred)
            b_h += b_delta[:, None] * b_k[None, :]
            b_b += b_delta
            b_o = tl.sum(b_h * b_q[None, :], 1) - b_lambda_q * b_b
        else:
            pred = tl.sum(b_h * b_k[:, None], 0) - b_lambda_k * b_b
            b_delta = b_beta * (b_v - pred)
            b_h += b_k[:, None] * b_delta[None, :]
            b_b += b_delta
            b_o = tl.sum(b_h * b_q[:, None], 0) - b_lambda_q * b_b
        if USE_PHASE:
            b_o = _rotate_phase_vec(b_o, -b_phase, o_v, num_phase_channels)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        if USE_PHASE:
            p_phase += HV * V
        p_lambda_q += HV
        p_lambda_k += HV
        if USE_G:
            p_g += HV
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV * V

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = ht + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
    if STORE_FINAL_BIAS_STATE:
        p_bt = bt + i_nh * V + o_v
        tl.store(p_bt, b_b.to(p_bt.dtype.element_ty), mask=mask_v)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_INITIAL_BIAS_STATE': lambda args: args['b0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_FINAL_BIAS_STATE_GRADIENT': lambda args: args['dbt'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_delta_rule_rank1_dc_bwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    lambda_q,
    lambda_k,
    h_states,
    b_states,
    h0,
    b0,
    dht,
    dbt,
    do,
    dq,
    dk,
    dv,
    dg,
    dbeta,
    dlambda_q,
    dlambda_k,
    dh0,
    db0,
    cu_seqlens,
    scale,
    B,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_BIAS_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_BIAS_STATE_GRADIENT: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
):
    tl.static_assert(not TRANSPOSE_STATE, "transpose_state backward is not implemented yet")
    tl.static_assert(not USE_EXP2, "use_exp2 backward is not implemented yet")
    tl.static_assert(IS_BETA_HEADWISE, "only headwise beta is supported")

    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_nh // H, i_nh % H
    all = B * T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_q = q + ((i_b * T + (T - 1)) * H + i_h) * K + o_k
    p_k = k + ((i_b * T + (T - 1)) * H + i_h) * K + o_k
    p_v = v + ((i_b * T + (T - 1)) * H + i_h) * V + o_v
    p_do = do + ((i_b * T + (T - 1)) * H + i_h) * V + o_v
    p_g = g + (i_b * T + (T - 1)) * H + i_h if USE_G else None
    p_beta = beta + (i_b * T + (T - 1)) * H + i_h
    p_lambda_q = lambda_q + (i_b * T + (T - 1)) * H + i_h
    p_lambda_k = lambda_k + (i_b * T + (T - 1)) * H + i_h

    p_h_prev = h_states + (((i_b * H + i_h) * (T + 1) + (T - 1)) * K * V + o_k[:, None] * V + o_v[None, :])
    p_h_t = p_h_prev + K * V
    p_b_prev = b_states + (((i_b * H + i_h) * (T + 1) + (T - 1)) * V + o_v)
    p_b_t = p_b_prev + V

    p_dq = dq + ((i_v * all + i_b * T + (T - 1)) * H + i_h) * K + o_k
    p_dk = dk + ((i_v * all + i_b * T + (T - 1)) * H + i_h) * K + o_k
    p_dv = dv + ((i_b * T + (T - 1)) * H + i_h) * V + o_v
    p_dg = dg + (i_v * all + i_b * T + (T - 1)) * H + i_h if USE_G else None
    p_dbeta = dbeta + (i_v * all + i_b * T + (T - 1)) * H + i_h
    p_dlambda_q = dlambda_q + (i_v * all + i_b * T + (T - 1)) * H + i_h
    p_dlambda_k = dlambda_k + (i_v * all + i_b * T + (T - 1)) * H + i_h

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    b_db = tl.zeros([BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + (i_b * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        b_dh += tl.load(p_dht, mask=mask_h, other=0).to(tl.float32)
    if USE_FINAL_BIAS_STATE_GRADIENT:
        p_dbt = dbt + (i_b * H + i_h) * V + o_v
        b_db += tl.load(p_dbt, mask=mask_v, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_h_prev = tl.load(p_h_prev, mask=mask_h, other=0).to(tl.float32)
        b_h_t = tl.load(p_h_t, mask=mask_h, other=0).to(tl.float32)
        b_b_prev = tl.load(p_b_prev, mask=mask_v, other=0).to(tl.float32)
        b_b_t = tl.load(p_b_t, mask=mask_v, other=0).to(tl.float32)

        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)
        b_lambda_q = tl.load(p_lambda_q).to(tl.float32)
        b_lambda_k = tl.load(p_lambda_k).to(tl.float32)
        if USE_G:
            b_alpha = exp(tl.load(p_g).to(tl.float32))
        else:
            b_alpha = 1.0

        b_dq = tl.sum(b_h_t * b_do[None, :], axis=1) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

        b_dh += b_q[:, None] * (b_do * scale)[None, :]
        b_dlambda_q = -tl.sum(b_b_t * b_do)
        b_db = b_db - b_lambda_q * b_do

        b_h1 = b_alpha * b_h_prev
        b_b1 = b_alpha * b_b_prev
        b_pred = tl.sum(b_h1 * b_k[:, None], axis=0) - b_lambda_k * b_b1
        b_innovation = b_v - b_pred

        b_dk = tl.sum(b_dh * (b_beta * b_innovation)[None, :], axis=1)
        b_ddelta = tl.sum(b_dh * b_k[:, None], axis=0) + b_db
        b_dbeta = tl.sum(b_ddelta * b_innovation)
        b_dv = b_beta * b_ddelta
        b_dpred = -b_beta * b_ddelta

        b_dh1 = b_dh + b_k[:, None] * b_dpred[None, :]
        b_dk += tl.sum(b_h1 * b_dpred[None, :], axis=1)
        b_dlambda_k = -tl.sum(b_b1 * b_dpred)
        b_db1 = b_db - b_lambda_k * b_dpred

        b_dalpha = tl.sum(b_dh1 * b_h_prev) + tl.sum(b_db1 * b_b_prev)
        b_dh = b_alpha * b_dh1
        b_db = b_alpha * b_db1
        b_dg_val = b_dalpha * b_alpha if USE_G else 0.0

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        if USE_G:
            tl.store(p_dg, b_dg_val.to(p_dg.dtype.element_ty))
        tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty))
        tl.store(p_dlambda_q, b_dlambda_q.to(p_dlambda_q.dtype.element_ty))
        tl.store(p_dlambda_k, b_dlambda_k.to(p_dlambda_k.dtype.element_ty))

        p_q -= H * K
        p_k -= H * K
        p_v -= H * V
        p_do -= H * V
        if USE_G:
            p_g -= H
        p_beta -= H
        p_lambda_q -= H
        p_lambda_k -= H
        p_h_prev -= K * V
        p_h_t -= K * V
        p_b_prev -= V
        p_b_t -= V
        p_dq -= H * K
        p_dk -= H * K
        p_dv -= H * V
        if USE_G:
            p_dg -= H
        p_dbeta -= H
        p_dlambda_q -= H
        p_dlambda_k -= H

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + (i_b * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_h)
    if USE_INITIAL_BIAS_STATE:
        p_db0 = db0 + (i_b * H + i_h) * V + o_v
        tl.store(p_db0, b_db.to(p_db0.dtype.element_ty), mask=mask_v)


def fused_recurrent_gated_delta_rule_rank1_dc_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    dbt: torch.Tensor | None,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    h_states: torch.Tensor | None = None,
    b_states: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if cu_seqlens is not None:
        raise NotImplementedError("Triton backward for fused recurrent rank-1 DC does not support `cu_seqlens` yet.")
    if transpose_state_layout:
        raise NotImplementedError("Triton backward for fused recurrent rank-1 DC does not support transposed layout.")
    if use_exp2:
        raise NotImplementedError("Triton backward for fused recurrent rank-1 DC does not support use_exp2.")
    if q.shape[2] != v.shape[2]:
        raise NotImplementedError("Triton backward for fused recurrent rank-1 DC requires q and v to have the same head count.")
    if beta.ndim == v.ndim:
        raise NotImplementedError("Triton backward for fused recurrent rank-1 DC only supports headwise beta.")
    if h_states is None or b_states is None:
        raise ValueError("h_states and b_states must be provided for Triton backward.")

    B, T, H, K, V = *q.shape, v.shape[-1]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)

    dq = q.new_empty(NV, *q.shape)
    dk = q.new_empty(NV, *k.shape)
    dv = q.new_empty(*v.shape)
    dg = q.new_empty(NV, *g.shape) if g is not None else None
    dbeta = q.new_empty(NV, *beta.shape)
    dlambda_q = q.new_empty(NV, *lambda_q.shape)
    dlambda_k = q.new_empty(NV, *lambda_k.shape)

    if initial_state is not None:
        dh0 = q.new_empty(B, H, K, V, dtype=torch.float32)
        db0 = q.new_empty(B, H, V, dtype=torch.float32)
    else:
        dh0 = None
        db0 = None

    grid = (NV, B * H)
    fused_recurrent_gated_delta_rule_rank1_dc_bwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        lambda_q=lambda_q,
        lambda_k=lambda_k,
        h_states=h_states,
        b_states=b_states,
        h0=initial_state[0] if initial_state is not None else None,
        b0=initial_state[1] if initial_state is not None else None,
        dht=dht,
        dbt=dbt,
        do=do,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        dbeta=dbeta,
        dlambda_q=dlambda_q,
        dlambda_k=dlambda_k,
        dh0=dh0,
        db0=db0,
        cu_seqlens=None,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=True,
        USE_EXP2=False,
        TRANSPOSE_STATE=False,
        num_warps=2,
        num_stages=1,
    )

    dq = dq.sum(0)
    dk = dk.sum(0)
    dbeta = dbeta.sum(0)
    dlambda_q = dlambda_q.sum(0)
    dlambda_k = dlambda_k.sum(0)
    if dg is not None:
        dg = dg.sum(0)

    return dq, dk, dv, dg, dbeta, dlambda_q, dlambda_k, dh0, db0


def fused_recurrent_gated_delta_rule_rank1_dc_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None = None,
    phase: torch.Tensor | None = None,
    lambda_q: torch.Tensor | None = None,
    lambda_k: torch.Tensor | None = None,
    scale: float = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
    num_phase_channels: int = 0,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    h0, b0 = (None, None) if initial_state is None else initial_state
    if output_final_state:
        if transpose_state_layout:
            final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
        final_bias_state = q.new_empty(N, HV, V, dtype=torch.float32)
    else:
        final_state = None
        final_bias_state = None

    grid = (NV, N * HV)
    fused_recurrent_gated_delta_rule_rank1_dc_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        phase=phase,
        lambda_q=lambda_q,
        lambda_k=lambda_k,
        o=o,
        h0=h0,
        b0=b0,
        ht=final_state,
        bt=final_bias_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        num_phase_channels=num_phase_channels,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state_layout,
        num_warps=1,
        num_stages=3,
    )
    if not output_final_state:
        return o, None
    return o, (final_state, final_bias_state)


def _dense_rank1_dc_forward_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    scale: float,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None,
):
    qh, kh, vh, gh, betah, lqh, lkh = (
        q.permute(0, 2, 1, 3).contiguous().to(torch.float32),
        k.permute(0, 2, 1, 3).contiguous().to(torch.float32),
        v.permute(0, 2, 1, 3).contiguous().to(torch.float32),
        g.permute(0, 2, 1).contiguous().to(torch.float32),
        beta.permute(0, 2, 1).contiguous().to(torch.float32),
        lambda_q.permute(0, 2, 1).contiguous().to(torch.float32),
        lambda_k.permute(0, 2, 1).contiguous().to(torch.float32),
    )
    batch_size, num_heads, seq_len, k_dim = qh.shape
    v_dim = vh.shape[-1]
    if initial_state is None:
        h = vh.new_zeros(batch_size, num_heads, k_dim, v_dim)
        b = vh.new_zeros(batch_size, num_heads, v_dim)
    else:
        h = initial_state[0].to(torch.float32)
        b = initial_state[1].to(torch.float32)

    outputs = []
    h_states = [h]
    b_states = [b]
    for t in range(seq_len):
        alpha_t = gh[:, :, t].exp()
        h1 = h * alpha_t[..., None, None]
        b1 = b * alpha_t[..., None]
        pred = torch.einsum('bhkv,bhk->bhv', h1, kh[:, :, t]) - lkh[:, :, t][..., None] * b1
        delta = betah[:, :, t][..., None] * (vh[:, :, t] - pred)
        h = h1 + torch.einsum('bhk,bhv->bhkv', kh[:, :, t], delta)
        b = b1 + delta
        outputs.append(scale * torch.einsum('bhk,bhkv->bhv', qh[:, :, t], h) - lqh[:, :, t][..., None] * b)
        h_states.append(h)
        b_states.append(b)
    o = torch.stack(outputs, dim=2).permute(0, 2, 1, 3).contiguous()
    return o, (h, b), torch.stack(h_states, dim=2), torch.stack(b_states, dim=2)


def _dense_rank1_dc_backward_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    scale: float,
    h_states: torch.Tensor,
    b_states: torch.Tensor,
    do: torch.Tensor,
    dstate: torch.Tensor | None,
    dbias_state: torch.Tensor | None,
):
    qh, kh, vh, gh, betah, lqh, lkh, doh = (
        q.permute(0, 2, 1, 3).contiguous().to(torch.float32),
        k.permute(0, 2, 1, 3).contiguous().to(torch.float32),
        v.permute(0, 2, 1, 3).contiguous().to(torch.float32),
        g.permute(0, 2, 1).contiguous().to(torch.float32),
        beta.permute(0, 2, 1).contiguous().to(torch.float32),
        lambda_q.permute(0, 2, 1).contiguous().to(torch.float32),
        lambda_k.permute(0, 2, 1).contiguous().to(torch.float32),
        do.permute(0, 2, 1, 3).contiguous().to(torch.float32),
    )
    batch_size, num_heads, seq_len, k_dim = qh.shape
    v_dim = vh.shape[-1]
    dh = h_states.new_zeros(batch_size, num_heads, k_dim, v_dim) if dstate is None else dstate.to(torch.float32)
    db = b_states.new_zeros(batch_size, num_heads, v_dim) if dbias_state is None else dbias_state.to(torch.float32)

    dq = torch.zeros_like(qh)
    dk = torch.zeros_like(kh)
    dv = torch.zeros_like(vh)
    dg = torch.zeros_like(gh)
    dbeta = torch.zeros_like(betah)
    dlambda_q = torch.zeros_like(lqh)
    dlambda_k = torch.zeros_like(lkh)

    for t in range(seq_len - 1, -1, -1):
        h_prev = h_states[:, :, t]
        h_t = h_states[:, :, t + 1]
        b_prev = b_states[:, :, t]
        b_t = b_states[:, :, t + 1]

        q_t = qh[:, :, t]
        k_t = kh[:, :, t]
        v_t = vh[:, :, t]
        do_t = doh[:, :, t]
        alpha_t = gh[:, :, t].exp()
        beta_t = betah[:, :, t]
        lambda_q_t = lqh[:, :, t]
        lambda_k_t = lkh[:, :, t]

        dq[:, :, t] = scale * torch.einsum('bhv,bhkv->bhk', do_t, h_t)
        dh = dh + scale * torch.einsum('bhk,bhv->bhkv', q_t, do_t)
        dlambda_q[:, :, t] = -(b_t * do_t).sum(-1)
        db = db - lambda_q_t[..., None] * do_t

        h1 = alpha_t[..., None, None] * h_prev
        b1 = alpha_t[..., None] * b_prev
        pred = torch.einsum('bhkv,bhk->bhv', h1, k_t) - lambda_k_t[..., None] * b1
        innovation = v_t - pred

        dk[:, :, t] = torch.einsum('bhkv,bhv->bhk', dh, beta_t[..., None] * innovation)
        ddelta = torch.einsum('bhkv,bhk->bhv', dh, k_t) + db
        dbeta[:, :, t] = (ddelta * innovation).sum(-1)
        dv[:, :, t] = beta_t[..., None] * ddelta
        dpred = -beta_t[..., None] * ddelta

        dh1 = dh + torch.einsum('bhk,bhv->bhkv', k_t, dpred)
        dk[:, :, t] = dk[:, :, t] + torch.einsum('bhkv,bhv->bhk', h1, dpred)
        dlambda_k[:, :, t] = -(b1 * dpred).sum(-1)
        db1 = db - lambda_k_t[..., None] * dpred

        dalpha = (dh1 * h_prev).sum(dim=(-1, -2)) + (db1 * b_prev).sum(dim=-1)
        dh = alpha_t[..., None, None] * dh1
        db = alpha_t[..., None] * db1
        dg[:, :, t] = dalpha * alpha_t

    return (
        dq.permute(0, 2, 1, 3).contiguous(),
        dk.permute(0, 2, 1, 3).contiguous(),
        dv.permute(0, 2, 1, 3).contiguous(),
        dg.permute(0, 2, 1).contiguous(),
        dbeta.permute(0, 2, 1).contiguous(),
        dlambda_q.permute(0, 2, 1).contiguous(),
        dlambda_k.permute(0, 2, 1).contiguous(),
        dh,
        db,
    )


class FusedRecurrentRank1DCFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None,
        beta: torch.Tensor,
        lambda_q: torch.Tensor,
        lambda_k: torch.Tensor,
        scale: float,
        state0: torch.Tensor,
        bias_state0: torch.Tensor,
        has_initial_state: bool,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
        cu_seqlens: torch.LongTensor | None,
        use_exp2: bool,
        transpose_state_layout: bool,
    ):
        if cu_seqlens is not None:
            raise NotImplementedError("Backward-capable fused rank-1 DC currently supports dense inputs only.")
        if use_qk_l2norm_in_kernel:
            raise NotImplementedError("Backward-capable fused rank-1 DC does not support q/k l2 norm in kernel.")
        if use_exp2:
            raise NotImplementedError("Backward-capable fused rank-1 DC does not support use_exp2 yet.")
        if transpose_state_layout:
            raise NotImplementedError("Backward-capable fused rank-1 DC does not support transposed state layout yet.")
        initial_state = (state0, bias_state0) if has_initial_state else None
        ctx.scale = scale
        ctx.has_initial_state = has_initial_state
        ctx.output_final_state = output_final_state
        ctx.use_triton_backward = _USE_TRITON_RANK1_DC_BACKWARD
        ctx.save_for_backward(q, k, v, g, beta, lambda_q, lambda_k, state0, bias_state0)
        with torch.no_grad():
            o, final_state = fused_recurrent_gated_delta_rule_rank1_dc_fwd(
                q=q,
                k=k,
                v=v,
                beta=beta,
                g=g,
                lambda_q=lambda_q,
                lambda_k=lambda_k,
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=None,
                use_exp2=False,
                transpose_state_layout=False,
            )
        state, bias_state = final_state
        if not output_final_state:
            empty = q.new_zeros(0, dtype=torch.float32)
            return o, empty, empty
        return o, state, bias_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dstate, dbias_state):
        q, k, v, g, beta, lambda_q, lambda_k, state0, bias_state0 = ctx.saved_tensors
        initial_state = (state0, bias_state0) if ctx.has_initial_state else None
        if dstate is not None and dstate.numel() == 0:
            dstate = None
        if dbias_state is not None and dbias_state.numel() == 0:
            dbias_state = None
        with torch.no_grad():
            _, _, h_states, b_states = _dense_rank1_dc_forward_batched(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                lambda_q=lambda_q,
                lambda_k=lambda_k,
                scale=ctx.scale,
                initial_state=initial_state,
            )
        if ctx.use_triton_backward:
            dq, dk, dv, dg, dbeta, dlambda_q, dlambda_k, dstate0, dbias0 = fused_recurrent_gated_delta_rule_rank1_dc_bwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                lambda_q=lambda_q,
                lambda_k=lambda_k,
                do=do,
                dht=dstate,
                dbt=dbias_state,
                scale=ctx.scale,
                initial_state=initial_state,
                h_states=h_states,
                b_states=b_states,
                cu_seqlens=None,
                use_exp2=False,
                transpose_state_layout=False,
            )
        else:
            dq, dk, dv, dg, dbeta, dlambda_q, dlambda_k, dstate0, dbias0 = _dense_rank1_dc_backward_batched(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                lambda_q=lambda_q,
                lambda_k=lambda_k,
                scale=ctx.scale,
                h_states=h_states,
                b_states=b_states,
                do=do,
                dstate=dstate,
                dbias_state=dbias_state,
            )
        if not ctx.has_initial_state:
            dstate0 = None
            dbias0 = None
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dg.to(g.dtype) if g is not None else None,
            dbeta.to(beta.dtype),
            dlambda_q.to(lambda_q.dtype),
            dlambda_k.to(lambda_k.dtype),
            None,
            dstate0,
            dbias0,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    phase: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
    num_phase_channels: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    if output_final_state:
        if transpose_state_layout:
            final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None

    grid = (NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        phase=phase,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        num_phase_channels=num_phase_channels,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state_layout,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        gv: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        use_exp2: bool = False,
        transpose_state_layout: bool = False,
        num_phase_channels: int = 0,
    ):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            gv=gv,
            beta=beta,
            phase=phase,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            use_exp2=use_exp2,
            transpose_state_layout=transpose_state_layout,
            num_phase_channels=num_phase_channels,
        )

        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it "
            "because we haven't figured out how to compute dg without materializing the full "
            "hidden states for all time steps.",
        )


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
    phase: torch.Tensor | None = None,
    num_phase_channels: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`. Default: `None`.
        gk (torch.Tensor):
            gk (decays) of shape `[B, T, HV, K]`. Default: `None`.
        gv (torch.Tensor):
            gv (decays) of shape `[B, T, HV, V]`. Default: `None`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        transpose_state_layout (bool):
            Whether to use transposed state layout `[V, K]` instead of `[K, V]`. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])

    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        gk,
        gv,
        beta,
        phase,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        use_exp2,
        transpose_state_layout,
        num_phase_channels,
    )
    return o, final_state




def fused_recurrent_gated_delta_rule_rank1_dc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
    phase: torch.Tensor | None = None,
    num_phase_channels: int = 0,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state[0].shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state[0].shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    needs_backward = torch.is_grad_enabled() and any(
        x.requires_grad for x in (q, k, v, g, beta, lambda_q, lambda_k) if x is not None
    )
    if phase is not None and num_phase_channels > 0 and needs_backward:
        raise NotImplementedError(
            "Phase-aware fused_recurrent_gated_delta_rule_rank1_dc does not support backward yet."
        )
    if needs_backward:
        if cu_seqlens is not None:
            raise NotImplementedError(
                "Backward-capable fused_recurrent_gated_delta_rule_rank1_dc does not support `cu_seqlens` yet."
            )
        if initial_state is None:
            state0 = q.new_zeros(0, dtype=torch.float32)
            bias_state0 = q.new_zeros(0, dtype=torch.float32)
            has_initial_state = False
        else:
            state0, bias_state0 = initial_state
            has_initial_state = True
        o, state, bias_state = FusedRecurrentRank1DCFunction.apply(
            q,
            k,
            v,
            g,
            beta,
            lambda_q,
            lambda_k,
            scale,
            state0,
            bias_state0,
            has_initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
            cu_seqlens,
            use_exp2,
            transpose_state_layout,
        )
        if not output_final_state:
            return o, None
        return o, (state, bias_state)
    return fused_recurrent_gated_delta_rule_rank1_dc_fwd(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        phase=phase,
        lambda_q=lambda_q,
        lambda_k=lambda_k,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
        num_phase_channels=num_phase_channels,
    )


fused_recurrent_gdn = fused_recurrent_gated_delta_rule
fused_recurrent_gdn_rank1_dc = fused_recurrent_gated_delta_rule_rank1_dc
