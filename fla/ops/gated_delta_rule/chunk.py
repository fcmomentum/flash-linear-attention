# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang

import warnings

import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.cp import FLACPContext
from fla.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu_pre_process,
    chunk_gated_delta_rule_fwd_h_pre_process,
    compress_h0,
    expand_h0,
)
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.naive import (
    naive_chunk_gated_delta_rule_phase_transport,
    naive_recurrent_gated_delta_rule,
    naive_recurrent_gated_delta_rule_rank1_dc,
)
from fla.ops.gated_delta_rule.phase_utils import rotate_phase_channels
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
):
    g = chunk_local_cumsum(
        g,
        chunk_size=64,
        scale=RCP_LN2 if use_exp2 else None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    # obtain WY representation. u is actually the new v.
    # fused kkt + solve_tril + recompute_w_u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
    )

    if cp_context is not None:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=k,
            w=w,
            u=u,
            g=g,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            context=cp_context,
            use_exp2=use_exp2,
            transpose_state_layout=transpose_state_layout,
        )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )

    if cp_context is not None:
        initial_state = compress_h0(initial_state, context=cp_context)

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )
    return g, o, A, final_state, initial_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
):
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
    )

    if cp_context is not None:
        initial_state = expand_h0(initial_state, context=cp_context)

    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
    )

    if cp_context is not None:
        # initial_state is None in the CP mode
        # We only need to compute dht of current rank and pass it to the backward kernel
        dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            dht=dht,
            initial_state=initial_state,
            context=cp_context,
            use_exp2=use_exp2,
            transpose_state_layout=transpose_state_layout,
        )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        cp_context: FLACPContext | None = None,
        transpose_state_layout: bool = False,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        chunk_indices = prepare_chunk_indices(
            cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None
        g, o, A, final_state, initial_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
            chunk_indices=chunk_indices,
            transpose_state_layout=transpose_state_layout,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens, chunk_indices)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cp_context = cp_context
        ctx.transpose_state_layout = transpose_state_layout
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            cp_context=ctx.cp_context,
            chunk_indices=chunk_indices,
            transpose_state_layout=ctx.transpose_state_layout,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None, None, None


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
    phase: torch.Tensor | None = None,
    num_phase_channels: int = 0,
    **kwargs,
):
    if phase is not None and num_phase_channels > 0:
        if cp_context is not None:
            raise NotImplementedError("Phase-aware chunk_gated_delta_rule does not support context parallelism yet.")
        q = rotate_phase_channels(q, phase, num_phase_channels=num_phase_channels)
        k = rotate_phase_channels(k, phase, num_phase_channels=num_phase_channels)
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, Hq, K]` where `Hq` is the number of query/key heads.
        k (torch.Tensor):
            keys of shape `[B, T, Hq, K]` where `Hq` is the number of query/key heads.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` where `H` is the number of value/output heads.
            For standard attention, `Hq == H`. For GQA, `H % Hq == 0`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        cp_context (Optional[FLACPContext]):
            Context parallel context for distributed training across multiple devices.
            When provided, `initial_state` and `output_final_state` are not supported,
            and `cu_seqlens` will be overridden by the context. Default: `None`.
        transpose_state_layout (Optional[bool]):
            Whether to use the transposed state layout for the hidden state.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    # Validate GQA head divisibility
    Hq, H = q.shape[2], v.shape[2]
    if H % Hq != 0:
        raise ValueError(
            f"For GQA, num_heads (H={H}) must be evenly divisible by "
            f"num_kv_heads (Hq={Hq}), but got H % Hq = {H % Hq}"
        )

    if 'head_first' in kwargs:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )

    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
        cu_seqlens = cp_context.cu_seqlens
        if cp_context.cu_seqlens_cpu is not None:
            cu_seqlens_cpu = cp_context.cu_seqlens_cpu

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
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        use_qk_l2norm_in_kernel,
        cp_context,
        transpose_state_layout,
    )
    return o, final_state


@triton.jit
def _rank1_dc_project_cols_kernel(
    z_ptr,
    u_ptr,
    out_ptr,
    n,
    d,
    stride_z0,
    stride_z1,
    stride_u0,
    stride_o0,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d0 in range(0, tl.cdiv(d, BLOCK_D)):
        offs_d = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d
        z = tl.load(
            z_ptr + offs_d[:, None] * stride_z0 + offs_n[None, :] * stride_z1,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.,
        )
        u = tl.load(u_ptr + offs_d * stride_u0, mask=mask_d, other=0.)
        acc += tl.sum(z * u[:, None], axis=0)
    tl.store(out_ptr + offs_n * stride_o0, acc, mask=mask_n)


@triton.jit
def _rank1_dc_update_cols_kernel(
    z_ptr,
    c_ptr,
    proj_ptr,
    alpha,
    beta,
    n,
    d,
    stride_z0,
    stride_z1,
    stride_c0,
    stride_p0,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_d = offs_d < d
    mask_n = offs_n < n
    z = tl.load(
        z_ptr + offs_d[:, None] * stride_z0 + offs_n[None, :] * stride_z1,
        mask=mask_d[:, None] & mask_n[None, :],
        other=0.,
    )
    c = tl.load(c_ptr + offs_d * stride_c0, mask=mask_d, other=0.)
    proj = tl.load(proj_ptr + offs_n * stride_p0, mask=mask_n, other=0.)
    out = alpha * z - beta * c[:, None] * proj[None, :]
    tl.store(
        z_ptr + offs_d[:, None] * stride_z0 + offs_n[None, :] * stride_z1,
        out,
        mask=mask_d[:, None] & mask_n[None, :],
    )


@triton.jit
def _rank1_dc_row_dot_kernel(
    z_ptr,
    r_ptr,
    out_ptr,
    n,
    d,
    stride_z0,
    stride_z1,
    stride_r0,
    stride_o0,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d0 in range(0, tl.cdiv(d, BLOCK_D)):
        offs_d = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d
        z = tl.load(
            z_ptr + offs_d[:, None] * stride_z0 + offs_n[None, :] * stride_z1,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.,
        )
        r = tl.load(r_ptr + offs_d * stride_r0, mask=mask_d, other=0.)
        acc += tl.sum(z * r[:, None], axis=0)
    tl.store(out_ptr + offs_n * stride_o0, acc, mask=mask_n)


def _rank1_dc_project_cols_triton(z: torch.Tensor, u: torch.Tensor):
    n = z.shape[1]
    out = torch.empty(n, device=z.device, dtype=torch.float32)
    block_n = 32
    block_d = 32
    grid = (triton.cdiv(n, block_n),)
    _rank1_dc_project_cols_kernel[grid](
        z, u, out,
        n, z.shape[0],
        z.stride(0), z.stride(1),
        u.stride(0), out.stride(0),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out.to(z.dtype)


def _rank1_dc_update_cols_triton(z: torch.Tensor, c_vec: torch.Tensor, proj: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
    n = z.shape[1]
    block_n = 32
    block_d = 32
    grid = (triton.cdiv(z.shape[0], block_d), triton.cdiv(n, block_n))
    _rank1_dc_update_cols_kernel[grid](
        z, c_vec, proj, alpha.item(), beta.item(),
        n, z.shape[0],
        z.stride(0), z.stride(1),
        c_vec.stride(0), proj.stride(0),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )


def _rank1_dc_row_dot_triton(z: torch.Tensor, r: torch.Tensor):
    n = z.shape[1]
    out = torch.empty(n, device=z.device, dtype=torch.float32)
    block_n = 32
    block_d = 32
    grid = (triton.cdiv(n, block_n),)
    _rank1_dc_row_dot_kernel[grid](
        z, r, out,
        n, z.shape[0],
        z.stride(0), z.stride(1),
        r.stride(0), out.stride(0),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out.to(z.dtype)


@triton.jit
def _rank1_dc_update_cols_fused_kernel(
    z_ptr,
    u_ptr,
    c_ptr,
    alpha,
    beta,
    n,
    d,
    stride_z0,
    stride_z1,
    stride_u0,
    stride_c0,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n
    proj = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d0 in range(0, tl.cdiv(d, BLOCK_D)):
        offs_d = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d
        z = tl.load(
            z_ptr + offs_d[:, None] * stride_z0 + offs_n[None, :] * stride_z1,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.,
        )
        u = tl.load(u_ptr + offs_d * stride_u0, mask=mask_d, other=0.)
        proj += tl.sum(z * u[:, None], axis=0)
    for d0 in range(0, tl.cdiv(d, BLOCK_D)):
        offs_d = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d
        z = tl.load(
            z_ptr + offs_d[:, None] * stride_z0 + offs_n[None, :] * stride_z1,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.,
        )
        c = tl.load(c_ptr + offs_d * stride_c0, mask=mask_d, other=0.)
        out = alpha * z - beta * c[:, None] * proj[None, :]
        tl.store(
            z_ptr + offs_d[:, None] * stride_z0 + offs_n[None, :] * stride_z1,
            out,
            mask=mask_d[:, None] & mask_n[None, :],
        )


@triton.jit
def _rank1_dc_update_cols_and_row_dot_fused_kernel(
    z_ptr,
    u_ptr,
    c_ptr,
    r_ptr,
    out_ptr,
    alpha,
    beta,
    n,
    d,
    stride_z0,
    stride_z1,
    stride_u0,
    stride_c0,
    stride_r0,
    stride_o0,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n
    proj = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d0 in range(0, tl.cdiv(d, BLOCK_D)):
        offs_d = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d
        z = tl.load(
            z_ptr + offs_d[:, None] * stride_z0 + offs_n[None, :] * stride_z1,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.,
        )
        u = tl.load(u_ptr + offs_d * stride_u0, mask=mask_d, other=0.)
        proj += tl.sum(z * u[:, None], axis=0)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d0 in range(0, tl.cdiv(d, BLOCK_D)):
        offs_d = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d
        z = tl.load(
            z_ptr + offs_d[:, None] * stride_z0 + offs_n[None, :] * stride_z1,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.,
        )
        c = tl.load(c_ptr + offs_d * stride_c0, mask=mask_d, other=0.)
        r = tl.load(r_ptr + offs_d * stride_r0, mask=mask_d, other=0.)
        out = alpha * z - beta * c[:, None] * proj[None, :]
        tl.store(
            z_ptr + offs_d[:, None] * stride_z0 + offs_n[None, :] * stride_z1,
            out,
            mask=mask_d[:, None] & mask_n[None, :],
        )
        acc += tl.sum(out * r[:, None], axis=0)
    tl.store(out_ptr + offs_n * stride_o0, acc, mask=mask_n)


def _rank1_dc_update_cols_fused_triton(z: torch.Tensor, u: torch.Tensor, c_vec: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
    n = z.shape[1]
    block_n = 32
    block_d = 32
    grid = (triton.cdiv(n, block_n),)
    _rank1_dc_update_cols_fused_kernel[grid](
        z, u, c_vec, alpha.item(), beta.item(),
        n, z.shape[0],
        z.stride(0), z.stride(1),
        u.stride(0), c_vec.stride(0),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )


def _rank1_dc_update_cols_and_row_dot_fused_triton(z: torch.Tensor, u: torch.Tensor, c_vec: torch.Tensor, r: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
    n = z.shape[1]
    out = torch.empty(n, device=z.device, dtype=torch.float32)
    block_n = 32
    block_d = 32
    grid = (triton.cdiv(n, block_n),)
    _rank1_dc_update_cols_and_row_dot_fused_kernel[grid](
        z, u, c_vec, r, out, alpha.item(), beta.item(),
        n, z.shape[0],
        z.stride(0), z.stride(1),
        u.stride(0), c_vec.stride(0), r.stride(0), out.stride(0),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out.to(z.dtype)


@triton.jit
def _rank1_dc_build_t_kernel(
    a_ptr,
    b_ptr,
    t_ptr,
    c,
    d,
    stride_a0,
    stride_a1,
    stride_b0,
    stride_b1,
    stride_t0,
    stride_t1,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    offs_i = pid_i * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_j = pid_j * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_i = offs_i < c
    mask_j = offs_j < c
    acc = tl.zeros([BLOCK_C, BLOCK_C], dtype=tl.float32)
    for d0 in range(0, tl.cdiv(d, BLOCK_D)):
        offs_d = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d
        a = tl.load(a_ptr + offs_i[:, None] * stride_a0 + offs_d[None, :] * stride_a1, mask=mask_i[:, None] & mask_d[None, :], other=0.)
        b = tl.load(b_ptr + offs_j[:, None] * stride_b0 + offs_d[None, :] * stride_b1, mask=mask_j[:, None] & mask_d[None, :], other=0.)
        acc += tl.dot(a, tl.trans(b))
    upper = offs_i[:, None] < offs_j[None, :]
    diag = offs_i[:, None] == offs_j[None, :]
    out = tl.where(diag, 1.0, tl.where(upper, acc, 0.0))
    tl.store(t_ptr + offs_i[:, None] * stride_t0 + offs_j[None, :] * stride_t1, out, mask=mask_i[:, None] & mask_j[None, :])


def _build_rank1_dc_wy_factors_triton(
    k_chunk: torch.Tensor,
    g_chunk: torch.Tensor,
    beta_chunk: torch.Tensor,
    lambda_k_chunk: torch.Tensor,
):
    c, k_dim = k_chunk.shape
    d = k_dim + 1
    device = k_chunk.device
    dtype = k_chunk.dtype

    alpha = g_chunk.exp()
    gamma = beta_chunk / alpha.clamp_min(1e-6)
    a = torch.empty(c, d, device=device, dtype=dtype)
    b = torch.empty(c, d, device=device, dtype=dtype)
    a[:, :k_dim] = gamma[:, None] * k_chunk
    a[:, k_dim] = -gamma * lambda_k_chunk
    b[:, :k_dim] = k_chunk
    b[:, k_dim] = 1

    t = torch.empty(c, c, device=device, dtype=torch.float32)
    block_c = 32
    block_d = 32
    grid = (triton.cdiv(c, block_c), triton.cdiv(c, block_c))
    _rank1_dc_build_t_kernel[grid](
        a, b, t,
        c, d,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        t.stride(0), t.stride(1),
        BLOCK_C=block_c,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    alpha_bar = alpha.prod()
    return alpha_bar, a, b, t


def _build_rank1_dc_chunk_transfer(
    q_chunk: torch.Tensor,
    k_chunk: torch.Tensor,
    g_chunk: torch.Tensor,
    beta_chunk: torch.Tensor,
    lambda_q_chunk: torch.Tensor,
    lambda_k_chunk: torch.Tensor,
    scale: float,
):
    c, k_dim = k_chunk.shape
    d = k_dim + 1
    device = k_chunk.device
    dtype = k_chunk.dtype

    eye_d = torch.eye(d, device=device, dtype=dtype)
    alpha = g_chunk.exp()

    c_vecs = torch.empty(c, d, device=device, dtype=dtype)
    u_vecs = torch.empty(c, d, device=device, dtype=dtype)
    p_vecs = torch.empty(c, d, device=device, dtype=dtype)
    rs = torch.empty(c, d, device=device, dtype=dtype)

    c_vecs[:, :k_dim] = k_chunk
    c_vecs[:, k_dim] = -lambda_k_chunk
    u_vecs[:, :k_dim] = k_chunk
    u_vecs[:, k_dim] = 1
    p_vecs[:, :k_dim] = beta_chunk[:, None] * k_chunk
    p_vecs[:, k_dim] = beta_chunk
    rs[:, :k_dim] = scale * q_chunk
    rs[:, k_dim] = -lambda_q_chunk

    # The exact augmented recurrence is X_t = (alpha I - beta u_t c_t^T) X_{t-1} + beta u_t v_t^T.
    # Keep the exact PyTorch path as ground truth until the WY factors are re-derived for this orientation.
    a_chunk = eye_d.clone()
    r_chunk = torch.empty(c, d, device=device, dtype=dtype)
    for t in range(c):
        proj = torch.matmul(c_vecs[t], a_chunk)
        a_chunk = alpha[t] * (a_chunk - beta_chunk[t] * torch.outer(u_vecs[t], proj))
        r_chunk[t] = torch.matmul(rs[t], a_chunk)

    use_triton_cols = False

    l_chunk = torch.zeros(c, c, device=device, dtype=dtype)
    z_cols = torch.empty(d, c, device=device, dtype=dtype)
    for t in range(c):
        if t > 0:
            if use_triton_cols:
                l_chunk[t, :t] = _rank1_dc_update_cols_and_row_dot_fused_triton(
                    z_cols[:, :t], c_vecs[t], u_vecs[t], rs[t], alpha[t], beta_chunk[t]
                )
            else:
                proj = torch.matmul(c_vecs[t], z_cols[:, :t])
                z_cols[:, :t] = alpha[t] * (z_cols[:, :t] - beta_chunk[t] * u_vecs[t][:, None] * proj[None, :])
                l_chunk[t, :t] = torch.matmul(rs[t], z_cols[:, :t])
        z_cols[:, t] = p_vecs[t]
        l_chunk[t, t] = torch.dot(rs[t], p_vecs[t])

    p_cols = p_vecs.transpose(0, 1).contiguous()
    for t in range(c - 1, 0, -1):
        if use_triton_cols:
            _rank1_dc_update_cols_fused_triton(p_cols[:, :t], c_vecs[t], u_vecs[t], alpha[t], beta_chunk[t])
        else:
            proj = torch.matmul(c_vecs[t], p_cols[:, :t])
            p_cols[:, :t] = alpha[t] * (p_cols[:, :t] - beta_chunk[t] * u_vecs[t][:, None] * proj[None, :])
    p_chunk = p_cols.transpose(0, 1).contiguous()

    return a_chunk, r_chunk, l_chunk, p_chunk


def _chunk_rank1_dc_triton_local(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    scale: float,
    chunk_size: int,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None,
    output_final_state: bool,
):
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_rank1_dc

    batch_size, seq_len = q.shape[:2]
    v_dim = v.shape[-1]
    o = torch.empty(batch_size, seq_len, q.shape[2], v_dim, device=v.device, dtype=v.dtype)
    if initial_state is None:
        state = torch.zeros(batch_size, q.shape[2], q.shape[3], v_dim, device=v.device, dtype=torch.float32)
        bias_state = torch.zeros(batch_size, q.shape[2], v_dim, device=v.device, dtype=torch.float32)
    else:
        state, bias_state = initial_state

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        o_chunk, (state, bias_state) = fused_recurrent_gated_delta_rule_rank1_dc(
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            lambda_q=lambda_q[:, start:end],
            lambda_k=lambda_k[:, start:end],
            scale=scale,
            initial_state=(state, bias_state),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        o[:, start:end] = o_chunk

    if not output_final_state:
        return o, None
    return o, (state, bias_state)


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

    h_states = [h]
    b_states = [b]
    outputs = []

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
    h_states = torch.stack(h_states, dim=2)
    b_states = torch.stack(b_states, dim=2)
    return o, (h, b), h_states, b_states


def _dense_rank1_dc_boundary_states(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    scale: float,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None,
    chunk_size: int,
):
    del q, scale
    kh, vh, gh, betah, lkh = (
        k.permute(0, 2, 1, 3).contiguous().to(torch.float32),
        v.permute(0, 2, 1, 3).contiguous().to(torch.float32),
        g.permute(0, 2, 1).contiguous().to(torch.float32),
        beta.permute(0, 2, 1).contiguous().to(torch.float32),
        lambda_k.permute(0, 2, 1).contiguous().to(torch.float32),
    )
    _, _, seq_len, k_dim = kh.shape
    v_dim = vh.shape[-1]
    if initial_state is None:
        h = vh.new_zeros(kh.shape[0], kh.shape[1], k_dim, v_dim)
        b = vh.new_zeros(kh.shape[0], kh.shape[1], v_dim)
    else:
        h = initial_state[0].to(torch.float32)
        b = initial_state[1].to(torch.float32)

    boundary_states = [(h, b)]
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        for t in range(start, end):
            alpha_t = gh[:, :, t].exp()
            h1 = h * alpha_t[..., None, None]
            b1 = b * alpha_t[..., None]
            pred = torch.einsum('bhkv,bhk->bhv', h1, kh[:, :, t]) - lkh[:, :, t][..., None] * b1
            delta = betah[:, :, t][..., None] * (vh[:, :, t] - pred)
            h = h1 + torch.einsum('bhk,bhv->bhkv', kh[:, :, t], delta)
            b = b1 + delta
        boundary_states.append((h, b))
    return boundary_states


def _chunk_rank1_dc_boundary_states_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    scale: float,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None,
    chunk_size: int,
):
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_rank1_dc_fwd

    del q
    seq_len = k.shape[1]
    state = initial_state
    boundary_states = [state]
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        _, state = fused_recurrent_gated_delta_rule_rank1_dc_fwd(
            q=k[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            lambda_q=lambda_q[:, start:end],
            lambda_k=lambda_k[:, start:end],
            scale=scale,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
            use_exp2=False,
            transpose_state_layout=False,
        )
        boundary_states.append(state)
    return boundary_states


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

    dq = dq.permute(0, 2, 1, 3).contiguous()
    dk = dk.permute(0, 2, 1, 3).contiguous()
    dv = dv.permute(0, 2, 1, 3).contiguous()
    dg = dg.permute(0, 2, 1).contiguous()
    dbeta = dbeta.permute(0, 2, 1).contiguous()
    dlambda_q = dlambda_q.permute(0, 2, 1).contiguous()
    dlambda_k = dlambda_k.permute(0, 2, 1).contiguous()
    return dq, dk, dv, dg, dbeta, dlambda_q, dlambda_k, dh, db


def _chunk_rank1_dc_single_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    scale: float,
    chunk_size: int,
    state: torch.Tensor,
    bias_state: torch.Tensor,
):
    h_list = [state[head_idx].clone() for head_idx in range(state.shape[0])]
    b_list = [bias_state[head_idx].clone() for head_idx in range(bias_state.shape[0])]
    num_heads, seq_len, k_dim = q.shape
    v_dim = v.shape[-1]
    o_chunks = [[None for _ in range((seq_len + chunk_size - 1) // chunk_size)] for _ in range(num_heads)]

    for chunk_idx, start in enumerate(range(0, seq_len, chunk_size)):
        end = min(start + chunk_size, seq_len)
        for head_idx in range(num_heads):
            a_chunk, r_chunk, l_chunk, p_chunk = _build_rank1_dc_chunk_transfer(
                q_chunk=q[head_idx, start:end],
                k_chunk=k[head_idx, start:end],
                g_chunk=g[head_idx, start:end],
                beta_chunk=beta[head_idx, start:end],
                lambda_q_chunk=lambda_q[head_idx, start:end],
                lambda_k_chunk=lambda_k[head_idx, start:end],
                scale=scale,
            )

            x_in = torch.cat([h_list[head_idx], b_list[head_idx].unsqueeze(0)], dim=0)
            v_chunk = v[head_idx, start:end]
            x_out = a_chunk @ x_in + p_chunk.transpose(0, 1) @ v_chunk
            y_chunk = r_chunk @ x_in + l_chunk @ v_chunk

            h_list[head_idx] = x_out[:k_dim]
            b_list[head_idx] = x_out[k_dim]
            o_chunks[head_idx][chunk_idx] = y_chunk

    o = torch.stack([torch.cat(chunks, dim=0) for chunks in o_chunks], dim=0)
    h = torch.stack(h_list, dim=0)
    b = torch.stack(b_list, dim=0)
    return o, h, b


def _chunk_gated_delta_rule_rank1_dc_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    scale: float,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
):
    q, k, v, g, beta, lambda_q, lambda_k = map(
        lambda x: x.to(torch.float32),
        [q, k, v, g, beta, lambda_q, lambda_k],
    )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        num_seq = len(cu_seqlens) - 1
        num_heads, k_dim, v_dim = q.shape[2], q.shape[3], v.shape[-1]
        init_state = torch.zeros(num_seq, num_heads, k_dim, v_dim, device=v.device, dtype=torch.float32)
        init_bias_state = torch.zeros(num_seq, num_heads, v_dim, device=v.device, dtype=torch.float32)
        if initial_state is not None:
            init_state, init_bias_state = initial_state
            init_state = init_state.to(torch.float32).clone()
            init_bias_state = init_bias_state.to(torch.float32).clone()
        o_segments = []
        state_segments = []
        bias_segments = []
        spans = []
        for n in range(num_seq):
            bos = int(cu_seqlens[n].item())
            eos = int(cu_seqlens[n + 1].item())
            o_n, state_n, bias_state_n = _chunk_rank1_dc_single_sequence(
                q=q[0, bos:eos].transpose(0, 1).contiguous(),
                k=k[0, bos:eos].transpose(0, 1).contiguous(),
                v=v[0, bos:eos].transpose(0, 1).contiguous(),
                g=g[0, bos:eos].transpose(0, 1).contiguous(),
                beta=beta[0, bos:eos].transpose(0, 1).contiguous(),
                lambda_q=lambda_q[0, bos:eos].transpose(0, 1).contiguous(),
                lambda_k=lambda_k[0, bos:eos].transpose(0, 1).contiguous(),
                scale=scale,
                chunk_size=chunk_size,
                state=init_state[n],
                bias_state=init_bias_state[n],
            )
            spans.append((bos, eos))
            o_segments.append(o_n.transpose(0, 1))
            state_segments.append(state_n)
            bias_segments.append(bias_state_n)
        o = torch.zeros(1, q.shape[1], num_heads, v_dim, device=v.device, dtype=torch.float32)
        for (bos, eos), o_n in zip(spans, o_segments):
            o = o.clone()
            o[0, bos:eos] = o_n
        state = torch.stack(state_segments, dim=0)
        bias_state = torch.stack(bias_segments, dim=0)
        return o, state, bias_state

    q, k, v, g, beta, lambda_q, lambda_k = map(
        lambda x: x.transpose(1, 2).contiguous(),
        [q, k, v, g, beta, lambda_q, lambda_k],
    )
    batch_size, num_heads, seq_len, k_dim, v_dim = *k.shape, v.shape[-1]
    init_state = torch.zeros(batch_size, num_heads, k_dim, v_dim, device=v.device, dtype=torch.float32)
    init_bias_state = torch.zeros(batch_size, num_heads, v_dim, device=v.device, dtype=torch.float32)
    if initial_state is not None:
        init_state, init_bias_state = initial_state
        init_state = init_state.to(torch.float32).clone()
        init_bias_state = init_bias_state.to(torch.float32).clone()
    o_list = []
    state_list = []
    bias_state_list = []
    for batch_idx in range(batch_size):
        o_i, state_i, bias_state_i = _chunk_rank1_dc_single_sequence(
            q=q[batch_idx],
            k=k[batch_idx],
            v=v[batch_idx],
            g=g[batch_idx],
            beta=beta[batch_idx],
            lambda_q=lambda_q[batch_idx],
            lambda_k=lambda_k[batch_idx],
            scale=scale,
            chunk_size=chunk_size,
            state=init_state[batch_idx],
            bias_state=init_bias_state[batch_idx],
        )
        o_list.append(o_i)
        state_list.append(state_i)
        bias_state_list.append(bias_state_i)
    o = torch.stack(o_list, dim=0).transpose(1, 2).contiguous()
    state = torch.stack(state_list, dim=0)
    bias_state = torch.stack(bias_state_list, dim=0)
    return o, state, bias_state


def _rank1_dc_backward_single_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    scale: float,
    h0: torch.Tensor,
    b0: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dbt: torch.Tensor,
):
    seq_len, k_dim = q.shape
    v_dim = v.shape[-1]

    saved = []
    h = h0
    b = b0
    for t in range(seq_len):
        q_t = q[t]
        k_t = k[t]
        v_t = v[t]
        alpha_t = g[t].exp()
        beta_t = beta[t]
        lambda_q_t = lambda_q[t]
        lambda_k_t = lambda_k[t]

        h_prev = h
        b_prev = b
        h1 = alpha_t * h_prev
        b1 = alpha_t * b_prev
        pred = (h1 * k_t[:, None]).sum(0) - lambda_k_t * b1
        innovation = v_t - pred
        delta = beta_t * innovation
        h = h1 + k_t[:, None] * delta[None, :]
        b = b1 + delta
        saved.append((h_prev, b_prev, h1, b1, h, b, innovation, q_t, k_t, v_t, alpha_t, beta_t, lambda_q_t, lambda_k_t))

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dg = torch.zeros_like(g)
    dbeta = torch.zeros_like(beta)
    dlambda_q = torch.zeros_like(lambda_q)
    dlambda_k = torch.zeros_like(lambda_k)

    dh = dht
    db = dbt

    for t in range(seq_len - 1, -1, -1):
        h_prev, b_prev, h1, b1, h_t, b_t, innovation, q_t, k_t, v_t, alpha_t, beta_t, lambda_q_t, lambda_k_t = saved[t]
        do_t = do[t]

        dq[t] = scale * (h_t @ do_t)
        dh = dh + scale * q_t[:, None] * do_t[None, :]
        dlambda_q[t] = -(b_t * do_t).sum()
        db = db - lambda_q_t * do_t

        dh1 = dh
        dk[t] = dk[t] + dh.reshape(k_dim, v_dim) @ (beta_t * innovation)
        ddelta = (dh * k_t[:, None]).sum(0) + db
        db1 = db

        dbeta[t] = (ddelta * innovation).sum()
        dv[t] = beta_t * ddelta
        dpred = -beta_t * ddelta

        dh1 = dh1 + k_t[:, None] * dpred[None, :]
        dk[t] = dk[t] + (h1 * dpred[None, :]).sum(1)
        dlambda_k[t] = -(b1 * dpred).sum()
        db1 = db1 - lambda_k_t * dpred

        dalpha = (dh1 * h_prev).sum() + (db1 * b_prev).sum()
        dh = alpha_t * dh1
        db = alpha_t * db1
        dg[t] = dalpha * alpha_t

    return dq, dk, dv, dg, dbeta, dlambda_q, dlambda_k, dh, db


def _chunk_gated_delta_rule_rank1_dc_backward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    scale: float,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None,
    cu_seqlens: torch.LongTensor | None,
    do: torch.Tensor,
    dstate: torch.Tensor,
    dbias_state: torch.Tensor,
):
    qf, kf, vf, gf, betaf, lqf, lkf, dof = [x.to(torch.float32) for x in (q, k, v, g, beta, lambda_q, lambda_k, do)]
    if dstate is None:
        dstate = torch.zeros((0,), device=q.device, dtype=torch.float32)
    if dbias_state is None:
        dbias_state = torch.zeros((0,), device=q.device, dtype=torch.float32)
    dstatef = dstate.to(torch.float32)
    dbiasf = dbias_state.to(torch.float32)

    if cu_seqlens is not None:
        num_seq = len(cu_seqlens) - 1
        total, num_heads, k_dim = qf.shape[1], qf.shape[2], qf.shape[3]
        v_dim = vf.shape[-1]
        dq = torch.zeros_like(qf)
        dk = torch.zeros_like(kf)
        dv = torch.zeros_like(vf)
        dg = torch.zeros_like(gf)
        dbeta = torch.zeros_like(betaf)
        dlq = torch.zeros_like(lqf)
        dlk = torch.zeros_like(lkf)
        if initial_state is None:
            h0 = torch.zeros(num_seq, num_heads, k_dim, v_dim, device=q.device, dtype=torch.float32)
            b0 = torch.zeros(num_seq, num_heads, v_dim, device=q.device, dtype=torch.float32)
        else:
            h0 = initial_state[0].to(torch.float32)
            b0 = initial_state[1].to(torch.float32)
        dh0 = torch.zeros_like(h0)
        db0 = torch.zeros_like(b0)
        for n in range(num_seq):
            bos = int(cu_seqlens[n].item())
            eos = int(cu_seqlens[n + 1].item())
            for h_idx in range(num_heads):
                grads = _rank1_dc_backward_single_sequence(
                    q=qf[0, bos:eos, h_idx],
                    k=kf[0, bos:eos, h_idx],
                    v=vf[0, bos:eos, h_idx],
                    g=gf[0, bos:eos, h_idx],
                    beta=betaf[0, bos:eos, h_idx],
                    lambda_q=lqf[0, bos:eos, h_idx],
                    lambda_k=lkf[0, bos:eos, h_idx],
                    scale=scale,
                    h0=h0[n, h_idx],
                    b0=b0[n, h_idx],
                    do=dof[0, bos:eos, h_idx],
                    dht=dstatef[n, h_idx] if dstatef.numel() > 0 else torch.zeros_like(h0[n, h_idx]),
                    dbt=dbiasf[n, h_idx] if dbiasf.numel() > 0 else torch.zeros_like(b0[n, h_idx]),
                )
                dq_i, dk_i, dv_i, dg_i, dbeta_i, dlq_i, dlk_i, dh0_i, db0_i = grads
                dq[0, bos:eos, h_idx] = dq_i
                dk[0, bos:eos, h_idx] = dk_i
                dv[0, bos:eos, h_idx] = dv_i
                dg[0, bos:eos, h_idx] = dg_i
                dbeta[0, bos:eos, h_idx] = dbeta_i
                dlq[0, bos:eos, h_idx] = dlq_i
                dlk[0, bos:eos, h_idx] = dlk_i
                dh0[n, h_idx] = dh0_i
                db0[n, h_idx] = db0_i
        return dq, dk, dv, dg, dbeta, dlq, dlk, dh0, db0

    batch_size, seq_len, num_heads, k_dim = qf.shape
    v_dim = vf.shape[-1]
    dq = torch.zeros_like(qf)
    dk = torch.zeros_like(kf)
    dv = torch.zeros_like(vf)
    dg = torch.zeros_like(gf)
    dbeta = torch.zeros_like(betaf)
    dlq = torch.zeros_like(lqf)
    dlk = torch.zeros_like(lkf)
    if initial_state is None:
        h0 = torch.zeros(batch_size, num_heads, k_dim, v_dim, device=q.device, dtype=torch.float32)
        b0 = torch.zeros(batch_size, num_heads, v_dim, device=q.device, dtype=torch.float32)
    else:
        h0 = initial_state[0].to(torch.float32)
        b0 = initial_state[1].to(torch.float32)
    dh0 = torch.zeros_like(h0)
    db0 = torch.zeros_like(b0)
    for b_idx in range(batch_size):
        for h_idx in range(num_heads):
            grads = _rank1_dc_backward_single_sequence(
                q=qf[b_idx, :, h_idx],
                k=kf[b_idx, :, h_idx],
                v=vf[b_idx, :, h_idx],
                g=gf[b_idx, :, h_idx],
                beta=betaf[b_idx, :, h_idx],
                lambda_q=lqf[b_idx, :, h_idx],
                lambda_k=lkf[b_idx, :, h_idx],
                scale=scale,
                h0=h0[b_idx, h_idx],
                b0=b0[b_idx, h_idx],
                do=dof[b_idx, :, h_idx],
                dht=dstatef[b_idx, h_idx] if dstatef.numel() > 0 else torch.zeros_like(h0[b_idx, h_idx]),
                dbt=dbiasf[b_idx, h_idx] if dbiasf.numel() > 0 else torch.zeros_like(b0[b_idx, h_idx]),
            )
            dq_i, dk_i, dv_i, dg_i, dbeta_i, dlq_i, dlk_i, dh0_i, db0_i = grads
            dq[b_idx, :, h_idx] = dq_i
            dk[b_idx, :, h_idx] = dk_i
            dv[b_idx, :, h_idx] = dv_i
            dg[b_idx, :, h_idx] = dg_i
            dbeta[b_idx, :, h_idx] = dbeta_i
            dlq[b_idx, :, h_idx] = dlq_i
            dlk[b_idx, :, h_idx] = dlk_i
            dh0[b_idx, h_idx] = dh0_i
            db0[b_idx, h_idx] = db0_i
    return dq, dk, dv, dg, dbeta, dlq, dlk, dh0, db0


class ChunkGatedDeltaRuleRank1DCFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        lambda_q: torch.Tensor,
        lambda_k: torch.Tensor,
        state0: torch.Tensor,
        bias_state0: torch.Tensor,
        cu_seqlens: torch.Tensor,
        scale: float,
        chunk_size: int,
        has_initial_state: bool,
    ):
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.has_initial_state = has_initial_state
        ctx.has_cu_seqlens = cu_seqlens.numel() > 0
        if (not q.is_cuda) or ctx.has_cu_seqlens:
            raise NotImplementedError(
                "ChunkGatedDeltaRuleRank1DCFunction only supports CUDA dense inputs. "
                "Variable-length inputs and non-CUDA execution are not supported in the optimized path.",
            )
        ctx.save_for_backward(q, k, v, g, beta, lambda_q, lambda_k, state0, bias_state0, cu_seqlens)
        initial_state = (state0, bias_state0) if has_initial_state else None
        with torch.no_grad():
            o, final_state, _, _ = _dense_rank1_dc_forward_batched(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                lambda_q=lambda_q,
                lambda_k=lambda_k,
                scale=scale,
                initial_state=initial_state,
            )
            state, bias_state = final_state
        return o, state, bias_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, dstate: torch.Tensor, dbias_state: torch.Tensor):
        from fla.ops.gated_delta_rule.fused_recurrent import (
            fused_recurrent_gated_delta_rule_rank1_dc_bwd,
            fused_recurrent_gated_delta_rule_rank1_dc_fwd,
        )

        q, k, v, g, beta, lambda_q, lambda_k, state0, bias_state0, cu_seqlens = ctx.saved_tensors
        if dstate is None:
            dstate = None
        if dbias_state is None:
            dbias_state = None
        initial_state = (state0, bias_state0) if ctx.has_initial_state else None
        with torch.no_grad():
            boundary_states = _chunk_rank1_dc_boundary_states_recurrent(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                lambda_q=lambda_q,
                lambda_k=lambda_k,
                scale=ctx.scale,
                initial_state=initial_state,
                chunk_size=ctx.chunk_size,
            )
        seq_len = q.shape[1]
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        dg = torch.zeros_like(g, dtype=torch.float32)
        dbeta = torch.zeros_like(beta, dtype=torch.float32)
        dlambda_q = torch.zeros_like(lambda_q, dtype=torch.float32)
        dlambda_k = torch.zeros_like(lambda_k, dtype=torch.float32)
        dh = None if dstate is None else dstate.to(torch.float32)
        db = None if dbias_state is None else dbias_state.to(torch.float32)

        num_chunks = len(boundary_states) - 1
        for chunk_idx in range(num_chunks - 1, -1, -1):
            start = chunk_idx * ctx.chunk_size
            end = min(start + ctx.chunk_size, seq_len)
            chunk_initial_state = boundary_states[chunk_idx]
            with torch.no_grad():
                out_i, final_state_i = fused_recurrent_gated_delta_rule_rank1_dc_fwd(
                    q=q[:, start:end],
                    k=k[:, start:end],
                    v=v[:, start:end],
                    g=g[:, start:end],
                    beta=beta[:, start:end],
                    lambda_q=lambda_q[:, start:end],
                    lambda_k=lambda_k[:, start:end],
                    scale=ctx.scale,
                    initial_state=chunk_initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=False,
                    cu_seqlens=None,
                    use_exp2=False,
                    transpose_state_layout=False,
                )
                _, _, h_states, b_states = _dense_rank1_dc_forward_batched(
                    q=q[:, start:end],
                    k=k[:, start:end],
                    v=v[:, start:end],
                    g=g[:, start:end],
                    beta=beta[:, start:end],
                    lambda_q=lambda_q[:, start:end],
                    lambda_k=lambda_k[:, start:end],
                    scale=ctx.scale,
                    initial_state=chunk_initial_state,
                )
            grad_h = dh if dh is not None else torch.zeros_like(final_state_i[0])
            grad_b = db if db is not None else torch.zeros_like(final_state_i[1])
            dq_i, dk_i, dv_i, dg_i, dbeta_i, dlambda_q_i, dlambda_k_i, dh, db = fused_recurrent_gated_delta_rule_rank1_dc_bwd(
                q=q[:, start:end],
                k=k[:, start:end],
                v=v[:, start:end],
                g=g[:, start:end],
                beta=beta[:, start:end],
                lambda_q=lambda_q[:, start:end],
                lambda_k=lambda_k[:, start:end],
                do=do[:, start:end],
                dht=grad_h,
                dbt=grad_b,
                scale=ctx.scale,
                initial_state=chunk_initial_state,
                h_states=h_states,
                b_states=b_states,
                cu_seqlens=None,
                use_exp2=False,
                transpose_state_layout=False,
            )
            dq[:, start:end] = dq_i.to(torch.float32)
            dk[:, start:end] = dk_i.to(torch.float32)
            dv[:, start:end] = dv_i.to(torch.float32)
            dg[:, start:end] = dg_i.to(torch.float32)
            dbeta[:, start:end] = dbeta_i.to(torch.float32)
            dlambda_q[:, start:end] = dlambda_q_i.to(torch.float32)
            dlambda_k[:, start:end] = dlambda_k_i.to(torch.float32)
            if chunk_initial_state is None:
                dh = None
                db = None
        dstate0, dbias0 = dh, db
        dq = dq.to(q.dtype)
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)
        dg = dg.to(g.dtype)
        dbeta = dbeta.to(beta.dtype)
        dlambda_q = dlambda_q.to(lambda_q.dtype)
        dlambda_k = dlambda_k.to(lambda_k.dtype)
        if not ctx.has_initial_state:
            dstate0 = None
            dbias0 = None
        return dq, dk, dv, dg, dbeta, dlambda_q, dlambda_k, dstate0, dbias0, None, None, None, None


@torch.compiler.disable
def chunk_gated_delta_rule_rank1_dc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    scale: float = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    phase: torch.Tensor | None = None,
    num_phase_channels: int = 0,
    **kwargs,
):
    if phase is not None and num_phase_channels > 0:
        raise NotImplementedError(
            "Operator-level `phase` for chunk_gated_delta_rule_rank1_dc still matches the legacy value/state-phase "
            "design and is incompatible with the current q/k-phase design. Rotate q/k upstream, recompute "
            "lambda_q/lambda_k from the rotated addresses, and call with `phase=None`.",
        )
    orig_dtype = v.dtype
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if (not q.is_cuda) or cu_seqlens is not None:
        raise NotImplementedError(
            "chunk_gated_delta_rule_rank1_dc only supports CUDA dense inputs. "
            "Non-CUDA and `cu_seqlens` fallback paths were removed.",
        )

    use_triton_chunk_local = (
        (not torch.is_grad_enabled())
        and not any(x.requires_grad for x in (q, k, v, g, beta, lambda_q, lambda_k))
    )

    if use_triton_chunk_local:
        o, final_state = _chunk_rank1_dc_triton_local(
            q=q.to(torch.float32),
            k=k.to(torch.float32),
            v=v.to(torch.float32),
            g=g.to(torch.float32),
            beta=beta.to(torch.float32),
            lambda_q=lambda_q.to(torch.float32),
            lambda_k=lambda_k.to(torch.float32),
            scale=scale,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        return o.to(orig_dtype), final_state

    needs_explicit_backward = torch.is_grad_enabled() and (
        any(x.requires_grad for x in (q, k, v, g, beta, lambda_q, lambda_k))
        or (initial_state is not None and any(x.requires_grad for x in initial_state))
    )

    use_autograd_wrapper = needs_explicit_backward

    if use_autograd_wrapper:
        if initial_state is None:
            state0 = q.new_zeros(0, dtype=torch.float32)
            bias_state0 = q.new_zeros(0, dtype=torch.float32)
            has_initial_state = False
        else:
            state0, bias_state0 = initial_state
            has_initial_state = True
        cu_seqlens_tensor = q.new_empty((0,), dtype=torch.long)
        o, state, bias_state = ChunkGatedDeltaRuleRank1DCFunction.apply(
            q, k, v, g, beta, lambda_q, lambda_k,
            state0, bias_state0, cu_seqlens_tensor,
            scale, chunk_size, has_initial_state,
        )
    else:
        o, final_state = _chunk_rank1_dc_triton_local(
            q=q.to(torch.float32),
            k=k.to(torch.float32),
            v=v.to(torch.float32),
            g=g.to(torch.float32),
            beta=beta.to(torch.float32),
            lambda_q=lambda_q.to(torch.float32),
            lambda_k=lambda_k.to(torch.float32),
            scale=scale,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        state, bias_state = (None, None) if final_state is None else final_state

    if not output_final_state:
        return o.to(orig_dtype), None
    return o.to(orig_dtype), (state, bias_state)


chunk_gdn = chunk_gated_delta_rule
