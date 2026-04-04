# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang

import warnings

import torch

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
    **kwargs,
):
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

    eye_k = torch.eye(k_dim, device=device, dtype=dtype)
    eye_d = torch.eye(d, device=device, dtype=dtype)
    ms = []
    us = []
    rs = []

    for t in range(c):
        phi_t = k_chunk[t]
        alpha_t = g_chunk[t].exp()
        beta_t = beta_chunk[t]
        lambda_k_t = lambda_k_chunk[t]

        m_t = torch.zeros(d, d, device=device, dtype=dtype)
        m_t[:k_dim, :k_dim] = alpha_t * eye_k - beta_t * torch.outer(phi_t, phi_t)
        m_t[:k_dim, k_dim] = beta_t * lambda_k_t * phi_t
        m_t[k_dim, :k_dim] = -beta_t * phi_t
        m_t[k_dim, k_dim] = alpha_t + beta_t * lambda_k_t
        ms.append(m_t)

        u_t = torch.empty(d, device=device, dtype=dtype)
        u_t[:k_dim] = beta_t * phi_t
        u_t[k_dim] = beta_t
        us.append(u_t)

        r_t = torch.empty(d, device=device, dtype=dtype)
        r_t[:k_dim] = scale * q_chunk[t]
        r_t[k_dim] = -lambda_q_chunk[t]
        rs.append(r_t)

    a_chunk = eye_d
    r_chunk = torch.empty(c, d, device=device, dtype=dtype)
    for t in range(c):
        a_chunk = ms[t] @ a_chunk
        r_chunk[t] = rs[t] @ a_chunk

    l_chunk = torch.zeros(c, c, device=device, dtype=dtype)
    p_chunk = torch.zeros(c, d, device=device, dtype=dtype)
    for m in range(c):
        z = us[m]
        for t in range(m, c):
            l_chunk[t, m] = torch.dot(rs[t], z)
            if t == c - 1:
                p_chunk[m] = z
            else:
                z = ms[t + 1] @ z

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
    h = state.clone()
    b = bias_state.clone()
    num_heads, seq_len, k_dim = q.shape
    v_dim = v.shape[-1]
    o = torch.zeros(num_heads, seq_len, v_dim, device=v.device, dtype=torch.float32)

    for start in range(0, seq_len, chunk_size):
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

            x_in = torch.cat([h[head_idx], b[head_idx].unsqueeze(0)], dim=0)
            v_chunk = v[head_idx, start:end]
            x_out = a_chunk @ x_in + p_chunk.transpose(0, 1) @ v_chunk
            y_chunk = r_chunk @ x_in + l_chunk @ v_chunk

            h[head_idx] = x_out[:k_dim]
            b[head_idx] = x_out[k_dim]
            o[head_idx, start:end] = y_chunk
    return o, h, b


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
    **kwargs,
):
    orig_dtype = v.dtype
    if scale is None:
        scale = k.shape[-1] ** -0.5

    use_triton_chunk_local = (
        cu_seqlens is None
        and q.is_cuda
        and (not torch.is_grad_enabled())
        and not any(x.requires_grad for x in (q, k, v, g, beta, lambda_q, lambda_k))
    )

    q, k, v, g, beta, lambda_q, lambda_k = map(
        lambda x: x.to(torch.float32),
        [q, k, v, g, beta, lambda_q, lambda_k],
    )

    if use_triton_chunk_local:
        return _chunk_rank1_dc_triton_local(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            lambda_q=lambda_q,
            lambda_k=lambda_k,
            scale=scale,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        num_seq = len(cu_seqlens) - 1
        num_heads, k_dim, v_dim = q.shape[2], q.shape[3], v.shape[-1]
        o = torch.zeros(1, q.shape[1], num_heads, v_dim, device=v.device, dtype=torch.float32)
        if initial_state is None:
            state = torch.zeros(num_seq, num_heads, k_dim, v_dim, device=v.device, dtype=torch.float32)
            bias_state = torch.zeros(num_seq, num_heads, v_dim, device=v.device, dtype=torch.float32)
        else:
            state, bias_state = initial_state
            state = state.to(torch.float32)
            bias_state = bias_state.to(torch.float32)
        for n in range(num_seq):
            bos = int(cu_seqlens[n].item())
            eos = int(cu_seqlens[n + 1].item())
            o_n, state[n], bias_state[n] = _chunk_rank1_dc_single_sequence(
                q=q[0, bos:eos].transpose(0, 1).contiguous(),
                k=k[0, bos:eos].transpose(0, 1).contiguous(),
                v=v[0, bos:eos].transpose(0, 1).contiguous(),
                g=g[0, bos:eos].transpose(0, 1).contiguous(),
                beta=beta[0, bos:eos].transpose(0, 1).contiguous(),
                lambda_q=lambda_q[0, bos:eos].transpose(0, 1).contiguous(),
                lambda_k=lambda_k[0, bos:eos].transpose(0, 1).contiguous(),
                scale=scale,
                chunk_size=chunk_size,
                state=state[n],
                bias_state=bias_state[n],
            )
            o[0, bos:eos] = o_n.transpose(0, 1)
    else:
        q, k, v, g, beta, lambda_q, lambda_k = map(
            lambda x: x.transpose(1, 2).contiguous(),
            [q, k, v, g, beta, lambda_q, lambda_k],
        )
        batch_size, num_heads, seq_len, k_dim, v_dim = *k.shape, v.shape[-1]
        o = torch.zeros(batch_size, num_heads, seq_len, v_dim, device=v.device, dtype=torch.float32)
        if initial_state is None:
            state = torch.zeros(batch_size, num_heads, k_dim, v_dim, device=v.device, dtype=torch.float32)
            bias_state = torch.zeros(batch_size, num_heads, v_dim, device=v.device, dtype=torch.float32)
        else:
            state, bias_state = initial_state
            state = state.to(torch.float32)
            bias_state = bias_state.to(torch.float32)
        for batch_idx in range(batch_size):
            o[batch_idx], state[batch_idx], bias_state[batch_idx] = _chunk_rank1_dc_single_sequence(
                q=q[batch_idx],
                k=k[batch_idx],
                v=v[batch_idx],
                g=g[batch_idx],
                beta=beta[batch_idx],
                lambda_q=lambda_q[batch_idx],
                lambda_k=lambda_k[batch_idx],
                scale=scale,
                chunk_size=chunk_size,
                state=state[batch_idx],
                bias_state=bias_state[batch_idx],
            )
        o = o.transpose(1, 2).contiguous()

    if not output_final_state:
        return o.to(orig_dtype), None
    return o.to(orig_dtype), (state, bias_state)


chunk_gdn = chunk_gated_delta_rule
