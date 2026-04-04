# Copyright (c) 2026, Liang Ge

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import triton

from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule_rank1_dc,
    fused_recurrent_gated_delta_rule_rank1_dc,
    naive_recurrent_gated_delta_rule_rank1_dc,
)


DEFAULT_SHAPES = [
    (1, 1024, 8, 64, 64),
    (1, 4096, 8, 64, 64),
    (2, 2048, 8, 64, 64),
    (1, 8192, 16, 64, 64),
]
DEFAULT_PROVIDERS = ['naive', 'fused_recurrent', 'chunk_exact', 'chunk_triton']
DTYPE_MAP = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
}


def parse_shapes(spec: str):
    if not spec:
        return DEFAULT_SHAPES
    shapes = []
    for item in spec.split(';'):
        item = item.strip()
        if not item:
            continue
        parts = [int(x) for x in item.split(',')]
        if len(parts) != 5:
            raise ValueError(f"Invalid shape '{item}'. Expected B,T,H,K,V")
        shapes.append(tuple(parts))
    return shapes


def make_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    q = F.normalize(
        torch.randn(batch_size, seq_len, num_heads, head_k_dim, dtype=dtype, device=device, generator=generator),
        dim=-1,
    )
    k = F.normalize(
        torch.randn(batch_size, seq_len, num_heads, head_k_dim, dtype=dtype, device=device, generator=generator),
        dim=-1,
    )
    v = torch.randn(batch_size, seq_len, num_heads, head_v_dim, dtype=dtype, device=device, generator=generator)
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_heads, dtype=dtype, device=device, generator=generator))
    g = F.logsigmoid(torch.randn(batch_size, seq_len, num_heads, dtype=dtype, device=device, generator=generator))

    nu = F.softplus(torch.randn(num_heads, head_k_dim, dtype=torch.float32, device=device, generator=generator)) + 1e-6
    nu = F.normalize(nu, dim=-1)
    nu_norm_sq = nu.square().sum(-1).clamp_min(1e-6)
    scale = head_k_dim ** -0.5
    q_scaled = q.float() * scale
    lambda_k = (k.float() * nu.unsqueeze(0).unsqueeze(0)).sum(-1) / nu_norm_sq.unsqueeze(0).unsqueeze(0)
    lambda_q = (q_scaled * nu.unsqueeze(0).unsqueeze(0)).sum(-1) / nu_norm_sq.unsqueeze(0).unsqueeze(0)

    return {
        'q': q,
        'k': k,
        'v': v,
        'g': g,
        'beta': beta,
        'lambda_q': lambda_q.to(dtype),
        'lambda_k': lambda_k.to(dtype),
        'scale': scale,
    }


def clone_inputs(inputs: dict[str, torch.Tensor | float]):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}


def run_provider(provider: str, inputs: dict[str, torch.Tensor | float], chunk_size: int):
    kwargs = clone_inputs(inputs)
    if provider == 'naive':
        with torch.no_grad():
            return naive_recurrent_gated_delta_rule_rank1_dc(output_final_state=True, **kwargs)
    if provider == 'fused_recurrent':
        with torch.no_grad():
            return fused_recurrent_gated_delta_rule_rank1_dc(
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
                **kwargs,
            )
    if provider == 'chunk_exact':
        with torch.enable_grad():
            return chunk_gated_delta_rule_rank1_dc(output_final_state=True, chunk_size=chunk_size, **kwargs)
    if provider == 'chunk_triton':
        with torch.no_grad():
            return chunk_gated_delta_rule_rank1_dc(output_final_state=True, chunk_size=chunk_size, **kwargs)
    raise ValueError(f'Unknown provider: {provider}')


def bench_provider(provider: str, inputs: dict[str, torch.Tensor | float], chunk_size: int, warmup_ms: int, rep_ms: int):
    def fn():
        run_provider(provider, inputs, chunk_size)
    torch.cuda.synchronize()
    median_ms, p20_ms, p80_ms = triton.testing.do_bench(
        fn,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.5, 0.2, 0.8],
    )
    return median_ms, p20_ms, p80_ms


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def correctness_report(providers: list[str], inputs: dict[str, torch.Tensor | float], chunk_size: int):
    baseline_o, baseline_state = run_provider('naive', inputs, chunk_size)
    base_h, base_b = baseline_state
    rows = []
    for provider in providers:
        o, state = run_provider(provider, inputs, chunk_size)
        state_h, state_b = state
        rows.append({
            'provider': provider,
            'output_max_abs': max_diff(o, baseline_o),
            'state_max_abs': max_diff(state_h, base_h),
            'bias_max_abs': max_diff(state_b, base_b),
        })
    return rows


def print_correctness(rows: list[dict[str, float]]):
    print('\nCorrectness vs naive_recurrent')
    print('-' * 78)
    print(f"{'provider':<16} {'output_max_abs':>16} {'state_max_abs':>16} {'bias_max_abs':>16}")
    for row in rows:
        print(f"{row['provider']:<16} {row['output_max_abs']:>16.6e} {row['state_max_abs']:>16.6e} {row['bias_max_abs']:>16.6e}")


def print_bench_header():
    gpu = torch.cuda.get_device_name(0)
    print(f'GPU: {gpu} | torch={torch.__version__} | triton={triton.__version__}')
    print('-' * 98)
    print(f"{'provider':<16} {'B':>3} {'T':>6} {'H':>4} {'K':>4} {'V':>4} {'median_ms':>12} {'p20_ms':>12} {'p80_ms':>12}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark rank-1 DC removal Gated Delta Rule paths.')
    parser.add_argument('--shapes', type=str, default='', help='Semicolon-separated shapes: B,T,H,K,V;...')
    parser.add_argument('--providers', nargs='*', default=DEFAULT_PROVIDERS, choices=DEFAULT_PROVIDERS)
    parser.add_argument('--dtype', type=str, default='bf16', choices=sorted(DTYPE_MAP))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--warmup-ms', type=int, default=25)
    parser.add_argument('--rep-ms', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--check', action='store_true', help='Run correctness checks against naive recurrent first.')
    args = parser.parse_args()

    if args.device != 'cuda':
        raise ValueError('This benchmark harness currently expects --device cuda.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark harness.')

    dtype = DTYPE_MAP[args.dtype]
    shapes = parse_shapes(args.shapes)

    print_bench_header()
    for shape_idx, (batch_size, seq_len, num_heads, head_k_dim, head_v_dim) in enumerate(shapes):
        inputs = make_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            dtype=dtype,
            device=args.device,
            seed=args.seed + shape_idx,
        )
        if args.check:
            print_correctness(correctness_report(args.providers, inputs, args.chunk_size))
        for provider in args.providers:
            median_ms, p20_ms, p80_ms = bench_provider(
                provider=provider,
                inputs=inputs,
                chunk_size=args.chunk_size,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
            print(
                f"{provider:<16} {batch_size:>3} {seq_len:>6} {num_heads:>4} {head_k_dim:>4} {head_v_dim:>4} "
                f"{median_ms:>12.4f} {p20_ms:>12.4f} {p80_ms:>12.4f}"
            )
        print('-' * 98)


if __name__ == '__main__':
    main()
