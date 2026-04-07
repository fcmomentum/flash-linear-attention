from __future__ import annotations

import argparse

import torch

from fla.ops.gated_delta_rule import (
    naive_chunk_gated_delta_rule_phase_transport,
    naive_recurrent_gated_delta_rule,
)


def compare_tensor(name: str, ref: torch.Tensor, test: torch.Tensor, rtol: float, atol: float) -> None:
    diff = (ref - test).abs()
    max_abs = diff.max().item()
    max_rel = (diff / test.abs().clamp_min(1e-12)).max().item()
    print(f"{name}: max_abs={max_abs:.6e} max_rel={max_rel:.6e}")
    torch.testing.assert_close(ref, test, rtol=rtol, atol=atol)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check transported-frame chunk phased DeltaNet forward against the dense phased recurrence.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--k-dim", type=int, default=16)
    parser.add_argument("--v-dim", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--num-phase-channels", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--with-initial-state", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    if args.num_phase_channels % 2 != 0:
        raise ValueError("--num-phase-channels must be even.")
    if args.num_phase_channels > args.v_dim:
        raise ValueError("--num-phase-channels cannot exceed --v-dim.")

    torch.manual_seed(0)

    q = torch.randn(args.batch_size, args.seq_len, args.num_heads, args.k_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(args.batch_size, args.seq_len, args.num_heads, args.v_dim, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(args.batch_size, args.seq_len, args.num_heads, device=device, dtype=dtype))
    g = torch.randn(args.batch_size, args.seq_len, args.num_heads, device=device, dtype=dtype) - 1.0
    phase = torch.randn(args.batch_size, args.seq_len, args.num_heads, args.v_dim, device=device, dtype=dtype)
    if args.num_phase_channels < args.v_dim:
        phase = phase.clone()
        phase[..., args.num_phase_channels:] = 0

    initial_state = None
    if args.with_initial_state:
        initial_state = torch.randn(
            args.batch_size, args.num_heads, args.k_dim, args.v_dim, device=device, dtype=dtype,
        )

    ref_out, ref_state = naive_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        initial_state=initial_state,
        output_final_state=True,
        phase=phase,
        num_phase_channels=args.num_phase_channels,
    )
    test_out, test_state = naive_chunk_gated_delta_rule_phase_transport(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        phase=phase,
        num_phase_channels=args.num_phase_channels,
        chunk_size=args.chunk_size,
        initial_state=initial_state,
        output_final_state=True,
    )

    compare_tensor("forward_output", ref_out, test_out, args.rtol, args.atol)
    compare_tensor("final_state", ref_state, test_state, args.rtol, args.atol)
    print("all checks passed")


if __name__ == "__main__":
    main()
