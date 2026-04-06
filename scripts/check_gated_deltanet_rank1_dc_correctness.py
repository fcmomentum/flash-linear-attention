#!/usr/bin/env python3

from __future__ import annotations

import argparse

import torch

from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_rank1_dc
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule_rank1_dc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare rank-1 DC chunked GatedDeltaNet against the naive recurrent reference.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--k-dim", type=int, default=32)
    parser.add_argument("--v-dim", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-initial-state", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def clone_for_grad(x: torch.Tensor) -> torch.Tensor:
    y = x.clone().detach()
    y.requires_grad_(True)
    return y


def compare_tensor(name: str, ref: torch.Tensor, test: torch.Tensor, rtol: float, atol: float) -> None:
    max_abs = (ref - test).abs().max().item()
    max_rel = ((ref - test).abs() / test.abs().clamp_min(1e-12)).max().item()
    print(f"{name}: max_abs={max_abs:.6e} max_rel={max_rel:.6e}")
    torch.testing.assert_close(ref, test, rtol=rtol, atol=atol)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    shape_qk = (args.batch_size, args.seq_len, args.num_heads, args.k_dim)
    shape_v = (args.batch_size, args.seq_len, args.num_heads, args.v_dim)
    shape_gate = (args.batch_size, args.seq_len, args.num_heads)
    shape_h0 = (args.batch_size, args.num_heads, args.k_dim, args.v_dim)
    shape_b0 = (args.batch_size, args.num_heads, args.v_dim)

    q = torch.randn(shape_qk, device=device, dtype=dtype)
    k = torch.randn(shape_qk, device=device, dtype=dtype)
    v = torch.randn(shape_v, device=device, dtype=dtype)
    g = torch.randn(shape_gate, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(shape_gate, device=device, dtype=dtype))
    lambda_q = torch.randn(shape_gate, device=device, dtype=dtype)
    lambda_k = torch.randn(shape_gate, device=device, dtype=dtype)

    initial_state = None
    if args.with_initial_state:
        initial_state = (
            torch.randn(shape_h0, device=device, dtype=dtype),
            torch.randn(shape_b0, device=device, dtype=dtype),
        )

    ref_inputs = [clone_for_grad(x) for x in (q, k, v, g, beta, lambda_q, lambda_k)]
    test_inputs = [clone_for_grad(x) for x in (q, k, v, g, beta, lambda_q, lambda_k)]

    ref_state = None
    test_state = None
    if initial_state is not None:
        ref_state = tuple(clone_for_grad(x) for x in initial_state)
        test_state = tuple(clone_for_grad(x) for x in initial_state)

    ref_out, ref_final = naive_recurrent_gated_delta_rule_rank1_dc(
        q=ref_inputs[0],
        k=ref_inputs[1],
        v=ref_inputs[2],
        beta=ref_inputs[4],
        g=ref_inputs[3],
        lambda_q=ref_inputs[5],
        lambda_k=ref_inputs[6],
        initial_state=ref_state,
        output_final_state=True,
    )
    test_out, test_final = chunk_gated_delta_rule_rank1_dc(
        q=test_inputs[0],
        k=test_inputs[1],
        v=test_inputs[2],
        beta=test_inputs[4],
        g=test_inputs[3],
        lambda_q=test_inputs[5],
        lambda_k=test_inputs[6],
        initial_state=test_state,
        output_final_state=True,
        chunk_size=args.chunk_size,
    )

    compare_tensor("forward_output", ref_out, test_out, args.rtol, args.atol)
    compare_tensor("final_state_h", ref_final[0], test_final[0], args.rtol, args.atol)
    compare_tensor("final_state_b", ref_final[1], test_final[1], args.rtol, args.atol)

    grad_out = torch.randn_like(ref_out)
    grad_h = torch.randn_like(ref_final[0])
    grad_b = torch.randn_like(ref_final[1])

    ref_grads = torch.autograd.grad(
        outputs=(ref_out, ref_final[0], ref_final[1]),
        inputs=ref_inputs + ([] if ref_state is None else list(ref_state)),
        grad_outputs=(grad_out, grad_h, grad_b),
        retain_graph=False,
        allow_unused=False,
    )
    test_grads = torch.autograd.grad(
        outputs=(test_out, test_final[0], test_final[1]),
        inputs=test_inputs + ([] if test_state is None else list(test_state)),
        grad_outputs=(grad_out, grad_h, grad_b),
        retain_graph=False,
        allow_unused=False,
    )

    grad_names = ["dq", "dk", "dv", "dg", "dbeta", "dlambda_q", "dlambda_k"]
    if ref_state is not None:
        grad_names.extend(["dh0", "db0"])

    for name, ref_grad, test_grad in zip(grad_names, ref_grads, test_grads):
        compare_tensor(name, ref_grad, test_grad, args.rtol, args.atol)

    print("All checks passed.")


if __name__ == "__main__":
    main()
