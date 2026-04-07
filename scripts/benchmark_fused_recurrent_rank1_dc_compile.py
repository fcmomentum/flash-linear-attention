#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_rank1_dc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark torch.compile on fused_recurrent_gated_delta_rule_rank1_dc.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--k-dim", type=int, default=64)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--timed-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-initial-state", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Run inference-only benchmark.")
    parser.add_argument("--use-triton-backward", action="store_true")
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


class FusedRecurrentRank1DCModule(nn.Module):

    def __init__(self, with_initial_state: bool) -> None:
        super().__init__()
        self.with_initial_state = with_initial_state

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        lambda_q: torch.Tensor,
        lambda_k: torch.Tensor,
        state: torch.Tensor | None = None,
        bias_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial_state = None if state is None else (state, bias_state)
        o, final_state = fused_recurrent_gated_delta_rule_rank1_dc(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            lambda_q=lambda_q,
            lambda_k=lambda_k,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
            use_exp2=False,
            transpose_state_layout=False,
        )
        return o, final_state[0], final_state[1]


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_run(fn, device: torch.device):
    sync_if_needed(device)
    start = time.perf_counter()
    result = fn()
    sync_if_needed(device)
    return result, time.perf_counter() - start


def main() -> None:
    args = parse_args()
    if args.use_triton_backward:
        os.environ["FLA_USE_TRITON_RANK1_DC_BWD"] = "1"

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)
    training = not args.eval

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    module = FusedRecurrentRank1DCModule(with_initial_state=args.with_initial_state).to(device=device)
    module.train(training)

    compiled_module = torch.compile(
        module,
        backend=args.compile_backend,
        fullgraph=args.fullgraph,
        dynamic=args.dynamic,
    )

    q = 0.25 * torch.randn(args.batch_size, args.seq_len, args.num_heads, args.k_dim, device=device, dtype=dtype)
    k = 0.25 * torch.randn(args.batch_size, args.seq_len, args.num_heads, args.k_dim, device=device, dtype=dtype)
    v = 0.25 * torch.randn(args.batch_size, args.seq_len, args.num_heads, args.v_dim, device=device, dtype=dtype)
    g = torch.nn.functional.logsigmoid(torch.randn(args.batch_size, args.seq_len, args.num_heads, device=device, dtype=dtype))
    beta = 0.25 + 0.5 * torch.sigmoid(torch.randn(args.batch_size, args.seq_len, args.num_heads, device=device, dtype=dtype))
    lambda_q = 0.1 * torch.randn(args.batch_size, args.seq_len, args.num_heads, device=device, dtype=dtype)
    lambda_k = 0.1 * torch.randn(args.batch_size, args.seq_len, args.num_heads, device=device, dtype=dtype)

    for tensor in (q, k, v, g, beta, lambda_q, lambda_k):
        tensor.requires_grad_(training)

    state = None
    bias_state = None
    if args.with_initial_state:
        state = 0.05 * torch.randn(args.batch_size, args.num_heads, args.k_dim, args.v_dim, device=device, dtype=torch.float32)
        bias_state = 0.05 * torch.randn(args.batch_size, args.num_heads, args.v_dim, device=device, dtype=torch.float32)
        state.requires_grad_(training)
        bias_state.requires_grad_(training)

    def run_eval():
        return compiled_module(q, k, v, g, beta, lambda_q, lambda_k, state, bias_state)

    def run_train():
        compiled_module.zero_grad(set_to_none=True)
        outputs = compiled_module(q, k, v, g, beta, lambda_q, lambda_k, state, bias_state)
        loss = sum(t.float().square().mean() for t in outputs)
        loss.backward()
        return loss.detach()

    run_step = run_eval if args.eval else run_train

    first_result, first_time = timed_run(run_step, device)
    print(f"first_run_time_s={first_time:.6f}")
    if torch.is_tensor(first_result):
        scalar = first_result.float().mean().item() if first_result.numel() > 1 else first_result.item()
        print(f"first_run_value={scalar:.6f}")
    elif isinstance(first_result, tuple):
        scalar = sum(t.float().mean().item() for t in first_result)
        print(f"first_run_value={scalar:.6f}")

    warmup_times = []
    for _ in range(args.warmup_iters):
        _, elapsed = timed_run(run_step, device)
        warmup_times.append(elapsed)
    if warmup_times:
        print(f"warmup_time_s_avg={sum(warmup_times) / len(warmup_times):.6f}")

    timed = []
    for _ in range(args.timed_iters):
        _, elapsed = timed_run(run_step, device)
        timed.append(elapsed)
    if timed:
        print(f"timed_time_s_avg={sum(timed) / len(timed):.6f}")
        print(f"timed_time_s_min={min(timed):.6f}")
        print(f"timed_time_s_max={max(timed):.6f}")


if __name__ == "__main__":
    main()
