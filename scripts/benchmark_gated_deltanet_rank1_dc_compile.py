#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.models.utils import Cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark torch.compile on a single GatedDeltaNet layer with rank-1 DC removal.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--expand-v", type=float, default=2.0)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["chunk", "fused_recurrent"], default="chunk")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--timed-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true", help="Run inference-only benchmark.")
    parser.add_argument("--use-short-conv", action="store_true", default=False)
    parser.add_argument("--use-gate", action="store_true", default=False)
    parser.add_argument("--allow-neg-eigval", action="store_true", default=False)
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


class SingleLayerRank1DCModel(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        expand_v: float,
        mode: str,
        use_short_conv: bool,
        use_gate: bool,
        allow_neg_eigval: bool,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.layer = GatedDeltaNet(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            expand_v=expand_v,
            mode=mode,
            use_gate=use_gate,
            use_short_conv=use_short_conv,
            allow_neg_eigval=allow_neg_eigval,
            use_rank1_dc_removal=True,
            layer_idx=0,
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states, _, _ = self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        return self.out_proj(hidden_states)


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_run(fn, device: torch.device) -> tuple[torch.Tensor, float]:
    sync_if_needed(device)
    start = time.perf_counter()
    result = fn()
    sync_if_needed(device)
    return result, time.perf_counter() - start


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)
    training = not args.eval

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.hidden_size != args.num_heads * args.head_dim:
        raise ValueError("hidden_size must equal num_heads * head_dim for this single-layer benchmark.")

    model = SingleLayerRank1DCModel(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        expand_v=args.expand_v,
        mode=args.mode,
        use_short_conv=args.use_short_conv,
        use_gate=args.use_gate,
        allow_neg_eigval=args.allow_neg_eigval,
    ).to(device=device, dtype=dtype)
    model.train(training)

    compiled_model = torch.compile(
        model,
        backend=args.compile_backend,
        fullgraph=args.fullgraph,
        dynamic=args.dynamic,
    )

    hidden_states = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device, dtype=dtype)
    attention_mask = torch.ones(args.batch_size, args.seq_len, device=device, dtype=torch.bool)
    attention_mask[:, : args.seq_len // 8] = False

    def run_eval() -> torch.Tensor:
        return compiled_model(hidden_states, attention_mask=attention_mask, use_cache=False)

    def run_train() -> torch.Tensor:
        compiled_model.zero_grad(set_to_none=True)
        out = compiled_model(hidden_states, attention_mask=attention_mask, use_cache=False)
        loss = out.float().square().mean()
        loss.backward()
        return loss.detach()

    run_step = run_eval if args.eval else run_train

    first_result, first_time = timed_run(run_step, device)
    print(f"first_run_time_s={first_time:.6f}")
    if torch.is_tensor(first_result):
        scalar = first_result.float().mean().item() if first_result.numel() > 1 else first_result.item()
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
