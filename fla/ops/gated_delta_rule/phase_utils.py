import torch
import torch.nn.functional as F
from einops import rearrange


def apply_nonphase_activation(x: torch.Tensor, *, num_phase_channels: int, head_v_dim: int, num_v_heads: int) -> torch.Tensor:
    if num_phase_channels == 0:
        return F.silu(x)
    if num_phase_channels == head_v_dim:
        return x
    x = rearrange(x, '... (h d) -> ... h d', h=num_v_heads, d=head_v_dim)
    x_nonphase = x[..., num_phase_channels:]
    x = torch.cat((x[..., :num_phase_channels], F.silu(x_nonphase)), dim=-1)
    return rearrange(x, '... h d -> ... (h d)')


def restrict_phase_channels(x: torch.Tensor, *, num_phase_channels: int) -> torch.Tensor:
    if num_phase_channels == 0:
        return x
    if num_phase_channels >= x.shape[-1]:
        return x
    result = x.clone()
    result[..., num_phase_channels:] = 0
    return result


def rotate_phase_channels(x: torch.Tensor, phase: torch.Tensor | None, *, num_phase_channels: int) -> torch.Tensor:
    if num_phase_channels == 0 or phase is None:
        return x
    # shape assumptions: phase[...] broadcasts to x[..., :num_phase_channels]
    x_phase = x[..., :num_phase_channels]
    phase_pairs = phase[..., :num_phase_channels:2]
    while phase_pairs.ndim < x_phase.ndim:
        phase_pairs = phase_pairs.unsqueeze(-2)
    cos = phase_pairs.float().cos().to(x.dtype).repeat_interleave(2, dim=-1)
    sin = phase_pairs.float().sin().to(x.dtype).repeat_interleave(2, dim=-1)
    x_even = x_phase[..., ::2]
    x_odd = x_phase[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    x_phase = x_phase * cos + x_rot * sin
    if num_phase_channels == x.shape[-1]:
        return x_phase
    return torch.cat((x_phase, x[..., num_phase_channels:]), dim=-1)


def phase_prefix_sum(
    phase: torch.Tensor,
    *,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is None:
        phase_cumsum = phase.cumsum(dim=1)
        phase_prefix = phase_cumsum - phase
        phase_total = phase_cumsum[:, -1]
        return phase_prefix, phase_total

    if phase.shape[0] != 1:
        raise ValueError("phase must have batch size 1 when cu_seqlens is provided.")

    phase_prefix = torch.zeros_like(phase)
    num_seqs = cu_seqlens.numel() - 1
    phase_total = phase.new_zeros((num_seqs, phase.shape[2], phase.shape[3]))
    for i in range(num_seqs):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        if end <= start:
            continue
        seq_phase = phase[:, start:end]
        seq_cumsum = seq_phase.cumsum(dim=1)
        phase_prefix[:, start:end] = seq_cumsum - seq_phase
        phase_total[i] = seq_cumsum[0, -1]
    return phase_prefix, phase_total


def build_fixed_rope_phase(
    *,
    inv_freq: torch.Tensor,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_v_dim: int,
    num_phase_channels: int,
    device: torch.device,
    dtype: torch.dtype,
    cu_seqlens: torch.LongTensor | None = None,
    offset: int = 0,
) -> torch.Tensor | None:
    if num_phase_channels == 0:
        return None

    pair_dim = num_phase_channels // 2
    inv_freq = inv_freq[:pair_dim].to(device=device)

    if cu_seqlens is None:
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=inv_freq.dtype)
    else:
        if batch_size != 1:
            raise ValueError("cu_seqlens with fixed rope phase requires batch_size == 1.")
        positions = torch.empty(seq_len, device=device, dtype=inv_freq.dtype)
        num_sequences = cu_seqlens.numel() - 1
        for i in range(num_sequences):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            if end <= start:
                continue
            positions[start:end] = torch.arange(end - start, device=device, dtype=inv_freq.dtype)

    freqs = torch.outer(positions, inv_freq).to(dtype)
    phase = torch.zeros(batch_size, seq_len, num_heads, head_v_dim, device=device, dtype=dtype)
    phase[..., :num_phase_channels:2] = freqs.unsqueeze(0).unsqueeze(2)
    return phase
