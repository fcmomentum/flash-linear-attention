import torch

from fla.models import PhasedDeltaNetConfig
from fla.utils import assert_close, device

from .test_modeling_utils import create_model_and_config


def test_phased_deltanet_forward_varlen_and_generation():
    model, config = create_model_and_config(
        PhasedDeltaNetConfig,
        L=2,
        H=2,
        D=16,
        dtype=torch.float32,
        use_l2warp=False,
    )
    model.eval()

    batch_size = 2
    seq_len = 12
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), device=device)

    full = model(input_ids=input_ids, output_hidden_states=True)
    assert full.hidden_states[-1].shape == (batch_size, seq_len, config.hidden_size)

    cu_seqlens = torch.arange(0, batch_size * seq_len + 1, seq_len, dtype=torch.int32, device=device)
    ragged = model(input_ids=input_ids.view(1, batch_size * seq_len), output_hidden_states=True, cu_seqlens=cu_seqlens)
    assert_close("varlen_hidden", full.hidden_states[-1].view(1, batch_size * seq_len, -1), ragged.hidden_states[-1], 1e-4)

    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    attention_mask[0, :3] = False
    attention_mask[1, :1] = False

    ref_logits = torch.cat([
        model(input_ids=input_ids[i:i + 1, start:], use_cache=False).logits
        for i, start in enumerate([3, 1])
    ], dim=1)

    out = model(
        input_ids=input_ids[:, :4],
        attention_mask=attention_mask[:, :4],
        use_cache=True,
        past_key_values=None,
    )
    logits = [out.logits]
    past_key_values = out.past_key_values
    for step in range(4, seq_len):
        out = model(
            input_ids=input_ids[:, step:step + 1],
            attention_mask=attention_mask[:, :step + 1],
            use_cache=True,
            past_key_values=past_key_values,
        )
        logits.append(out.logits)
        past_key_values = out.past_key_values
    gen_logits = torch.cat(logits, dim=1)
    gen_logits = torch.cat([gen_logits[i:i + 1, start:] for i, start in enumerate([3, 1])], dim=1)
    assert_close("cached_logits", ref_logits, gen_logits, 1e-4)
