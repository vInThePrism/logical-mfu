import math

import pytest

from logical_mfu import (LMSpec, TrainSpec, VisionSpec, attention_pairs, batch_stats, mfu,
                         peak_flops, step_flops)

# Public configs (config.json on the Hugging Face Hub).
QWEN3_8B = dict(hidden_size=4096, num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8,
                head_dim=128, intermediate_size=12288, vocab_size=151936, tie_word_embeddings=False)
QWEN3_30B_A3B = dict(hidden_size=2048, num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=4,
                     head_dim=128, intermediate_size=6144, moe_intermediate_size=768, num_experts=128,
                     num_experts_per_tok=8, decoder_sparse_step=1, vocab_size=151936,
                     tie_word_embeddings=False)


def no_attn(lm, train, T, **kw):
    """FLOPs with the attention-core term removed (1-token documents => ~0 pairs)."""
    f = step_flops(lm, train, [1] * T, **kw)
    return f.model_flops - f.parts["llm.attn_core"].total


def test_param_counts_match_published_sizes():
    dense = LMSpec.from_hf_config(QWEN3_8B)
    assert dense.num_params() / 1e9 == pytest.approx(8.19, rel=0.01)
    moe = LMSpec.from_hf_config(QWEN3_30B_A3B)
    assert moe.num_params() / 1e9 == pytest.approx(30.5, rel=0.01)
    assert moe.num_params(active_only=True) / 1e9 == pytest.approx(3.3, rel=0.02)


def test_full_finetune_is_6N_without_attention():
    lm = LMSpec.from_hf_config(QWEN3_8B)
    T = 4096
    n = lm.num_params(include_embeddings=False) + lm.vocab_size * lm.hidden_size  # + lm_head
    assert no_attn(lm, TrainSpec(llm="full"), T) == pytest.approx(6 * n * T, rel=1e-9)


def test_moe_counts_active_params():
    lm = LMSpec.from_hf_config(QWEN3_30B_A3B)
    T = 4096
    active = lm.num_params(active_only=True, include_embeddings=False) + lm.vocab_size * lm.hidden_size
    assert no_attn(lm, TrainSpec(llm="full"), T) == pytest.approx(6 * active * T, rel=1e-9)


def test_frozen_llm_is_forward_only():
    lm = LMSpec.from_hf_config(QWEN3_8B)
    f = step_flops(lm, TrainSpec(llm="frozen"), [2048])
    assert all(t.dgrad == 0 and t.wgrad == 0 for t in f.parts.values())


def test_lora_base_weights_cost_4N_except_first_group():
    """Frozen base linear: forward + input grad (4N). No weight grad.
    Layer 0's q/k/v have nothing trainable upstream, so they are forward only."""
    lm = LMSpec.from_hf_config(QWEN3_8B)
    T = 1000
    f = step_flops(lm, TrainSpec(llm="lora", lora_rank=16), [1] * T)
    base_qkv_one_layer = 2 * T * lm.hidden_size * (lm.q_dim + 2 * lm.kv_dim)
    lora_qkv_one_layer = 2 * T * 16 * (3 * lm.hidden_size + lm.q_dim + 2 * lm.kv_dim)
    qkv = f.parts["llm.qkv"]
    assert qkv.fwd == pytest.approx(lm.num_layers * (base_qkv_one_layer + lora_qkv_one_layer))
    # base dgrad runs in layers 1..L-1 only
    lora_b_dgrad = 2 * T * 16 * (lm.q_dim + 2 * lm.kv_dim)
    lora_a_dgrad = 2 * T * 16 * 3 * lm.hidden_size
    expected_dgrad = (lm.num_layers - 1) * (base_qkv_one_layer + lora_a_dgrad) + lm.num_layers * lora_b_dgrad
    assert qkv.dgrad == pytest.approx(expected_dgrad)
    assert f.parts["llm.lm_head"].wgrad == 0  # frozen head
    assert f.parts["llm.lm_head"].dgrad > 0  # but gradient must flow back through it


def test_packing_counts_documents_not_rows():
    one_row = attention_pairs([6144], causal=True, causal_count="full")
    packed = attention_pairs([2048, 2048, 2048], causal=True, causal_count="full")
    assert one_row / packed == pytest.approx(3.0)


def test_numerator_is_invariant_to_how_the_batch_is_laid_out():
    """Same documents, padded vs packed: identical logical FLOPs."""
    lm = LMSpec.from_hf_config(QWEN3_8B)
    padded = batch_stats({"attention_mask": [[1] * 700 + [0] * 324, [1] * 1024]})
    packed = batch_stats({"position_ids": [list(range(700)) + list(range(1024))]})
    assert sorted(padded.seg_lens) == sorted(packed.seg_lens)
    a = step_flops(lm, TrainSpec(), padded.seg_lens).model_flops
    b = step_flops(lm, TrainSpec(), packed.seg_lens).model_flops
    assert a == b


def test_neat_packing_mask_and_cu_seqlens():
    assert batch_stats({"attention_mask": [[1, 1, 2, 2, 2, 3, 0, 0]]}).seg_lens == [2, 3, 1]
    assert batch_stats({"cu_seqlens": [0, 5, 9, 9, 12]}).seg_lens == [5, 4, 3]


def test_frozen_vision_tower_is_forward_only_and_merger_trains():
    lm = LMSpec.from_hf_config(QWEN3_8B)
    vit = VisionSpec(hidden_size=1280, num_layers=32, num_heads=16, intermediate_size=3420,
                     out_hidden_size=4096, patch_embed_in=3 * 2 * 14 * 14, gated_mlp=True,
                     window_tokens=64, full_attn_layers=(7, 15, 23, 31))
    st = batch_stats({"input_ids": [[0] * 2048], "image_grid_thw": [[1, 32, 32], [1, 16, 48]]})
    assert st.image_patches == [1024, 768]
    f = step_flops(lm, TrainSpec(vision="frozen", merger="full"), st.seg_lens, st.image_patches, vit)
    for k, t in f.parts.items():
        if k.startswith("vision.") and k != "vision.merger":
            assert t.dgrad == 0 and t.wgrad == 0, k
    # Merger = Linear(5120->5120) -> Linear(5120->4096). Both train. The first one has nothing
    # trainable upstream (no dgrad); the second one does (its input comes from the first).
    merged = sum(st.image_patches) / 4
    second_fwd = 2 * merged * 5120 * 4096
    m = f.parts["vision.merger"]
    assert m.wgrad == m.fwd and m.dgrad == pytest.approx(second_fwd)


def test_window_attention_is_cheaper_than_full():
    vit_full = VisionSpec(1280, 32, 16, 3420, 4096, 1176)
    vit_win = VisionSpec(1280, 32, 16, 3420, 4096, 1176, window_tokens=64, full_attn_layers=(7, 15, 23, 31))
    lm = LMSpec.from_hf_config(QWEN3_8B)
    a = step_flops(lm, TrainSpec(), [512], [4096], vit_full).parts["vision.attn_core"].fwd
    b = step_flops(lm, TrainSpec(), [512], [4096], vit_win).parts["vision.attn_core"].fwd
    assert b < a / 4


def test_hfu_full_recompute_adds_one_forward():
    lm = LMSpec.from_hf_config(QWEN3_8B)
    f0 = step_flops(lm, TrainSpec(), [4096])
    f1 = step_flops(lm, TrainSpec(), [4096], recompute="full")
    assert f0.model_flops == f1.model_flops  # MFU numerator unchanged
    layers_fwd = sum(t.fwd for k, t in f1.parts.items() if k != "llm.lm_head")
    assert f1.hardware_flops - f1.model_flops == pytest.approx(layers_fwd)


def test_mfu_ratio_equals_throughput_ratio():
    """The sanity check: fixed numerator => MFU moves exactly with step time."""
    lm = LMSpec.from_hf_config(QWEN3_30B_A3B)
    work = step_flops(lm, TrainSpec(), [8192] * 4).model_flops
    p = peak_flops("H800")
    before, after = mfu(work, 14.4, 32, p), mfu(work, 7.2, 32, p)
    assert after / before == pytest.approx(14.4 / 7.2)


def test_device_lookup():
    assert peak_flops("NVIDIA H800") == pytest.approx(989.4e12)
    assert peak_flops("H100", "fp8") == pytest.approx(2 * 989.4e12, rel=1e-3)
    with pytest.raises(KeyError):
        peak_flops("H100", "fp4")
