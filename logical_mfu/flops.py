"""Logical FLOPs per step, split into forward / input-gradient / weight-gradient.

Rules (the whole method is these four lines):

1. Every matmul-bearing module runs **forward** (2 * T * in * out for a linear).
2. It runs **weight-gradient** (another 2 * T * in * out) only if its weight is trainable.
3. It runs **input-gradient** (another 2 * T * in * out) only if *something upstream of it*
   is trainable. Otherwise no gradient needs to flow past it.
4. Attention's score/value products (QK^T and PV) have no weights: forward is
   4 * pairs * q_dim, backward is 2x forward, and it runs only if something upstream is trainable.

"6N" is the special case where every module is trainable, so all three terms run.
LoRA, frozen towers and MoE all fall out of the same rules without special cases.

Nothing in this file looks at which kernels run. Swapping FA2 -> FA3 -> FA4, a
per-expert loop -> grouped GEMM, or full logits -> Cut Cross-Entropy cannot change
the result. Only step time can move MFU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .spec import LMSpec, TrainSpec, VisionSpec


@dataclass
class Terms:
    fwd: float = 0.0
    dgrad: float = 0.0
    wgrad: float = 0.0

    @property
    def total(self) -> float:
        return self.fwd + self.dgrad + self.wgrad

    def __iadd__(self, o: "Terms") -> "Terms":
        self.fwd += o.fwd
        self.dgrad += o.dgrad
        self.wgrad += o.wgrad
        return self


@dataclass
class StepFlops:
    """Per-component breakdown for one optimizer step (or one micro-batch)."""

    parts: Dict[str, Terms] = field(default_factory=dict)
    recompute_fwd: float = 0.0  # extra forward FLOPs from activation checkpointing (HFU only)

    def add(self, name: str, t: Terms) -> None:
        self.parts.setdefault(name, Terms())
        self.parts[name] += t

    @property
    def model_flops(self) -> float:
        """Numerator of MFU."""
        return sum(t.total for t in self.parts.values())

    @property
    def hardware_flops(self) -> float:
        """Numerator of HFU (adds recomputation)."""
        return self.model_flops + self.recompute_fwd

    def __iadd__(self, o: "StepFlops") -> "StepFlops":
        for k, v in o.parts.items():
            self.add(k, v)
        self.recompute_fwd += o.recompute_fwd
        return self

    def table(self) -> str:
        rows = [f"{'component':<22}{'fwd':>12}{'dgrad':>12}{'wgrad':>12}{'total':>12}"]
        for k, t in self.parts.items():
            rows.append(f"{k:<22}{t.fwd/1e12:>11.2f}T{t.dgrad/1e12:>11.2f}T{t.wgrad/1e12:>11.2f}T{t.total/1e12:>11.2f}T")
        rows.append(f"{'MODEL (MFU)':<22}{'':>36}{self.model_flops/1e12:>11.2f}T")
        rows.append(f"{'HARDWARE (HFU)':<22}{'':>36}{self.hardware_flops/1e12:>11.2f}T")
        return "\n".join(rows)


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------

class _Walker:
    """Walks modules in graph order and tracks whether anything upstream is trainable."""

    def __init__(self, out: StepFlops):
        self.out = out
        self.upstream_trainable = False

    def linear_group(self, name: str, tokens: float, shapes: Sequence[Tuple[int, int]],
                     trainable: bool, lora_rank: int = 0, lora_mask: Optional[Sequence[bool]] = None) -> Terms:
        """A set of *parallel* linears that read the same input (e.g. q/k/v, or gate/up).

        Parallel matters: a LoRA adapter on q is not upstream of k.
        """
        flow_in = self.upstream_trainable
        t = Terms()
        any_trainable = trainable
        for j, (fan_in, fan_out) in enumerate(shapes):
            mm = 2.0 * tokens * fan_in * fan_out
            t.fwd += mm
            if flow_in:
                t.dgrad += mm
            if trainable:
                t.wgrad += mm
            if lora_rank and (lora_mask is None or lora_mask[j]):
                any_trainable = True
                a = 2.0 * tokens * fan_in * lora_rank   # x @ A
                b = 2.0 * tokens * lora_rank * fan_out  # (xA) @ B
                t.fwd += a + b
                t.wgrad += a + b                       # both adapters are trainable
                t.dgrad += b                           # grad must reach A, so B's dgrad always runs
                if flow_in:
                    t.dgrad += a                       # A's dgrad only if something upstream wants it
        self.out.add(name, t)
        if any_trainable:
            self.upstream_trainable = True
        return t

    def attention_core(self, name: str, pairs: float, q_dim: int) -> Terms:
        t = Terms()
        t.fwd = 4.0 * pairs * q_dim          # QK^T and PV, 2 FLOPs per MAC
        if self.upstream_trainable:
            t.dgrad = 2.0 * t.fwd            # dQ, dK (from dS) and dV, dP
        self.out.add(name, t)
        return t


def attention_pairs(seg_lens: Iterable[int], causal: bool, causal_count: str = "exact") -> float:
    """Number of (query, key) pairs actually needed.

    Packing matters: three 2K documents in one 6K row cost 3 * 2K^2, not 6K^2.
    """
    total = 0.0
    for L in seg_lens:
        if L <= 0:
            continue
        if causal and causal_count == "exact":
            total += L * (L + 1) / 2.0
        elif causal and causal_count == "half":
            total += L * L / 2.0
        else:  # non-causal, or causal counted as full square (Megatron convention)
            total += float(L) * L
    return total


def _windows(n: int, w: int) -> List[int]:
    return [min(w, n - s) for s in range(0, n, w)]


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def step_flops(
    lm: LMSpec,
    train: TrainSpec,
    seg_lens: Sequence[int],
    image_patches: Sequence[int] = (),
    vision: Optional[VisionSpec] = None,
    *,
    lm_head_tokens: Optional[int] = None,
    causal_count: str = "exact",
    recompute: str = "none",
) -> StepFlops:
    """Logical FLOPs for one micro-batch.

    Args:
        seg_lens: length of every *document* the LM sees, after packing (real tokens only,
            padding excluded). For an unpacked batch, it is one entry per sample.
        image_patches: number of ViT patches per image (t * h * w before merging).
        lm_head_tokens: positions that go through the LM head. Default: every token.
            (Pick one convention and keep it fixed. Cut Cross-Entropy must not change it.)
        causal_count: "exact" (L(L+1)/2), "half" (L^2/2) or "full" (L^2).
        recompute: "none" | "selective" (attention core) | "full" (whole LM layer). HFU only.
    """
    out = StepFlops()
    w = _Walker(out)
    T = float(sum(seg_lens))

    # -------- vision tower (upstream of everything) --------
    if vision is not None and image_patches:
        v = vision
        P = float(sum(image_patches))
        vt = train.vision == "full"
        w.linear_group("vision.patch_embed", P, [(v.patch_embed_in, v.hidden_size)], vt)
        for i in range(v.num_layers):
            w.linear_group("vision.qkv", P, [(v.hidden_size, 3 * v.hidden_size)], vt)
            if v.window_tokens and i not in set(v.full_attn_layers):
                segs = [x for n in image_patches for x in _windows(n, v.window_tokens)]
            else:
                segs = list(image_patches)
            w.attention_core("vision.attn_core", attention_pairs(segs, causal=False), v.hidden_size)
            w.linear_group("vision.o", P, [(v.hidden_size, v.hidden_size)], vt)
            if v.gated_mlp:
                w.linear_group("vision.mlp_in", P, [(v.hidden_size, v.intermediate_size)] * 2, vt)
            else:
                w.linear_group("vision.mlp_in", P, [(v.hidden_size, v.intermediate_size)], vt)
            w.linear_group("vision.mlp_out", P, [(v.intermediate_size, v.hidden_size)], vt)
        m2 = v.spatial_merge_size ** 2
        merged = P / m2
        mh = v.hidden_size * m2
        mt = train.merger == "full"
        w.linear_group("vision.merger", merged, [(mh, mh)], mt)
        w.linear_group("vision.merger", merged, [(mh, v.out_hidden_size)], mt)

    # -------- language model --------
    full = train.llm == "full"
    lora_r = train.lora_rank if train.llm == "lora" else 0
    tg = set(train.lora_targets)
    emb_trainable = train.embeddings_trainable if train.embeddings_trainable is not None else full
    if emb_trainable:
        w.upstream_trainable = True  # lookup is 0 FLOPs, but layer-0 dgrad must now run

    h = lm.hidden_size
    pairs = attention_pairs(seg_lens, causal=True, causal_count=causal_count)
    for i in range(lm.num_layers):
        snap = {k: Terms(t.fwd, t.dgrad, t.wgrad) for k, t in out.parts.items()}
        w.linear_group("llm.qkv", T, [(h, lm.q_dim), (h, lm.kv_dim), (h, lm.kv_dim)], full,
                       lora_r, [x in tg for x in ("q", "k", "v")])
        w.attention_core("llm.attn_core", pairs, lm.q_dim)
        w.linear_group("llm.o", T, [(lm.q_dim, h)], full, lora_r, ["o" in tg])

        def mlp(name: str, tokens: float, inter: int):
            if lm.gated_mlp:
                w.linear_group(name + ".gate_up", tokens, [(h, inter), (h, inter)], full,
                               lora_r, ["gate" in tg, "up" in tg])
            else:
                w.linear_group(name + ".up", tokens, [(h, inter)], full, lora_r, ["up" in tg])
            # down reads the gate/up output, so gate/up are upstream of it
            w.linear_group(name + ".down", tokens, [(inter, h)], full, lora_r, ["down" in tg])

        if lm.is_moe_layer(i):
            w.linear_group("llm.router", T, [(h, lm.num_experts)], full)
            # Dropless routing: exactly T * k token-expert assignments, however they are executed.
            saved = w.upstream_trainable
            mlp("llm.experts", T * lm.experts_per_token, lm.moe_intermediate_size)
            after_experts = w.upstream_trainable
            if lm.shared_expert_intermediate_size:
                w.upstream_trainable = saved  # shared expert runs in parallel with routed experts
                mlp("llm.shared_expert", T, lm.shared_expert_intermediate_size)
                w.upstream_trainable = w.upstream_trainable or after_experts
        else:
            mlp("llm.mlp", T, lm.intermediate_size)

        if recompute != "none":
            layer_fwd = sum(t.fwd - snap.get(k, Terms()).fwd for k, t in out.parts.items())
            layer_bwd = sum((t.dgrad + t.wgrad) - (snap.get(k, Terms()).dgrad + snap.get(k, Terms()).wgrad)
                            for k, t in out.parts.items())
            if layer_bwd > 0:  # only layers that actually run backward recompute
                if recompute == "full":
                    out.recompute_fwd += layer_fwd
                elif recompute == "selective":
                    out.recompute_fwd += 4.0 * pairs * lm.q_dim
                else:
                    raise ValueError(recompute)

    head_trainable = full if train.lm_head is None else train.lm_head == "full"
    n_head = float(lm_head_tokens) if lm_head_tokens is not None else T
    w.linear_group("llm.lm_head", n_head, [(h, lm.vocab_size)], head_trainable)
    return out


def mfu(model_flops: float, step_time_s: float, n_gpus: int, peak_flops_per_gpu: float) -> float:
    """MFU = model FLOPs per step / (step time * #GPUs * dense peak for the training precision)."""
    return model_flops / (step_time_s * n_gpus * peak_flops_per_gpu)
