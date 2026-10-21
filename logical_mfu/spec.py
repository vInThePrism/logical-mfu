"""Model and training specs: the *static* half of the logical-FLOPs numerator.

Everything here is read once from the model config. Nothing here depends on
which kernels run (FlashAttention vs SDPA, fused LoRA vs unfused, grouped GEMM
vs a per-expert loop, Cut Cross-Entropy vs full logits). That is the point:
the execution path must not be able to change the numerator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class LMSpec:
    """Decoder-only language model (dense, MoE, or mixed)."""

    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    # Dense MLP (used by dense layers, or by every layer when num_experts == 0).
    intermediate_size: int = 0
    gated_mlp: bool = True  # SwiGLU / GeGLU: gate, up, down. False: up, down.
    # MoE
    num_experts: int = 0
    experts_per_token: int = 0  # top-k
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0  # 0 = no shared expert
    moe_layer_freq: int = 1  # every k-th layer is MoE
    first_k_dense: int = 0  # first k layers use the dense MLP
    tie_word_embeddings: bool = False
    qkv_bias: bool = False  # bias FLOPs are ignored (they are O(T*d), not O(T*d^2))

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    def is_moe_layer(self, i: int) -> bool:
        if self.num_experts == 0:
            return False
        if i < self.first_k_dense:
            return False
        return (i - self.first_k_dense) % self.moe_layer_freq == 0

    # --- parameter counts (matmul weights only; norms and biases ignored) ---
    def attn_params(self) -> int:
        h = self.hidden_size
        return h * self.q_dim + 2 * h * self.kv_dim + self.q_dim * h

    def mlp_params(self, inter: int) -> int:
        return (3 if self.gated_mlp else 2) * self.hidden_size * inter

    def layer_params(self, i: int, active_only: bool) -> int:
        p = self.attn_params()
        if self.is_moe_layer(i):
            n = self.experts_per_token if active_only else self.num_experts
            p += self.hidden_size * self.num_experts  # router
            p += n * self.mlp_params(self.moe_intermediate_size)
            if self.shared_expert_intermediate_size:
                p += self.mlp_params(self.shared_expert_intermediate_size)
        else:
            p += self.mlp_params(self.intermediate_size)
        return p

    def num_params(self, active_only: bool = False, include_embeddings: bool = True) -> int:
        p = sum(self.layer_params(i, active_only) for i in range(self.num_layers))
        if include_embeddings:
            p += self.vocab_size * self.hidden_size  # input embedding
            if not self.tie_word_embeddings:
                p += self.vocab_size * self.hidden_size  # lm_head
        return p

    @classmethod
    def from_hf_config(cls, cfg: dict) -> "LMSpec":
        """Build from a Hugging Face ``config.json`` dict (Llama/Qwen2/Qwen3/Qwen3-MoE/Mixtral-style).

        For multimodal configs, pass ``cfg["text_config"]``.
        """
        g = cfg.get
        hidden = g("hidden_size")
        heads = g("num_attention_heads")
        head_dim = g("head_dim") or hidden // heads
        num_experts = g("num_experts") or g("num_local_experts") or g("n_routed_experts") or 0
        sparse_step = g("decoder_sparse_step") or g("moe_layer_freq") or 1
        mlp_only = g("mlp_only_layers") or []
        if mlp_only:
            raise NotImplementedError("mlp_only_layers: express it with first_k_dense / moe_layer_freq")
        shared = g("shared_expert_intermediate_size") or 0
        if not shared and g("n_shared_experts"):
            shared = g("n_shared_experts") * g("moe_intermediate_size")
        return cls(
            hidden_size=hidden,
            num_layers=g("num_hidden_layers"),
            num_heads=heads,
            num_kv_heads=g("num_key_value_heads") or heads,
            head_dim=head_dim,
            vocab_size=g("vocab_size"),
            intermediate_size=g("intermediate_size") or 0,
            num_experts=num_experts,
            experts_per_token=g("num_experts_per_tok") or 0,
            moe_intermediate_size=g("moe_intermediate_size") or (g("intermediate_size") if num_experts else 0) or 0,
            shared_expert_intermediate_size=shared,
            moe_layer_freq=sparse_step,
            first_k_dense=g("first_k_dense_replace") or 0,
            tie_word_embeddings=bool(g("tie_word_embeddings", False)),
        )


@dataclass(frozen=True)
class VisionSpec:
    """ViT-style vision tower with a patch merger that projects into the LM hidden size."""

    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    out_hidden_size: int  # LM hidden size after the merger
    patch_embed_in: int  # in_channels * temporal_patch * patch * patch
    spatial_merge_size: int = 2  # merger groups merge^2 patches into one LM token
    gated_mlp: bool = False
    # Windowed attention (Qwen2.5-VL style). None = full attention in every layer.
    window_tokens: Optional[int] = None
    full_attn_layers: Sequence[int] = field(default_factory=tuple)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass(frozen=True)
class TrainSpec:
    """Which weights receive gradients. This decides which of the three terms run per module."""

    llm: str = "full"  # "full" | "lora" | "frozen"
    lora_rank: int = 0
    lora_targets: Sequence[str] = ("q", "k", "v", "o", "gate", "up", "down")
    vision: str = "frozen"  # "full" | "frozen" (only used when a VisionSpec is given)
    merger: str = "full"  # "full" | "frozen"
    lm_head: Optional[str] = None  # default: follows llm ("lora" -> frozen)
    embeddings_trainable: Optional[bool] = None  # default: llm == "full"

    def __post_init__(self):
        for name, v in (("llm", self.llm), ("vision", self.vision), ("merger", self.merger)):
            if v not in ("full", "lora", "frozen"):
                raise ValueError(f"{name}={v!r}")
        if self.llm == "lora" and self.lora_rank <= 0:
            raise ValueError("lora_rank must be > 0 when llm='lora'")
