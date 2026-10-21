"""Runtime half of the numerator: what each micro-batch actually contains.

Read from the batch, not from the config:
  * real tokens (padding excluded),
  * document lengths after packing (attention cost is sum(L_i^2), not (sum L_i)^2),
  * image patch counts (vision cost scales with patches, not with images).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional


@dataclass
class BatchStats:
    seg_lens: List[int] = field(default_factory=list)
    image_patches: List[int] = field(default_factory=list)
    loss_tokens: Optional[int] = None

    @property
    def tokens(self) -> int:
        return sum(self.seg_lens)


def _tolist(x: Any) -> list:
    return x.tolist() if hasattr(x, "tolist") else list(x)


def _segments_from_position_ids(pos_row: list, valid_row: Optional[list]) -> List[int]:
    """Packed rows restart position_ids at 0 for every document."""
    segs, cur = [], 0
    for j, p in enumerate(pos_row):
        if valid_row is not None and not valid_row[j]:
            continue
        if p == 0 and cur:
            segs.append(cur)
            cur = 0
        cur += 1
    if cur:
        segs.append(cur)
    return segs


def batch_stats(batch: Mapping[str, Any]) -> BatchStats:
    """Infer stats from a Hugging Face-style batch.

    Priority: explicit ``cu_seqlens`` > document-id masks > 2-D ``position_ids``
    resets > a binary mask or input row length. Position resets may be used only
    when they represent attention-isolated documents. Multimodal 3-D position
    IDs are ambiguous: provide explicit segment boundaries for those batches.
    Image patches come from ``image_grid_thw``.
    """
    st = BatchStats()
    mask = batch.get("attention_mask")
    mask_rows = _tolist(mask) if mask is not None and getattr(mask, "dim", lambda: 2)() == 2 else None

    if batch.get("cu_seqlens") is not None:
        cu = _tolist(batch["cu_seqlens"])
        st.seg_lens = [b - a for a, b in zip(cu[:-1], cu[1:]) if b > a]
    elif mask_rows is not None and any(v > 1 for row in mask_rows for v in row):
        for row in mask_rows:
            ids = [v for v in row if v]
            st.seg_lens.extend(ids.count(d) for d in sorted(set(ids)))
    elif batch.get("position_ids") is not None:
        pos = _tolist(batch["position_ids"])
        if pos and isinstance(pos[0], list) and pos[0] and isinstance(pos[0][0], list):
            raise ValueError(
                "3-D position_ids do not identify attention segments reliably; "
                "provide cu_seqlens or a document-id attention_mask"
            )
        for r, row in enumerate(pos):
            valid = [bool(v) for v in mask_rows[r]] if mask_rows else None
            st.seg_lens += _segments_from_position_ids(row, valid)
    elif mask_rows is not None:
        for row in mask_rows:
            ids = [v for v in row if v]
            if ids and max(ids) > 1:  # document-id mask from neat packing
                for d in sorted(set(ids)):
                    st.seg_lens.append(ids.count(d))
            else:
                st.seg_lens.append(len(ids))
    elif batch.get("input_ids") is not None:
        st.seg_lens = [len(r) for r in _tolist(batch["input_ids"])]

    if batch.get("image_grid_thw") is not None:
        st.image_patches = [int(t * h * w) for t, h, w in _tolist(batch["image_grid_thw"])]
    if batch.get("labels") is not None:
        st.loss_tokens = sum(1 for row in _tolist(batch["labels"]) for v in row if v != -100)
    return st
