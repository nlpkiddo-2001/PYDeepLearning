"""Turn-boundary cutting for documents longer than the training sequence length."""
from __future__ import annotations

from typing import Optional

import numpy as np


def truncate_at_turn(mask: np.ndarray, limit: int, min_trainable: int) -> Optional[int]:
    """Largest cut <= limit that ends a trained (assistant) span, keeping >= min_trainable trained tokens.

    Cutting only at span ends means a tool call / assistant reply is never cut in half and every kept target is whole.
    Returns None when no complete assistant turn fits (the document is unusable at this length).
    """
    n = len(mask)
    head = mask[:limit]
    nxt = np.append(mask[1:limit], mask[limit] if n > limit else 0)
    ends = np.flatnonzero((head == 1) & (nxt == 0)) + 1
    if len(ends) == 0:
        return None
    cut = int(ends[-1])
    return cut if int(mask[:cut].sum()) >= min_trainable else None
