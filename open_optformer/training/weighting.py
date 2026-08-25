"""Per-position loss-weight utilities for advantage-reweighted SFT.

This is deliberately kept separate from `syne_tune_data.py`'s vocab-level
`loss_weights` mechanism: that reweights by token *identity* (same weight for
a given token id everywhere), whereas the utilities here build a per-sequence-
-position weight array aligned with a specific tokenized prompt, so the same
token id can carry a different weight depending on which trial it came from.
"""

from typing import List, Tuple


def segments_to_token_weights(
    segments: List[Tuple[str, float]],
    tokenizer,
) -> Tuple[List[int], List[float]]:
    """Tokenize a list of (text_chunk, weight) segments and repeat each
    segment's scalar weight once per token it produces.

    Segments are tokenized independently (rather than tokenizing the full
    joined string once and trying to re-align weights after the fact) so
    that trial-boundary weights stay exact regardless of how the tokenizer's
    BPE/subword merges interact with neighboring characters. Only the first
    segment (the shared benchmark/algorithm/search-space header) is tokenized
    with the tokenizer's default `bos` behavior, matching how
    `preprocess_data.py` tokenizes a full prompt line today; every subsequent
    segment is tokenized with `bos=False` since it's a mid-sequence
    continuation, not the start of a new example.
    """
    token_ids: List[int] = []
    weights: List[float] = []
    for i, (text, weight) in enumerate(segments):
        bos = None if i == 0 else False
        ids = tokenizer.encode(text, bos=bos, eos=False).tolist()
        token_ids.extend(ids)
        weights.extend([weight] * len(ids))
    return token_ids, weights
