from typing import Union, List, Optional

import torch


def chunked_cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_index: int = -100,
    weight: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude
    #
    # `weight` is the standard F.cross_entropy per-vocab-id class weight (same weight for a given
    # token id everywhere). `sample_weights`, if given, is a per-position weight shaped like
    # `targets` (e.g. built from per-trial advantages) and is combined multiplicatively with the
    # per-token loss instead of by class id. When `sample_weights` is None, behavior is unchanged.

    # lm_head was chunked (we are fine-tuning)
    if isinstance(logits, list):
        # don't want to chunk cross entropy
        if chunk_size == 0 and sample_weights is None:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index, weight=weight)

        # chunk cross entropy
        if chunk_size == 0:
            logit_chunks = [torch.cat(logits, dim=1).reshape(-1, logits[0].size(-1))]
            target_chunks = [targets.reshape(-1)]
            weight_chunks = [sample_weights.reshape(-1)] if sample_weights is not None else None
        else:
            logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
            target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
            # sample_weights must be chunked identically to targets (dim=1 splits matching the
            # lm_head chunk boundaries), not via a single global reshape, so that after
            # concatenation each weight still lines up with the loss/target it belongs to.
            weight_chunks = (
                [wc.reshape(-1) for wc in sample_weights.split(logits[0].size(1), dim=1)]
                if sample_weights is not None
                else None
            )
        loss_chunks = [
            torch.nn.functional.cross_entropy(
                logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none", weight=weight
            )
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        flat_targets = torch.cat(target_chunks)
        flat_loss = torch.cat(loss_chunks)
        if weight_chunks is not None:
            flat_sample_weights = torch.cat(weight_chunks)
            mask = (flat_targets != ignore_index).to(flat_sample_weights.dtype)
            denom = (flat_sample_weights * mask).sum().clamp_min(1e-6)
            return (flat_loss * flat_sample_weights).sum() / denom
        non_masked_elems = (flat_targets != ignore_index).sum()
        # See [non_masked_elems div note]
        return flat_loss.sum() / non_masked_elems.maximum(torch.ones_like(non_masked_elems))

    # no chunking at all
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if sample_weights is not None:
        sample_weights = sample_weights.reshape(-1)
    if chunk_size == 0 and sample_weights is None:
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index, weight=weight)

    # lm_head wasn't chunked, chunk cross entropy
    if chunk_size == 0:
        logit_chunks = [logits]
        target_chunks = [targets]
    else:
        logit_chunks = logits.split(chunk_size)
        target_chunks = targets.split(chunk_size)
    loss_chunks = [
        torch.nn.functional.cross_entropy(
            logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none", weight=weight
        )
        for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
    ]
    flat_loss = torch.cat(loss_chunks)
    if sample_weights is not None:
        mask = (targets != ignore_index).to(sample_weights.dtype)
        denom = (sample_weights * mask).sum().clamp_min(1e-6)
        return (flat_loss * sample_weights).sum() / denom
    non_masked_elems = (targets != ignore_index).sum()
    # [non_masked_elems div note]:
    #   max(1, non_masked_elems) would be more ergonomic to avoid a division by zero. However that
    #   results in a python int which is then passed back to torch division. By using the
    #   `x.maximum(torch.ones_like(x))` pattern we avoid a cudaStreamSynchronize.
    return flat_loss.sum() / non_masked_elems.maximum(torch.ones_like(non_masked_elems))
