import torch
import torch.nn.functional as F

from open_optformer.training.chuncked_cross_entropy import chunked_cross_entropy


def _make_inputs(seed=0):
    torch.manual_seed(seed)
    B, T, V = 2, 10, 50
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    targets[0, -2:] = -100  # ignored positions
    return logits, targets


def test_sample_weights_none_matches_baseline_tensor_branch():
    logits, targets = _make_inputs()
    for chunk_size in (0, 4):
        loss = chunked_cross_entropy(logits, targets, chunk_size=chunk_size)
        ref = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        assert torch.allclose(loss, ref, atol=1e-5)


def test_sample_weights_all_ones_matches_none():
    logits, targets = _make_inputs()
    ones = torch.ones_like(targets, dtype=torch.float)
    for chunk_size in (0, 4):
        loss_none = chunked_cross_entropy(logits, targets, chunk_size=chunk_size)
        loss_ones = chunked_cross_entropy(logits, targets, chunk_size=chunk_size, sample_weights=ones)
        assert torch.allclose(loss_none, loss_ones, atol=1e-5)


def test_sample_weights_zeroing_subset_matches_manual_reference():
    logits, targets = _make_inputs()
    weights = torch.ones_like(targets, dtype=torch.float)
    weights[1, :5] = 0.0
    for chunk_size in (0, 3):
        loss = chunked_cross_entropy(logits, targets, chunk_size=chunk_size, sample_weights=weights)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        flat_weights = weights.reshape(-1)
        keep = (flat_weights != 0) & (flat_targets != -100)
        ref = F.cross_entropy(flat_logits[keep], flat_targets[keep], reduction="mean")
        assert torch.allclose(loss, ref, atol=1e-5)


def test_list_branch_matches_tensor_branch_with_and_without_weights():
    logits, targets = _make_inputs()
    weights = torch.ones_like(targets, dtype=torch.float)
    weights[1, :5] = 0.0
    half = logits.size(1) // 2
    logits_list = [logits[:, :half, :], logits[:, half:, :]]

    for chunk_size in (0, 4):
        a = chunked_cross_entropy(logits_list, targets, chunk_size=chunk_size)
        b = chunked_cross_entropy(logits, targets, chunk_size=chunk_size)
        assert torch.allclose(a, b, atol=1e-5)

        aw = chunked_cross_entropy(logits_list, targets, chunk_size=chunk_size, sample_weights=weights)
        bw = chunked_cross_entropy(logits, targets, chunk_size=chunk_size, sample_weights=weights)
        assert torch.allclose(aw, bw, atol=1e-5)
