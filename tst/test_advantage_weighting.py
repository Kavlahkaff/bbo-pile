from syne_tune.config_space import randint, uniform, choice

from open_optformer.history import History


class _CharTokenizer:
    """Minimal char-level stand-in tokenizer for testing token/weight alignment
    without depending on a real litgpt Tokenizer asset."""

    vocab_size = 256

    def encode(self, text, bos=None, eos=False):
        import torch
        return torch.tensor([ord(c) for c in text], dtype=torch.int)

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


def _build_history():
    config_space = {
        "x": uniform(0, 1),
        "y": randint(0, 10),
        "z": choice(["a", "b", "c"]),
    }
    history = History(name="test", algorithm="test", config_space=config_space, num_numeric_tokens=1000)
    history.add_trial({"x": 0.5, "y": 5, "z": "a"}, 0.5)
    history.add_trial({"x": 0.6, "y": 6, "z": "b"}, 0.6)
    return history


def test_get_prompt_with_weights_matches_get_prompt_string():
    history = _build_history()
    prompt = history.get_prompt()
    weighted_prompt, segments = history.get_prompt_with_weights(advantages=[1.0, 9.0])
    assert weighted_prompt == prompt
    assert "".join(text for text, _ in segments) == prompt


def test_get_prompt_unchanged_after_refactor():
    history = _build_history()
    prompt = history.get_prompt()
    assert "500,500,<0>*0|599,599,<1>*999|" in prompt


def test_segments_to_token_weights_alignment():
    from open_optformer.training.weighting import segments_to_token_weights

    history = _build_history()
    _, segments = history.get_prompt_with_weights(advantages=[1.0, 9.0], base_weight=0.2)

    tokenizer = _CharTokenizer()
    token_ids, token_weights = segments_to_token_weights(segments, tokenizer)

    assert len(token_ids) == len(token_weights)

    # Decode contiguous same-weight runs and check they land in the right span.
    runs = []
    start = 0
    for i in range(1, len(token_weights) + 1):
        if i == len(token_weights) or token_weights[i] != token_weights[start]:
            runs.append((token_weights[start], tokenizer.decode(token_ids[start:i])))
            start = i

    weight_9_text = "".join(text for w, text in runs if w == 9.0)
    weight_1_text = "".join(text for w, text in runs if w == 1.0)
    weight_header_text = "".join(text for w, text in runs if w == 0.2)

    assert "599,599,<1>*999|" == weight_9_text
    assert "500,500,<0>*0|" == weight_1_text
    assert weight_header_text.startswith("benchmark:test,algorithm:test,search-space:")
    assert weight_header_text.endswith(",history:")
