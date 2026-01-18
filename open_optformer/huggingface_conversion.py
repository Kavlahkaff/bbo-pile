"""
Convert LitGPT Qwen3 checkpoint to HuggingFace format.
"""
import sys
import torch
import yaml
from pathlib import Path
from typing import Union

from transformers import LlamaTokenizerFast


def load_litgpt(path: Union[str, Path]):
    """Load a LitGPT model from checkpoint directory."""
    from litgpt.config import Config
    from litgpt.model import GPT

    path = Path(path)
    config = Config.from_file(str(path / 'model_config.yaml'))
    model = GPT(config)

    state_dict = torch.load(
        str(path / 'lit_model.pth'),
        weights_only=True,
        map_location='cpu'
    )
    if 'model' in state_dict:
        state_dict = state_dict['model']

    model.load_state_dict(state_dict)
    model.eval()
    return model


def convert_to_huggingface(path: Union[str, Path], output_dir: Union[str, Path] = "qwen3-hf"):
    """Convert LitGPT checkpoint to HuggingFace Qwen3 model and save it."""
    from transformers import Qwen3ForCausalLM, Qwen3Config

    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = Path(path)

    # Load LitGPT config
    with open(path / 'model_config.yaml', 'r') as f:
        litgpt_config = yaml.safe_load(f)

    # Load LitGPT weights
    state_dict = torch.load(
        str(path / 'lit_model.pth'),
        weights_only=True,
        map_location='cpu'
    )
    if 'model' in state_dict:
        state_dict = state_dict['model']

    # Create HuggingFace config
    hf_config = Qwen3Config(
        vocab_size=litgpt_config['vocab_size'],
        hidden_size=litgpt_config['n_embd'],
        intermediate_size=litgpt_config['intermediate_size'],
        num_hidden_layers=litgpt_config['n_layer'],
        num_attention_heads=litgpt_config['n_head'],
        num_key_value_heads=litgpt_config['n_query_groups'],
        head_dim=litgpt_config['head_size'],
        max_position_embeddings=litgpt_config['block_size'],
        rms_norm_eps=litgpt_config['norm_eps'],
        rope_theta=litgpt_config['rope_base'],
        tie_word_embeddings=True,  # From hyperparameters.yaml
        attention_bias=litgpt_config['attn_bias'],
    )

    # Convert weights
    hf_state_dict = {}

    # Embeddings
    hf_state_dict['model.embed_tokens.weight'] = state_dict['transformer.wte.weight']

    # Final layer norm
    hf_state_dict['model.norm.weight'] = state_dict['transformer.ln_f.weight']

    # LM head (tied with embeddings, but we include it anyway)
    hf_state_dict['lm_head.weight'] = state_dict['lm_head.weight']

    # Per-layer weights
    n_layer = litgpt_config['n_layer']
    n_head = litgpt_config['n_head']
    n_query_groups = litgpt_config['n_query_groups']
    head_size = litgpt_config['head_size']

    for i in range(n_layer):
        # Layer norms
        hf_state_dict[f'model.layers.{i}.input_layernorm.weight'] = state_dict[f'transformer.h.{i}.norm_1.weight']
        hf_state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = state_dict[f'transformer.h.{i}.norm_2.weight']

        # QKV split
        # LitGPT fuses QKV with shape [q_size + k_size + v_size, hidden_size]
        # where q_size = n_head * head_size, k_size = v_size = n_query_groups * head_size
        # LitGPT uses CONTIGUOUS [Q, K, V] layout (not interleaved)
        qkv_weight = state_dict[f'transformer.h.{i}.attn.qkv.weight']

        q_size = n_head * head_size
        k_size = n_query_groups * head_size
        v_size = n_query_groups * head_size

        # Simple contiguous split
        q_weight = qkv_weight[:q_size, :]
        k_weight = qkv_weight[q_size:q_size + k_size, :]
        v_weight = qkv_weight[q_size + k_size:, :]

        hf_state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = q_weight
        hf_state_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = k_weight
        hf_state_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = v_weight

        # Output projection
        hf_state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = state_dict[f'transformer.h.{i}.attn.proj.weight']

        # QK norms (Qwen3 has these)
        hf_state_dict[f'model.layers.{i}.self_attn.q_norm.weight'] = state_dict[f'transformer.h.{i}.attn.norm_q.weight']
        hf_state_dict[f'model.layers.{i}.self_attn.k_norm.weight'] = state_dict[f'transformer.h.{i}.attn.norm_k.weight']

        # MLP
        # LitGPT: fc_1 is gate, fc_2 is up, proj is down
        hf_state_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = state_dict[f'transformer.h.{i}.mlp.fc_1.weight']
        hf_state_dict[f'model.layers.{i}.mlp.up_proj.weight'] = state_dict[f'transformer.h.{i}.mlp.fc_2.weight']
        hf_state_dict[f'model.layers.{i}.mlp.down_proj.weight'] = state_dict[f'transformer.h.{i}.mlp.proj.weight']

    # Create model and load weights
    model = Qwen3ForCausalLM(hf_config)
    info = model.load_state_dict(hf_state_dict, strict=False)

    if info.missing_keys:
        print("WARNING: Missing keys:", info.missing_keys)
    if info.unexpected_keys:
        print("WARNING: Unexpected keys:", info.unexpected_keys)

    model.eval()

    # Save model + config
    print(f"Saving HuggingFace model to {output_dir}")
    model.save_pretrained(output_dir)

    # Use LlamaTokenizer, because AutoTokenizer can load binary files
    tokenizer = LlamaTokenizerFast(
        vocab_file=str(path / "tokenizer.model"),
        local_files_only=True
    )

    tokenizer.eos_token = "|"

    # Save the converted tokenizer back to the directory
    # This generates 'tokenizer.json', 'tokenizer_config.json', and 'special_tokens_map.json'
    tokenizer.save_pretrained(output_dir)

    print("Save complete.")

    return model


def do_inference_litgpt(context: list[int], litgpt_model) -> torch.Tensor:
    """Run inference with LitGPT model, returns logits for next token."""
    with torch.no_grad():
        input_tensor = torch.tensor([context], dtype=torch.long)
        logits = litgpt_model(input_tensor)
        # Return logits for the last position
        return logits[0, -1, :]


def do_inference_huggingface(context: list[int], hf_model) -> torch.Tensor:
    """Run inference with HuggingFace model, returns logits for next token."""
    with torch.no_grad():
        input_tensor = torch.tensor([context], dtype=torch.long)
        outputs = hf_model(input_tensor)
        # Return logits for the last position
        return outputs.logits[0, -1, :]


if __name__ == '__main__':
    import random

    print("Loading LitGPT model...")
    litgpt_model = load_litgpt(sys.argv[1])

    print("Converting to HuggingFace...")
    hf_model = convert_to_huggingface(sys.argv[1], sys.argv[2])

    # Generate random context
    vocab_size = litgpt_model.config.vocab_size


    context_length =  10
    random.seed(42)
    torch.manual_seed(42)
    random_context = [random.randint(0, vocab_size - 1) for _ in range(context_length)]
    print(f"Random context: {random_context}")

    print("Running LitGPT inference...")
    litgpt_logits = do_inference_litgpt(random_context, litgpt_model)

    print("Running HuggingFace inference...")
    hf_logits = do_inference_huggingface(random_context, hf_model)

    # Compare logits
    print(f"LitGPT logits shape: {litgpt_logits.shape}")
    print(f"HuggingFace logits shape: {hf_logits.shape}")

    print(f"LitGPT logits (first 10): {litgpt_logits[:10]}")
    print(f"HuggingFace logits (first 10): {hf_logits[:10]}")

    # Check if predictions match
    max_diff = torch.max(torch.abs(litgpt_logits - hf_logits)).item()
    print(f"Max absolute difference: {max_diff}")

    # Check argmax matches
    litgpt_pred = torch.argmax(litgpt_logits).item()
    hf_pred = torch.argmax(hf_logits).item()
    print(f"LitGPT prediction: {litgpt_pred}")
    print(f"HuggingFace prediction: {hf_pred}")

    # Assert close enough (small numerical differences are expected)
    assert torch.allclose(litgpt_logits, hf_logits, atol=1e-4), \
        f"Logits don't match! Max diff: {max_diff}"

    print("\nSUCCESS: Predictions match!")

