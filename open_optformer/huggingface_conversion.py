"""
Convert LitGPT Qwen3 checkpoint to HuggingFace format.
"""
import sys
import torch
import yaml
from pathlib import Path
from typing import Union
import os
import json

from transformers import LlamaTokenizer, LlamaTokenizerFast


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

    print(f"Loading SentencePiece model from: {path}")
    slow_tokenizer = LlamaTokenizer(vocab_file=path / "tokenizer.model", legacy=False)

    # 2. Save "slow" version to directory
    # This generates the initial config files
    print(f"Saving temporary files to: {output_dir}")
    slow_tokenizer.save_pretrained(output_dir)

    # 3. Convert to "Fast" (Rust-based) version
    # This creates the critical tokenizer.json file
    print(f"Generating Fast tokenizer...")
    fast_tokenizer = LlamaTokenizerFast.from_pretrained(output_dir)
    fast_tokenizer.save_pretrained(output_dir)

    # 4. MANUALLY FIX THE tokenizer.json
    # This removes the Pre-Tokenizer and Normalizer to prevent ID mismatches
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")

    with open(tokenizer_json_path, "r") as f:
        data = json.load(f)

    print(f"Patching tokenizer.json (Setting pre_tokenizer and normalizer to null)...")
    # Bypass HF's default splitting logic to match LitGPT's raw SentencePiece behavior
    data["pre_tokenizer"] = None
    data["normalizer"] = None

    with open(tokenizer_json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Success! Converted tokenizer saved to: {output_dir}")

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
    import sys
    import torch
    import sentencepiece as spm
    import json
    from transformers import AutoTokenizer

    # Check for correct number of arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py <litgpt_checkpoint_path> <hf_output_path>")
        sys.exit(1)

    litgpt_path = sys.argv[1]
    hf_path = sys.argv[2]

    print(f"Loading LitGPT model from: {litgpt_path}")
    litgpt_model = load_litgpt(litgpt_path)

    print(f"Converting to HuggingFace at: {hf_path}")
    hf_model = convert_to_huggingface(litgpt_path, hf_path)

    # Generate random context
    vocab_size = litgpt_model.config.vocab_size
    context_length = 10
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

    # Assert close enough
    assert torch.allclose(litgpt_logits, hf_logits, atol=1e-4), \
        f"Logits don't match! Max diff: {max_diff}"

    print("\nSUCCESS: Model predictions match!")

    # --- TOKENIZER COMPARISON ---
    print("\n--- Verifying Tokenizer Parity ---")
    # Path to the original .model file within the checkpoint folder
    sp_model_path = f"{litgpt_path}/tokenizer.model"
    sp_processor = spm.SentencePieceProcessor(model_file=sp_model_path)

    # Load your newly converted Hugging Face Tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_path)

    # Test string with your categorical and special tokens
    test_text = "<algorithm>:RS<type>:<UNI>,<min value>:0,<max value>:1.0,<log-scale>&<type>:<INT>,<min value>:1,<max value>:5,<linear-scale>&<type>:<CATEGORICAL>,<categories>:['l1', 'l2']120,200,<1>*300|60,50,<0>*200|"

    # Check ID 1035
    try:
        print(f"Token ID 1035 represents: '{sp_processor.decode([1035])}'")
    except:
        print("Token ID 1035 out of bounds for SentencePiece processor.")

    litgpt_ids = sp_processor.encode(test_text)
    hf_ids = hf_tokenizer.encode(test_text, add_special_tokens=False)

    print(f"LitGPT IDs: {litgpt_ids}")
    print(f"HF IDs:     {hf_ids}")

    if litgpt_ids == hf_ids:
        print("✅ Success! Token IDs match perfectly.")
    else:
        print("❌ Warning: ID mismatch detected.")

    # --- VOCAB SIZE CHECK ---
    print("\n--- Verifying Vocab Size ---")
    tokenizer_vocab_size = len(hf_tokenizer)

    # Load model config from the converted path
    hf_config_path = f"{hf_path}/config.json"
    with open(hf_config_path, "r") as f:
        config = json.load(f)
    model_vocab_size = config.get("vocab_size")

    print(f"Tokenizer Vocab Size: {tokenizer_vocab_size}")
    print(f"Model Config Vocab Size: {model_vocab_size}")

    if tokenizer_vocab_size == model_vocab_size:
        print("✅ Match! Vocab sizes are consistent.")
    elif tokenizer_vocab_size < model_vocab_size:
        print("⚠️ Warning: Model config vocab_size is larger than tokenizer.")
    else:
        print("❌ Error: Tokenizer vocab is larger than model config!")