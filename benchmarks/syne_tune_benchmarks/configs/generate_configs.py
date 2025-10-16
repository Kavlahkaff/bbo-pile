
import yaml
from pathlib import Path

def generate_configs():
    model_name = 'pythia-72M'
    base_config_path = Path(__file__).parent / f"{model_name}_template.yaml"
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    token_counts = {
        "50M": 50_000_000,
        "100M": 100_000_000,
        "200M": 200_000_000,
        "400M": 400_000_000,
        "600M": 600_000_000,
        "800M": 800_000_000,
        "1B": 1_000_000_000,
    }

    for name, tokens in token_counts.items():
        new_config = base_config.copy()
        new_config['train']['max_tokens'] = tokens
        
        run_name = f"{model_name}_token_{name}"
        new_config['log']['run'] = run_name
        
        new_filename = f"{run_name}.yaml"
        new_filepath = base_config_path.parent / new_filename
        
        with open(new_filepath, 'w') as f:
            yaml.dump(new_config, f, sort_keys=False)
        
        print(f"Generated {new_filepath}")

if __name__ == "__main__":
    generate_configs()
