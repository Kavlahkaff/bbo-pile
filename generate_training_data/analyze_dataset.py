import json
import argparse
from collections import Counter
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Analyze the compiled dataset method distributions.")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the dataset output folder containing dataset_summary.json")
    args = parser.parse_args()

    summary_path = Path(args.dataset_path) / "dataset_summary.json"
    
    if not summary_path.exists():
        print(f"Error: {summary_path} not found. Please ensure the dataset was compiled with --only_best")
        return

    with open(summary_path, 'r') as f:
        summary_data = json.load(f)

    if not summary_data:
        print("Dataset summary is empty.")
        return

    df = pd.DataFrame(summary_data)
    
    print("="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    print(f"Total experiments (trajectories) in dataset: {len(df)}")
    print(f"Total unique benchmarks: {df['benchmark'].nunique()}")
    
    dist = Counter(df['algorithm'])
    print("\n" + "-"*50)
    print("OVERALL METHOD DISTRIBUTION")
    print("-"*50)
    for alg, count in dist.most_common():
        print(f"  {alg:10s} : {count:4d} ({count/len(df)*100:.2f}%)")
        
    print("\n" + "-"*50)
    print("METHOD DISTRIBUTION PER BENCHMARK")
    print("-"*50)
    for benchmark in sorted(df['benchmark'].unique()):
        b_df = df[df['benchmark'] == benchmark]
        b_dist = Counter(b_df['algorithm'])
        methods_str = ", ".join([f"{alg}: {count}" for alg, count in b_dist.most_common()])
        print(f"  {benchmark:40s} | {methods_str}")

if __name__ == "__main__":
    main()
