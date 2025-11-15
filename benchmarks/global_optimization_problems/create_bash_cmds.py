import os

from baselines import methods
from benchmarks_definitions import benchmark_definitions

output_path = '/home/aaron/experiments/open_optformer/global_optimization_problems/'
os.makedirs(output_path, exist_ok=True)
fh = open('bash_commands.sh', 'w')
counter = 0
for benchmark in list(benchmark_definitions.keys()):
    for method in methods:
        for seed in range(30):
            cmd = f'python run_synthetic_functions_without_backend.py --method {method} --benchmark {benchmark} --seed {seed} --output_path {output_path}'
            counter += 1
            fh.write(cmd + '\n')

print(f'Total commands: {counter}')
fh.close()