import os

from baselines import methods
from benchmarks_definitions import benchmark_definitions

output_path = os.environ['OUTPUT_PATH']
os.makedirs(output_path, exist_ok=True)
fh = open('bash_commands.sh', 'w')
counter = 0
for benchmark in list(benchmark_definitions.keys()):
    for method in methods:
        for seed in range(30):
            cmd = f'python run_benchmark.py --method {method} --benchmark {benchmark} --seed {seed} --output_path {output_path}'
            counter += 1
            fh.write(cmd + '\n')

print(f'Total commands: {counter}')
fh.close()