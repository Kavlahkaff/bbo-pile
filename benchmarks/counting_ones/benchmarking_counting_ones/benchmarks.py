from dataclasses import dataclass
from typing import Optional, List, Dict, Callable

from syne_tune.config_space import choice


@dataclass
class BenchmarkDefinition:
    objective: str
    metric: str
    mode: str
    dim: int
    max_num_evaluations: Optional[int] = None

benchmark_definitions = {
    "co-5": BenchmarkDefinition(objective='benchmarking_counting_ones/counting_ones_5D.py',
                                dim=5,
                                metric="feval",
                                mode='max',
                                max_num_evaluations=20),
    "co-10": BenchmarkDefinition(objective='benchmarking_counting_ones/counting_ones_10D.py',
                                dim=10,
                                metric="feval",
                                mode='max',
                                max_num_evaluations=20),
    "co-20": BenchmarkDefinition(objective='benchmarking_counting_ones/counting_ones_20D.py',
                                dim=20,
                                metric="feval",
                                mode='max',
                                max_num_evaluations=20)
}
