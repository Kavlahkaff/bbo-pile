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
    "co-5": BenchmarkDefinition(objective='counting_ones_5D.py',
                                dim=5,
                                metric="feval",
                                mode='max',
                                max_num_evaluations=20)
}
