from dataclasses import dataclass
from typing import Optional, List, Dict, Callable

from syne_tune.config_space import choice


def counting_ones(dim, **kwargs):
    """
    The function to be tuned, note that import must be in PythonBackend and no global variable are allowed,
    more details on requirements of tuned functions can be found in
    :class:`~syne_tune.backend.PythonBackend`.
    """
    from syne_tune import Reporter
    import numpy as np
    reporter = Reporter()
    config = []
    for i in range(dim):
        config.append(kwargs[f'x_{i}'])
    ones = np.sum(config)
    reporter(feval=ones)

@dataclass
class BenchmarkDefinition:
    objective: Callable
    metric: str
    mode: str
    dim: int
    max_num_evaluations: Optional[int] = None

benchmark_definitions = {
    "co-5": BenchmarkDefinition(objective=counting_ones,
                                dim=5,
                                metric="feval",
                                mode='max',
                                max_num_evaluations=20)
}
