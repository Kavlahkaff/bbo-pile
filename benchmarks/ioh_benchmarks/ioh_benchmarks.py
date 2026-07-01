import ioh
import numpy as np
from syne_tune.blackbox_repository.blackbox import Blackbox
from syne_tune.config_space import uniform, randint

class IOHExperimenterProblem(Blackbox):
    def __init__(self, problem_id: int, dimension: int, problem_class: ioh.ProblemClass, instance: int = 1):
        self.problem = ioh.get_problem(problem_id, instance, dimension, problem_class)
        self.dimension = dimension
        
        lb, ub = self.problem.bounds.lb, self.problem.bounds.ub
        is_int = problem_class in (
            getattr(ioh.ProblemClass, "PBO", None),
            getattr(ioh.ProblemClass, "STAR_INTEGER", None),
            getattr(ioh.ProblemClass, "GRAPH", None)
        )
        
        cs = {
            f"x{i}": randint(int(lb[i]), int(ub[i])) if is_int or np.issubdtype(type(lb[i]), np.integer) 
            else uniform(float(lb[i]), float(ub[i]))
            for i in range(dimension)
        }
        super().__init__(configuration_space=cs, objectives_names=["y"])

    def _objective_function(self, configuration, fidelity=None, seed=None):
        val = self.problem([configuration[f"x{i}"] for i in range(self.dimension)])
        if self.problem.meta_data.optimization_type == ioh.OptimizationType.MAX:
            val = -val
        return {"y": float(val)}
