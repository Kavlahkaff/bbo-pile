import numpy as np
from syne_tune.blackbox_repository.blackbox import Blackbox
from syne_tune.config_space import uniform, randint, choice

class NevergradExperimenterProblem(Blackbox):
    def __init__(self, problem):
        """
        Wraps a nevergrad.functions ArtificialFunction or ExperimentFunction.
        """
        self.problem = problem
        self.dimension = problem.dimension
        self.name = problem.name
        
        # We need to extract the parameter bounds to define the configuration space.
        cs = {}
        
        # Most Nevergrad core functions have a Parameter object with bounds.
        # If it's a generic tuple or dictionary parameter, it might have sub-parameters.
        # For simplicity, we assume single Array parameter or multiple Array parameters 
        # combined into a tuple/kwargs.
        
        # If it's a standard ArtificialFunction, bounded=True ensures bounds are set.
        # We attempt to safely read bounds.
        bounds = None
        if hasattr(self.problem, "parametrization") and hasattr(self.problem.parametrization, "bounds"):
            bounds = self.problem.parametrization.bounds
            
        lb = bounds[0] if bounds is not None and bounds[0] is not None else np.full(self.dimension, -5.0)
        ub = bounds[1] if bounds is not None and bounds[1] is not None else np.full(self.dimension, 5.0)
        
        if np.isscalar(lb): lb = np.array([lb])
        if np.isscalar(ub): ub = np.array([ub])
        
        # Broadcast single bounds to full dimension if needed
        if len(lb) == 1 and self.dimension > 1:
            lb = np.full(self.dimension, lb[0])
            ub = np.full(self.dimension, ub[0])
            
        for i in range(self.dimension):
            # We treat them as continuous variables by default. 
            # If discrete domains are needed, they would map differently.
            # Nevergrad's corefuncs are predominantly continuous.
            cs[f"x{i}"] = uniform(float(lb[i]), float(ub[i]))

        super().__init__(configuration_space=cs, objectives_names=["y"])

    def _objective_function(self, configuration, fidelity=None, seed=None):
        # We parse the config space variables back into an array to pass to the function
        x = np.array([configuration[f"x{i}"] for i in range(self.dimension)])
        
        # Evaluate the nevergrad problem
        try:
            val = self.problem(x)
        except TypeError:
            # Some nevergrad functions take *args instead of a list/array
            val = self.problem(*x.tolist())
            
        # Nevergrad functions aim to minimize, so we map exactly to y
        return {"y": float(val)}

