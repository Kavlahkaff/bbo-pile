import nevergrad as ng
from nevergrad.functions import ArtificialFunction, corefuncs
from nevergrad_benchmarks import NevergradExperimenterProblem

benchmark_definitions = {}

# 1. Load the Core Mathematical Functions (includes components of BBOB/YABBOB, deceptive, illcond, etc.)
# We explicitly set bounded=True to force Nevergrad to define standard bounds (typically [-5, 5])
dims = [2, 5, 10]
for name in corefuncs.registry.keys():
    for d in dims:
        try:
            func = ArtificialFunction(name=name, block_dimension=d, bounded=True)
            key = f"nevergrad_core_{name}_{d}d"
            benchmark_definitions[key] = NevergradExperimenterProblem(func)
        except Exception:
            pass

# Helper to safely load and register other functions that may or may not have dependencies installed
def try_add_function(suite_name, func_callable, *args, **kwargs):
    try:
        func = func_callable(*args, **kwargs)
        # Attempt to create the problem (this will fail if bounds extraction fails or instantiation fails)
        problem = NevergradExperimenterProblem(func)
        key = f"nevergrad_{suite_name}_{func.name}_{func.dimension}d"
        if key not in benchmark_definitions:
            benchmark_definitions[key] = problem
    except Exception as e:
        # Silently skip if dependency fails or parameterization is unsupported
        pass

# 2. Dynamically extract all possible benchmark problems from the experiments registry
# This ensures we get all variants (YABBOB, ms_bbob, noisy, constrained, discrete, RL, physics)
import warnings
warnings.filterwarnings("ignore")

try:
    import nevergrad.benchmark.experiments as exp
    # We skip a few specific suites that attempt to download massive datasets (e.g. asterweb DEM)
    # or have highly brittle dependencies which cause extreme delays during instantiation.
    skip_suites = {'mlda', 'mldakmeans', 'image_similarity', 'image_quality', 'ceviche', 'realworld'}
    
    for suite_name, suite_generator in exp.registry.items():
        if any(s in suite_name for s in skip_suites):
            continue
        try:
            gen = suite_generator()
            for i, e in enumerate(gen):
                # Ensure the function has a name
                if not hasattr(e.function, "name"):
                    continue
                    
                func = e.function
                name = func.name
                d = func.dimension
                key = f"nevergrad_{suite_name}_{name}_{d}d"
                
                if key not in benchmark_definitions:
                    benchmark_definitions[key] = NevergradExperimenterProblem(func)
                
                # To avoid an explosion of duplicates (since experiments loop over optimizers/budgets),
                # we cap the search depth per suite once we've sampled enough unique functions.
                if i > 50:
                    break
        except Exception:
            # Safely skip suites where external dependencies (mujoco, etc.) are missing
            pass
except Exception:
    pass

