import os
import ioh
from ioh_benchmarks import IOHExperimenterProblem

benchmark_definitions = {}

suites = [
    ("bbob", ioh.problem.BBOB.problems, [2, 5, 10], ioh.ProblemClass.BBOB),
    ("cec2013", ioh.problem.CEC2013.problems, [2, 5, 10], ioh.ProblemClass.CEC2013),
    ("cec2022", ioh.problem.CEC2022.problems, [2, 5, 10], ioh.ProblemClass.CEC2022),
    ("sbox", ioh.problem.SBOX.problems, [2, 5, 10], ioh.ProblemClass.SBOX),
    ("star_real", ioh.problem.RealStarDiscrepancy.problems, [2, 5, 10], ioh.ProblemClass.STAR_REAL),
    ("pbo", ioh.problem.PBO.problems, [10, 50, 100], ioh.ProblemClass.PBO),
    ("star_integer", ioh.problem.IntegerStarDiscrepancy.problems, [10, 50, 100], ioh.ProblemClass.STAR_INTEGER),
]

if hasattr(ioh.problem, "GraphProblem"):
    suites.append(("graph", ioh.problem.GraphProblem.problems, [10, 50, 100], ioh.ProblemClass.GRAPH))

# Suppress verbose C++ stderr output from ioh during instantiation
devnull = os.open(os.devnull, os.O_WRONLY)
old_stderr = os.dup(2)
os.dup2(devnull, 2)

for name, probs, dims, pclass in suites:
    for pid, pname in probs.items():
        for d in dims:
            try:
                _ = ioh.get_problem(pid, 1, d, pclass)
                benchmark_definitions[f"ioh_{name}_{pid}_{pname.lower()}_{d}d"] = IOHExperimenterProblem(pid, d, pclass)
            except Exception:
                pass

os.dup2(old_stderr, 2)
os.close(devnull)
