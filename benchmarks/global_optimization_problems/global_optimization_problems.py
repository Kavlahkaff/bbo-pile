from typing import Any, Dict, List

import numpy as np

from syne_tune.blackbox_repository.blackbox import Blackbox, ObjectiveFunctionResult
from syne_tune.config_space import uniform


class SyntheticFunction(Blackbox):
    """
    Base class for synthetic blackbox functions. These are defined by a formula and
    are cheap to evaluate. They are mainly used for testing and to prototype new
    algorithms.
    """

    def __init__(
        self,
        dimension: int,
        configuration_space: Dict[str, Any] | None = None,
        objectives_names: List[str] | None = None,
    ):
        if configuration_space is None:
            configuration_space = {
                f"x{i}": uniform(-5.0, 10.0) for i in range(dimension)
            }
        if objectives_names is None:
            objectives_names = ["y"]
        super().__init__(
            configuration_space=configuration_space,
            objectives_names=objectives_names,
        )
        self.dimension = dimension

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        raise NotImplementedError


class Rosenbrock(SyntheticFunction):
    """
    The Rosenbrock function is a non-convex function used as a performance test
    problem for optimization algorithms.
    See https://www.sfu.ca/~ssurjano/rosen.html for details.
    """

    def __init__(
        self, dimension: int, lower_bound: float = -5.0, upper_bound: float = 10.0
    ):
        self.configuration_space = {
            f"x{i}": uniform(lower_bound, upper_bound) for i in range(dimension)
        }
        super().__init__(
            dimension=dimension,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = np.array([configuration[f"x{i}"] for i in range(self.dimension)])
        val = np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (x[:-1] - 1.0) ** 2.0)
        return {self.objectives_names[0]: float(val)}


class Michalewicz(SyntheticFunction):
    """
    The Michalewicz function is a multimodal function with d! local minima.
    See https://www.sfu.ca/~ssurjano/michal.html for details.
    """

    def __init__(self, dimension: int, m: float = 10.0):
        self.configuration_space = {
            f"x{i}": uniform(0, np.pi) for i in range(dimension)
        }
        super().__init__(
            dimension=dimension,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )
        self.m = m

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = np.array([configuration[f"x{i}"] for i in range(self.dimension)])
        i = np.arange(1, self.dimension + 1)
        val = -np.sum(np.sin(x) * (np.sin(i * x**2 / np.pi)) ** (2 * self.m))
        return {self.objectives_names[0]: float(val)}


class Ackley(SyntheticFunction):
    """
    The Ackley function is a multimodal function with a global optimum surrounded
    by many local optima.
    See https://www.sfu.ca/~ssurjano/ackley.html for details.
    """

    def __init__(self, dimension: int):
        self.configuration_space = {
            f"x{i}": uniform(-32.768, 32.768) for i in range(dimension)
        }
        super().__init__(
            dimension=dimension,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = np.array([configuration[f"x{i}"] for i in range(self.dimension)])
        term1 = -self.a * np.exp(-self.b * np.sqrt(np.mean(x**2)))
        term2 = -np.exp(np.mean(np.cos(self.c * x)))
        val = term1 + term2 + self.a + np.exp(1)
        return {self.objectives_names[0]: float(val)}


class Branin(SyntheticFunction):
    """
    The Branin function is a common benchmark function for optimization.
    See https://www.sfu.ca/~ssurjano/branin.html for details.
    """

    def __init__(self):
        self.configuration_space = {
            "x0": uniform(-5, 10),
            "x1": uniform(0, 15),
        }
        super().__init__(
            dimension=2,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )
        self.a = 1
        self.b = 5.1 / (4 * np.pi**2)
        self.c = 5 / np.pi
        self.r = 6
        self.s = 10
        self.t = 1 / (8 * np.pi)

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x0 = configuration["x0"]
        x1 = configuration["x1"]
        term1 = self.a * (x1 - self.b * x0**2 + self.c * x0 - self.r) ** 2
        term2 = self.s * (1 - self.t) * np.cos(x0)
        val = term1 + term2 + self.s
        return {self.objectives_names[0]: float(val)}


class Hartman(SyntheticFunction):
    """
    The Hartman function is a multimodal benchmark function.
    See https://www.sfu.ca/~ssurjano/hart3.html for details.
    """

    def __init__(self, dimension: int, alpha: np.ndarray, A: np.ndarray, P: np.ndarray):
        self.configuration_space = {f"x{i}": uniform(0, 1) for i in range(dimension)}
        super().__init__(
            dimension=dimension,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )
        self.alpha = alpha
        self.A = A
        self.P = P

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = np.array([configuration[f"x{i}"] for i in range(self.dimension)])
        outer_sum = 0
        for i in range(4):
            inner_sum = 0
            for j in range(self.dimension):
                inner_sum += self.A[i, j] * (x[j] - self.P[i, j]) ** 2
            outer_sum += self.alpha[i] * np.exp(-inner_sum)
        val = -outer_sum
        return {self.objectives_names[0]: float(val)}


class Hartman3(Hartman):
    def __init__(self):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [3.0, 10, 30],
                [0.1, 10, 35],
                [3.0, 10, 30],
                [0.1, 10, 35],
            ]
        )
        P = np.array(
            [
                [0.3689, 0.1170, 0.2673],
                [0.4699, 0.4387, 0.7470],
                [0.1091, 0.8732, 0.5547],
                [0.03815, 0.5743, 0.8828],
            ]
        )
        super().__init__(dimension=3, alpha=alpha, A=A, P=P)


class Hartman6(Hartman):
    def __init__(self):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = np.array(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )
        super().__init__(dimension=6, alpha=alpha, A=A, P=P)


class GoldsteinPrice(SyntheticFunction):
    """
    The Goldstein-Price function is a global optimization benchmark.
    See https://www.sfu.ca/~ssurjano/goldpr.html for details.
    """

    def __init__(self):
        self.configuration_space = {
            "x0": uniform(-2, 2),
            "x1": uniform(-2, 2),
        }
        super().__init__(
            dimension=2,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x0 = configuration["x0"]
        x1 = configuration["x1"]
        term1 = 1 + (x0 + x1 + 1) ** 2 * (
            19 - 14 * x0 + 3 * x0**2 - 14 * x1 + 6 * x0 * x1 + 3 * x1**2
        )
        term2 = 30 + (2 * x0 - 3 * x1) ** 2 * (
            18 - 32 * x0 + 12 * x0**2 + 48 * x1 - 36 * x0 * x1 + 27 * x1**2
        )
        val = term1 * term2
        return {self.objectives_names[0]: float(val)}


class Forrester(SyntheticFunction):
    """
    The Forrester function is a 1D function used for testing optimization algorithms.
    See https://www.sfu.ca/~ssurjano/forretal08.html for details.
    """

    def __init__(self):
        self.configuration_space = {
            "x0": uniform(0, 1),
        }
        super().__init__(
            dimension=1,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = configuration["x0"]
        val = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        return {self.objectives_names[0]: float(val)}


class SixHumpCamel(SyntheticFunction):
    """
    The Six-hump Camel function is a 2D function with six local minima, two of
    which are global.
    See https://www.sfu.ca/~ssurjano/camel6.html for details.
    """

    def __init__(self):
        self.configuration_space = {
            "x0": uniform(-3, 3),
            "x1": uniform(-2, 2),
        }
        super().__init__(
            dimension=2,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x0 = configuration["x0"]
        x1 = configuration["x1"]
        term1 = (4 - 2.1 * x0**2 + (x0**4) / 3) * x0**2
        term2 = x0 * x1
        term3 = (-4 + 4 * x1**2) * x1**2
        val = term1 + term2 + term3
        return {self.objectives_names[0]: float(val)}


class Rastrigin(SyntheticFunction):
    """
    The Rastrigin function is a highly multimodal function with a global minimum
    at the origin.
    See https://www.sfu.ca/~ssurjano/rastr.html for details.
    """

    def __init__(self, dimension: int):
        self.configuration_space = {
            f"x{i}": uniform(-5.12, 5.12) for i in range(dimension)
        }
        super().__init__(
            dimension=dimension,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = np.array([configuration[f"x{i}"] for i in range(self.dimension)])
        val = 10 * self.dimension + np.sum(
            x**2 - 10 * np.cos(2 * np.pi * x)
        )
        return {self.objectives_names[0]: float(val)}


class Eggholder(SyntheticFunction):
    """
    The Eggholder function is a difficult to optimize function with many local
    minima.
    See https://www.sfu.ca/~ssurjano/egg.html for details.
    """

    def __init__(self):
        self.configuration_space = {
            "x0": uniform(-512, 512),
            "x1": uniform(-512, 512),
        }
        super().__init__(
            dimension=2,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x0 = configuration["x0"]
        x1 = configuration["x1"]
        term1 = -(x1 + 47) * np.sin(np.sqrt(np.abs(x1 + x0 / 2 + 47)))
        term2 = -x0 * np.sin(np.sqrt(np.abs(x0 - (x1 + 47))))
        val = term1 + term2
        return {self.objectives_names[0]: float(val)}


class SumPowers(SyntheticFunction):
    """
    The Sum Powers function is a simple polynomial function.
    See https://www.sfu.ca/~ssurjano/sumpow.html for details.
    """

    def __init__(self, dimension: int):
        self.configuration_space = {
            f"x{i}": uniform(-1, 1) for i in range(dimension)
        }
        super().__init__(
            dimension=dimension,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = np.array([configuration[f"x{i}"] for i in range(self.dimension)])
        i = np.arange(1, self.dimension + 1)
        val = np.sum(np.abs(x) ** (i + 1))
        return {self.objectives_names[0]: float(val)}
