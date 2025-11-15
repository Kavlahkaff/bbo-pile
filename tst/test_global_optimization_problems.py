import numpy as np

from benchmarks.global_optimization_problems.global_optimization_problems import (
    Rosenbrock,
    Michalewicz,
    Ackley,
    Branin,
    Hartman3,
    Hartman6,
    GoldsteinPrice,
    Forrester,
    SixHumpCamel,
    Eggholder,
    Rastrigin,
    SumPowers,
    StyblinskiTang,
)


def test_rosenbrock():
    # The Rosenbrock function has a global minimum of 0 at (1, 1, ..., 1).
    rosenbrock = Rosenbrock(dimension=5)
    configuration = {f"x{i}": 1.0 for i in range(5)}
    result = rosenbrock.objective_function(configuration)
    assert np.isclose(result["y"], 0.0)

    # Test another point
    configuration = {f"x{i}": 0.0 for i in range(5)}
    result = rosenbrock.objective_function(configuration)
    assert np.isclose(result["y"], 4.0)


def test_michalewicz():
    # The Michalewicz function has a global minimum that depends on the dimension.
    # For d=2, the minimum is approx -1.8013.
    michalewicz = Michalewicz(dimension=2)
    configuration = {"x0": 2.20, "x1": 1.57}
    result = michalewicz.objective_function(configuration)
    assert np.isclose(result["y"], -1.8013, atol=1e-3)


def test_ackley():
    # The Ackley function has a global minimum of 0 at (0, 0, ..., 0).
    ackley = Ackley(dimension=5)
    configuration = {f"x{i}": 0.0 for i in range(5)}
    result = ackley.objective_function(configuration)
    assert np.isclose(result["y"], 0.0)

    # Test another point
    configuration = {f"x{i}": 1.0 for i in range(5)}
    result = ackley.objective_function(configuration)
    assert not np.isclose(result["y"], 0.0)


def test_branin():
    # The Branin function has three global minima.
    branin = Branin()
    minima = [
        (-np.pi, 12.275),
        (np.pi, 2.275),
        (9.42478, 2.475),
    ]
    min_value = 0.397887

    for x0, x1 in minima:
        configuration = {"x0": x0, "x1": x1}
        result = branin.objective_function(configuration)
        assert np.isclose(result["y"], min_value, atol=1e-4)


def test_hartman3():
    # The Hartman3 function has a global minimum of -3.86278.
    hartman3 = Hartman3()
    configuration = {"x0": 0.114614, "x1": 0.555649, "x2": 0.852547}
    result = hartman3.objective_function(configuration)
    assert np.isclose(result["y"], -3.86278, atol=1e-5)


def test_hartman6():
    # The Hartman6 function has a global minimum of -3.32237.
    hartman6 = Hartman6()
    configuration = {
        "x0": 0.20169,
        "x1": 0.150011,
        "x2": 0.476874,
        "x3": 0.275332,
        "x4": 0.311652,
        "x5": 0.6573,
    }
    result = hartman6.objective_function(configuration)
    assert np.isclose(result["y"], -3.32237, atol=1e-5)

def test_goldstein_price():
    # Global minimum is at (0, -1) with value 3
    goldstein_price = GoldsteinPrice()
    config = {"x0": 0.0, "x1": -1.0}
    result = goldstein_price.objective_function(config)
    assert np.isclose(result["y"], 3.0, atol=1e-4)


def test_forrester():
    # The Forrester function has a global minimum of -6.02074 at x = 0.75725.
    forrester = Forrester()
    configuration = {"x0": 0.75725}
    result = forrester.objective_function(configuration)
    assert np.isclose(result["y"], -6.02074, atol=1e-5)

    # Test another point
    configuration = {"x0": 0.0}
    result = forrester.objective_function(configuration)
    assert np.isclose(result["y"], 3.027, atol=1e-3)


def test_six_hump_camel():
    # The Six-hump Camel function has two global minima.
    six_hump_camel = SixHumpCamel()
    minima = [
        (0.0898, -0.7126),
        (-0.0898, 0.7126),
    ]
    min_value = -1.0316

    for x0, x1 in minima:
        configuration = {"x0": x0, "x1": x1}
        result = six_hump_camel.objective_function(configuration)
        assert np.isclose(result["y"], min_value, atol=1e-4)


def test_eggholder():
    # The Eggholder function has a global minimum of -959.6407 at (512, 404.2319).
    eggholder = Eggholder()
    configuration = {"x0": 512, "x1": 404.2319}
    result = eggholder.objective_function(configuration)
    assert np.isclose(result["y"], -959.6407, atol=1e-4)


def test_rastrigin():
    # The Rastrigin function has a global minimum of 0 at (0, 0, ..., 0).
    rastrigin = Rastrigin(dimension=5)
    configuration = {f"x{i}": 0.0 for i in range(5)}
    result = rastrigin.objective_function(configuration)
    assert np.isclose(result["y"], 0.0)

    # Test another point
    configuration = {f"x{i}": 1.0 for i in range(5)}
    result = rastrigin.objective_function(configuration)
    assert not np.isclose(result["y"], 0.0)


def test_sum_powers():
    # The Sum Powers function has a global minimum of 0 at (0, 0, ..., 0).
    sum_powers = SumPowers(dimension=5)
    configuration = {f"x{i}": 0.0 for i in range(5)}
    result = sum_powers.objective_function(configuration)
    assert np.isclose(result["y"], 0.0)

    # Test another point
    sum_powers = SumPowers(dimension=2)
    configuration = {f"x{i}": 1.0 for i in range(2)}
    result = sum_powers.objective_function(configuration)
    assert np.isclose(result["y"], 2.0)


def test_styblinski_tang():
    # The Styblinski-Tang function has a global minimum of -39.16599 * d.
    styblinski_tang = StyblinskiTang(dimension=2)
    configuration = {f"x{i}": -2.903534 for i in range(2)}
    result = styblinski_tang.objective_function(configuration)
    assert np.isclose(result["y"], -39.16599 * 2, atol=1e-4)

    # Test another point
    configuration = {f"x{i}": 0.0 for i in range(2)}
    result = styblinski_tang.objective_function(configuration)
    assert np.isclose(result["y"], 0.0)
