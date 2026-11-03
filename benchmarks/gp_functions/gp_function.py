"""
GP-sample test functions of the gp_traj corpus as a syne-tune Blackbox.

The grid is loaded from <corpus>/functions/<fid>.npy (test functions are stored) and evaluated with the same bilinear
interpolation as gp_traj.Fn. With symmetry=g the blackbox is f_g(x) = f(g^-1 x): the transformed initial design
g(X[:n_init]) then gives exactly the teacher's y values, and fmin / fmax are unchanged.
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np
from syne_tune.blackbox_repository.blackbox import Blackbox, ObjectiveFunctionResult
from syne_tune.config_space import uniform

from open_optformer.gp_corpus import apply_sym, bilinear, inverse, log_regret_from_y

CONFIG_SPACE = {"x0": uniform(0.0, 1.0), "x1": uniform(0.0, 1.0)}


class GPGridFunction(Blackbox):
    def __init__(self, grid: np.ndarray, spec: dict | None = None, symmetry: int = 0):
        super().__init__(configuration_space=dict(CONFIG_SPACE), objectives_names=["y"])
        self.grid = np.asarray(grid, dtype=float)
        self.spec = spec
        self.symmetry = int(symmetry)
        self._ginv = inverse(self.symmetry)
        self.fmin = float(spec["fmin"]) if spec else float(self.grid.min())
        self.fmax = float(spec["fmax"]) if spec else float(self.grid.max())

    @classmethod
    def from_corpus(cls, corpus_dir, spec: dict, symmetry: int = 0):
        path = Path(corpus_dir) / "functions" / f"{spec['fid']}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} (only test grids are stored; regenerate others with gp_traj.sample_gp)")
        return cls(np.load(path).astype(float), spec, symmetry)

    def evaluate(self, X) -> np.ndarray:
        """f_g at points X (n x 2)."""
        return bilinear(self.grid, apply_sym(X, self._ginv))

    def log_regret(self, y) -> np.ndarray:
        return log_regret_from_y(y, self.fmin, self.fmax)

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = np.array([[configuration["x0"], configuration["x1"]]])
        return {"y": float(self.evaluate(x)[0])}
