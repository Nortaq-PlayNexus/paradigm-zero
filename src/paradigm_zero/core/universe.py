import numpy as np
import sympy as sp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class UniverseConfig:
    name: str = "Universe A"
    seed: int = 42
    n_samples: int = 200
    noise: float = 0.1
    hidden_equation: str = "x1**2 + sin(x2) + 0.5*x1*x2"
    observed_vars: List[str] = field(default_factory=lambda: ["x1", "x2"])
    hidden_vars: List[str] = field(default_factory=lambda: ["h"])
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {"x1": (-3.0, 3.0), "x2": (-3.0, 3.0)})
    test_seed: int = 999


class Universe:
    def __init__(self, config: UniverseConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.symbols = {v: sp.symbols(v) for v in config.observed_vars + config.hidden_vars}
        self.expr = sp.sympify(config.hidden_equation)
        self.f = sp.lambdify(list(self.symbols.values()), self.expr, "numpy")

    def _sample_inputs(self, n, seed):
        rng = np.random.default_rng(seed)
        data = {}
        for v in self.config.observed_vars:
            lo, hi = self.config.ranges.get(v, (-1.0, 1.0))
            data[v] = rng.uniform(lo, hi, size=n)
        # hidden variable derived
        if "h" in self.config.hidden_vars:
            if "x1" in data and "x2" in data:
                data["h"] = data["x1"] * data["x2"]
            else:
                data["h"] = rng.normal(size=n)
        return data

    def generate(self, n=None, seed=None, include_hidden_obs=False):
        n = n or self.config.n_samples
        seed = self.config.seed if seed is None else seed
        data = self._sample_inputs(n, seed)
        args = [data[v] for v in self.config.observed_vars + self.config.hidden_vars if v in data]
        # build full arg list matching symbols order
        sym_order = self.config.observed_vars + self.config.hidden_vars
        full = []
        for v in sym_order:
            if v in data:
                full.append(data[v])
            else:
                full.append(np.zeros(n))
        y = np.asarray(self.f(*full), dtype=float).ravel()
        y = y + self.rng.normal(0, self.config.noise, size=n) if seed == self.config.seed else y + np.random.default_rng(seed + 7).normal(0, self.config.noise, size=n)
        obs = {v: data[v] for v in self.config.observed_vars}
        obs["y"] = y
        if include_hidden_obs:
            for v in self.config.hidden_vars:
                obs[v] = data[v]
        return obs

    def ground_truth_str(self):
        return str(self.expr)

    def evaluate(self, expr_str, data):
        e = sp.sympify(expr_str)
        f = sp.lambdify([sp.symbols(v) for v in self.config.observed_vars], e, "numpy")
        args = [data[v] for v in self.config.observed_vars]
        return np.asarray(f(*args), dtype=float).ravel()
