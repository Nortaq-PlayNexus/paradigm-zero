from paradigm_zero.core.universe import Universe, UniverseConfig
from paradigm_zero.core.symbolic_regression import SymbolicRegressor


def test_symbolic_regression_runs():
    cfg = UniverseConfig(noise=0.0, n_samples=200)
    u = Universe(cfg)
    d = u.generate()
    X = {v: d[v] for v in cfg.observed_vars}
    reg = SymbolicRegressor(cfg.observed_vars)
    cands = reg.fit(X, d["y"])
    assert len(cands) > 0
    assert any("x1" in c for c in cands)
