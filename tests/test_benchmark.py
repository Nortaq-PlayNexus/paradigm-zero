from paradigm_zero.core.universe import Universe, UniverseConfig
from paradigm_zero.core.benchmark import Benchmark


def test_benchmark():
    cfg = UniverseConfig(noise=0.0)
    u = Universe(cfg)
    te = u.generate(n=50, seed=999)
    bm = Benchmark(u)
    m = bm.run("x1**2 + sin(x2) + 0.5*x1*x2", te)
    assert "equation_recovery" in m
    assert m["prediction_accuracy_r2"] > 0.9
