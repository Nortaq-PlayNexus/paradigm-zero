from paradigm_zero.core.universe import Universe, UniverseConfig
from paradigm_zero.core.testing import HypothesisTester


def test_tester():
    cfg = UniverseConfig(noise=0.0, n_samples=100)
    u = Universe(cfg)
    tr = u.generate()
    te = u.generate(n=50, seed=999)
    t = HypothesisTester(u)
    res = t.test("x1**2 + sin(x2) + 0.5*x1*x2", tr)
    assert res["valid"] is True
    assert res["r2"] > 0.9
