import numpy as np
from paradigm_zero.core.universe import Universe, UniverseConfig


def test_universe_generate():
    cfg = UniverseConfig(n_samples=50, noise=0.0)
    u = Universe(cfg)
    d = u.generate()
    assert "x1" in d and "y" in d
    assert len(d["y"]) == 50


def test_ground_truth():
    cfg = UniverseConfig()
    u = Universe(cfg)
    assert "x1" in u.ground_truth_str()
