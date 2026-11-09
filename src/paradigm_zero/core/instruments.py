import numpy as np


class Instrument:
    def __init__(self, name, precision=0.05, bias=0.0):
        self.name = name
        self.precision = precision
        self.bias = bias

    def measure(self, values, seed=0):
        rng = np.random.default_rng(seed)
        return np.asarray(values, dtype=float) + self.bias + rng.normal(0, self.precision, size=np.shape(values))


class ObservationSystem:
    def __init__(self, instruments=None):
        self.instruments = instruments or []

    def observe(self, data):
        out = dict(data)
        for inst in self.instruments:
            if "y" in out:
                out["y_obs"] = inst.measure(out["y"])
        return out
