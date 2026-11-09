import numpy as np
import sympy as sp


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class HypothesisTester:
    def __init__(self, universe):
        self.universe = universe

    def test(self, expr_str, data):
        try:
            pred = self.universe.evaluate(expr_str, data)
        except Exception:
            return {"r2": -np.inf, "rmse": np.inf, "valid": False}
        y = np.asarray(data["y"], dtype=float)
        return {"r2": float(r2_score(y, pred)), "rmse": rmse(y, pred), "valid": True}

    def generalization(self, expr_str, train_data, test_data):
        tr = self.test(expr_str, train_data)
        te = self.test(expr_str, test_data)
        return {"train_r2": tr["r2"], "test_r2": te["r2"], "gap": tr["r2"] - te["r2"]}
