import numpy as np
import sympy as sp


def expression_similarity(expr_a, expr_b):
    try:
        a = sp.simplify(sp.sympify(expr_a) - sp.sympify(expr_b))
        return 1.0 if a == 0 else 0.0
    except Exception:
        return 0.0


class Benchmark:
    def __init__(self, universe):
        self.universe = universe

    def run(self, discovered_expr, test_data):
        tester = None
        from .testing import HypothesisTester
        tester = HypothesisTester(self.universe)
        gen = tester.generalization(discovered_expr, self.universe.generate(), test_data)
        rec = expression_similarity(discovered_expr, self.universe.ground_truth_str())
        return {
            "equation_recovery": rec,
            "prediction_accuracy_r2": gen["test_r2"],
            "generalization_gap": gen["gap"],
        }
