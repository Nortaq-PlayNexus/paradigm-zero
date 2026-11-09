import sympy as sp
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Candidate:
    expression: str
    r2: float = 0.0
    rmse: float = 0.0
    aic: float = 0.0
    gen_test_r2: float = 0.0


class HypothesisGenerator:
    def __init__(self, variables):
        self.variables = variables

    def templates(self):
        x, y = (sp.symbols(self.variables[0]), sp.symbols(self.variables[1])) if len(self.variables) >= 2 else (sp.symbols(self.variables[0]), None)
        cands = [
            f"{self.variables[0]}",
            f"{self.variables[0]}**2",
            f"sin({self.variables[0]})",
            f"{self.variables[0]} + {self.variables[1]}" if len(self.variables) > 1 else f"{self.variables[0]}",
            f"{self.variables[0]}**2 + {self.variables[1]}",
        ]
        if len(self.variables) > 1:
            cands.append(f"{self.variables[0]}**2 + sin({self.variables[1]})")
            cands.append(f"{self.variables[0]}**2 + sin({self.variables[1]}) + {self.variables[0]}*{self.variables[1]}")
        return cands
