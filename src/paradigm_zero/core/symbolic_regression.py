import numpy as np
import sympy as sp
from itertools import combinations


class SymbolicRegressor:
    def __init__(self, variables, max_terms=6, random_state=0):
        self.variables = variables
        self.max_terms = max_terms
        self.rng = np.random.default_rng(random_state)

    def _features(self, X):
        feats = {}
        names = []
        for v in self.variables:
            arr = np.asarray(X[v], dtype=float)
            feats[v] = arr
            names.append(v)
            feats[f"{v}**2"] = arr ** 2
            names.append(f"{v}**2")
            feats[f"sin({v})"] = np.sin(arr)
            names.append(f"sin({v})")
            feats[f"cos({v})"] = np.cos(arr)
            names.append(f"cos({v})")
        if len(self.variables) >= 2:
            for i in range(len(self.variables)):
                for j in range(i + 1, len(self.variables)):
                    a = np.asarray(X[self.variables[i]], dtype=float)
                    b = np.asarray(X[self.variables[j]], dtype=float)
                    key = f"{self.variables[i]}*{self.variables[j]}"
                    feats[key] = a * b
                    names.append(key)
        # constant
        feats["1"] = np.ones_like(np.asarray(X[self.variables[0]], dtype=float))
        names.append("1")
        return feats, names

    def fit(self, X, y, top_k=5):
        feats, names = self._features(X)
        M = np.column_stack([feats[n] for n in names])
        # solve least squares
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        # prune small
        threshold = 1e-2
        active = [(names[i], coef[i]) for i in range(len(names)) if abs(coef[i]) > threshold]
        # build expression
        terms = []
        for name, c in active:
            if name == "1":
                terms.append(f"{c:.6g}")
            else:
                terms.append(f"{c:.6g}*{name}")
        expr = " + ".join(terms) if terms else "0"
        # variants: try dropping each feature
        candidates = [expr]
        # add a couple random subsets
        for _ in range(top_k - 1):
            idx = self.rng.choice(len(names), size=min(self.max_terms, len(names)), replace=False)
            M2 = M[:, idx]
            c2, *_ = np.linalg.lstsq(M2, y, rcond=None)
            t2 = []
            for k, i in enumerate(idx):
                if abs(c2[k]) > threshold:
                    nm = names[i]
                    t2.append(f"{c2[k]:.6g}*{nm}" if nm != "1" else f"{c2[k]:.6g}")
            candidates.append(" + ".join(t2) if t2 else "0")
        # dedupe
        uniq = []
        for c in candidates:
            if c not in uniq:
                uniq.append(c)
        return uniq
