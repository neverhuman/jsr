#!/usr/bin/env python3
"""h45-wins: post-hoc EMA coefficients (Karras et al. 2024, exponential form) for A1's five EMA snapshots.
Least squares between the target EMA kernel (time constant tau, at step T) and a combination of the trained EMA
kernels (decay 0.9995, lerp after every step), coefficients >= 0 summing to 1."""
import json
import numpy as np
from scipy.optimize import minimize
beta, T = 0.9995, 49000
ts = [40000, 42500, 45000, 47500, 49000]
s = np.arange(0, T + 1, dtype=np.float64)
W = np.stack([np.where(s <= t, (1 - beta) * beta ** (t - s), 0.0) for t in ts])
out = {}
for tau in (4000, 8000):
    b2 = 1 - 1 / tau
    w = (1 - b2) * b2 ** (T - s)
    G, g = W @ W.T, W @ w
    r = minimize(lambda c: float(c @ G @ c - 2 * g @ c), np.ones(5) / 5, method="SLSQP", bounds=[(0, None)] * 5,
                 constraints=[{"type": "eq", "fun": lambda c: c.sum() - 1}], options={"ftol": 1e-18, "maxiter": 2000})
    c = np.clip(r.x, 0, None); c = c / c.sum()
    out[str(tau)] = {"steps": ts, "coef": [round(float(x), 6) for x in c],
                     "relerr": float(np.linalg.norm(c @ W - w) / np.linalg.norm(w))}
json.dump(out, open("/mnt/xbabe3-external/jsr/h45-wins/recipes/posthoc_coefs.json", "w"), indent=1)
print(json.dumps(out))
