"""The model. AGENT-EDITABLE -- this is the file to rewrite for an architecture change.

MINIMAL PROFILE. This copy implements ONLY the official baseline: a second-order
Factorization Machine with a pointwise logloss step. There is deliberately no pairwise (BPR),
no listwise, and no alternative architecture here -- if a run wants a ranking-aligned
objective or a different model, the agent has to write it. The `inherited` profile
(`research_workspace/`) keeps those branches for historical reproduction.

Seeded with the official baseline: a second-order Factorization Machine over one shared
embedding table, Adam, numpy only. The forward pass and the pointwise gradient are the
kit's arithmetic, so the seed configuration reproduces the published baseline.

`step_*` methods differ only in how the per-row gradient COEFFICIENT is formed; they share
`_apply`. Keeping that split makes a loss change a small, honest diff instead of a rewrite.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Second-order FM. `dim` is the total vocabulary size from features.encode()."""

    def __init__(self, dim: int, k: int = 16, lr: float = 0.001, l2: float = 1e-6,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.k, self.lr, self.l2 = int(k), float(lr), float(l2)
        self.V = rng.normal(0, 0.01, (dim, self.k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    # ------------------------------------------------------------------ forward
    def logits(self, X):
        """Returns (z, E, S). E and S are handed back because the backward pass needs
        them and recomputing them would double the cost of every step."""
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def predict(self, X, bs: int = 200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0]
                               for i in range(0, len(X), bs)]) if len(X) else np.zeros(0)

    # ----------------------------------------------------------------- backward
    def _apply(self, X, g, E, S):
        """One Adam step given the per-row gradient coefficient `g`."""
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV),
                            (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()

    def step_pointwise(self, X, y, w=None):
        """Logloss on individual impressions. The kit's baseline objective."""
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        if w is not None:
            g = (g * w).astype(np.float32)
        self._apply(X, g, E, S)
        p = sigmoid(z)
        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    def state(self):
        """A checkpoint. Copies, so the caller can keep it across further steps."""
        return (self.V.copy(), self.W.copy(), np.float32(self.b))

    def load(self, st):
        self.V, self.W, self.b = st
