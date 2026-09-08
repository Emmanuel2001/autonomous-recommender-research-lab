"""The model. AGENT-EDITABLE -- this is the file to rewrite for an architecture change.

Second-order Factorization Machine over one shared embedding table, Adam, numpy only.
The forward pass is unchanged from the official baseline, so the pointwise configuration
still reproduces the published number.

THIS EDIT adds exactly one new gradient method: `step_pairwise`, a within-user BPR step.
The forward pass, `_apply` (the Adam update) and `step_pointwise` are byte-identical to
the parent, so the diff is the objective and nothing else.

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

    def step_pairwise(self, Xp, Xn):
        """BPR on (positive, negative) impression pairs drawn from the SAME user.

        Loss per pair is -log sigmoid(z_p - z_n), so
            dL/dz_p = -sigmoid(-(z_p - z_n)),   dL/dz_n = +sigmoid(-(z_p - z_n)).

        Both members of the pair are stacked into ONE feature matrix and pushed through
        the shared `_apply` with signed coefficients, so a pair contributes a single Adam
        update and feature ids shared by the two rows (user_id above all) cancel exactly
        inside `np.add.at`. The coefficients sum to zero, so the global bias -- the part
        of the score the within-user metric quotients out -- receives no gradient.
        """
        B = len(Xp)
        if B == 0 or len(Xn) != B:
            raise ValueError(f"step_pairwise needs equal, non-empty pair sides; "
                             f"got {len(Xp)} positives and {len(Xn)} negatives")
        X = np.concatenate([Xp, Xn], axis=0)
        z, E, S = self.logits(X)
        d = z[:B] - z[B:]
        s = sigmoid(-d).astype(np.float32) / np.float32(B)
        g = np.concatenate([-s, s]).astype(np.float32)
        self._apply(X, g, E, S)
        return float(np.mean(np.logaddexp(0.0, -d)))

    def state(self):
        """A checkpoint. Copies, so the caller can keep it across further steps."""
        return (self.V.copy(), self.W.copy(), np.float32(self.b))

    def load(self, st):
        self.V, self.W, self.b = st
