from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
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

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def predict(self, X, bs: int = 200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0]
                               for i in range(0, len(X), bs)]) if len(X) else np.zeros(0)

    def _apply(self, X, g, E, S):
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
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        if w is not None:
            g = (g * w).astype(np.float32)
        self._apply(X, g, E, S)
        p = sigmoid(z)
        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    def step_bpr(self, X_pos, X_neg):
        B = len(X_pos)
        z_pos, E_pos, S_pos = self.logits(X_pos)
        z_neg, E_neg, S_neg = self.logits(X_neg)
        diff = z_pos - z_neg
        sig = sigmoid(-diff)

        g = (sig / B).astype(np.float32)

        # Concatenate positive and negative items and gradients
        X_combined = np.concatenate([X_pos, X_neg], axis=0)
        g_combined = np.concatenate([g, -g], axis=0)
        E_combined = np.concatenate([E_pos, E_neg], axis=0)
        S_combined = np.concatenate([S_pos, S_neg], axis=0)

        self._apply(X_combined, g_combined, E_combined, S_combined)
        return float(-np.mean(np.log(sigmoid(diff) + 1e-9)))

    def state(self):
        return (self.V.copy(), self.W.copy(), np.float32(self.b))

    def load(self, st):
        self.V, self.W, self.b = st
