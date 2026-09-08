"""The model. AGENT-EDITABLE -- this is the file rewritten for the architecture change.

DeepFM-style hybrid, pure numpy, CPU: one shared embedding table feeds BOTH a
second-order Factorization Machine and a small dense tower, and the two scores are added.

    score(x) = b + sum_f W[x_f] + 0.5*(||sum_f E_f||^2 - sum_f ||E_f||^2)
                 + w_out . relu(W2 . relu(W1 . concat(E) + b1) + b2)

WHY (the hypothesis this file implements)
-----------------------------------------
The parent's dev primary is flat at 0.603-0.604 from epoch 4 while its TRAIN primary keeps
climbing. Either the bilinear form has run out of expressible structure, or the columns
have run out of signal. A nonlinear head on IDENTICAL inputs, IDENTICAL pairs and an
identical sampler separates those two. torch is not installed in this sandbox, so the
tower and its backward pass are written by hand in numpy.

WHAT IS DELIBERATELY UNCHANGED FROM THE PARENT
----------------------------------------------
  * the FM forward (linear + second-order term) and the BPR sign convention;
  * the embedding/linear optimiser: full-array Adam with l2 folded into the gradient,
    and plain SGD on the global bias -- exactly the parent's arithmetic;
  * the init order of the RNG: V is drawn FIRST, so V's initial values are identical to
    the parent's for the same seed. The tower's weights are drawn afterwards. The
    comparison to the champion is therefore one-factor.

STRUCTURE OF THE BACKWARD PASS
------------------------------
`_forward` hands back a cache (E, S, and the tower's activations/pre-activations),
`_grads` turns a per-row gradient coefficient `g` into every parameter gradient WITHOUT
touching the parameters, and `_apply` = `_grads` + Adam. The embedding gradient is the sum
of two terms:

    dL/dE_{r,f,:} = g_r * (S_r - E_{r,f,:})        # FM second order
                  + reshape(dL/dconcat(E))_{r,f,:}  # tower, via W1^T

and both are scattered with the same `np.add.at`, so a feature id shared by the two rows
of a within-user pair still cancels in the LINEAR table (g_pos + g_neg = 0) exactly as in
the parent, while its embedding keeps learning.

Splitting `_grads` out of `_apply` is what makes `grad_check` cheap and honest: it
finite-differences the SAME code path the optimiser uses, on a float64 copy of the model
so the check is not swamped by float32 noise. Hand-written backprop that is silently wrong
is the main risk of this candidate, so the check is run once per fit and recorded.

Sign convention (unchanged): loss = -log sigmoid(z+ - z-), dL/dz+ = -sigmoid(-(z+ - z-)),
dL/dz- = +sigmoid(-(z+ - z-)). A flipped sign shows up as dev primary <= 0.5 in epoch 1.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Second-order FM plus an optional dense tower over the same embeddings.

    `dim` is the total vocabulary size from features.encode(); `n_fields` is the number of
    encoded columns (the tower's input width is n_fields * k). `mlp_hidden = ()` disables
    the tower and recovers the parent model exactly.
    """

    def __init__(self, dim: int, k: int = 16, lr: float = 0.001, l2: float = 1e-6,
                 seed: int = 0, n_fields: int | None = None,
                 mlp_hidden=(64, 32), mlp_lr: float | None = None,
                 mlp_wd: float = 0.0):
        rng = np.random.default_rng(seed)
        self.k, self.lr, self.l2 = int(k), float(lr), float(l2)
        self.dtype = np.float32
        # --- FM half. Drawn FIRST so V matches the parent bit-for-bit for a given seed.
        self.V = rng.normal(0, 0.01, (dim, self.k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

        # --- dense tower half
        hidden = tuple(int(h) for h in (mlp_hidden or ()) if int(h) > 0)
        self.n_fields = int(n_fields) if n_fields else 0
        self.use_mlp = bool(hidden) and self.n_fields > 0
        self.hidden = hidden if self.use_mlp else ()
        self.mlp_lr = float(self.lr if mlp_lr is None else mlp_lr)
        self.mlp_wd = float(mlp_wd)
        self.Ws, self.bs = [], []
        if self.use_mlp:
            fan = self.n_fields * self.k
            for h in hidden:
                # He init. With embeddings at 0.01 scale the tower's initial contribution
                # to the logit is ~1e-2 of the FM's, so epoch 1 behaves like the parent.
                self.Ws.append(rng.normal(0.0, np.sqrt(2.0 / fan), (fan, h)).astype(np.float32))
                self.bs.append(np.zeros(h, dtype=np.float32))
                fan = h
            self.w_out = rng.normal(0.0, np.sqrt(2.0 / fan), fan).astype(np.float32)
        else:
            self.w_out = np.zeros(0, dtype=np.float32)
        self._adam = {}                      # dense param name -> [m, v]

    # ------------------------------------------------------------------- params
    def dense_params(self) -> dict:
        """Name -> the actual ndarray object (updates are in place)."""
        d = {}
        if not self.use_mlp:
            return d
        for i, (Wl, bl) in enumerate(zip(self.Ws, self.bs)):
            d[f"W{i}"] = Wl
            d[f"b{i}"] = bl
        d["w_out"] = self.w_out
        return d

    def n_params(self) -> int:
        n = self.V.size + self.W.size + 1
        for p in self.dense_params().values():
            n += p.size
        return int(n)

    # ------------------------------------------------------------------ forward
    def _forward(self, X, cache: bool = True):
        """Returns (z, cache_or_None). The cache holds everything the backward pass
        needs, so nothing is recomputed."""
        E = self.V[X]                                   # (B, F, k)
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        z = self.b + self.W[X].sum(1) + inter
        c = {"E": E, "S": S} if cache else None
        if self.use_mlp:
            if X.shape[1] != self.n_fields:
                raise ValueError(
                    f"tower was built for {self.n_fields} fields but X has {X.shape[1]}")
            B = X.shape[0]
            h = E.reshape(B, self.n_fields * self.k)
            acts, pre = [h], []
            for Wl, bl in zip(self.Ws, self.bs):
                a = h @ Wl + bl
                pre.append(a)
                h = np.maximum(a, 0)
                acts.append(h)
            z = z + h @ self.w_out
            if cache:
                c["acts"], c["pre"] = acts, pre
        return z, c

    def logits(self, X):
        """Back-compatible shim: (z, E, S)."""
        z, c = self._forward(X, cache=True)
        return z, c["E"], c["S"]

    def predict(self, X, bs: int = 32_768):
        if len(X) == 0:
            return np.zeros(0)
        return np.concatenate([self._forward(X[i:i + bs], cache=False)[0]
                               for i in range(0, len(X), bs)])

    # ----------------------------------------------------------------- backward
    def _grads(self, X, g, cache, l2: float | None = None) -> dict:
        """Every parameter gradient for a per-row coefficient `g`. Applies nothing."""
        l2 = self.l2 if l2 is None else float(l2)
        E, S = cache["E"], cache["S"]
        B = X.shape[0]
        gr = {}
        dE = g[:, None, None] * (S[:, None, :] - E)          # FM second-order term
        if self.use_mlp:
            acts, pre = cache["acts"], cache["pre"]
            gr["w_out"] = acts[-1].T @ g
            dh = g[:, None] * self.w_out[None, :]
            for i in range(len(self.Ws) - 1, -1, -1):
                da = dh * (pre[i] > 0)
                gr[f"W{i}"] = acts[i].T @ da
                gr[f"b{i}"] = da.sum(0)
                dh = da @ self.Ws[i].T
            dE = dE + dh.reshape(B, self.n_fields, self.k)   # tower term into the table
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, dE)
        if l2:
            gV += l2 * self.V
            gW += l2 * self.W
        gr["V"], gr["W"], gr["b"] = gV, gW, float(g.sum())
        return gr

    def _apply(self, X, g, E, S, cache=None):
        """One optimiser step. `cache` carries the tower activations when there is one."""
        if cache is None:
            cache = {"E": E, "S": S}
        grads = self._grads(X, g, cache)
        self._step(grads)

    def _step(self, grads: dict):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        # Embedding + linear: identical arithmetic to the parent.
        for P, G, M, Vv in ((self.V, grads["V"], self.mV, self.vV),
                            (self.W, grads["W"], self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        # Dense tower: own Adam moments, decoupled weight decay.
        for name, P in self.dense_params().items():
            G = grads[name]
            st = self._adam.get(name)
            if st is None:
                st = [np.zeros_like(P), np.zeros_like(P)]
                self._adam[name] = st
            M, Vv = st
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.mlp_lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
            if self.mlp_wd:
                P -= self.mlp_lr * self.mlp_wd * P
        self.b -= self.lr * grads["b"]

    # -------------------------------------------------------------------- steps
    def step_pointwise(self, X, y, w=None):
        """Logloss on individual impressions. The kit's baseline objective."""
        B = len(y)
        z, c = self._forward(X, cache=True)
        g = ((sigmoid(z) - y) / B).astype(self.dtype)
        if w is not None:
            g = (g * w).astype(self.dtype)
        self._apply(X, g, c["E"], c["S"], cache=c)
        p = sigmoid(z)
        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    def step_bpr(self, Xp, Xn):
        """Pairwise BPR on (positive row, negative row) from the SAME user."""
        B = len(Xp)
        if B == 0 or len(Xn) != B:
            raise ValueError(f"step_bpr needs aligned non-empty pairs, got {len(Xp)}/{len(Xn)}")
        X = np.concatenate((Xp, Xn), axis=0)
        z, c = self._forward(X, cache=True)
        d = z[:B] - z[B:]
        s = sigmoid(-d)
        g = np.empty(2 * B, dtype=self.dtype)
        g[:B] = (-s / B).astype(self.dtype)          # push the positive row UP
        g[B:] = (s / B).astype(self.dtype)           # push the negative row DOWN
        self._apply(X, g, c["E"], c["S"], cache=c)
        return float(np.mean(np.log1p(np.exp(-np.clip(d, -30, 30)))))

    # -------------------------------------------------------------- grad check
    def as_float64(self):
        """A parameter copy in float64. For finite differences only -- no optimiser state."""
        m = FM.__new__(FM)
        m.__dict__.update(self.__dict__)
        m.dtype = np.float64
        m.V = self.V.astype(np.float64)
        m.W = self.W.astype(np.float64)
        m.b = np.float64(self.b)
        m.Ws = [w.astype(np.float64) for w in self.Ws]
        m.bs = [b.astype(np.float64) for b in self.bs]
        m.w_out = self.w_out.astype(np.float64)
        m._adam = {}
        return m

    def grad_check(self, Xp, Xn, rng, per_param: int = 3, eps: float = 1e-5) -> dict:
        """Central finite differences of the BPR loss against `_grads`, on a float64 copy.

        The model itself is untouched. Returns a telemetry dict; the caller decides what to
        do with it. l2 is excluded on both sides so the comparison is of the data gradient.
        """
        m = self.as_float64()
        B = len(Xp)
        X = np.concatenate((Xp, Xn), axis=0)

        def loss():
            zz = m._forward(X, cache=False)[0]
            dd = zz[:B] - zz[B:]
            return float(np.mean(np.log1p(np.exp(-np.clip(dd, -30, 30)))))

        z, c = m._forward(X, cache=True)
        d = z[:B] - z[B:]
        s = sigmoid(-d)
        g = np.concatenate((-s / B, s / B))
        grads = m._grads(X, g, c, l2=0.0)

        targets = {"V": m.V, "W": m.W}
        targets.update(m.dense_params())
        checks, rels = [], []
        for name, P in targets.items():
            G = grads[name]
            flat = P.reshape(-1)
            gflat = np.asarray(G).reshape(-1)
            nz = np.flatnonzero(np.abs(gflat) > 1e-12)
            if nz.size == 0:
                checks.append({"param": name, "status": "all_zero_grad"})
                continue
            pick = rng.choice(nz, size=int(min(per_param, nz.size)), replace=False)
            for j in np.atleast_1d(pick):
                j = int(j)
                old = float(flat[j])
                flat[j] = old + eps; lp = loss()
                flat[j] = old - eps; lm = loss()
                flat[j] = old
                num = (lp - lm) / (2.0 * eps)
                ana = float(gflat[j])
                rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
                rels.append(rel)
                checks.append({"param": name, "idx": j,
                               "analytic": float(f"{ana:.6g}"),
                               "numeric": float(f"{num:.6g}"),
                               "rel_err": float(f"{rel:.3g}")})
        if not rels:
            return {"ran": True, "passed": False, "note": "no non-zero gradients sampled",
                    "checks": checks}
        rels = np.asarray(rels)
        max_rel = float(rels.max())
        med_rel = float(np.median(rels))
        frac_ok = float(np.mean(rels < 1e-4))
        return {"ran": True,
                "n_checked": int(rels.size),
                "max_rel_err": float(f"{max_rel:.3g}"),
                "median_rel_err": float(f"{med_rel:.3g}"),
                "frac_rel_err_below_1e-4": round(frac_ok, 3),
                # ReLU kinks can make a single sampled coordinate disagree, so the pass
                # rule is "max is tight OR essentially all coordinates are tight".
                "passed": bool(max_rel < 1e-4 or (frac_ok >= 0.9 and med_rel < 1e-6)),
                "eps": eps,
                "checks": checks}

    # ------------------------------------------------------------- checkpointing
    def state(self):
        return {"V": self.V.copy(), "W": self.W.copy(), "b": np.float32(self.b),
                "Ws": [w.copy() for w in self.Ws],
                "bs": [b.copy() for b in self.bs],
                "w_out": self.w_out.copy()}

    def load(self, st):
        if isinstance(st, tuple):                       # parent's 3-tuple format
            self.V, self.W, self.b = st
            return
        self.V, self.W, self.b = st["V"], st["W"], st["b"]
        self.Ws = [w.copy() for w in st["Ws"]]
        self.bs = [b.copy() for b in st["bs"]]
        self.w_out = st["w_out"].copy()
        self._adam = {}
