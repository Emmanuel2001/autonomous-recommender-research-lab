"""The mutable layer for the CURRENT task: within-user ranking of logged impressions.

WHY THIS REPLACES mle_strategy.py RATHER THAN EXTENDING IT
---------------------------------------------------------
`mle_strategy.py` holds EASE / ItemKNN / ALS / BPR-MF / Popularity -- RETRIEVAL models,
which score (user, item) pairs in order to pick a top-K from 7,583 candidates. This task
ranks ~6.9 impressions the platform already chose for that user. Measured 2026-08-27
with the official harness: our best retrieval model scores test primary 0.5340, BELOW
the trivial popularity baseline (0.5715) and 0.0606 below the official FM (0.5946).
Collaborative similarity is near-constant across a user's own impressions -- being
similar to their history is *why they were shown* -- so it carries almost no ordering
signal. EASE survives here only as a FEATURE, never as the model.

THE STARTING INCUMBENT IS THE OFFICIAL BASELINE
-----------------------------------------------
`Config(model="fm")` reproduces the kit's FM. The agent therefore starts level with the
baseline instead of 0.06 below it, and every action it proposes is measured against a
reference we have independently reproduced (test primary 0.5953 vs published 0.5946,
std 0.0008).

TWO CONSTRAINTS THE ORGANISERS MEASURED, ENCODED HERE
----------------------------------------------------
1. More static feature fields do not help (.5940 with 13 fields vs .5950 with 5) and
   more capacity does not help (k=8/16/32 -> .5895/.5902/.5887). The bottleneck is
   neither.
2. **Pure user-side first-order terms are mathematically inert.** Ranking happens
   within a user, so any term constant across that user's rows cannot change their
   order. A user bias vector is a no-op by construction, not by measurement. `FM` here
   therefore keeps user-side signal only in the interaction term.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

LOSSES = ("pointwise", "pairwise", "listwise")
MODELS = ("fm", "popularity", "random", "patch")

#: Which Config fields actually change a given model's output. Anything omitted is
#: inert for that model and must not enter the config hash, or the agent pays a full
#: fit for a provably identical result (STATE.md bug 4).
RELEVANT = {
    "fm": ("model", "loss", "fields", "k", "lr", "l2", "epochs", "batch", "patience",
           "neg_per_pos", "listwise_temp", "aux_weights", "use_random_log",
           "ensemble_seeds", "ensemble_combine"),
    "popularity": ("model", "pop_prior"),
    "random": ("model",),
    # A generated strategy is identified by its code and its params, nothing else --
    # so two textually identical patches dedup for free, exactly like a typed config.
    "patch": ("model", "patch_code", "patch_params"),
}


@dataclass
class Config:
    """The agent's action surface for the current task."""
    model: str = "fm"
    #: pointwise logloss is the kit's baseline. The metrics are RANKING metrics, and the
    #: organisers name loss replacement as their most promising unexplored direction.
    loss: str = "pointwise"
    fields: tuple = ("user_id", "video_id", "author_id", "tab", "dur_bucket")
    k: int = 16
    lr: float = 0.001
    l2: float = 1e-6
    epochs: int = 40
    batch: int = 8192
    patience: int = 4
    seed: int = 0
    # pairwise / listwise
    neg_per_pos: int = 1
    listwise_temp: float = 1.0
    # multi-objective: auxiliary label -> weight, added to the primary long_view loss
    aux_weights: dict = field(default_factory=dict)
    # opt-in unbiased validation log (kit direction 7). Never training data by default.
    use_random_log: bool = False
    pop_prior: float = 20.0
    #: Fit this many seeds (seed, seed+1, ...) and combine their scores. 1 = no ensemble.
    #: Costs n_seeds x the fit time, which is why it is a DECLARED action with a visible
    #: cost rather than a silent default.
    ensemble_seeds: int = 1
    ensemble_combine: str = "rank"          # "rank" | "score"
    #: set by ProposePatch. When model == "patch" the strategy is this source, run
    #: sandboxed via agent.patch.run_sandboxed rather than by anything in this file.
    patch_code: str = ""
    patch_params: dict = field(default_factory=dict)

    def hash(self) -> str:
        keep = RELEVANT[self.model]
        d = {k: v for k, v in asdict(self).items() if k in keep}
        return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":"),
                                         default=str).encode()).hexdigest()


def within_user_ranks(user_id: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Rank scores within each user, normalised to [0, 1], best = 1.

    Ensembling by RANK rather than by raw score because two models fit to different
    objectives put their logits on different scales -- averaging those lets the
    wider-scaled model dominate for no reason. Ranks are scale-free, and the metric only
    ever looks at within-user order anyway.
    """
    out = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(user_id, kind="stable")
    su = user_id[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(su)) + 1, len(su)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        sc = scores[idx]
        r = np.empty(len(sc))
        r[np.argsort(-sc, kind="stable")] = np.arange(len(sc))
        out[idx] = 1.0 - (r / max(len(sc) - 1, 1))
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ------------------------------------------------------------------ feature encoding
def encode(splits, edges, fields) -> tuple[dict, int]:
    """Categorical fields -> contiguous ids with per-field offsets, vocab from TRAIN.

    Mirrors the kit's `data.encode`: unseen values fall into a per-field UNK slot, and
    each field gets its own contiguous block so one embedding table serves all of them.
    """
    from core.official_data import dur_bucket

    def col(rows, name):
        if name == "dur_bucket":
            return dur_bucket(rows.duration_ms, edges)
        if name == "hourmin_bucket":
            # Hour of day, from the `hourmin` column the loader already carries. Advertised
            # by SetFields since the action space was written and never implemented, so any
            # agent that proposed it got an AttributeError. Measured worth: +0.001192 valid
            # (paired, 5 seeds, t=14.04, 5/5 wins) -- larger than any hyperparameter the
            # agent has found, and it takes the combined delta past the organisers'
            # epsilon=0.002 for the first time. One line, no join, no new data.
            return (np.asarray(rows.hourmin, dtype=np.int64) // 100)
        try:
            return getattr(rows, name)
        except AttributeError:
            # `agent.official_actions.SetFields` advertises fields the LOADER may not
            # carry -- 9 of its 14 raise here. A bare AttributeError told the agent
            # nothing, and the driver recorded it only as `strategy_crashed`, so a
            # proposal was burned with no way to learn why. Name the field and say what
            # is actually available.
            have = sorted(f for f in ("user_id", "video_id", "author_id", "tab",
                                      "duration_ms", "date", "hourmin")
                          if hasattr(rows, f))
            raise ValueError(
                f"field {name!r} is not carried by the data loader, so it cannot be "
                f"encoded. The loader provides {have} (plus 'dur_bucket', derived from "
                f"duration_ms). Adding a field means extending core.official_data.Rows "
                f"and the loader that fills it -- the action space alone cannot make a "
                f"column exist.") from None

    tr = splits["train"]
    vocabs, dims = [], []
    for f in fields:
        vals = np.unique(col(tr, f))
        vocabs.append({int(v): i for i, v in enumerate(vals)})
        dims.append(len(vals) + 1)                      # +1 UNK
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    enc = {}
    for name, rows in splits.items():
        X = np.empty((len(rows), len(fields)), dtype=np.int32)
        for i, f in enumerate(fields):
            c = col(rows, f)
            vmap, unk = vocabs[i], len(vocabs[i])
            X[:, i] = np.fromiter((vmap.get(int(v), unk) for v in c),
                                  np.int64, len(rows)) + offsets[i]
        enc[name] = X
    return enc, int(sum(dims))


# ------------------------------------------------------------------------------ FM
class FM:
    """Factorization Machine, Adam, numpy only. The kit's model, with a loss switch.

    `logits` and the pointwise gradient are the kit's arithmetic verbatim so that
    `loss="pointwise"` reproduces the official baseline. The pairwise and listwise
    branches reuse the same forward pass and only change how the per-row gradient
    coefficient is formed -- which keeps the comparison between losses honest.
    """

    def __init__(self, dim: int, cfg: Config):
        rng = np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.V = rng.normal(0, 0.01, (dim, cfg.k)).astype(np.float32)
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

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0]
                               for i in range(0, len(X), bs)]) if len(X) else np.zeros(0)

    def _apply(self, X, g, E, S):
        """One Adam step given the per-row gradient coefficient g."""
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.cfg.l2 * self.V; gW += self.cfg.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.cfg.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.cfg.lr * g.sum()

    def step_pointwise(self, X, y, w=None):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        if w is not None:
            g = (g * w).astype(np.float32)
        self._apply(X, g, E, S)
        p = sigmoid(z)
        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    def step_pairwise(self, Xp, Xn):
        """BPR: maximise sigma(z_pos - z_neg) for pairs drawn WITHIN a user."""
        B = len(Xp)
        zp, Ep, Sp = self.logits(Xp)
        zn, En, Sn = self.logits(Xn)
        d = sigmoid(-(zp - zn)) / B                     # dL/dz_pos = -sigma(-(zp-zn))
        self._apply(Xp, (-d).astype(np.float32), Ep, Sp)
        self._apply(Xn, (+d).astype(np.float32), En, Sn)
        return float(-np.mean(np.log(sigmoid(zp - zn) + 1e-9)))

    def step_listwise(self, X, y, group_starts, temp):
        """Softmax cross-entropy over each user's own impressions.

        Gradient of -sum_i y_i log softmax(z)_i is (p - y_norm) within each group, which
        is exactly the metric's own within-user framing.
        """
        z, E, S = self.logits(X)
        g = np.zeros(len(z), dtype=np.float64)
        loss = 0.0
        ngroups = 0
        for a, bnd in zip(group_starts[:-1], group_starts[1:]):
            zz = z[a:bnd] / temp
            yy = y[a:bnd]
            npos = yy.sum()
            if npos == 0 or npos == len(yy):
                continue                                 # no ordering signal in-group
            zz = zz - zz.max()
            p = np.exp(zz); p /= p.sum()
            tgt = yy / npos
            g[a:bnd] = (p - tgt) / temp
            loss += float(-(tgt * np.log(p + 1e-12)).sum())
            ngroups += 1
        if ngroups == 0:
            return 0.0
        g /= ngroups
        self._apply(X, g.astype(np.float32), E, S)
        return loss / ngroups


# ------------------------------------------------------------------------- the loop
def _group_bounds(users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous group boundaries for rows sorted by user. Returns (order, starts)."""
    order = np.argsort(users, kind="stable")
    su = users[order]
    starts = np.r_[0, np.flatnonzero(np.diff(su)) + 1, len(su)]
    return order, starts


def _pairs_within_user(y: np.ndarray, order: np.ndarray, starts: np.ndarray,
                       rng, neg_per_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Positive/negative row-index pairs drawn WITHIN each user.

    Across-user pairs would be meaningless: the metric never compares two users'
    impressions, so a model trained on cross-user pairs optimises a quantity nobody
    scores.
    """
    P, N = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]
        yy = y[idx]
        pos, neg = idx[yy == 1], idx[yy == 0]
        if pos.size == 0 or neg.size == 0:
            continue
        rep = np.repeat(pos, neg_per_pos)
        P.append(rep)
        N.append(rng.choice(neg, size=rep.size, replace=True))
    if not P:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(P), np.concatenate(N)


def run(splits, edges, cfg: Config, verbose: bool = False) -> tuple[dict, dict]:
    """Fit on train, return {split: scores aligned to that split's rows} + info.

    When cfg.ensemble_seeds > 1 this fits that many seeds and combines them. Seeds are
    cfg.seed, cfg.seed+1, ... so the ensemble is fully determined by the config and
    reproduces exactly.

    Scores are returned for valid AND test, but the caller decides what may be looked
    at. The strategy never sees an eval label -- `core.official_data.load_for_strategy`
    does not return one.
    """
    from dataclasses import replace
    if cfg.model == "fm" and cfg.ensemble_seeds > 1:
        if cfg.ensemble_combine not in ("rank", "score"):
            raise ValueError(f"unknown ensemble_combine {cfg.ensemble_combine!r}; "
                             f"choose rank|score")
        parts, infos = [], []
        for i in range(cfg.ensemble_seeds):
            sub = replace(cfg, seed=cfg.seed + i, ensemble_seeds=1)
            sc, inf = _run_single(splits, edges, sub, verbose)
            parts.append(sc); infos.append(inf)
        out = {}
        for name in parts[0]:
            if cfg.ensemble_combine == "rank":
                uid = splits[name].user_id
                out[name] = np.mean([within_user_ranks(uid, p[name]) for p in parts], axis=0)
            else:
                out[name] = np.mean([p[name] for p in parts], axis=0)
        return out, {"model": "fm", "loss": cfg.loss,
                     "ensemble_seeds": cfg.ensemble_seeds,
                     "ensemble_combine": cfg.ensemble_combine,
                     "member_valid_primary": [i.get("valid_primary_best") for i in infos],
                     "epochs_run": [i.get("epochs_run") for i in infos],
                     # per-member curves, kept so the report can show the seed spread
                     # that motivates ensembling at all. Reporting only, never read back.
                     "member_valid_history": [i.get("valid_history") for i in infos],
                     "member_train_loss_history": [i.get("train_loss_history")
                                                   for i in infos],
                     "deterministic_given_config": True}
    return _run_single(splits, edges, cfg, verbose)


def _run_single(splits, edges, cfg: Config, verbose: bool = False) -> tuple[dict, dict]:
    """One fit, one seed. The ensemble path above composes over this."""
    from core.metrics import primary

    if cfg.model == "random":
        rng = np.random.default_rng(cfg.seed)
        return ({s: rng.random(len(splits[s])) for s in splits},
                {"model": "random", "deterministic": False})

    if cfg.model == "popularity":
        tr = splits["train"]
        pos = np.zeros(int(tr.video_id.max()) + 2, np.float64)
        imp = np.zeros_like(pos)
        np.add.at(imp, tr.video_id, 1.0)
        np.add.at(pos, tr.video_id, tr.label)
        gmean = pos.sum() / imp.sum()
        rate = np.where(imp > 0, (pos + cfg.pop_prior * gmean) / (imp + cfg.pop_prior), gmean)
        out = {}
        for s, rows in splits.items():
            v = np.clip(rows.video_id, 0, len(rate) - 1)
            out[s] = rate[v]
        return out, {"model": "popularity", "prior": cfg.pop_prior, "deterministic": True}

    if cfg.loss not in LOSSES:
        raise ValueError(f"unknown loss {cfg.loss!r}; choose from {LOSSES}")

    enc, dim = encode(splits, edges, cfg.fields)
    tr = splits["train"]
    Xtr, ytr = enc["train"], tr.label
    m = FM(dim, cfg)
    rng = np.random.default_rng(cfg.seed)

    # auxiliary multi-objective rows (kit direction 3): extra pointwise targets, added
    # as separate weighted passes so the primary objective's gradient scale is untouched.
    aux = [(k, w, tr.aux[k]) for k, w in (cfg.aux_weights or {}).items()
           if w and k in tr.aux]

    order, starts = _group_bounds(tr.user_id)
    Xva, yva, uva = enc["valid"], splits["valid"].label, splits["valid"].user_id
    best, best_state, bad, hist, loss_hist = -1.0, None, 0, [], []

    for ep in range(1, cfg.epochs + 1):
        if cfg.loss == "pointwise":
            idx = rng.permutation(len(ytr))
            losses = [m.step_pointwise(Xtr[idx[i:i + cfg.batch]], ytr[idx[i:i + cfg.batch]])
                      for i in range(0, len(idx), cfg.batch)]
        elif cfg.loss == "pairwise":
            P, N = _pairs_within_user(ytr, order, starts, rng, cfg.neg_per_pos)
            perm = rng.permutation(len(P))
            losses = [m.step_pairwise(Xtr[P[perm[i:i + cfg.batch]]],
                                      Xtr[N[perm[i:i + cfg.batch]]])
                      for i in range(0, len(P), cfg.batch)]
        else:                                            # listwise
            losses = []
            gs = starts
            # batch whole users so each softmax group stays intact
            b = 0
            while b < len(gs) - 1:
                e = b
                rows = 0
                while e < len(gs) - 1 and rows < cfg.batch:
                    rows += gs[e + 1] - gs[e]; e += 1
                sel = order[gs[b]:gs[e]]
                local = np.r_[0, np.cumsum(np.diff(gs[b:e + 1]))]
                losses.append(m.step_listwise(Xtr[sel], ytr[sel], local, cfg.listwise_temp))
                b = e
        for name, w, ay in aux:
            idx = rng.permutation(len(ay))[: cfg.batch * 4]
            m.step_pointwise(Xtr[idx], ay[idx], w=np.full(len(idx), w, np.float32))

        va = primary(uva, yva, m.predict(Xva))
        hist.append(round(va["primary"], 6))
        # mean minibatch loss for this epoch -- the TRAINING half of the report's two
        # curves (the line above is the validation half). A non-finite mean is stored
        # as None on purpose: json.dumps writes bare NaN, which is not valid JSON and
        # would break every reader of the ledger it lands in.
        ml = float(np.mean(losses)) if len(losses) else 0.0
        loss_hist.append(round(ml, 6) if np.isfinite(ml) else None)
        if verbose:
            print(f"  ep {ep:2d} loss {ml:.4f} valid primary {va['primary']:.4f}")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        m.V, m.W, m.b = best_state
    return ({s: m.predict(enc[s]) for s in splits},
            {"model": "fm", "loss": cfg.loss, "epochs_run": len(hist),
             "valid_primary_best": round(best, 6), "valid_history": hist,
             "train_loss_history": loss_hist,
             "n_features": dim, "deterministic": False})
