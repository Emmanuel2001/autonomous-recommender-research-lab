"""THE MUTABLE LAYER. This is the only file the agent rewrites.

It may see: TRAIN rows (features + labels) and the user lists to rank for.
It may NOT see: any validation or test label. `core.splits.load_for_strategy`
enforces that by never returning them.

Contract with the scorer:
  * produce a (n_users, N) array of RAW video_ids, ranked best-first, N >= 50
  * no duplicates within a user's row
  * if cfg.mask == "exclude_train_seen", no train-seen item may appear -- the
    scorer VERIFIES this, so getting it wrong is a refusal, not a silent penalty
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch

LABEL = "is_click"
SIGNAL_COLS = ("is_click", "is_like", "is_follow", "is_comment", "is_forward",
               "is_hate", "long_view", "is_profile_enter", "play_ratio")
# Resolved through core.splits.DATA so an isolated process reads the prepared root
# (symlinks to permitted files only) rather than the real dataset directory.
def _video_features_path():
    from core.splits import DATA
    return DATA / "video_features_basic_pure.csv"


# Which Config fields actually change each model's output. Anything omitted here is
# inert for that model, so it must not enter the config hash. train_dev_boundary is
# absent everywhere because nothing consumes train_dev yet -- add it when something does.
RELEVANT_FIELDS = {
    "ease":       ("model", "mask", "n_ranked", "use_log_random", "ease_lambda",
                   "signal_weights", "content_alpha", "content_fields", "content_topk"),
    "itemknn":    ("model", "mask", "n_ranked", "use_log_random", "knn_k"),
    "als":        ("model", "mask", "n_ranked", "use_log_random", "dim", "als_iters",
                   "als_alpha", "reg_als"),
    "bpr_mf":     ("model", "mask", "n_ranked", "use_log_random", "dim", "epochs",
                   "lr", "reg", "neg_per_pos", "batch_size"),
    "popularity": ("model", "mask", "n_ranked", "use_log_random"),
}


@dataclass
class Config:
    """The agent's action surface. Every field here is something it may change."""
    model: str = "bpr_mf"
    dim: int = 64
    epochs: int = 3
    lr: float = 0.05
    reg: float = 1e-5
    neg_per_pos: int = 4
    batch_size: int = 8192
    train_dev_boundary: int = 20220418      # strategy parameter, not core
    mask: str = "exclude_train_seen"
    n_ranked: int = 50
    seed: int = 0
    # append log_random (1.19M randomly-exposed rows) to training data.
    # DEFAULT OFF -- a rules question, see core.splits.load_for_strategy
    use_log_random: bool = False
    # multi-signal X: which feedback signals build the interaction matrix, and how
    # strongly. Only is_click is SCORED, but 11 other signals exist and the brief
    # explicitly endorses learning from them jointly.
    signal_weights: dict = field(default_factory=lambda: {"is_click": 1.0})
    # content blending: B = (1-a)*B_ease + a*B_content from video side-features
    content_alpha: float = 0.0
    content_fields: tuple = ("tag", "upload_type", "music_type", "video_type", "duration_bin")
    content_topk: int = 50
    # model-specific
    ease_lambda: float = 250.0
    knn_k: int = 200
    als_iters: int = 8
    als_alpha: float = 40.0
    reg_als: float = 10.0

    def hash(self) -> str:
        """CANONICAL hash: seed excluded, and only fields RELEVANT to this model.

        Seed excluded because the config identifies the experiment and the seed is a
        replicate -- two seeds of one config must share a config_hash or seed
        variance cannot be separated from between-config variance.

        Model-relevant fields only because otherwise a change to an irrelevant
        parameter yields a NEW hash and a full evaluation for a provably identical
        result. Measured on a 14-iteration random-proposer run: 5 of 6 early
        iterations were no-ops of exactly this kind (ChangeKnnK/ChangeDim/ChangeLr/
        ChangeAlsIters applied to an EASE config), each costing a ~20s refit plus a
        gate call. Canonicalising turns those into free duplicate rejections.
        """
        keep = RELEVANT_FIELDS[self.model]
        d = {k: v for k, v in asdict(self).items() if k in keep}
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()


class BprMF:
    """Implicit-feedback MF with a BPR objective.

    Chosen deliberately: it is genuinely stochastic in BOTH initialisation and
    negative sampling, which is what makes seed variance measurable. ALS or a
    truncated SVD would be near-deterministic and would understate the noise floor.
    """

    def __init__(self, cfg: Config, n_users: int, n_items: int, device: str = "cpu"):
        self.cfg, self.n_users, self.n_items = cfg, n_users, n_items
        self.device = torch.device(device)
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        scale = 0.1 / np.sqrt(cfg.dim)
        self.U = (torch.randn(n_users, cfg.dim, generator=g) * scale).to(self.device).requires_grad_(True)
        self.V = (torch.randn(n_items, cfg.dim, generator=g) * scale).to(self.device).requires_grad_(True)
        self.bi = torch.zeros(n_items, device=self.device, requires_grad=True)

    def fit(self, u_pos: np.ndarray, i_pos: np.ndarray) -> dict:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)          # drives negative sampling
        opt = torch.optim.Adam([self.U, self.V, self.bi], lr=cfg.lr, weight_decay=cfg.reg)
        n = len(u_pos)
        hist = []
        for ep in range(cfg.epochs):
            order = rng.permutation(n)
            tot, nb = 0.0, 0
            for s in range(0, n, cfg.batch_size):
                idx = order[s: s + cfg.batch_size]
                bu = torch.from_numpy(u_pos[idx].astype(np.int64)).to(self.device)
                bi = torch.from_numpy(i_pos[idx].astype(np.int64)).to(self.device)
                # uniform negatives; a collision with a true positive is rare and is
                # standard BPR practice to tolerate rather than reject
                neg = rng.integers(0, self.n_items, size=(len(idx), cfg.neg_per_pos))
                bn = torch.from_numpy(neg.astype(np.int64)).to(self.device)
                pu = self.U[bu]                                    # (B, d)
                s_pos = (pu * self.V[bi]).sum(1) + self.bi[bi]      # (B,)
                s_neg = torch.einsum("bd,bkd->bk", pu, self.V[bn]) + self.bi[bn]
                loss = -torch.nn.functional.logsigmoid(
                    s_pos.unsqueeze(1) - s_neg).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.detach()); nb += 1
            hist.append(round(tot / max(nb, 1), 6))
        return {"loss_per_epoch": hist}

    @torch.no_grad()
    def rank(self, users: np.ndarray, train_seen: dict[int, np.ndarray] | None,
             n_ranked: int, batch: int = 4096) -> np.ndarray:
        """Full-catalog top-N. Masking by -inf guarantees the scorer's two
        structural invariants: no duplicates, and no masked item present."""
        out = np.empty((len(users), n_ranked), dtype=np.int64)
        U, V, bi = self.U.detach(), self.V.detach(), self.bi.detach()
        for s in range(0, len(users), batch):
            ub = users[s: s + batch]
            S = (U[torch.from_numpy(ub.astype(np.int64)).to(self.device)] @ V.T + bi)
            if train_seen is not None:
                # one scatter for the whole batch; a per-user loop of index
                # assignments is ~20x slower and this runs on every iteration
                rows, cols = [], []
                for r, u in enumerate(ub):
                    seen = train_seen.get(int(u))
                    if seen is not None and seen.size:
                        rows.append(np.full(seen.size, r, dtype=np.int64)); cols.append(seen)
                if rows:
                    ri = torch.from_numpy(np.concatenate(rows)).to(self.device)
                    ci = torch.from_numpy(np.concatenate(cols).astype(np.int64)).to(self.device)
                    S[ri, ci] = float("-inf")
            top = torch.topk(S, n_ranked, dim=1).indices.cpu().numpy().astype(np.int64)
            out[s: s + len(ub)] = top
        return out


# --------------------------------------------------------------- item-item models
class _ItemItem:
    """Base for models scoring as  X_user_row @ B  where B is item-item.

    Cold users (no train interactions) get a zero row, which would rank
    arbitrarily -- they fall back to popularity. 3.1% of eval users are cold.
    """
    def __init__(self, cfg: Config, n_users: int, n_items: int, device: str = "cpu"):
        self.cfg, self.n_users, self.n_items = cfg, n_users, n_items
        self.B = None; self.pop = None; self.X = None

    def _matrix(self, u_pos, i_pos):
        from scipy.sparse import csr_matrix
        X = csr_matrix((np.ones(len(u_pos), np.float32), (u_pos, i_pos)),
                       shape=(self.n_users, self.n_items))
        X.data[:] = 1.0                      # BINARY: duplicate clicks are one signal
        self.pop = np.asarray(X.sum(0)).ravel()
        return X

    def _matrix_multisignal(self, train: pd.DataFrame) -> "object":
        """X from a WEIGHTED blend of feedback signals rather than clicks alone.

        Duplicate (user,item) rows are summed then clipped, so a user who clicked
        the same video twice still contributes a bounded amount.
        """
        from scipy.sparse import csr_matrix
        w = self.cfg.signal_weights
        unknown = set(w) - set(SIGNAL_COLS)
        if unknown:
            raise ValueError(f"unknown signal(s) {sorted(unknown)}; "
                             f"choose from {SIGNAL_COLS}")
        val = np.zeros(len(train), dtype=np.float32)
        for sig, weight in w.items():
            if sig == "play_ratio":
                r = (train["play_time_ms"].to_numpy(np.float32) /
                     np.maximum(train["duration_ms"].to_numpy(np.float32), 1.0))
                val += float(weight) * np.clip(r, 0.0, 2.0)
            else:
                val += float(weight) * train[sig].to_numpy(np.float32)
        keep = val > 0
        X = csr_matrix((val[keep],
                        (train.user_id.to_numpy(np.int64)[keep],
                         train.video_id.to_numpy(np.int64)[keep])),
                       shape=(self.n_users, self.n_items))
        X.sum_duplicates()
        X.data = np.clip(X.data, 0.0, float(max(w.values()) * 2))
        self.pop = np.asarray(X.sum(0)).ravel()
        return X

    def _content_similarity(self) -> np.ndarray:
        """Item-item similarity from video side-features.

        Uses only categoricals with real overlap. author_id (6,510 distinct over
        7,583 videos) and music_id (7,202) are near-unique per video, so they
        produce almost no shared structure and are deliberately excluded.
        """
        from scipy.sparse import csr_matrix
        v = pd.read_csv(_video_features_path())
        v = v.sort_values("video_id").reset_index(drop=True)
        v["duration_bin"] = pd.qcut(v.video_duration.rank(method="first"),
                                    20, labels=False, duplicates="drop")
        rows, cols, off = [], [], 0
        for fname in self.cfg.content_fields:
            if fname not in v.columns:
                raise ValueError(f"unknown content field {fname!r}")
            # .codes is int8 for small cardinalities; adding the block offset
            # overflows it, so widen before the arithmetic
            codes = pd.Categorical(v[fname].astype("string").fillna("NA")).codes.astype(np.int64)
            rows.append(v.video_id.to_numpy(np.int64)); cols.append(codes + off)
            off += int(codes.max()) + 1
        r = np.concatenate(rows); c = np.concatenate(cols)
        F = csr_matrix((np.ones(len(r), np.float32), (r, c)),
                       shape=(self.n_items, off))
        norm = np.sqrt(np.asarray(F.multiply(F).sum(1)).ravel()); norm[norm == 0] = 1.0
        S = np.asarray((F @ F.T).todense(), dtype=np.float32) / norm[:, None] / norm[None, :]
        np.fill_diagonal(S, 0.0)
        k = self.cfg.content_topk
        if k < self.n_items:
            cut = np.partition(S, -k, axis=1)[:, -k][:, None]
            S[S < cut] = 0.0
        rs = S.sum(1, keepdims=True); rs[rs == 0] = 1.0
        return S / rs                        # row-normalised so alpha is comparable

    def rank(self, users, train_seen, n_ranked, batch=2048):
        out = np.empty((len(users), n_ranked), dtype=np.int64)
        for s in range(0, len(users), batch):
            ub = users[s: s + batch]
            S = np.asarray(self.X[ub] @ self.B, dtype=np.float32)
            cold = np.asarray(self.X[ub].sum(1)).ravel() == 0
            if cold.any():
                S[cold] = self.pop                      # popularity fallback
            if train_seen is not None:
                for r, u in enumerate(ub):
                    seen = train_seen.get(int(u))
                    if seen is not None and seen.size:
                        S[r, seen] = -np.inf
            idx = np.argpartition(-S, n_ranked - 1, axis=1)[:, :n_ranked]
            rows = np.arange(len(ub))[:, None]
            out[s: s + len(ub)] = idx[rows, np.argsort(-S[rows, idx], axis=1)]
        return out


class EASE(_ItemItem):
    """Embarrassingly Shallow AutoEncoder (Steck, WWW'19). Closed form, no seed."""
    def fit(self, u_pos, i_pos, train: pd.DataFrame | None = None):
        multi = set(self.cfg.signal_weights) != {"is_click"}
        if multi:
            if train is None:
                raise ValueError("multi-signal X needs the train frame")
            self.X = self._matrix_multisignal(train)
        else:
            self.X = self._matrix(u_pos, i_pos)
        G = np.asarray((self.X.T @ self.X).todense(), dtype=np.float64)
        G[np.diag_indices_from(G)] += self.cfg.ease_lambda
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        B[np.diag_indices_from(B)] = 0.0            # no self-recommendation
        B = B.astype(np.float32)
        a = float(self.cfg.content_alpha)
        if a > 0.0:
            # scale-match before blending: EASE rows and content rows live on
            # different scales, so a raw convex combination would be dominated
            # by whichever is larger
            Bc = self._content_similarity()
            sb = np.abs(B).sum(1, keepdims=True); sb[sb == 0] = 1.0
            B = (1.0 - a) * (B / sb) + a * Bc
        self.B = B
        return {"model": "ease", "lambda": self.cfg.ease_lambda,
                "multi_signal": multi, "signal_weights": dict(self.cfg.signal_weights),
                "content_alpha": a, "deterministic": True}


class ItemKNN(_ItemItem):
    """Cosine item-item, top-k neighbours retained. Deterministic."""
    def fit(self, u_pos, i_pos):
        self.X = self._matrix(u_pos, i_pos)
        C = np.asarray((self.X.T @ self.X).todense(), dtype=np.float32)
        d = np.sqrt(np.diag(C)).copy(); d[d == 0] = 1.0
        S = C / d[:, None] / d[None, :]
        np.fill_diagonal(S, 0.0)
        k = self.cfg.knn_k
        if k < self.n_items:                        # keep only the k strongest per item
            cut = np.partition(S, -k, axis=1)[:, -k][:, None]
            S[S < cut] = 0.0
        self.B = S
        return {"model": "itemknn", "k": k, "deterministic": True}


class ALS(_ItemItem):
    """Implicit-feedback ALS (Hu, Koren & Volinsky 2008). Stochastic via init."""
    def fit(self, u_pos, i_pos):
        self.X = self._matrix(u_pos, i_pos)
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        d = cfg.dim
        U = rng.normal(0, 0.01, (self.n_users, d))
        V = rng.normal(0, 0.01, (self.n_items, d))
        Xc = self.X.tocsr(); Xt = self.X.T.tocsr()
        for _ in range(cfg.als_iters):
            for A, B_, M in ((U, V, Xc), (V, U, Xt)):
                BtB = B_.T @ B_ + cfg.reg_als * np.eye(d)
                for row in range(A.shape[0]):
                    idx = M.indices[M.indptr[row]: M.indptr[row + 1]]
                    if idx.size == 0:
                        A[row] = 0.0; continue
                    Bi = B_[idx]
                    Ai = BtB + cfg.als_alpha * (Bi.T @ Bi)
                    bi = (1.0 + cfg.als_alpha) * Bi.sum(0)
                    A[row] = np.linalg.solve(Ai, bi)
        # expose as an item-item-shaped scorer: scores = X @ (V @ U-ish) is wrong,
        # so override rank via factors instead
        self.U, self.V = U.astype(np.float32), V.astype(np.float32)
        self.B = None
        return {"model": "als", "iters": cfg.als_iters, "deterministic": False}

    def rank(self, users, train_seen, n_ranked, batch=4096):
        out = np.empty((len(users), n_ranked), dtype=np.int64)
        for s in range(0, len(users), batch):
            ub = users[s: s + batch]
            S = self.U[ub] @ self.V.T
            if train_seen is not None:
                for r, u in enumerate(ub):
                    seen = train_seen.get(int(u))
                    if seen is not None and seen.size:
                        S[r, seen] = -np.inf
            idx = np.argpartition(-S, n_ranked - 1, axis=1)[:, :n_ranked]
            rows = np.arange(len(ub))[:, None]
            out[s: s + len(ub)] = idx[rows, np.argsort(-S[rows, idx], axis=1)]
        return out


class Popularity(_ItemItem):
    def fit(self, u_pos, i_pos):
        self.X = self._matrix(u_pos, i_pos)
        self.B = None
        return {"model": "popularity", "deterministic": True}

    def rank(self, users, train_seen, n_ranked, batch=8192):
        out = np.empty((len(users), n_ranked), dtype=np.int64)
        for s in range(0, len(users), batch):
            ub = users[s: s + batch]
            S = np.tile(self.pop, (len(ub), 1)).astype(np.float32)
            if train_seen is not None:
                for r, u in enumerate(ub):
                    seen = train_seen.get(int(u))
                    if seen is not None and seen.size:
                        S[r, seen] = -np.inf
            idx = np.argpartition(-S, n_ranked - 1, axis=1)[:, :n_ranked]
            rows = np.arange(len(ub))[:, None]
            out[s: s + len(ub)] = idx[rows, np.argsort(-S[rows, idx], axis=1)]
        return out


MODELS = {"bpr_mf": None, "ease": EASE, "itemknn": ItemKNN, "als": ALS,
          "popularity": Popularity}


def build_train_seen(train: pd.DataFrame) -> dict[int, np.ndarray]:
    """Items each user has ANY interaction with in train.

    Note: interactions, not just clicks. That matches the scorer's definition; a
    mismatch here is the most likely way to trip the masking refusal.
    """
    a = train[["user_id", "video_id"]].to_numpy(dtype=np.int64)
    a = a[np.lexsort((a[:, 1], a[:, 0]))]
    u, st = np.unique(a[:, 0], return_index=True)
    return {int(k): np.unique(g[:, 1]) for k, g in zip(u, np.split(a, st[1:]))}


def run(train: pd.DataFrame, users_to_rank: np.ndarray, cfg: Config,
        n_items: int, n_users: int, device: str = "cpu") -> tuple[np.ndarray, dict]:
    """Train and produce rankings. The whole mutable pipeline in one entry point."""
    pos = train[train[LABEL] == 1]
    u_pos = pos.user_id.to_numpy(np.int64)
    i_pos = pos.video_id.to_numpy(np.int64)
    if cfg.model not in MODELS:
        raise ValueError(f"unknown model {cfg.model!r}; choose from {sorted(MODELS)}")
    cls = MODELS[cfg.model]
    m = BprMF(cfg, n_users, n_items, device) if cls is None else cls(cfg, n_users, n_items, device)
    try:
        info = m.fit(u_pos, i_pos, train)          # EASE accepts the frame
    except TypeError:
        info = m.fit(u_pos, i_pos)
    seen = build_train_seen(train) if cfg.mask == "exclude_train_seen" else None
    ranked = m.rank(users_to_rank, seen, cfg.n_ranked)
    info["n_train_positives"] = int(len(pos))
    return ranked, info
