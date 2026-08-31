"""Metrics.

CURRENT TASK  -> `primary()` at the bottom of this file. It delegates to
`core.official_evaluate`, which is the organiser's `evaluate.py` vendored verbatim.
GAUC + nDCG@5 over within-user rankings of logged impressions, label `long_view`.
**We do not reimplement it.** See that module's header for why.

SUPERSEDED    -> everything else here. NDCG@10 / Recall@50 under full-catalog
retrieval with `is_click`, which is what the task was before 2026-08-26 18:33. The
functions are kept because the superseded experiments in experiments/ still import
them and their numbers are still reproducible. They are NOT the graded metric.

--- superseded docstring follows -------------------------------------------------

NDCG@K and Recall@K under BOTH candidate-set protocols.

Protocol A ("rerank")  - order each user's own logged impressions. This is what
                         CWM/evaluate.py actually does (groupby user, rank by pred).
Protocol B ("catalog") - rank the full item catalog per user. The only protocol
                         under which Recall@50 is non-trivial.
The brief specifies NDCG@10 / Recall@50 but does not say which protocol, and CWM
implements no Recall at all -- so we measure both and let the numbers decide.
"""
from __future__ import annotations
import numpy as np

def _discount(n: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, n + 2))      # position 1 -> 1/log2(2) = 1

def ndcg_at_k(hits_in_rank_order: np.ndarray, k: int, n_rel: int | None = None) -> float:
    """Binary relevance.

    n_rel MUST be the user's TOTAL number of (distinct, reachable) relevant items.
    Deriving it from hits_in_rank_order is only correct when that array covers the
    whole candidate list -- under full-catalog eval it covers just the retrieved
    window, which silently shrinks the ideal denominator and inflates NDCG.
    """
    r = hits_in_rank_order[:k]
    if r.size == 0:
        return 0.0
    dcg = float((r * _discount(r.size)).sum())
    if n_rel is None:
        n_rel = int(hits_in_rank_order.sum())
    if n_rel == 0:
        return 0.0
    # Invariant: you cannot retrieve more relevant items than exist. A violation
    # means duplicated relevance, a mismatched n_rel, or a join bug -- all of which
    # silently produce NDCG > 1. Fail loudly instead.
    n_hit = int(hits_in_rank_order.sum())
    if n_hit > n_rel:
        raise ValueError(f"inconsistent input: {n_hit} hits but n_rel={n_rel}. "
                         "Relevance is probably duplicated or misaligned.")
    idcg = float(_discount(min(k, n_rel)).sum())
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------- protocol A
def eval_rerank(u: np.ndarray, rel: np.ndarray, score: np.ndarray,
                k_ndcg: int = 10, k_recall: int = 50) -> dict:
    """Sort each user's logged rows by score; score the resulting list."""
    order = np.lexsort((-score, u))            # by user, then score desc
    u_s, rel_s = u[order], rel[order]
    bounds = np.flatnonzero(np.diff(u_s)) + 1
    ndcgs, recalls, n_eval = [], [], 0
    for seg_rel in np.split(rel_s, bounds):
        if seg_rel.sum() == 0:                 # no positive -> undefined, skipped
            continue
        n_eval += 1
        ndcgs.append(ndcg_at_k(seg_rel, k_ndcg))
        recalls.append(float(seg_rel[:k_recall].sum()) / float(seg_rel.sum()))
    return {"protocol": "rerank", f"ndcg@{k_ndcg}": float(np.mean(ndcgs)),
            f"recall@{k_recall}": float(np.mean(recalls)), "n_users_scored": n_eval,
            "mean_candidates": float(len(u) / max(1, len(np.unique(u))))}


# ---------------------------------------------------------------- protocol B
def eval_catalog(score_fn, eval_users: np.ndarray, relevant: dict,
                 n_items: int, exclude: dict | None = None,
                 k_ndcg: int = 10, k_recall: int = 50, batch: int = 4096) -> dict:
    """score_fn(user_batch) -> (len(batch), n_items) float32 score matrix.
    exclude: user -> array of item ids to mask out (e.g. seen in train)."""
    kmax = max(k_ndcg, k_recall)
    ndcgs, recalls, n_unreachable = [], [], [0]
    for s in range(0, len(eval_users), batch):
        ub = eval_users[s:s + batch]
        S = np.asarray(score_fn(ub), dtype=np.float32).copy()
        if exclude is not None:
            for r, usr in enumerate(ub):
                ex = exclude.get(usr)
                if ex is not None and len(ex):
                    S[r, ex] = -np.inf
        top = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
        rows = np.arange(len(ub))[:, None]
        top = top[rows, np.argsort(-S[rows, top], axis=1)]     # exact order in top-kmax
        for r, usr in enumerate(ub):
            rel_items = np.unique(relevant[usr])          # dedupe: repeat clicks are one item
            if exclude is not None:
                ex = exclude.get(usr)
                if ex is not None and len(ex):
                    rel_items = np.setdiff1d(rel_items, ex)   # masked -> unreachable
                    if rel_items.size == 0:
                        n_unreachable[0] += 1
                        continue
            hits = np.isin(top[r], rel_items).astype(np.float64)
            n_rel = int(rel_items.size)
            ndcgs.append(ndcg_at_k(hits, k_ndcg, n_rel=n_rel))
            recalls.append(float(hits[:k_recall].sum()) / float(n_rel))
    return {"protocol": "catalog", f"ndcg@{k_ndcg}": float(np.mean(ndcgs)),
            f"recall@{k_recall}": float(np.mean(recalls)),
            "n_users_scored": len(ndcgs), "n_users_all_rel_masked": n_unreachable[0],
            "mean_candidates": float(n_items)}


# ============================================================ CURRENT TASK (official)
def primary(user_ids, labels, scores, k: int = 5) -> dict:
    """THE graded metric: {'GAUC', 'nDCG@5', 'primary', 'users', 'rows'}.

    A thin pass-through to the vendored organiser code. It exists so the rest of our
    codebase has one import site for "the official metric", and so that swapping in a
    newer kit release is a one-file change.

    Deliberately NOT reimplemented. Three conventions here are trap-shaped and getting
    any of them wrong moves the number by tens of a percent:
      * users with zero positives score nDCG = 0.0 AND are counted in the mean
        (27.1% of test users -- dropping them inflates the score by ~37%)
      * GAUC covers only users with 0 < npos < n_impressions, weighted by npos
      * gain is 2^rel - 1, discount log2(rank+1), ideal over the user's own list
    """
    from core.official_evaluate import evaluate as _official
    return _official(list(user_ids), list(labels), list(scores), k=k)


#: Published reference points on the official test split (kuairand-starter-kit/
#: baseline_scores.json). `random` doubles as the harness self-check: if our pipeline
#: cannot reproduce ~0.4753 from random scores, the harness is broken, not the model.
OFFICIAL_TEST = {"random": 0.4753, "item_popularity": 0.5715,
                 "fm_official": 0.5946, "oracle_ceiling": 0.8645}
OFFICIAL_VALID = {"random": 0.4834, "item_popularity": 0.5807,
                  "fm_official": 0.6016, "oracle_ceiling": 0.8484}
#: FM's seed std, published. The convergence rule eps=0.002 is ~2.5 sigma of this.
OFFICIAL_FM_SEED_STD = 0.0008
OFFICIAL_EPSILON, OFFICIAL_N = 0.002, 3
