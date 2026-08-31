"""Golden vectors for NDCG/Recall, plus an equivalence check between the two
implementations (metrics.eval_catalog vs scorer._score_lists) so they cannot drift."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.metrics import ndcg_at_k, eval_catalog, eval_rerank
from core.scorer import _score_lists, K_NDCG, K_RECALL

R=[]
def check(n,c,extra=""): R.append((n,bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

d = 1.0 / np.log2(np.arange(2, 80))

# --- golden vectors, hand-computed
cases = [
    ("perfect: 3 relevant, all at top",        np.r_[1,1,1,np.zeros(47)], 3, d[:3].sum()/d[:3].sum()),
    ("1 hit at rank 4, 5 relevant total",      np.r_[0,0,0,1,np.zeros(46)], 5, d[3]/d[:5].sum()),
    ("no hits, 4 relevant total",              np.zeros(50), 4, 0.0),
    ("1 hit rank 1, 1 relevant total",         np.r_[1,np.zeros(49)], 1, 1.0),
    ("2 hits at ranks 1,2 but 6 relevant",     np.r_[1,1,np.zeros(48)], 6, (d[0]+d[1])/d[:6].sum()),
    ("hit beyond k=10 does not count in DCG",  np.r_[np.zeros(15),1,np.zeros(34)], 1, 0.0),
    ("n_rel > k caps the ideal at k",          np.r_[1,np.zeros(49)], 99, d[0]/d[:K_NDCG].sum()),
]
for name, hits, n_rel, exp in cases:
    got = ndcg_at_k(hits, K_NDCG, n_rel=n_rel)
    check(name, abs(got-exp) < 1e-12, f"got={got:.6f} exp={exp:.6f}")

check("ndcg is 0 when n_rel is 0", ndcg_at_k(np.r_[1,np.zeros(49)], 10, n_rel=0) == 0.0)
def _consistent(seed, n_rel):
    """hits.sum() <= n_rel, i.e. an input that can physically occur."""
    g = np.random.default_rng(seed)
    h = np.zeros(50)
    h[g.choice(50, size=int(g.integers(0, n_rel + 1)), replace=False)] = 1.0
    return h
check("ndcg in [0,1] over 500 physically-possible inputs",
      all(0.0 <= ndcg_at_k(_consistent(s, 7), 10, n_rel=7) <= 1.0 + 1e-12 for s in range(500)))
try:
    ndcg_at_k(np.ones(50), 10, n_rel=3)
    check("GUARD: more hits than relevant items raises", False)
except ValueError as e:
    check("GUARD: more hits than relevant items raises", "inconsistent input" in str(e))
check("REGRESSION: deriving n_rel from a truncated window inflates NDCG",
      ndcg_at_k(np.r_[0,0,0,1,np.zeros(46)], 10) > ndcg_at_k(np.r_[0,0,0,1,np.zeros(46)], 10, n_rel=5))

# --- the two implementations must agree exactly
rng = np.random.default_rng(7)
n_users, n_items = 300, 500
users = np.arange(n_users, dtype=np.int64)
rel = {int(u): np.unique(rng.choice(n_items, size=int(rng.integers(1, 8)), replace=False)).astype(np.int64)
       for u in users}
scores = rng.normal(size=(n_users, n_items)).astype(np.float32)
ranked = np.argsort(-scores, axis=1)[:, :K_RECALL].astype(np.int64)

a_nd, a_rc, a_n = _score_lists(ranked, users, rel)
b = eval_catalog(lambda ub: scores[ub], users, rel, n_items, k_ndcg=K_NDCG, k_recall=K_RECALL)
check("scorer._score_lists == metrics.eval_catalog on NDCG",
      abs(a_nd - b[f"ndcg@{K_NDCG}"]) < 1e-12, f"{a_nd:.9f} vs {b[f'ndcg@{K_NDCG}']:.9f}")
check("scorer._score_lists == metrics.eval_catalog on Recall",
      abs(a_rc - b[f"recall@{K_RECALL}"]) < 1e-12, f"{a_rc:.9f} vs {b[f'recall@{K_RECALL}']:.9f}")
check("both agree on the number of users scored", a_n == b["n_users_scored"])

# --- recall bounds and the dedup property
one = {0: np.array([5], np.int64)}
check("recall is 1.0 when the single relevant item is retrieved",
      abs(_score_lists(np.array([[5]+list(range(100,149))], np.int64), np.array([0]), one)[1] - 1.0) < 1e-12)
check("recall is 0.0 when it is not",
      _score_lists(np.array([[list(range(100,150))][0]], np.int64), np.array([0]), one)[1] == 0.0)

# --- rerank protocol derives n_rel from the FULL list, which is correct there
u = np.array([0,0,0,1,1], np.int64); y = np.array([1.,0.,1.,0.,1.]); sc = np.array([3.,2.,1.,2.,1.])
r = eval_rerank(u, y, sc)
check("eval_rerank scores only users with a positive", r["n_users_scored"] == 2)
check("eval_rerank recall@50 is 1.0 on short lists (degenerate, as measured)",
      abs(r["recall@50"] - 1.0) < 1e-12)

print()
bad=[n for n,ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
