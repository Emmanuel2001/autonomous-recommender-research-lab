"""Tests for the partition. The label-quarantine test is the important one."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from core import splits as S

R=[]
def check(n,c,extra=""): R.append((n,bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

train_raw, val, test = S._load_raw()
sv, av = S._partition_val(val)

# 1 determinism
check("manifest is deterministic across builds",
      S.build_manifest()["splits"] == S.build_manifest()["splits"])
b1 = S._row_bucket(val.user_id.to_numpy()[:500], val.video_id.to_numpy()[:500], val.time_ms.to_numpy()[:500])
b2 = S._row_bucket(val.user_id.to_numpy()[:500], val.video_id.to_numpy()[:500], val.time_ms.to_numpy()[:500])
check("row bucketing is deterministic", np.array_equal(b1, b2))

# 2 partition is exact: disjoint and complete
check("search_val + audit_val == val (exact partition)", len(sv) + len(av) == len(val))
check("search_val and audit_val are row-disjoint",
      len(set(sv.index) & set(av.index)) == 0)
check("union of indices covers val exactly",
      set(sv.index) | set(av.index) == set(val.index))

# 3 THE QUARANTINE TEST -----------------------------------------------------
tr, part = S.load_for_strategy()
val_keys  = set(map(tuple, val[S.KEYS + ["time_ms"]].to_numpy()))
test_keys = set(map(tuple, test[S.KEYS + ["time_ms"]].to_numpy()))
tr_keys   = set(map(tuple, tr[S.KEYS + ["time_ms"]].to_numpy()))
check("strategy sees NO validation row", len(tr_keys & val_keys) == 0)
check("strategy sees NO quarantined-test row", len(tr_keys & test_keys) == 0)
check("strategy's train rows == the train file exactly", len(tr) == len(train_raw))
# the Partition object must carry user ids only -- no items, no labels
leaked = []
for fname in ("train_users", "search_val_users", "audit_val_users"):
    arr = getattr(part, fname)
    if arr.ndim != 1:
        leaked.append(fname)
check("Partition carries 1-D user arrays only (no items/labels)", not leaked)
check("Partition exposes no label field",
      not any(k in part.manifest for k in ("relevance", "labels", "y")))

# 4 relevance sets are deduped and positive-only
u, rel = S._relevance(sv)
check("relevance sets contain no duplicate items",
      all(len(v) == len(np.unique(v)) for v in rel.values()))
check("every relevance user has >=1 item", all(len(v) >= 1 for v in rel.values()))
raw_pos = sv[sv[S.LABEL] == 1]
check("dedup actually removed repeats",
      sum(len(v) for v in rel.values()) < len(raw_pos))
check("relevance users == users with a positive",
      set(rel) == set(raw_pos.user_id.unique()))

# 5 scorer view
sc = S.load_for_scorer()
check("scorer holds search_val + audit_val relevance",
      set(sc.relevance) == {"search_val", "audit_val"})
check("scorer excludes quarantine by default", "test_quarantine" not in sc.relevance)
check("scorer manifest hash is 64 hex chars", len(sc.manifest_sha256) == 64)
one = next(iter(sc.train_seen))
seen_true = np.unique(train_raw.loc[train_raw.user_id == one, "video_id"].to_numpy())
check("train_seen matches the train file for a sampled user",
      np.array_equal(np.sort(sc.train_seen[one]), seen_true))

# 5b log_random opt-in must work AND must not break the quarantine
tr_lr, part_lr = S.load_for_strategy(use_log_random=True)
check("log_random opt-in adds rows", len(tr_lr) > len(train_raw),
      f"{len(train_raw):,} -> {len(tr_lr):,}")
check("log_random rows are source-tagged",
      "_source" in tr_lr and set(tr_lr["_source"].unique()) == {"log_standard", "log_random"})
lr_keys = set(map(tuple, tr_lr[S.KEYS + ["time_ms"]].to_numpy()))
check("QUARANTINE HOLDS with log_random enabled: no test row",
      len(lr_keys & test_keys) == 0)
check("QUARANTINE HOLDS with log_random enabled: no validation row",
      len(lr_keys & val_keys) == 0)
check("log_random is OFF by default", "_source" not in S.load_for_strategy()[0])

# 6 quarantine access warns loudly
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always"); S.load_quarantined_test()
    check("quarantined access emits a warning", len(w) == 1 and "QUARANTINED" in str(w[0].message))

# 7 exchangeability diagnostic -- the check whose absence caused the earlier error
def jsd(p, q):
    p, q = p / p.sum(), q / q.sum(); m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
nb = 27285
hs = np.bincount(sv.user_id.to_numpy(), minlength=nb).astype(float)
ha = np.bincount(av.user_id.to_numpy(), minlength=nb).astype(float)
d = jsd(np.bincount(np.minimum(hs.astype(int), 30), minlength=31).astype(float) + 1e-9,
        np.bincount(np.minimum(ha.astype(int), 30), minlength=31).astype(float) + 1e-9)
m = S.build_manifest()["splits"]
print(f"\n  [diagnostic] search_val click={m['search_val']['click_rate']:.4f}  "
      f"audit_val click={m['audit_val']['click_rate']:.4f}  "
      f"delta={abs(m['search_val']['click_rate']-m['audit_val']['click_rate'])*100:.2f}pp")
print(f"  [diagnostic] user-frequency JSD(search,audit) = {d:.4f}  "
      f"(val-vs-test was 0.128 -- lower is more exchangeable)")
print(f"  [diagnostic] audit_val scoreable users = {m['audit_val']['users_with_positive']:,} "
      f"-> SE of mean NDCG ~ {0.10/np.sqrt(m['audit_val']['users_with_positive']):.5f} at per-user sd 0.10")


# ---------------------------------------------------------------- split rule
# Added 2026-08-27. The eval-window cut was hardcoded until the organisers' own
# reference (CWM train_model2.py:86-88) was found to cut TEMPORALLY while the brief's
# wording says "first 50% / last 50%". The rule is now explicit, and these guard it.
from core.splits import _cut_eval, VALID_SPLIT_RULES, TEMPORAL_VAL, TEMPORAL_TEST
_ev = pd.read_csv(S.DATA / S.EVAL_F, usecols=S.KEYS + S.CONTEXT + S.ALL_LABELS)

for _rule in VALID_SPLIT_RULES:
    _v, _t = _cut_eval(_ev, _rule)
    check(f"[{_rule}] cut loses no rows", len(_v) + len(_t) == len(_ev),
          f"{len(_v):,}+{len(_t):,}={len(_v)+len(_t):,} vs {len(_ev):,}")
    check(f"[{_rule}] halves are row-disjoint", len(set(_v.index) & set(_t.index)) == 0)
    check(f"[{_rule}] halves cover the file exactly",
          set(_v.index) | set(_t.index) == set(_ev.index))

_pv, _pt = _cut_eval(_ev, "positional")
check("positional cut is exactly 50/50 (this is why it matches the brief's wording)",
      abs(len(_pv) - len(_pt)) <= 1, f"{len(_pv):,} vs {len(_pt):,}")

_tv, _tt = _cut_eval(_ev, "temporal")
check("temporal val is 4/22-4/28 only",
      _tv.date.min() == TEMPORAL_VAL[0] and _tv.date.max() == TEMPORAL_VAL[1])
check("temporal test is 4/29-5/08 only",
      _tt.date.min() == TEMPORAL_TEST[0] and _tt.date.max() == TEMPORAL_TEST[1])
check("temporal cut is NOT 50/50 (which is why the two readings differ)",
      abs(len(_tv) - len(_tt)) > 1000, f"{len(_tv):,} vs {len(_tt):,}")
def _keys(d):
    return set(map(tuple, d[["user_id", "video_id", "time_ms"]].to_numpy()))
_j = len(_keys(_pt) & _keys(_tt)) / len(_keys(_pt) | _keys(_tt))
check("the two readings genuinely disagree on the test half (Jaccard < 0.6)",
      _j < 0.6, f"Jaccard {_j:.4f}")

try:
    _cut_eval(_ev, "by_vibes")
    _raised = False
except ValueError:
    _raised = True
check("an unknown split rule raises rather than silently defaulting", _raised)

check("the manifest records the split rule, so its hash separates the two worlds",
      S.build_manifest().get("split_rule") == S.SPLIT_RULE,
      f"rule={S.SPLIT_RULE!r}")

print()
bad=[n for n,ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
