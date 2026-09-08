"""Feature encoding. AGENT-EDITABLE.

WHAT THIS EDIT ADDS (over the parent)
-------------------------------------
The parent encoded absolute-value fields only: id fields, item attributes, coarse profile
attributes and TRAIN-quantile buckets of continuous columns. Every user therefore shares
one global bucket-to-score mapping, and a user whose whole impression list sits inside a
single duration decile receives no discriminative signal from duration at all -- even
though the metric (GAUC / nDCG@5) is decided *exclusively* by variation across that user's
own impressions.

Four WITHIN-USER relative context columns are added, all derived, all label-free:

  * `dur_pct_bucket`  -- the impression's duration percentile INSIDE that user's own
                         impression list (mid-rank, so ties share a percentile), cut into
                         `PCT_BINS` equal-width percentile bins.
  * `dur_dev_bucket`  -- signed deviation from the user's OWN median duration, expressed as
                         log((dur+1)/(median+1)) and bucketed with fixed symmetric edges.
                         The middle bucket is a narrow band around zero, so "this video is
                         typical for me" is its own id rather than an arbitrary side.
  * `user_nimp_bucket`-- how heavy the user is: the percentile of that user's impression
                         COUNT among the users of the same split, in `CNT_BINS` bins.
                         A percentile and not log-count on purpose: raw per-user counts are
                         much larger in train than in dev/heldout, so a count vocabulary
                         built on train would be mostly UNK at scoring time, whereas the
                         percentile is comparable across splits.
  * `upos_bucket`      -- fractional position of the row inside the user's own block of
                         rows in ROW ORDER (KuaiRand's log is time ordered), `POS_BINS`
                         bins. This is the only column here that varies monotonically
                         along a user's session.

STATISTICS ARE PER-SPLIT AND FEATURE-ONLY
-----------------------------------------
Each of these is computed from the rows of the split being encoded, using only `user_id`
and `duration_ms` / row order. No labels are read (they do not exist for heldout in this
process) and no train statistic is carried across, so the same code path produces the
heldout column as produces the train column. What IS still train-only, exactly as in the
parent, is the VOCABULARY: the small integer bucket ids are mapped through a train-built
vocab with a per-field UNK slot.

Invariants kept from the seed and the parent:
  * vocabulary and continuous quantile edges come from train only;
  * `column()` raises with the available names, so a bad field costs one cheap refusal;
  * mapping is vectorised (sorted unique + searchsorted + equality check for UNK).

The leakage guards below (`POST_IMPRESSION`, `DECLINED`) are untouched.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- leakage guards
# Outcome / post-impression columns. NEVER admitted, not configurable.
POST_IMPRESSION = frozenset({
    "label", "long_view", "is_click", "is_like", "is_follow", "is_comment",
    "is_forward", "is_hate", "is_profile_enter", "play_time_ms", "play_duration",
    "profile_stay_time", "comment_stay_time", "long_view_ratio", "play_progress",
    "effective_view", "is_share", "is_collect", "y", "target",
})

# Pre-impression, but refused on purpose:
#   time_ms / timestamp / date -- raw clock values. KuaiRand's splits are time ordered, so
#     a date vocabulary built on train is almost entirely UNK on dev/heldout: it cannot
#     help and it can only fit train-specific drift. Hour-of-day is admitted instead, via
#     the derived `hourmin_bucket`.
#   is_rand -- which logging policy produced the impression. Not an outcome, but it is the
#     provenance of the split itself; keeping it out makes the leakage story unarguable.
DECLINED = frozenset({"time_ms", "timestamp", "date", "is_rand", "user_id_hash"})

# Within-user relative context columns added by this candidate.
WITHIN_USER_FIELDS = ("dur_pct_bucket", "dur_dev_bucket", "user_nimp_bucket",
                      "upos_bucket")

# Derived columns and the raw columns they need.
DERIVED = {
    "dur_bucket": frozenset({"duration_ms"}),
    "hourmin_bucket": frozenset({"hourmin"}),
    "dur_pct_bucket": frozenset({"duration_ms", "user_id"}),
    "dur_dev_bucket": frozenset({"duration_ms", "user_id"}),
    "user_nimp_bucket": frozenset({"user_id"}),
    "upos_bucket": frozenset({"user_id"}),
}

# Whitelisted columns treated as CONTINUOUS -> train-quantile buckets.
CONTINUOUS = frozenset({
    "video_duration", "duration_ms", "follow_user_num", "fans_user_num",
    "friend_user_num", "register_days", "server_width", "server_height",
})

# ---- fixed bucket edges for the within-user columns (NOT fitted on any split) ----
PCT_BINS = 10
PCT_EDGES = np.linspace(0.0, 1.0, PCT_BINS + 1)[1:-1]
POS_BINS = 8
POS_EDGES = np.linspace(0.0, 1.0, POS_BINS + 1)[1:-1]
CNT_BINS = 8
CNT_EDGES = np.linspace(0.0, 1.0, CNT_BINS + 1)[1:-1]
# log-ratio to the user's own median duration; middle bucket is |log ratio| < 0.02.
DEV_EDGES = np.array([-1.5, -0.7, -0.35, -0.15, -0.02, 0.02, 0.15, 0.35, 0.7, 1.5],
                     dtype=np.float64)

# Ordered by expected within-user ranking value (see module docstring). The within-user
# columns come first; when they are already in the config's base `fields` they are skipped
# here without consuming the max_new budget.
CANDIDATE_FIELDS = (
    # within-user relative context
    "dur_pct_bucket", "dur_dev_bucket", "upos_bucket", "user_nimp_bucket",
    # item-side descriptors: shared parameters that transfer to rare video_ids
    "video_type", "upload_type", "music_type", "visible_status",
    # context
    "hourmin_bucket",
    # coarse user-profile attributes (cross with item fields inside the FM)
    "user_active_degree", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
    "is_live_streamer", "is_video_author", "is_lowactive_period",
    # item-side, coarser / higher cardinality
    "server_width", "server_height", "tag", "music_id", "upload_dt_bucket",
    # the wide profile one-hot block
    "onehot_feat0", "onehot_feat1", "onehot_feat2", "onehot_feat3", "onehot_feat4",
    "onehot_feat5", "onehot_feat6", "onehot_feat7", "onehot_feat8", "onehot_feat9",
    "onehot_feat10", "onehot_feat11", "onehot_feat12", "onehot_feat13",
    "onehot_feat14", "onehot_feat15", "onehot_feat16", "onehot_feat17",
    # continuous user counters, and the duplicate of dur_bucket, last
    "follow_user_num", "fans_user_num", "friend_user_num", "register_days",
    "video_duration",
)


# ------------------------------------------------------------------- primitives
def dur_bucket(duration_ms, edges) -> np.ndarray:
    """The kit's duration bucket: searchsorted against train deciles."""
    return np.searchsorted(edges, np.asarray(duration_ms, dtype=np.float64)).astype(np.int64)


# ------------------------------------------------- within-user helpers (label free)
# One memo per row-object so the two duration-relative columns share a single sort.
_WU_CACHE: dict = {}


def _cache_get(rows, tag):
    return _WU_CACHE.get((id(rows), len(rows), tag))


def _cache_put(rows, tag, value):
    if len(_WU_CACHE) > 64:
        _WU_CACHE.clear()
    _WU_CACHE[(id(rows), len(rows), tag)] = value
    return value


def _runs(flag_new: np.ndarray, n: int):
    """For a boolean 'starts a new run' mask over n sorted elements, return per-element
    (run_start, run_end) index arrays."""
    starts = np.flatnonzero(flag_new)
    rid = np.cumsum(flag_new) - 1
    rs = starts[rid]
    re = np.r_[starts[1:], n][rid]
    return rs, re


def _within_user_stats(u: np.ndarray, x: np.ndarray):
    """(percentile inside the user's own values, the user's own median value) per row.

    Percentile uses MID-RANKS, so equal values inside a user share one percentile instead
    of being ordered arbitrarily by the sort. Users with a single row get 0.5.
    """
    n = len(x)
    if n == 0:
        return np.zeros(0, np.float64), np.zeros(0, np.float64)
    o = np.lexsort((x, u))
    su, sx = u[o], x[o]
    g_new = np.r_[True, su[1:] != su[:-1]] if n > 1 else np.array([True])
    gs, ge = _runs(g_new, n)
    size = (ge - gs).astype(np.int64)
    if n > 1:
        r_new = g_new | np.r_[True, sx[1:] != sx[:-1]]
    else:
        r_new = np.array([True])
    rs, re = _runs(r_new, n)
    midpos = 0.5 * (rs + re - 1) - gs
    pct_sorted = np.where(size > 1, midpos / np.maximum(size - 1, 1), 0.5)
    i1 = gs + (size - 1) // 2
    i2 = gs + size // 2
    med_sorted = 0.5 * (sx[i1] + sx[i2])
    pct = np.empty(n, np.float64); pct[o] = pct_sorted
    med = np.empty(n, np.float64); med[o] = med_sorted
    return pct, med


def _user_count_pct(u: np.ndarray) -> np.ndarray:
    """Per-row percentile of the user's impression COUNT among the users of this split."""
    n = len(u)
    if n == 0:
        return np.zeros(0, np.float64)
    order = np.argsort(u, kind="stable")
    su = u[order]
    g_new = np.r_[True, su[1:] != su[:-1]] if n > 1 else np.array([True])
    starts = np.flatnonzero(g_new)
    sizes = np.diff(np.r_[starts, n]).astype(np.int64)
    nu = len(sizes)
    so = np.argsort(sizes, kind="stable")
    ss = sizes[so]
    s_new = np.r_[True, ss[1:] != ss[:-1]] if nu > 1 else np.array([True])
    rs, re = _runs(s_new, nu)
    mid = 0.5 * (rs + re - 1)
    ranks = np.empty(nu, np.float64)
    ranks[so] = mid / max(nu - 1, 1)
    out = np.empty(n, np.float64)
    out[order] = np.repeat(ranks, sizes)
    return out


def _user_pos_frac(u: np.ndarray) -> np.ndarray:
    """Per-row fractional position inside the user's own block of rows, in ROW ORDER."""
    n = len(u)
    if n == 0:
        return np.zeros(0, np.float64)
    order = np.argsort(u, kind="stable")
    su = u[order]
    g_new = np.r_[True, su[1:] != su[:-1]] if n > 1 else np.array([True])
    starts = np.flatnonzero(g_new)
    sizes = np.diff(np.r_[starts, n]).astype(np.int64)
    pos = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    per = np.repeat(sizes, sizes)
    frac = np.where(per > 1, pos / np.maximum(per - 1, 1), 0.0)
    out = np.empty(n, np.float64)
    out[order] = frac
    return out


def _user_ids(rows) -> np.ndarray:
    v = getattr(rows, "user_id", None)
    if v is None:
        raise ValueError("within-user context features need a 'user_id' column")
    a = np.asarray(v)
    return a if a.dtype.kind in "iuf" else a.astype(str)


def column(rows, name: str, edges) -> np.ndarray:
    """One feature column by name. Derived columns are computed here.

    The within-user columns use ONLY this split's `user_id` / `duration_ms` / row order --
    no labels, no statistic borrowed from another split.
    """
    if name == "dur_bucket":
        if edges is None:
            raise ValueError("'dur_bucket' needs duration edges; data.edges is None")
        return dur_bucket(rows.duration_ms, edges)
    if name == "hourmin_bucket":
        # hour of day from `hourmin` (HHMM as an integer).
        return np.asarray(rows.hourmin, dtype=np.int64) // 100
    if name in ("dur_pct_bucket", "dur_dev_bucket"):
        cached = _cache_get(rows, "dur_stats")
        if cached is None:
            d = getattr(rows, "duration_ms", None)
            if d is None:
                raise ValueError("'%s' needs a 'duration_ms' column" % name)
            x = np.asarray(d, dtype=np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            cached = _cache_put(rows, "dur_stats",
                                _within_user_stats(_user_ids(rows), x) + (x,))
        pct, med, x = cached
        if name == "dur_pct_bucket":
            return np.searchsorted(PCT_EDGES, pct).astype(np.int64)
        r = np.log((np.maximum(x, 0.0) + 1.0) / (np.maximum(med, 0.0) + 1.0))
        return np.searchsorted(DEV_EDGES, r).astype(np.int64)
    if name == "user_nimp_bucket":
        cp = _cache_get(rows, "cnt_pct")
        if cp is None:
            cp = _cache_put(rows, "cnt_pct", _user_count_pct(_user_ids(rows)))
        return np.searchsorted(CNT_EDGES, cp).astype(np.int64)
    if name == "upos_bucket":
        pf = _cache_get(rows, "pos_frac")
        if pf is None:
            pf = _cache_put(rows, "pos_frac", _user_pos_frac(_user_ids(rows)))
        return np.searchsorted(POS_EDGES, pf).astype(np.int64)
    try:
        v = getattr(rows, name)
    except AttributeError:
        v = None
    if v is None:
        from contract import columns_of
        raise ValueError(
            f"field {name!r} is not carried by this dataset. Available columns: "
            f"{list(columns_of(rows))}, plus derived {sorted(DERIVED)}. "
            f"Add a derived column in features.column() rather than assuming one exists.")
    return np.asarray(v)


def available_columns(rows) -> list:
    """Every raw column this dataset carries, best effort."""
    try:
        from contract import columns_of
        return sorted(str(c) for c in columns_of(rows))
    except Exception:
        return sorted(n for n in dir(rows) if not n.startswith("_"))


def _keys(raw, force_str: bool = False) -> np.ndarray:
    """Hashable/sortable key array for a categorical column.

    Integers and booleans stay integral. Floats that are integral (the usual encoding of
    small profile codes, sometimes with NaN) are rounded to int with a sentinel for
    non-finite. Anything else becomes a string, so mixed/object/str columns still encode.
    """
    a = np.asarray(raw)
    if force_str:
        return a.astype(str)
    if a.dtype.kind in "iu":
        return a.astype(np.int64)
    if a.dtype.kind == "b":
        return a.astype(np.int64)
    if a.dtype.kind == "f":
        b = np.where(np.isfinite(a), a, np.float64(-2147483648.0))
        r = np.rint(b)
        if np.all(np.abs(b - r) < 1e-9):
            return r.astype(np.int64)
        return a.astype(str)
    return a.astype(str)


def _bucket_edges(raw, nbins: int) -> np.ndarray:
    """Train-only quantile edges for a continuous column (deduplicated)."""
    x = np.asarray(raw, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    nbins = max(2, int(nbins))
    qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    return np.unique(np.quantile(x, qs))


def _bucketise(raw, edges) -> np.ndarray:
    return np.searchsorted(edges, np.asarray(raw, dtype=np.float64)).astype(np.int64)


def _map_to_vocab(vals: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """keys -> index into sorted `vals`, misses -> UNK slot len(vals). Vectorised."""
    unk = len(vals)
    if unk == 0:
        return np.zeros(len(keys), dtype=np.int64)
    idx = np.clip(np.searchsorted(vals, keys), 0, unk - 1)
    hit = vals[idx] == keys
    return np.where(hit, idx, unk).astype(np.int64)


# ------------------------------------------------------------- field resolution
def resolve_fields(train_rows, edges, base_fields, max_card: int = 2000,
                   max_new: int = 10, cont_bins: int = 16, exclude=(),
                   candidates=None):
    """Base fields plus every admissible pre-impression column this dataset carries.

    Returns (fields, admitted, report). `report` is one dict per considered column and is
    meant to be dropped straight into telemetry: it is the record of what the dataset
    offered and why each column was taken or refused.
    """
    base = list(dict.fromkeys(str(f) for f in base_fields))
    hard = POST_IMPRESSION | DECLINED
    soft = {str(e) for e in (exclude or ())}
    avail = set(available_columns(train_rows))
    report, admitted = [], []

    for name in (candidates if candidates is not None else CANDIDATE_FIELDS):
        if len(admitted) >= int(max_new):
            break
        if name in base or name in admitted:
            continue
        if name in hard:
            report.append({"column": name, "status": "refused_post_impression_or_declined"})
            continue
        if name in soft:
            report.append({"column": name, "status": "refused_by_config"})
            continue
        if name in DERIVED:
            if not DERIVED[name].issubset(avail):
                continue                        # dependency not carried; silent
        elif name not in avail:
            continue                            # dataset does not carry it; silent
        try:
            raw = column(train_rows, name, edges)
        except Exception as exc:                # noqa: BLE001 - one cheap refusal
            report.append({"column": name, "status": "error", "detail": str(exc)[:120]})
            continue
        try:
            if name in CONTINUOUS:
                ce = _bucket_edges(raw, cont_bins)
                if len(ce) < 2:
                    report.append({"column": name, "status": "dropped_constant"})
                    continue
                admitted.append(name)
                report.append({"column": name, "status": "admitted_bucketed",
                               "bins": int(len(ce) + 1)})
            else:
                k = _keys(raw)
                card = int(np.unique(k).size)
                if card < 2:
                    report.append({"column": name, "status": "dropped_constant"})
                    continue
                if card > int(max_card):
                    report.append({"column": name, "status": "dropped_high_cardinality",
                                   "cardinality": card})
                    continue
                admitted.append(name)
                report.append({"column": name, "status": "admitted", "cardinality": card})
        except Exception as exc:                # noqa: BLE001
            report.append({"column": name, "status": "error", "detail": str(exc)[:120]})

    return tuple(base + admitted), admitted, report


# -------------------------------------------------------------------- encoding
def encode(splits: dict, edges, fields, cont_bins: int = 16) -> tuple[dict, int]:
    """Categorical fields -> contiguous ids with per-field offsets. Vocab from TRAIN.

    Continuous whitelisted fields are first bucketed by TRAIN-only quantiles, then treated
    exactly like any other categorical field. Derived within-user columns arrive as small
    integers already (their statistics are per-split, by construction) and go through the
    same train-built vocabulary + UNK slot as everything else.

    `splits` maps a name to a row object and MUST contain "train". Returns
    ({name: int32 array (rows, n_fields)}, total_vocab_size).
    """
    fields = tuple(fields)
    if not fields:
        raise ValueError("no fields to encode")
    if "train" not in splits:
        raise ValueError("encode() needs splits['train'] to build the vocabulary")
    tr = splits["train"]

    cont_edges, keys_tr, as_str = {}, [], []
    for f in fields:
        raw = column(tr, f, edges)
        if f in CONTINUOUS:
            ce = _bucket_edges(raw, cont_bins)
            if len(ce) < 1:
                raise ValueError(f"continuous field {f!r} has no finite values in train")
            cont_edges[f] = ce
            k = _bucketise(raw, ce)
        else:
            k = _keys(raw)
        keys_tr.append(k)
        as_str.append(k.dtype.kind in "US")

    vocabs = [np.unique(k) for k in keys_tr]
    dims = [len(v) + 1 for v in vocabs]                  # +1 UNK per field
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    enc = {}
    for name, rows in splits.items():
        X = np.empty((len(rows), len(fields)), dtype=np.int32)
        for i, f in enumerate(fields):
            if name == "train":
                k = keys_tr[i]
            else:
                raw = column(rows, f, edges)
                if f in cont_edges:
                    k = _bucketise(raw, cont_edges[f])
                else:
                    k = _keys(raw, force_str=as_str[i])
            X[:, i] = _map_to_vocab(vocabs[i], k) + offsets[i]
        enc[name] = X
    _WU_CACHE.clear()
    return enc, int(sum(dims))
