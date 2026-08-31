"""Feature encoding. AGENT-EDITABLE.

Seeded with the official baseline's encoding: five categorical fields mapped to one
vocabulary from TRAIN only, unseen values falling into a per-field
UNK slot. This is a starting point, not a constraint -- new columns, crosses, bucketings
and count features all belong here.

Two things to keep if you rewrite it:
  * the vocabulary comes from train only. Building it from train+dev leaks dev structure
    into the model and the gain will not survive the held-out split.
  * `column()` raises with the available names rather than AttributeError, so a field that
    does not exist costs one cheap refusal instead of a whole fit.
"""
from __future__ import annotations

import numpy as np


def dur_bucket(duration_ms, edges) -> np.ndarray:
    """The kit's duration bucket: searchsorted against train deciles."""
    return np.searchsorted(edges, np.asarray(duration_ms, dtype=np.float64)).astype(np.int64)


def compute_user_long_view_rates(train_rows) -> dict[int, int]:
    """Compute smoothed user historical positive rate from train only."""
    u_ids = np.asarray(train_rows.user_id)
    labels = np.asarray(train_rows.label, dtype=np.float64)

    global_mean = float(np.mean(labels)) if len(labels) > 0 else 0.5
    m = 10.0

    from collections import defaultdict
    sums = defaultdict(float)
    counts = defaultdict(int)
    for u, l in zip(u_ids, labels):
        sums[int(u)] += l
        counts[int(u)] += 1

    rates = {}
    for u, c in counts.items():
        s = sums[u]
        smoothed = (s + m * global_mean) / (c + m)
        bin_idx = int(np.clip(np.floor(smoothed * 10.0), 0, 9))
        rates[u] = bin_idx
    return rates


def compute_author_long_view_rates(train_rows) -> dict[int, int]:
    """Compute smoothed author historical positive rate from train only."""
    a_ids = np.asarray(train_rows.author_id)
    labels = np.asarray(train_rows.label, dtype=np.float64)

    global_mean = float(np.mean(labels)) if len(labels) > 0 else 0.5
    m = 20.0

    from collections import defaultdict
    sums = defaultdict(float)
    counts = defaultdict(int)
    for a, l in zip(a_ids, labels):
        sums[int(a)] += l
        counts[int(a)] += 1

    rates = {}
    for a, c in counts.items():
        s = sums[a]
        smoothed = (s + m * global_mean) / (c + m)
        bin_idx = int(np.clip(np.floor(smoothed * 10.0), 0, 9))
        rates[a] = bin_idx
    return rates


def compute_user_impression_counts(train_rows) -> dict[int, int]:
    """Compute binned user total impression frequency from train only."""
    u_ids = np.asarray(train_rows.user_id)
    from collections import Counter
    counts = Counter(int(u) for u in u_ids)

    bins = {}
    for u, c in counts.items():
        val = int(np.floor(np.log1p(c) * 3.0))
        bins[u] = int(np.clip(val, 0, 9))
    return bins


_USER_RATES_CACHE = {}
_AUTHOR_RATES_CACHE = {}
_USER_COUNTS_CACHE = {}


def column(rows, name: str, edges, train_rows=None) -> np.ndarray:
    """One feature column by name. Derived columns are computed here."""
    if name == "dur_bucket":
        if edges is None:
            raise ValueError("'dur_bucket' needs duration edges; data.edges is None")
        return dur_bucket(rows.duration_ms, edges)
    if name == "hourmin_bucket":
        return np.asarray(rows.hourmin, dtype=np.int64) // 100
    if name == "user_long_view_rate":
        global _USER_RATES_CACHE
        if id(train_rows) not in _USER_RATES_CACHE:
            if train_rows is None:
                u_arr = np.asarray(rows.user_id)
                return np.full(len(u_arr), 5, dtype=np.int64)
            _USER_RATES_CACHE[id(train_rows)] = compute_user_long_view_rates(train_rows)
        rates_map = _USER_RATES_CACHE[id(train_rows)]
        u_arr = np.asarray(rows.user_id)
        return np.fromiter((rates_map.get(int(u), 5) for u in u_arr), np.int64, len(u_arr))
    if name == "author_long_view_rate":
        global _AUTHOR_RATES_CACHE
        if id(train_rows) not in _AUTHOR_RATES_CACHE:
            if train_rows is None:
                a_arr = np.asarray(rows.author_id)
                return np.full(len(a_arr), 5, dtype=np.int64)
            _AUTHOR_RATES_CACHE[id(train_rows)] = compute_author_long_view_rates(train_rows)
        rates_map = _AUTHOR_RATES_CACHE[id(train_rows)]
        a_arr = np.asarray(rows.author_id)
        return np.fromiter((rates_map.get(int(a), 5) for a in a_arr), np.int64, len(a_arr))
    if name == "user_impression_count":
        global _USER_COUNTS_CACHE
        if id(train_rows) not in _USER_COUNTS_CACHE:
            if train_rows is None:
                u_arr = np.asarray(rows.user_id)
                return np.full(len(u_arr), 3, dtype=np.int64)
            _USER_COUNTS_CACHE[id(train_rows)] = compute_user_impression_counts(train_rows)
        counts_map = _USER_COUNTS_CACHE[id(train_rows)]
        u_arr = np.asarray(rows.user_id)
        return np.fromiter((counts_map.get(int(u), 3) for u in u_arr), np.int64, len(u_arr))

    try:
        v = getattr(rows, name)
    except AttributeError:
        v = None
    if v is None:
        from contract import columns_of
        raise ValueError(
            f"field {name!r} is not carried by this dataset. Available columns: "
            f"{list(columns_of(rows))}, plus derived 'dur_bucket' and 'hourmin_bucket'. "
            f"Add a derived column in features.column() rather than assuming one exists.")
    return np.asarray(v)


def encode(splits: dict, edges, fields) -> tuple[dict, int]:
    fields = tuple(fields)
    if not fields:
        raise ValueError("no fields to encode")
    if "train" not in splits:
        raise ValueError("encode() needs splits['train'] to build the vocabulary")
    tr = splits["train"]

    vocabs, dims = [], []
    for f in fields:
        vals = np.unique(column(tr, f, edges, train_rows=tr))
        vocabs.append({int(v): i for i, v in enumerate(vals)})
        dims.append(len(vals) + 1)
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    enc = {}
    for name, rows in splits.items():
        X = np.empty((len(rows), len(fields)), dtype=np.int32)
        for i, f in enumerate(fields):
            c = column(rows, f, edges, train_rows=tr)
            vmap, unk = vocabs[i], len(vocabs[i])
            X[:, i] = np.fromiter((vmap.get(int(v), unk) for v in c),
                                  np.int64, len(rows)) + offsets[i]
        enc[name] = X
    return enc, int(sum(dims))
