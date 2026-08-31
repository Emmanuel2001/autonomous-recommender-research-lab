"""Feature encoding. AGENT-EDITABLE.

Seeded with the official baseline's encoding: five categorical fields mapped to one
contiguous id space, vocabulary from TRAIN only, unseen values falling into a per-field
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


_USER_RATES_CACHE = {}
_AUTHOR_RATES_CACHE = {}


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
    if name == "user_tab_x_author_id":
        u_arr = np.asarray(rows.user_id)
        t_arr = np.asarray(rows.tab)
        a_arr = np.asarray(rows.author_id)
        return np.array([f"{u}_{t}_{a}" for u, t, a in zip(u_arr, t_arr, a_arr)], dtype=object)

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
    """Categorical fields -> contiguous ids with per-field offsets. Vocab from TRAIN.

    `splits` maps a name to a row object and MUST contain "train". Returns
    ({name: int32 array (rows, n_fields)}, total_vocab_size).
    """
    fields = tuple(fields)
    if not fields:
        raise ValueError("no fields to encode")
    if "train" not in splits:
        raise ValueError("encode() needs splits['train'] to build the vocabulary")
    tr = splits["train"]

    vocabs, dims = [], []
    for f in fields:
        vals = np.unique(column(tr, f, edges, train_rows=tr))
        vocabs.append({v: i for i, v in enumerate(vals)})
        dims.append(len(vals) + 1)                       # +1 UNK
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    enc = {}
    for name, rows in splits.items():
        X = np.empty((len(rows), len(fields)), dtype=np.int32)
        for i, f in enumerate(fields):
            c = column(rows, f, edges, train_rows=tr)
            vmap, unk = vocabs[i], len(vocabs[i])
            X[:, i] = np.fromiter((vmap.get(v, unk) for v in c),
                                  np.int64, len(rows)) + offsets[i]
        enc[name] = X
    return enc, int(sum(dims))
