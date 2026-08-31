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


def compute_user_frequencies(train_user_ids: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    """Compute user interaction counts from train set only and define buckets."""
    unique_users, counts = np.unique(train_user_ids, return_counts=True)
    # Compute quantiles for bucketing (e.g., 10 bins)
    quantiles = np.percentile(counts, np.linspace(0, 100, 11))
    # Ensure unique edges for searchsorted
    edges = np.unique(quantiles[1:-1])

    user_to_freq = {}
    for uid, cnt in zip(unique_users, counts):
        bucket = int(np.searchsorted(edges, cnt, side='right'))
        user_to_freq[int(uid)] = bucket
    return user_to_freq, edges


# Global cache for train-derived user frequencies to use across splits
_TRAIN_FREQ_CACHE = {}


def user_freq_bucket(user_ids, tr_user_ids) -> np.ndarray:
    """Map user interaction count to a bucket based on train distribution."""
    u_arr = np.asarray(user_ids, dtype=np.int64)
    if id(tr_user_ids) not in _TRAIN_FREQ_CACHE:
        u_map, edges = compute_user_frequencies(np.asarray(tr_user_ids, dtype=np.int64))
        _TRAIN_FREQ_CACHE[id(tr_user_ids)] = (u_map, edges)

    u_map, edges = _TRAIN_FREQ_CACHE[id(tr_user_ids)]
    max_bucket = len(edges)
    return np.fromiter((u_map.get(int(u), max_bucket) for u in u_arr), np.int64, len(u_arr))


def column(rows, name: str, edges, tr_rows=None) -> np.ndarray:
    """One feature column by name. Derived columns are computed here."""
    if name == "dur_bucket":
        if edges is None:
            raise ValueError("'dur_bucket' needs duration edges; data.edges is None")
        return dur_bucket(rows.duration_ms, edges)
    if name == "hourmin_bucket":
        return np.asarray(rows.hourmin, dtype=np.int64) // 100
    if name == "user_freq_bucket":
        if tr_rows is None:
            raise ValueError("'user_freq_bucket' needs train rows to compute frequencies")
        return user_freq_bucket(rows.user_id, tr_rows.user_id)
    try:
        v = getattr(rows, name)
    except AttributeError:
        v = None
    if v is None:
        from contract import columns_of
        raise ValueError(
            f"field {name!r} is not carried by this dataset. Available columns: "
            f"{list(columns_of(rows))}, plus derived 'dur_bucket', 'hourmin_bucket', and 'user_freq_bucket'. "
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
        vals = np.unique(column(tr, f, edges, tr_rows=tr))
        vocabs.append({int(v): i for i, v in enumerate(vals)})
        dims.append(len(vals) + 1)                       # +1 UNK
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    enc = {}
    for name, rows in splits.items():
        X = np.empty((len(rows), len(fields)), dtype=np.int32)
        for i, f in enumerate(fields):
            c = column(rows, f, edges, tr_rows=tr)
            vmap, unk = vocabs[i], len(vocabs[i])
            X[:, i] = np.fromiter((vmap.get(int(v), unk) for v in c),
                                  np.int64, len(rows)) + offsets[i]
        enc[name] = X
    return enc, int(sum(dims))
