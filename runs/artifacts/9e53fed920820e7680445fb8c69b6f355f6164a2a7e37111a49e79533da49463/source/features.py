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


def compute_user_counts(splits) -> dict[int, np.ndarray]:
    """Compute the number of impressions per user in train, mapping to each split."""
    tr_users = np.asarray(splits["train"].user_id)
    unique_u, counts = np.unique(tr_users, return_counts=True)
    count_map = dict(zip(unique_u, counts))

    # Define bucket edges for counts based on train distribution
    edges = np.quantile(counts, np.linspace(0.1, 0.9, 9))

    res = {}
    for name, rows in splits.items():
        u = np.asarray(rows.user_id)
        raw_counts = np.array([count_map.get(int(x), 1) for x in u], dtype=np.float64)
        res[name] = np.searchsorted(edges, raw_counts).astype(np.int64)
    return res


def column(rows, name: str, edges, cached_derived=None) -> np.ndarray:
    """One feature column by name. Derived columns are computed here."""
    if name == "dur_bucket":
        if edges is None:
            raise ValueError("'dur_bucket' needs duration edges; data.edges is None")
        return dur_bucket(rows.duration_ms, edges)
    if name == "hourmin_bucket":
        return np.asarray(rows.hourmin, dtype=np.int64) // 100
    if name == "user_count":
        if cached_derived is not None and name in cached_derived:
            return cached_derived[name]
        # Fallback if called directly
        u = np.asarray(rows.user_id)
        return np.zeros_like(u, dtype=np.int64)

    try:
        v = getattr(rows, name)
    except AttributeError:
        v = None
    if v is None:
        from contract import columns_of
        raise ValueError(
            f"field {name!r} is not carried by this dataset. Available columns: "
            f"{list(columns_of(rows))}, plus derived 'dur_bucket', 'hourmin_bucket', and 'user_count'. "
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

    derived_cache = {}
    if "user_count" in fields:
        derived_cache["user_count"] = compute_user_counts(splits)

    vocabs, dims = [], []
    for f in fields:
        if f == "user_count":
            vals = np.unique(derived_cache["user_count"]["train"])
        else:
            vals = np.unique(column(tr, f, edges))
        vocabs.append({int(v): i for i, v in enumerate(vals)})
        dims.append(len(vals) + 1)                       # +1 UNK
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    enc = {}
    for name, rows in splits.items():
        X = np.empty((len(rows), len(fields)), dtype=np.int32)
        for i, f in enumerate(fields):
            if f == "user_count":
                c = derived_cache["user_count"][name]
            else:
                c = column(rows, f, edges)
            vmap, unk = vocabs[i], len(vocabs[i])
            X[:, i] = np.fromiter((vmap.get(int(v), unk) for v in c),
                                  np.int64, len(rows)) + offsets[i]
        enc[name] = X
    return enc, int(sum(dims))
