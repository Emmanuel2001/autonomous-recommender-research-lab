"""Feature encoding. AGENT-EDITABLE.

WHAT THIS EDIT ADDS
-------------------
The parent encoded exactly five categorical fields (user_id, video_id, author_id, tab,
dur_bucket). With that input the FM can only generalise across items through author_id
and the duration bucket, so a video_id seen twice in train is nearly a free parameter and
within-user ranking of tail impressions has nothing to lean on.

This file now *discovers* the columns the loader actually carries and admits the
PRE-IMPRESSION ones as extra encoded fields:

  * `POST_IMPRESSION` is an explicit blacklist of outcome / post-impression columns. It is
    checked before anything else and cannot be overridden by config: admitting any of
    these would leak the label.
  * `DECLINED` holds columns that are pre-impression but that we deliberately refuse for
    reasons written next to them (raw timestamps, calendar date, logging-policy flags).
  * `CANDIDATE_FIELDS` is an ORDERED priority list. Ordering matters because `max_new`
    caps how many extra fields are admitted (each field costs forward/backward time), so
    the item-side and context descriptors -- the ones that transfer across video_ids --
    are considered first, then coarse user-profile attributes (useless on their own for
    within-user ranking, but they cross with item fields inside the FM), then the wide
    onehot_feat block.
  * continuous whitelisted columns are bucketed by TRAIN-ONLY quantiles.

Two invariants kept from the seed:
  * the vocabulary comes from train only -- dev/heldout values that were never seen fall
    into a per-field UNK slot, exactly as before;
  * `column()` still raises with the available names, so a bad field costs one cheap
    refusal instead of a whole fit. `resolve_fields()` catches that and simply skips the
    column, recording why in its report.

ONE PERFORMANCE CHANGE, DECLARED
--------------------------------
The seed mapped values with a Python dict and `np.fromiter` -- one interpreted loop per
field per split. Going from 5 to ~15 fields would have spent a large slice of the fit
budget inside that loop, so the mapping is now vectorised (sorted unique + searchsorted +
an equality check to route misses to UNK). The SEMANTICS are identical: train-only vocab,
per-field UNK, per-field offsets into one shared id space.
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

# Derived columns and the raw columns they need.
DERIVED = {
    "dur_bucket": frozenset({"duration_ms"}),
    "hourmin_bucket": frozenset({"hourmin"}),
}

# Whitelisted columns treated as CONTINUOUS -> train-quantile buckets.
CONTINUOUS = frozenset({
    "video_duration", "duration_ms", "follow_user_num", "fans_user_num",
    "friend_user_num", "register_days", "server_width", "server_height",
})

# Ordered by expected within-user ranking value (see module docstring).
CANDIDATE_FIELDS = (
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


def column(rows, name: str, edges) -> np.ndarray:
    """One feature column by name. Derived columns are computed here."""
    if name == "dur_bucket":
        if edges is None:
            raise ValueError("'dur_bucket' needs duration edges; data.edges is None")
        return dur_bucket(rows.duration_ms, edges)
    if name == "hourmin_bucket":
        # hour of day from `hourmin` (HHMM as an integer).
        return np.asarray(rows.hourmin, dtype=np.int64) // 100
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
    exactly like any other categorical field.

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
    return enc, int(sum(dims))
