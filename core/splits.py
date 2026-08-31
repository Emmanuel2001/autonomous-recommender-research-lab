"""The four-way partition, with validation labels quarantined from the strategy.

WHY TWO LOADERS
---------------
Under full-catalog ranking the strategy does not need validation ROWS at all --
it needs training data and the list of users to produce rankings for. So the
quarantine is total rather than best-effort:

    load_for_strategy()  -> train rows (features+labels) + the user lists.
                            Contains NO validation or test label anywhere.
    load_for_scorer()     -> relevance sets and train-seen sets. Scorer only.

WHAT IS IMMUTABLE
-----------------
The partition train / search_val / audit_val / test_quarantine is CORE. It is
content-hashed into a manifest, and the scorer refuses to score against a
manifest it does not recognise. If the agent could move these boundaries its
numbers would stop being comparable to the baseline.

How the strategy subdivides TRAIN for its own early stopping is NOT core -- that
is a legitimate modelling choice, recorded in the config hash. See
`default_train_subdivision`.

THE BRIEF'S "test" IS QUARANTINED
---------------------------------
The brief calls the last 50% of the eval file "test" but also says the hidden
test is scored exactly once. Until the Starter Kit disambiguates, this module
never hands it out. `load_quarantined_test()` exists, is never called by the
loop, and logs a warning.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# The data directory is overridable so an ISOLATED strategy process can be pointed
# at a prepared root containing only the files it is permitted to read. The default
# is the real dataset, so nothing outside a sandbox changes behaviour.
DATA = Path(os.environ.get("TRACK2_DATA_ROOT")
            or Path(__file__).resolve().parents[1] / "KuaiRand-Pure" / "data")
TRAIN_F = "log_standard_4_08_to_4_21_pure.csv"
EVAL_F = "log_standard_4_22_to_5_08_pure.csv"
RAND_F = "log_random_4_22_to_5_08_pure.csv"   # randomly-exposed; opt-in only

LABEL = "is_click"                       # the brief: "click = positive (fixed)"
AUX_SIGNALS = ["is_like", "is_follow", "is_comment", "is_forward", "is_hate",
               "long_view", "is_profile_enter"]
CONTINUOUS = ["play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time"]
CONTEXT = ["date", "time_ms", "tab"]
ALL_LABELS = [LABEL] + AUX_SIGNALS + CONTINUOUS   # everything derived from the interaction
KEYS = ["user_id", "video_id"]

AUDIT_FRACTION = 30                      # percent of val held back for audit
TRAIN_DEV_BOUNDARY = 20220418            # default only; strategy may override


def _row_bucket(user_id: np.ndarray, video_id: np.ndarray, time_ms: np.ndarray) -> np.ndarray:
    """Deterministic 0-99 bucket per row. Order-independent and RNG-free, so the
    split is reproducible from the data alone -- no seed to lose."""
    out = np.empty(len(user_id), dtype=np.int64)
    for n, (u, v, t) in enumerate(zip(user_id, video_id, time_ms)):
        h = hashlib.sha256(f"{u}:{v}:{t}".encode()).digest()
        out[n] = int.from_bytes(h[:8], "big") % 100
    return out


def _content_hash(df: pd.DataFrame) -> str:
    """Hash of the split's identity: sorted (user, item, label) triples."""
    t = df[KEYS + [LABEL]].to_numpy(dtype=np.int64)
    t = t[np.lexsort((t[:, 2], t[:, 1], t[:, 0]))]
    return hashlib.sha256(t.tobytes()).hexdigest()


def _relevance(df: pd.DataFrame) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Deduped relevant item set per user, for users with >=1 positive.

    Dedup matters: repeat clicks on the same video are ONE relevant item, and
    counting them twice deflates Recall's denominator.
    """
    pos = df.loc[df[LABEL] == 1, KEYS].to_numpy(dtype=np.int64)
    if len(pos) == 0:
        return np.array([], dtype=np.int64), {}
    pos = pos[np.lexsort((pos[:, 1], pos[:, 0]))]
    users, starts = np.unique(pos[:, 0], return_index=True)
    rel = {int(u): np.unique(g[:, 1])
           for u, g in zip(users, np.split(pos, starts[1:]))}
    return users.astype(np.int64), rel


@dataclass(frozen=True)
class Partition:
    """Metadata only. Never holds a validation label."""
    manifest: dict
    train_users: np.ndarray
    search_val_users: np.ndarray = field(repr=False)
    audit_val_users: np.ndarray = field(repr=False)

    @property
    def manifest_sha256(self) -> str:
        from core.ledger import canonical
        return hashlib.sha256(canonical(self.manifest)).hexdigest()


# HOW THE EVAL WINDOW IS CUT INTO validation / test.
#
# This is the single most load-bearing assumption in the project and it was hardcoded
# until 2026-08-27. The brief and the organisers' own reference implementation DISAGREE:
#
#   positional  ev.iloc[:len//2] / ev.iloc[len//2:]
#               Matches the brief's wording literally -- "4/22-5/08 standard first 50%
#               -> validation, last 50% -> test" -- and lands on exactly 50.0% / 50.0%.
#               The file is USER-sorted, not date-sorted (dates jump 0422, 0425, 0429,
#               ...), so both halves span the whole window.
#
#   temporal    date 4/22-4/28 -> val, 4/29-5/08 -> test
#               What CWM actually does (`train_model2.py:86-88`), and the brief names
#               CWM as the reference for the evaluation script. But it yields
#               42.3% / 57.7%, which is NOT "first 50% / last 50%".
#
# MEASURED CONSEQUENCE of getting it wrong: the two readings' test sets share only
# 47% of their rows (Jaccard 0.474). If the hidden test is the temporal one, **28.0% of
# it sits inside our search_val** -- the set the agent optimises against -- so the label
# quarantine would be protecting a set defined by the wrong rule. The engineering is
# unaffected (everything downstream derives from here); the definition is the risk.
#
# Default stays `positional` because it matches the brief's own words exactly. The rule
# is recorded in the manifest, so its hash changes when the rule changes and ledger
# entries made under one rule can never be silently compared against the other.
SPLIT_RULE = os.environ.get("TRACK2_SPLIT_RULE", "positional")
VALID_SPLIT_RULES = ("positional", "temporal")
TEMPORAL_VAL = (20220422, 20220428)      # CWM train_model2.py:87
TEMPORAL_TEST = (20220429, 20220508)     # CWM train_model2.py:88


def _cut_eval(ev: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rule == "positional":
        cut = len(ev) // 2
        return ev.iloc[:cut].copy(), ev.iloc[cut:].copy()
    if rule == "temporal":
        v = ev[(ev.date >= TEMPORAL_VAL[0]) & (ev.date <= TEMPORAL_VAL[1])].copy()
        t = ev[(ev.date >= TEMPORAL_TEST[0]) & (ev.date <= TEMPORAL_TEST[1])].copy()
        if len(v) + len(t) != len(ev):
            raise ValueError(f"temporal cut lost rows: {len(v)}+{len(t)} != {len(ev)}")
        return v, t
    raise ValueError(f"unknown split rule {rule!r}; choose from {VALID_SPLIT_RULES}")


def _load_raw(rule: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = KEYS + CONTEXT + ALL_LABELS
    train = pd.read_csv(DATA / TRAIN_F, usecols=cols)
    ev = pd.read_csv(DATA / EVAL_F, usecols=cols)
    val, test = _cut_eval(ev, rule or SPLIT_RULE)
    return train, val, test


def _partition_val(val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = _row_bucket(val.user_id.to_numpy(), val.video_id.to_numpy(), val.time_ms.to_numpy())
    is_audit = b < AUDIT_FRACTION
    return val.loc[~is_audit].copy(), val.loc[is_audit].copy()


def build_manifest() -> dict:
    train, val, test = _load_raw()
    sv, av = _partition_val(val)
    n_items = int(pd.read_csv(DATA / "video_features_basic_pure.csv",
                              usecols=["video_id"]).video_id.nunique())
    m = {
        "dataset": "KuaiRand-Pure",
        "label": LABEL,
        "audit_fraction_pct": AUDIT_FRACTION,
        "n_items_catalog": n_items,
        # Recorded so the manifest hash differs between the two readings of the brief.
        # A ledger entry made under one rule can then never be silently compared with
        # one made under the other -- the scorer binds this hash into every entry.
        "split_rule": SPLIT_RULE,
        "splits": {},
    }
    for name, df in (("train", train), ("search_val", sv),
                     ("audit_val", av), ("test_quarantine", test)):
        u, rel = _relevance(df)
        m["splits"][name] = {
            "rows": int(len(df)),
            "users": int(df.user_id.nunique()),
            "users_with_positive": int(len(u)),
            "positives_deduped": int(sum(len(v) for v in rel.values())),
            "click_rate": round(float(df[LABEL].mean()), 6),
            "date_min": int(df.date.min()), "date_max": int(df.date.max()),
            "content_sha256": _content_hash(df),
        }
    return m


# ------------------------------------------------------------------ strategy side
def load_for_strategy(use_log_random: bool = False) -> tuple[pd.DataFrame, Partition]:
    """Everything the strategy container is allowed to see.

    Returns TRAIN rows (features + all labels -- legitimate) and the user lists
    it must produce rankings for. No validation or test label is present.

    use_log_random appends log_random_4_22_to_5_08 (1,186,059 rows of RANDOMLY
    EXPOSED interactions) to the training data. Measured: only 67 of 144,042
    quarantined-test pairs appear in it (0.05%), and its click rate is 17.6% vs
    44.3% in the standard logs -- that gap IS the exposure bias this dataset exists
    to study. It is in-dataset, not external.

    DEFAULT OFF. The brief's split names only the two log_standard files, so
    whether log_random may be trained on is a RULES question, not a measured one.
    Pair overlap being low is evidence about leakage, not permission.
    """
    train, val, test = _load_raw()
    if use_log_random:
        rnd = pd.read_csv(DATA / RAND_F, usecols=KEYS + CONTEXT + ALL_LABELS)
        rnd["_source"] = "log_random"
        train = train.copy(); train["_source"] = "log_standard"
        train = pd.concat([train, rnd], axis=0, ignore_index=True)
    sv, av = _partition_val(val)
    part = Partition(
        manifest=build_manifest(),
        train_users=np.sort(train.user_id.unique()),
        search_val_users=_relevance(sv)[0],
        audit_val_users=_relevance(av)[0],
    )
    del val, test, sv, av              # explicit: labels do not escape this function
    return train, part


#: Held-out window under the CURRENT task definition. A sandbox may legitimately expose
#: the eval file (the task ranks eval rows, so it needs their features) provided the
#: labels in this window are redacted.
HELD_OUT_RANGE = (20220429, 20220508)


def _assert_heldout_redacted(sample: int = 4000) -> None:
    """Inside a sandbox, verify the eval file's held-out labels are redacted.

    The original version of this guard refused if the eval file was PRESENT at all,
    which was right when the sandbox simply omitted it. The current task ranks eval rows,
    so the sandbox ships a redacted copy and presence is expected -- the invariant that
    still matters is that the held-out labels are gone. Checking absence instead of
    redaction would fail on a correctly-built sandbox, which is how this was found.
    """
    import csv
    f = DATA / EVAL_F
    if not f.exists():
        return                                  # omitted entirely: also fine
    from core.isolation import OUTCOME_COLUMNS
    lo, hi = HELD_OUT_RANGE
    seen = 0
    leaked: dict[str, int] = {}
    with f.open(newline="") as fh:
        rd = csv.DictReader(fh)
        cols = [c for c in OUTCOME_COLUMNS if c in (rd.fieldnames or [])]
        for r in rd:
            if not (lo <= int(r["date"]) <= hi):
                continue
            seen += 1
            for c in cols:
                if r[c] not in ("0", "0.0", ""):
                    leaked[c] = leaked.get(c, 0) + 1
            if seen >= sample:
                break
    if leaked:
        raise RuntimeError(
            f"sandbox misconfigured: held-out OUTCOME columns are not redacted in "
            f"{EVAL_F} -- {leaked} non-zero in the first {seen:,} held-out rows. Every "
            f"outcome must be blanked, not just the designated label: held-out is_click "
            f"correlates +0.75 with long_view (P(lv|click)=0.70 vs 0.002), so leaving it "
            f"readable is a near-oracle shortcut. Refusing to run.")


def load_train_only(use_log_random: bool = False) -> tuple[pd.DataFrame, dict]:
    """The ONLY loader an isolated strategy calls. Reads the train log and nothing else.

    WHY THIS IS SEPARATE FROM load_for_strategy
    -------------------------------------------
    `load_for_strategy()` calls `build_manifest()`, which calls `_load_raw()`, which
    reads all three csvs -- the eval file included. It never RETURNS a validation
    label, and the tests that assert so are correct, but every one of those labels
    still lands in the address space that runs the agent's mutable code. Under a
    kernel read-deny that call does not merely leak, it fails outright. Found while
    building core/isolation.py; see that module's docstring.

    This loader needs no channel from the label holder. Everything it returns comes
    from files a sandboxed process is allowed to open:

        n_items_catalog  video_features_basic_pure.csv
        rank_users       user_features_pure.csv -- ALL 27,285 users, contiguous 0..27284
        n_users_cap      the same

    Ranking every user rather than the scoreable subset is the point. The scoreable
    subset is "users with at least one click in the val window", which is derived
    from validation labels; handing it over would have been a real leak, small but
    not provably inert (a strategy could filter its training signal to those users
    and thereby use label information). Ranking all users removes the question
    instead of arguing about it, and the scorer subsets to the users it scores, so
    the metrics are unchanged -- asserted in tests/test_scorer.py.
    """
    cols = KEYS + CONTEXT + ALL_LABELS
    if os.environ.get("TRACK2_SANDBOX") == "1":
        _assert_heldout_redacted()
    train = pd.read_csv(DATA / TRAIN_F, usecols=cols)
    if use_log_random:
        rnd = pd.read_csv(DATA / RAND_F, usecols=cols)
        rnd["_source"] = "log_random"
        train = train.copy(); train["_source"] = "log_standard"
        train = pd.concat([train, rnd], axis=0, ignore_index=True)

    items = pd.read_csv(DATA / "video_features_basic_pure.csv", usecols=["video_id"])
    all_users = np.sort(pd.read_csv(DATA / "user_features_pure.csv",
                                    usecols=["user_id"]).user_id.unique()).astype(np.int64)
    meta = {
        "n_items_catalog": int(items.video_id.nunique()),
        "rank_users": all_users,
        "n_users_cap": int(all_users.max()) + 1,
        "train_rows": int(len(train)),
        "use_log_random": bool(use_log_random),
        "data_root": str(DATA),
    }
    return train, meta


def default_train_subdivision(train: pd.DataFrame, boundary: int = TRAIN_DEV_BOUNDARY):
    """Temporal fit/dev split inside TRAIN. A strategy parameter, not core.

    Temporal rather than random because the tail days' click rate (43.5-44.5%)
    brackets search_val's, so the dev signal tracks the eval distribution.
    """
    return train.loc[train.date < boundary].copy(), train.loc[train.date >= boundary].copy()


# -------------------------------------------------------------------- scorer side
@dataclass(frozen=True)
class ScorerView:
    manifest: dict
    manifest_sha256: str
    relevance: dict[str, dict[int, np.ndarray]]     # split -> user -> deduped items
    users: dict[str, np.ndarray]
    train_seen: dict[int, np.ndarray]               # user -> items seen in TRAIN
    n_items: int
    catalog_items: np.ndarray                       # raw video_ids, the rankable universe


def load_for_scorer(include_quarantine: bool = False) -> ScorerView:
    """Labels live here and nowhere else."""
    train, val, test = _load_raw()
    sv, av = _partition_val(val)
    man = build_manifest()

    seen_src = train[KEYS].to_numpy(dtype=np.int64)
    seen_src = seen_src[np.lexsort((seen_src[:, 1], seen_src[:, 0]))]
    su, ss = np.unique(seen_src[:, 0], return_index=True)
    train_seen = {int(u): np.unique(g[:, 1])
                  for u, g in zip(su, np.split(seen_src, ss[1:]))}

    rel, users = {}, {}
    parts = [("search_val", sv), ("audit_val", av)]
    if include_quarantine:
        parts.append(("test_quarantine", test))
    for name, df in parts:
        u, r = _relevance(df)
        users[name], rel[name] = u, r

    catalog = np.sort(pd.read_csv(DATA / "video_features_basic_pure.csv",
                                  usecols=["video_id"]).video_id.unique()).astype(np.int64)
    from core.ledger import canonical
    return ScorerView(man, hashlib.sha256(canonical(man)).hexdigest(),
                      rel, users, train_seen, man["n_items_catalog"], catalog)


def load_quarantined_test() -> pd.DataFrame:
    """NEVER called by the loop. Final scoring only, and it announces itself."""
    import warnings
    warnings.warn("QUARANTINED SPLIT ACCESSED -- this must happen exactly once, "
                  "at final scoring, and be recorded in the ledger.", stacklevel=2)
    return _load_raw()[2]
