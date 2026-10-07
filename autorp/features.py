from dataclasses import asdict, dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

# Generic, dataset-agnostic feature transforms. Which columns to use and how is not decided here: it comes in as a
# FeatureSpec (written by hand for the baseline, later produced by candidate configs / the agent).
#
# Safety model: the agent never writes code, it only outputs a spec as data (JSON). FeatureSpec.from_dict() only
# accepts the transform types in TRANSFORMS, validate() checks the spec against the actual dataset, and
# make_features() is the only code that applies transforms to data.

MISSING = "Missing"   # explicit level for missing values, so they get their own relativity instead of crashing the fit


@dataclass
class Band:
    """Continuous column -> ordered bands, one relativity per band (captures non-linear shapes, e.g. driver age).

    Bands are left-closed: [edges[i], edges[i+1]). Values below the first / above the last edge go to the
    first / last band, so new data outside the training range still gets a price.
    """
    kind: ClassVar[str] = "band"
    column: str
    edges: list[float]
    labels: list[str] | None = None   # default: "[18, 21)" style labels

    def apply(self, data):
        bins = [-np.inf, *self.edges[1:-1], np.inf]
        labels = self.labels or [f"[{lo}, {hi})" for lo, hi in zip(self.edges[:-1], self.edges[1:])]
        x = pd.cut(data[self.column], bins, right=False, labels=labels)
        return x.cat.add_categories(MISSING).fillna(MISSING) if x.isna().any() else x


@dataclass
class Categorical:
    """Categorical column, optionally with levels merged into groups (e.g. thin regions -> "Other")."""
    kind: ClassVar[str] = "categorical"
    column: str
    groups: dict[str, str] = field(default_factory=dict)   # level -> group; levels not listed are kept as they are

    def apply(self, data):
        raw = data[self.column]
        x = raw.astype(str).where(raw.notna(), MISSING)   # compare as strings, so JSON specs match numeric codes too
        return x.replace(self.groups).astype("category")


@dataclass
class Numeric:
    """Continuous column as a single linear term, optionally clipped and log-transformed.

    With a log link, log(x) as a term gives a power curve x^beta: a smooth effect with one coefficient.
    """
    kind: ClassVar[str] = "numeric"
    column: str
    clip: tuple[float, float] | None = None   # cap extreme values before the transform
    log: bool = False

    def apply(self, data):
        x = data[self.column].astype(float)
        if self.clip is not None:
            x = x.clip(*self.clip)
        return np.log(x) if self.log else x


TRANSFORMS = {t.kind: t for t in (Band, Categorical, Numeric)}   # the complete whitelist


@dataclass
class FeatureSpec:
    transforms: list[Band | Categorical | Numeric] = field(default_factory=list)

    @property
    def features(self):
        return [t.column for t in self.transforms]

    @classmethod
    def from_dict(cls, spec):
        # The entry point for untrusted specs (agent output, config files): unknown types or fields raise an error
        transforms = []
        for item in spec["transforms"]:
            item = dict(item)
            kind = item.pop("type", None)
            if kind not in TRANSFORMS:
                raise ValueError(f"Unknown transform type {kind!r}; allowed: {list(TRANSFORMS)}")
            transforms.append(TRANSFORMS[kind](**item))   # a field the transform doesn't have raises TypeError
        return cls(transforms)

    def to_dict(self):
        # JSON-ready form, e.g. for logging the spec to MLflow next to the model
        return {"transforms": [{"type": t.kind, **asdict(t)} for t in self.transforms]}


def validate(spec, data):
    # Check a spec against the actual dataset before fitting; raises with every problem found
    problems = []
    columns = spec.features
    problems += [f"column {c!r} is used more than once" for c in set(columns) if columns.count(c) > 1]
    for t in spec.transforms:
        if t.column not in data.columns:
            problems.append(f"{t.kind} {t.column!r}: column not in dataset")
            continue
        x = data[t.column]
        if isinstance(t, Band):
            edges = np.asarray(t.edges, dtype=float)
            if len(edges) < 3 or np.any(np.diff(edges) <= 0):
                problems.append(f"band {t.column!r}: need at least 3 strictly increasing edges")
            if t.labels is not None and len(t.labels) != len(edges) - 1:
                problems.append(f"band {t.column!r}: {len(t.labels)} labels for {len(edges) - 1} bands")
            if not pd.api.types.is_numeric_dtype(x):
                problems.append(f"band {t.column!r}: column is not numeric")
        elif isinstance(t, Categorical):
            unknown = set(t.groups) - set(x.dropna().astype(str).unique())
            if unknown:
                problems.append(f"categorical {t.column!r}: grouped levels not in data: {sorted(unknown)}")
        elif isinstance(t, Numeric):
            if not pd.api.types.is_numeric_dtype(x):
                problems.append(f"numeric {t.column!r}: column is not numeric")
                continue
            if x.isna().any():
                problems.append(f"numeric {t.column!r}: has missing values; use a band instead")
            if t.clip is not None and t.clip[0] >= t.clip[1]:
                problems.append(f"numeric {t.column!r}: clip lower bound must be below upper bound")
            lowest = max(x.min(), t.clip[0]) if t.clip is not None else x.min()
            if t.log and lowest <= 0:
                problems.append(f"numeric {t.column!r}: log needs positive values (min {lowest})")
    if problems:
        raise ValueError("Invalid feature spec:\n- " + "\n- ".join(problems))


def make_features(data, spec):
    # Apply a spec to raw data: one column per transform, ready for the GLM
    return pd.DataFrame({t.column: t.apply(data) for t in spec.transforms}, index=data.index)


def set_base_levels(X, exposure):
    # Reorder categories so the highest-exposure level comes first (glum's drop_first uses it as the reference)
    X = X.copy()
    for var in X.columns:
        if not isinstance(X[var].dtype, pd.CategoricalDtype):
            continue   # numeric terms have no reference level
        base = exposure.groupby(X[var], observed=True).sum().idxmax()
        cats = list(X[var].cat.categories)
        X[var] = X[var].cat.reorder_categories([base] + [c for c in cats if c != base])
    return X
