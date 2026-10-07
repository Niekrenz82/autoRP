from dataclasses import dataclass, field

import pandas as pd

# Generic, dataset-agnostic feature transforms. Which columns to use and how to band them is not decided here:
# it comes in as a FeatureSpec (written by hand for the baseline, later produced by candidate configs / the agent).


@dataclass
class FeatureSpec:
    # bands: column -> (left-closed band edges, band labels); categorical: columns used as-is as categories
    bands: dict[str, tuple[list[float], list[str]]] = field(default_factory=dict)
    categorical: list[str] = field(default_factory=list)

    @property
    def features(self):
        return list(self.bands) + list(self.categorical)


def make_features(data, spec):
    # Turn raw columns into the categorical design matrix described by `spec` (one column per rating factor)
    X = pd.DataFrame(index=data.index)
    for var, (edges, labels) in spec.bands.items():
        X[var] = pd.cut(data[var], edges, right=False, labels=labels)
    for var in spec.categorical:
        X[var] = data[var].astype("category")
    return X


def set_base_levels(X, exposure):
    # Reorder categories so the highest-exposure level comes first (glum's drop_first uses it as the reference)
    X = X.copy()
    for var in X.columns:
        base = exposure.groupby(X[var], observed=True).sum().idxmax()
        cats = list(X[var].cat.categories)
        X[var] = X[var].cat.reorder_categories([base] + [c for c in cats if c != base])
    return X
