import pandas as pd

# Band edges (left-closed) and labels for the continuous rating factors
BANDS = {
    "DrivAge":    ([18, 21, 26, 31, 41, 51, 71, 101], ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]),
    "BonusMalus": ([50, 51, 60, 70, 80, 100, 231],    ["50", "51-59", "60-69", "70-79", "80-99", "100+"]),
    "VehAge":     ([0, 1, 5, 10, 15, 101],            ["0", "1-4", "5-9", "10-14", "15+"]),
    "VehPower":   ([4, 5, 6, 7, 8, 9, 16],            ["4", "5", "6", "7", "8", "9+"]),
}
CATEGORICAL = ["Area", "VehGas"]
FEATURES = list(BANDS) + CATEGORICAL


def make_features(data):
    # Turn raw rating factors into the banded categorical design used by the GLMs
    X = pd.DataFrame(index=data.index)
    for var, (edges, labels) in BANDS.items():
        X[var] = pd.cut(data[var], edges, right=False, labels=labels)
    for var in CATEGORICAL:
        X[var] = data[var].astype(str).str.strip("'").astype("category")   # raw VehGas comes quoted, e.g. "'Regular'"
    return X


def set_base_levels(X, exposure):
    # Reorder categories so the highest-exposure level comes first (glum's drop_first uses it as the reference)
    X = X.copy()
    for var in X.columns:
        base = exposure.groupby(X[var], observed=True).sum().idxmax()
        cats = list(X[var].cat.categories)
        X[var] = X[var].cat.reorder_categories([base] + [c for c in cats if c != base])
    return X
