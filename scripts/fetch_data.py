from sklearn.datasets import fetch_openml
import pandas as pd
from pathlib import Path


freMTPL2freq = fetch_openml("freMTPL2freq", as_frame=True).frame
freMTPL2sev  = fetch_openml("freMTPL2sev", as_frame=True).frame

freMTPL2freq["IDpol"] = freMTPL2freq["IDpol"].astype(int)
freMTPL2sev["IDpol"] = freMTPL2sev["IDpol"].astype(int)


data_path = Path(__file__).resolve().parents[1] / "data"

for name, df in [("freMTPL2freq", freMTPL2freq), ("freMTPL2sev", freMTPL2sev)]:
    path = data_path / f"{name}.parquet"
    df.to_parquet(path, index=False)

print("done")