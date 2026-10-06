import pandas as pd
from pathlib import Path

data_path = Path(__file__).parents[1] / "data"

def load_sev():
    return pd.read_parquet(data_path / "freMTPL2sev.parquet")
    
def load_freq():
    return pd.read_parquet(data_path / "freMTPL2freq.parquet")