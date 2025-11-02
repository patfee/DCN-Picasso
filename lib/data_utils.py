from functools import lru_cache
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

@lru_cache(maxsize=32)
def load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame({"info": [f"CSV not found: {filename}"]})
    return pd.read_csv(path, encoding="utf-8")
