from functools import lru_cache
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def data_dir_exists() -> bool:
    return DATA_DIR.exists()

def list_data_files():
    if not data_dir_exists():
        return []
    return sorted([p.name for p in DATA_DIR.glob("*") if p.is_file()])

def _read_csv_smart(path: Path) -> pd.DataFrame:
    """
    Read CSV robustly:
      - try utf-8-sig (BOM) then utf-8
      - try delimiter auto-detect (engine='python', sep=None)
      - fallback to ';' then ','
    """
    if not path.exists():
        return pd.DataFrame()

    # 1) try sep=None (sniff), utf-8-sig
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        pass
    # 2) sep=None, utf-8
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception:
        pass
    # 3) semicolon
    try:
        return pd.read_csv(path, sep=";", encoding="utf-8-sig")
    except Exception:
        pass
    # 4) comma
    try:
        return pd.read_csv(path, sep=",", encoding="utf-8-sig")
    except Exception:
        pass

    return pd.DataFrame()

@lru_cache(maxsize=32)
def load_csv(filename: str) -> pd.DataFrame:
    """Simple wrapper using the smart reader."""
    path = DATA_DIR / filename
    return _read_csv_smart(path)

def _is_numberlike(s) -> bool:
    try:
        float(str(s).strip().replace("°", "").replace(",", "."))
        return True
    except Exception:
        return False

def _coerce_angle_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("°", "", regex=False).str.strip()
    # convert comma decimal to dot, then to float
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def _find_angle_col(df: pd.DataFrame) -> str:
    """
    Heuristic to find the column that holds folding-jib angles (row index):
    - Prefer a column whose values look numeric angles
    - Otherwise fall back to the first column
    """
    for c in df.columns:
        if df[c].dropna().astype(str).map(_is_numberlike).mean() > 0.8:
            return c
    return df.columns[0]

@lru_cache(maxsize=32)
def load_matrix_csv_flexible(possible_filenames: tuple) -> pd.DataFrame:
    """
    Try a list of filenames. Expect:
      - Rows keyed by folding angles (first angle-like column)
      - Columns named by main angles (header row)
    Return DataFrame indexed by folding_deg (float), columns = main_deg (float).
    """
    for name in possible_filenames:
        path = DATA_DIR / name
        df = _read_csv_smart(path)
        if not df.empty and df.shape[1] >= 2:
            # try to find angle column
            angle_col = _find_angle_col(df)
            # coerce folding angles
            folds = _coerce_angle_series(df[angle_col])
            df = df.drop(columns=[angle_col])
            # clean & coerce main-angle headers
            clean_cols = []
            for c in df.columns:
                cs = str(c).replace("°", "").strip().replace(",", ".")
                clean_cols.append(pd.to_numeric(cs, errors="coerce"))
            df.columns = clean_cols
            # coerce body
            df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
            # set index
            df.index = folds
            # drop non-numeric angle columns or rows
            df = df.loc[~df.index.isna(), [c for c in df.columns if pd.notna(c)]]
            if not df.empty:
                # sort
                return df.sort_index(axis=0).sort_index(axis=1)
    return pd.DataFrame()

def stack_height_outreach(height_df: pd.DataFrame, outreach_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine height and outreach matrices into long form with columns:
      main_deg, folding_deg, height_m, outreach_m
    Only keep finite pairs.
    """
    f_angles = sorted(set(height_df.index).union(set(outreach_df.index)))
    m_angles = sorted(set(height_df.columns).union(set(outreach_df.columns)))
    H = height_df.reindex(index=f_angles, columns=m_angles)
    R = outreach_df.reindex(index=f_angles, columns=m_angles)

    h_long = H.stack().rename("height_m").reset_index(names=["folding_deg", "main_deg"])
    r_long = R.stack().rename("outreach_m").reset_index(names=["folding_deg", "main_deg"])
    df = pd.merge(h_long, r_long, on=["folding_deg", "main_deg"], how="inner")
    df = df[np.isfinite(df["height_m"]) & np.isfinite(df["outreach_m"])]
    # ensure float
    df["folding_deg"] = df["folding_deg"].astype(float)
    df["main_deg"] = df["main_deg"].astype(float)
    return df.sort_values(["main_deg", "folding_deg"]).reset_index(drop=True)
