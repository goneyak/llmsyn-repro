import ast
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — all anchored to the repository root via __file__, so the script
# can be invoked from any working directory.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
SYN_IN_DIR = BASE_DIR / "outputs" / "synthetic"
# Real MIMIC flat-file must be placed here by the user (not committed to repo).
REAL_IN_PATH = BASE_DIR / "data" / "input" / "real_mimic.csv"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Synthetic files: fname → patient-ID prefix
SYN_FILES = {
    "syn_full.csv": "SYNFULL_",
    "syn_prior.csv": "SYNPRIOR_",
    "syn_base.csv": "SYNBASE_",
}

OUT_COLS = [
    "PATIENT_ID", "AGE", "LANGUAGE", "RELIGION", "MARITAL_STATUS", "ETHNICITY",
    "INSURANCE", "HOSPITAL_EXPIRE_FLAG", "MAIN_DIAGNOSIS", "COMORBIDITIES", "PROCEDURES",
]


def to_list(val):
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
    except Exception:
        return [s]
    return [s]


def normalize_df(df: pd.DataFrame, *, is_real: bool, prefix: str) -> pd.DataFrame:
    """Rename columns to canonical schema, assign PATIENT_ID, fill missing columns."""
    df = df.rename(
        columns={
            "MAIN_ICD9": "MAIN_DIAGNOSIS",
            "main_diagnosis": "MAIN_DIAGNOSIS",
            "CO-MORBIDITIES": "COMORBIDITIES",
            "co-morbidities": "COMORBIDITIES",
            "PROCEDURES_ICD9": "PROCEDURES",
            "procedures": "PROCEDURES",
        }
    )

    if is_real:
        if "SUBJECT_ID" in df.columns:
            df["PATIENT_ID"] = df["SUBJECT_ID"].astype(str)
        elif "PATIENT_ID" in df.columns:
            df["PATIENT_ID"] = df["PATIENT_ID"].astype(str)
        else:
            raise ValueError("Real MIMIC data has no SUBJECT_ID or PATIENT_ID column")
    else:
        df["PATIENT_ID"] = [f"{prefix}{i:05d}" for i in range(1, len(df) + 1)]

    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))] if col in ("COMORBIDITIES", "PROCEDURES") else ""

    df["COMORBIDITIES"] = df["COMORBIDITIES"].apply(to_list)
    df["PROCEDURES"] = df["PROCEDURES"].apply(to_list)

    return df[OUT_COLS].copy()


def main():
    # ---- Synthetic files ----
    for fname, prefix in SYN_FILES.items():
        path = SYN_IN_DIR / fname
        if not path.exists():
            print(f"[WARN] missing synthetic file: {path}")
            continue
        df = normalize_df(pd.read_csv(path), is_real=False, prefix=prefix)
        out_path = OUT_DIR / fname
        df.to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path}  shape {df.shape}")

    # ---- Real MIMIC (optional — skipped if the user has not placed the file) ----
    if REAL_IN_PATH.exists():
        df = normalize_df(pd.read_csv(REAL_IN_PATH), is_real=True, prefix="REAL_")
        out_path = OUT_DIR / "real_mimic.csv"
        df.to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path}  shape {df.shape}")
    else:
        print(f"[SKIP] real MIMIC file not found at {REAL_IN_PATH} — skipping real data alignment")


if __name__ == "__main__":
    main()
