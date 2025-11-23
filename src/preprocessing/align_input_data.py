import ast
import pandas as pd
from pathlib import Path

# Input and output dirs
# Synthetic CSVs live under outputs/synthetic (not data/output/synthetic).
IN_DIR = Path("outputs/synthetic")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(exist_ok=True, parents=True)

FILES = {
    "real_mimic.csv": {"is_real": True, "prefix": "REAL_"},
    "syn_full.csv": {"is_real": False, "prefix": "SYNFULL_"},
    "syn_prior.csv": {"is_real": False, "prefix": "SYNPRIOR_"},
    "syn_base.csv": {"is_real": False, "prefix": "SYNBASE_"},
}

OUT_COLS = ["PATIENT_ID", "AGE", "LANGUAGE", "RELIGION", "MARITAL_STATUS", "ETHNICITY", "INSURANCE",
            "HOSPITAL_EXPIRE_FLAG", "MAIN_DIAGNOSIS", "COMORBIDITIES", "PROCEDURES"]


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


def main():
    for fname, meta in FILES.items():
        path = IN_DIR / fname
        if not path.exists():
            print(f"[WARN] missing {path}")
            continue
        df = pd.read_csv(path)

        # Normalize column names
        df = df.rename(
            columns={
                "MAIN_ICD9": "MAIN_DIAGNOSIS",
                "main_diagnosis": "MAIN_DIAGNOSIS",
                "CO-MORBIDITIES": "COMORBIDITIES",
                "co-morbidities": "COMORBIDITIES",
                "PROCEDURES_ICD9": "PROCEDURES",
                "procedures": "PROCEDURES",
                "PROCEDURES": "PROCEDURES",
            }
        )

        # Ensure PATIENT_ID
        if meta["is_real"]:
            if "SUBJECT_ID" in df.columns:
                df["PATIENT_ID"] = df["SUBJECT_ID"].astype(str)
            elif "PATIENT_ID" in df.columns:
                df["PATIENT_ID"] = df["PATIENT_ID"].astype(str)
            else:
                raise ValueError(f"No SUBJECT_ID/PATIENT_ID in {fname}")
        else:
            df["PATIENT_ID"] = [f"{meta['prefix']}{i:05d}" for i in range(1, len(df) + 1)]

        # Fill missing required columns
        for col in OUT_COLS:
            if col not in df.columns:
                if col in ["COMORBIDITIES", "PROCEDURES"]:
                    df[col] = [[] for _ in range(len(df))]
                else:
                    df[col] = ""

        # Parse list columns
        df["COMORBIDITIES"] = df["COMORBIDITIES"].apply(to_list)
        df["PROCEDURES"] = df["PROCEDURES"].apply(to_list)

        out = df[OUT_COLS].copy()
        out_path = OUT_DIR / fname
        out.to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path} shape {out.shape}")


if __name__ == "__main__":
    main()
