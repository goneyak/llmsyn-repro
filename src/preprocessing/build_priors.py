import json
from pathlib import Path

import pandas as pd

# ---------- PATH ----------
BASE_DIR = Path(__file__).resolve().parents[2]
MIMIC_PATH = BASE_DIR / "data" / "input"
OUT_DIR = BASE_DIR / "data" / "priors"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LOAD ----------
admissions = pd.read_csv(MIMIC_PATH / "ADMISSIONS.csv", low_memory=False)
patients = pd.read_csv(MIMIC_PATH / "PATIENTS.csv", low_memory=False)
diagnoses = pd.read_csv(MIMIC_PATH / "DIAGNOSES_ICD.csv", low_memory=False)
icd_diag = pd.read_csv(MIMIC_PATH / "D_ICD_DIAGNOSES.csv", low_memory=False)

# ---------- Pre-processing ----------
admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce", utc=True).dt.tz_localize(None)
patients["DOB"] = pd.to_datetime(patients["DOB"], errors="coerce", utc=True).dt.tz_localize(None)

diagnoses["ICD9_CODE"] = diagnoses["ICD9_CODE"].astype(str).str.strip()
icd_diag["ICD9_CODE"]  = icd_diag["ICD9_CODE"].astype(str).str.strip()

# ---------- Excluding NEWBORN ----------
admissions = admissions[admissions["ADMISSION_TYPE"] != "NEWBORN"].copy()
non_newborn_hadm = admissions["HADM_ID"].unique()
diagnoses = diagnoses[diagnoses["HADM_ID"].isin(non_newborn_hadm)].copy()

# ---------- FIRST / LAST ADMISSION ----------
first_admit = (
    admissions.sort_values(["SUBJECT_ID", "ADMITTIME"])
              .groupby("SUBJECT_ID", as_index=False)
              .first()
)
last_admit = (
    admissions.sort_values(["SUBJECT_ID", "ADMITTIME"])
              .groupby("SUBJECT_ID", as_index=False)
              .last()
)

# ---------- AGE (first admission) ----------
demo = first_admit.merge(patients[["SUBJECT_ID", "GENDER", "DOB"]], on="SUBJECT_ID", how="left")

valid = demo["DOB"].notna() & demo["ADMITTIME"].notna() & (demo["ADMITTIME"] >= demo["DOB"])
delta_ns = demo.loc[valid, "ADMITTIME"].astype("int64") - demo.loc[valid, "DOB"].astype("int64")
demo.loc[valid, "AGE"] = (delta_ns / 1_000_000_000 / 86400.0 / 365.2425).clip(lower=0, upper=120)

# ---------- function ----------
def dist(series: pd.Series) -> dict:
    """normalized categorical distribution (NaN→'UNKNOWN')"""
    s = series.astype(str)
    s = s.replace("nan", "UNKNOWN").replace("", "UNKNOWN")
    return s.value_counts(normalize=True).to_dict()

def compute_demo(demo_df):
    """Demographics based on first admission"""
    bins = [0,18,35,50,65,80,120]
    labels = ["0-17","18-34","35-49","50-64","65-79","80+"]
    demo_df["AGE_BIN"] = pd.cut(demo_df["AGE"], bins=bins, labels=labels, right=False)
    
    demographics = {
        "LANGUAGE":       dist(demo_df["LANGUAGE"])       if "LANGUAGE" in demo_df else {},
        "RELIGION":       dist(demo_df["RELIGION"])       if "RELIGION" in demo_df else {},
        "MARITAL_STATUS": dist(demo_df["MARITAL_STATUS"]) if "MARITAL_STATUS" in demo_df else {},
        "ETHNICITY":      dist(demo_df["ETHNICITY"])      if "ETHNICITY" in demo_df else {},
        "INSURANCE":      dist(demo_df["INSURANCE"])      if "INSURANCE" in demo_df else {},
        "GENDER":         dist(demo_df["GENDER"]),
        "AGE_BIN":        dist(demo_df["AGE_BIN"]),
    }
    return demographics

def compute_mortality(last_admissions: pd.DataFrame) -> float:
    """Patient-level in-hospital mortality rate using pre-computed last-admission rows."""
    rate = (last_admissions["HOSPITAL_EXPIRE_FLAG"] == 1).mean()
    return float(rate)

# ---------- ICD-9 TOP100 (excluding newborn) ----------
icd_counts = diagnoses["ICD9_CODE"].value_counts().head(100).reset_index()
icd_counts.columns = ["ICD9_CODE", "COUNT"]
icd_counts = icd_counts.merge(icd_diag[["ICD9_CODE", "LONG_TITLE"]], on="ICD9_CODE", how="left")

icd_path = OUT_DIR / "top100_icd9.csv"
icd_counts.to_csv(icd_path, index=False)

# ---------- PRIOR JSON ----------
prior = {
    "mortality_rate": compute_mortality(last_admit),
    "top100_icd9_path": str(icd_path.relative_to(BASE_DIR)),
    "demographics": compute_demo(demo),
    "cohort": "no_newborn_filter_first_admission",
}

out_file = OUT_DIR / "prior.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(prior, f, indent=2)

print(f"[OK] Saved: {out_file}")
print(f"Mortality rate: {prior['mortality_rate']:.4f}")
print(f"Top100 ICD path: {prior['top100_icd9_path']}")