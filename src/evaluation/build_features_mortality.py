import os
import ast
import argparse
import numpy as np
import pandas as pd


def parse_list_cell(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return v
        except Exception:
            return [item.strip() for item in s.split(",") if item.strip()]
    return []


def build_mortality_features(csv_path: str, out_prefix: str):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.rename(
        columns={
            "MAIN_DIAGNOSIS": "MAIN_ICD9",
            "COMORBIDITIES": "OTHER_ICD9_LIST",
            "PROCEDURE": "PROCEDURES",
        }
    )

    if "HOSPITAL_EXPIRE_FLAG" not in df.columns:
        raise KeyError("HOSPITAL_EXPIRE_FLAG column missing for mortality task")
    y = df["HOSPITAL_EXPIRE_FLAG"].astype(int).values

    X_parts, feature_names = [], []

    age = df["AGE"].fillna(df["AGE"].median())
    X_parts.append(age.to_numpy().reshape(-1, 1))
    feature_names.append("AGE")

    for col in ["LANGUAGE", "ETHNICITY", "INSURANCE", "RELIGION", "MARITAL_STATUS"]:
        if col in df.columns:
            cat = df[col].fillna("UNKNOWN").astype("category")
            codes = cat.cat.codes
            X_parts.append(codes.to_numpy().reshape(-1, 1))
            feature_names.append(f"{col}_code")

    df["OTHER_ICD9_LIST"] = df["OTHER_ICD9_LIST"].apply(parse_list_cell)
    other_icd_len = df["OTHER_ICD9_LIST"].apply(len)
    X_parts.append(other_icd_len.to_numpy().reshape(-1, 1))
    feature_names.append("OTHER_ICD_COUNT")

    X = np.hstack(X_parts)

    base_dir = os.path.dirname(out_prefix)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    np.save(f"{out_prefix}_X.npy", X)
    np.save(f"{out_prefix}_y.npy", y)
    pd.Series(feature_names, name="feature_name").to_csv(f"{out_prefix}_feature_names.csv", index=False)


if __name__ == "__main__":
    _base = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(_base / "data" / "processed" / "real_mimic.csv"))
    parser.add_argument("--out-prefix", default=str(_base / "outputs" / "eval" / "real_mortality"))
    args = parser.parse_args()
    # Resolve relative paths from repo root so the script works from any cwd.
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = _base / input_path
    out_prefix = args.out_prefix
    if not Path(out_prefix).is_absolute():
        out_prefix = str(_base / out_prefix)
    build_mortality_features(str(input_path), out_prefix)
