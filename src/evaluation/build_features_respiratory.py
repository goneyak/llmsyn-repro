import os
import ast
import argparse
import numpy as np
import pandas as pd

def normalize_icd9(icd: str) -> str:
    if pd.isna(icd):
        return ""
    s = str(icd).strip()
    return s.replace(".", "")


def is_respiratory(main_icd: str) -> int:
    code = normalize_icd9(main_icd)
    try:
        prefix = int(code[:3])
    except ValueError:
        return 0
    return 1 if 460 <= prefix <= 519 else 0


def parse_list_cell(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return v
        except Exception:
            return [item.strip() for item in s.split(",") if item.strip() != ""]
    return []


def build_features(csv_path: str, out_prefix: str):
    print("Loading", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    df = df.rename(
        columns={
            "MAIN_DIAGNOSIS": "MAIN_ICD9",
            "COMORBIDITIES": "OTHER_ICD9_LIST",
            "PROCEDURE": "PROCEDURES",
        }
    )

    if "MAIN_ICD9" not in df.columns:
        raise KeyError("MAIN_ICD9 (or MAIN_DIAGNOSIS) column is required.")

    print("Making labels (respiratory 460-519 vs other)")
    y = df["MAIN_ICD9"].apply(is_respiratory).astype(int).values

    print("Building X")
    X_parts = []
    feature_names = []

    age = df["AGE"].fillna(df["AGE"].median())
    X_parts.append(age.to_numpy().reshape(-1, 1))
    feature_names.append("AGE")

    for col in ["LANGUAGE", "ETHNICITY", "INSURANCE", "RELIGION", "MARITAL_STATUS"]:
        if col in df.columns:
            cat = df[col].fillna("UNKNOWN").astype("category")
            codes = cat.cat.codes
            X_parts.append(codes.to_numpy().reshape(-1, 1))
            feature_names.append(f"{col}_code")
        else:
            print(f"[WARN] Column {col} not found; skipping.")

    df["OTHER_ICD9_LIST"] = df["OTHER_ICD9_LIST"].apply(parse_list_cell)
    other_icd_len = df["OTHER_ICD9_LIST"].apply(len)
    X_parts.append(other_icd_len.to_numpy().reshape(-1, 1))
    feature_names.append("OTHER_ICD_COUNT")

    X = np.hstack(X_parts)

    print("X shape", X.shape, "y shape", y.shape)

    base_dir = os.path.dirname(out_prefix)
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    x_path = f"{out_prefix}_X.npy"
    y_path = f"{out_prefix}_y.npy"
    feat_path = f"{out_prefix}_feature_names.csv"

    np.save(x_path, X)
    np.save(y_path, y)
    pd.Series(feature_names, name="feature_name").to_csv(feat_path, index=False)

    print("Saved", x_path, y_path, feat_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/real_mimic.csv",
        help="Input flat CSV (real or synthetic)",
    )
    parser.add_argument(
        "--out-prefix",
        default="../outputs/eval/real_resp",
        help="Prefix for output files",
    )
    args = parser.parse_args()

    build_features(args.input, args.out_prefix)
