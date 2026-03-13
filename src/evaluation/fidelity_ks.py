import os
import ast
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp


def parse_args():
    p = argparse.ArgumentParser(description="KS only between REAL and synthetic datasets.")
    p.add_argument("--real-path", required=True)
    p.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Synthetic dataset entry (repeatable). ex) --dataset syn_full=data/processed/syn_full.csv",
    )
    p.add_argument("--out-dir", default="outputs/eval")
    return p.parse_args()


def count_list(col):
    def _count(x):
        if isinstance(x, list):
            return len(x)
        if pd.isna(x):
            return 0
        s = str(x).strip()
        if s == "":
            return 0
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return len(parsed)
            except Exception:
                return 1
        return 1
    return col.apply(_count)


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    os.makedirs(out_dir, exist_ok=True)

    datasets = {}
    for item in args.dataset:
        name, path = item.split("=", 1)
        path_obj = Path(path.strip())
        if not path_obj.is_absolute():
            path_obj = base_dir / path_obj
        datasets[name.strip()] = path_obj

    real_path = Path(args.real_path)
    if not real_path.is_absolute():
        real_path = base_dir / real_path
    real_df = pd.read_csv(real_path, low_memory=False)

    ks_rows = []
    for name, path in datasets.items():
        if not path.exists():
            continue
        syn_df = pd.read_csv(path, low_memory=False)

        cat_cols = ["LANGUAGE", "RELIGION", "MARITAL_STATUS", "ETHNICITY", "INSURANCE", "HOSPITAL_EXPIRE_FLAG", "MAIN_DIAGNOSIS"]
        for col in cat_cols:
            if col not in real_df.columns or col not in syn_df.columns:
                continue
            r = real_df[col].fillna("UNKNOWN").astype(str)
            s = syn_df[col].fillna("UNKNOWN").astype(str)
            all_vals = pd.Index(r.tolist() + s.tolist()).unique()
            mapper = {v: i for i, v in enumerate(all_vals)}
            r_codes = r.map(mapper)
            s_codes = s.map(mapper)
            if len(r_codes) == 0 or len(s_codes) == 0:
                continue
            ks_stat, p_val = ks_2samp(r_codes, s_codes)
            ks_rows.append({"dataset": name, "feature": col, "KS_stat": ks_stat, "KS_pvalue": p_val})

        real_df["OTHER_ICD_COUNT"] = count_list(real_df["COMORBIDITIES"])
        syn_df["OTHER_ICD_COUNT"] = count_list(syn_df["COMORBIDITIES"])
        proc_col_real = "PROCEDURES" if "PROCEDURES" in real_df.columns else "PROCEDURE"
        proc_col_syn = "PROCEDURES" if "PROCEDURES" in syn_df.columns else "PROCEDURE"
        real_df["PROCEDURE_COUNT"] = count_list(real_df[proc_col_real])
        syn_df["PROCEDURE_COUNT"] = count_list(syn_df[proc_col_syn])
        num_cols = ["AGE", "OTHER_ICD_COUNT", "PROCEDURE_COUNT"]
        for col in num_cols:
            if col not in real_df.columns or col not in syn_df.columns:
                continue
            r_vals = pd.to_numeric(real_df[col], errors="coerce").dropna()
            s_vals = pd.to_numeric(syn_df[col], errors="coerce").dropna()
            if len(r_vals) == 0 or len(s_vals) == 0:
                continue
            ks_stat, p_val = ks_2samp(r_vals, s_vals)
            ks_rows.append({"dataset": name, "feature": col, "KS_stat": ks_stat, "KS_pvalue": p_val})

    out_csv = out_dir / "fidelity_ks.csv"
    if ks_rows:
        pd.DataFrame(ks_rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
