import os
import ast
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="MMD only between REAL and synthetic datasets.")
    p.add_argument("--real-path", required=True)
    p.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Synthetic dataset entry (repeatable). ex) --dataset syn_full=data/processed/syn_full.csv",
    )
    p.add_argument("--out-dir", default="outputs/eval")
    p.add_argument("--sigma", type=float, default=1.0, help="Gaussian kernel sigma")
    p.add_argument("--sample-size", type=int, default=None, help="Optional max rows to sample per dataset before MMD")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
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


def first_item(col):
    def _first(x):
        if isinstance(x, list):
            return str(x[0]) if x else "NONE"
        if pd.isna(x):
            return "NONE"
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list) and parsed:
                    return str(parsed[0])
            except Exception:
                pass
        return s if s else "NONE"
    return col.apply(_first)


def gaussian_kernel_mean(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0, chunk: int = 1000) -> float:
    """Compute the mean of the Gaussian kernel matrix K(X, Y) in row-blocks.

    Processing in chunks avoids materialising an n×m matrix in memory
    (the naive approach peaks at n×m×8 bytes, which exceeds available RAM
    when n=m=5000).  This version uses O(chunk × m) memory instead.
    """
    total = 0.0
    count = 0
    denom = 2.0 * sigma ** 2
    for i in range(0, len(X), chunk):
        Xi = X[i : i + chunk]
        Xi_sq = np.sum(Xi ** 2, axis=1, keepdims=True)          # (c, 1)
        Y_sq = np.sum(Y ** 2, axis=1, keepdims=True).T           # (1, m)
        dist_sq = Xi_sq + Y_sq - 2.0 * Xi.dot(Y.T)              # (c, m)
        total += float(np.exp(-dist_sq / denom).sum())
        count += Xi.shape[0] * Y.shape[0]
    return total / count


def mmd2_gaussian(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """Unbiased-style MMD² via chunked Gaussian kernel means."""
    return (
        gaussian_kernel_mean(X, X, sigma)
        + gaussian_kernel_mean(Y, Y, sigma)
        - 2.0 * gaussian_kernel_mean(X, Y, sigma)
    )


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
    real_df_full = pd.read_csv(real_path, low_memory=False)
    # Use a fixed integer seed for EVERY synthetic dataset so the real-data
    # sample is identical across all comparisons — essential for a fair MMD
    # ranking.  Do NOT advance a shared RNG state between datasets.
    if args.sample_size is not None and len(real_df_full) > args.sample_size:
        real_df_full = real_df_full.sample(n=args.sample_size, random_state=args.seed)

    mmd_rows = []
    for name, path in datasets.items():
        if not path.exists():
            continue

        syn_df = pd.read_csv(path, low_memory=False)
        # Each dataset gets its own fresh sample from the full real DataFrame
        # so comparisons are fair regardless of dataset iteration order.
        real_df = real_df_full.copy()
        if args.sample_size is not None and len(syn_df) > args.sample_size:
            syn_df = syn_df.sample(n=args.sample_size, random_state=args.seed)

        proc_col_real = "PROCEDURES" if "PROCEDURES" in real_df.columns else "PROCEDURE"
        proc_col_syn = "PROCEDURES" if "PROCEDURES" in syn_df.columns else "PROCEDURE"

        real_df["PROC_CODE"] = first_item(real_df[proc_col_real])
        syn_df["PROC_CODE"] = first_item(syn_df[proc_col_syn])
        real_df["MAIN_CODE"] = real_df["MAIN_DIAGNOSIS"].fillna("UNKNOWN").astype(str)
        syn_df["MAIN_CODE"] = syn_df["MAIN_DIAGNOSIS"].fillna("UNKNOWN").astype(str)
        real_df["COMORB_FIRST"] = first_item(real_df["COMORBIDITIES"])
        syn_df["COMORB_FIRST"] = first_item(syn_df["COMORBIDITIES"])

        pair_defs = {
            "MAIN+PROC": ("MAIN_CODE", "PROC_CODE"),
            "MAIN+COMORB": ("MAIN_CODE", "COMORB_FIRST"),
            "COMORB+PROC": ("COMORB_FIRST", "PROC_CODE"),
        }

        for pair_name, (col_a, col_b) in pair_defs.items():
            all_a = pd.Index(real_df[col_a].tolist() + syn_df[col_a].tolist()).unique()
            all_b = pd.Index(real_df[col_b].tolist() + syn_df[col_b].tolist()).unique()
            map_a = {v: i for i, v in enumerate(all_a)}
            map_b = {v: i for i, v in enumerate(all_b)}

            real_vec = np.column_stack([
                real_df[col_a].map(map_a).to_numpy(),
                real_df[col_b].map(map_b).to_numpy(),
            ]).astype(float)
            syn_vec = np.column_stack([
                syn_df[col_a].map(map_a).to_numpy(),
                syn_df[col_b].map(map_b).to_numpy(),
            ]).astype(float)

            mmd_rows.append(
                {
                    "dataset": name,
                    "pair": pair_name,
                    "sigma": args.sigma,
                    "MMD2": mmd2_gaussian(real_vec, syn_vec, sigma=args.sigma),
                }
            )

    out_csv = out_dir / "fidelity_mmd.csv"
    if mmd_rows:
        pd.DataFrame(mmd_rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
