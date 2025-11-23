import os
import ast
import argparse
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
        help="Synthetic dataset entry (repeatable). ex) --dataset syn_full=../data/input_processed/syn_full.csv",
    )
    p.add_argument("--out-dir", default="../outputs/eval")
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


def gaussian_kernel_matrix(X, Y, sigma=1.0):
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    dist_sq = X_norm + Y_norm - 2 * X.dot(Y.T)
    return np.exp(-dist_sq / (2 * sigma ** 2))


def mmd2_gaussian(X, Y, sigma=1.0):
    K_xx = gaussian_kernel_matrix(X, X, sigma)
    K_yy = gaussian_kernel_matrix(Y, Y, sigma)
    K_xy = gaussian_kernel_matrix(X, Y, sigma)
    return float(K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    datasets = {}
    for item in args.dataset:
        name, path = item.split("=", 1)
        datasets[name.strip()] = path.strip()

    real_df = pd.read_csv(args.real_path, low_memory=False)
    if args.sample_size is not None and len(real_df) > args.sample_size:
        real_df = real_df.sample(n=args.sample_size, random_state=rng)

    mmd_rows = []
    for name, path in datasets.items():
        if not os.path.exists(path):
            continue

        syn_df = pd.read_csv(path, low_memory=False)
        if args.sample_size is not None and len(syn_df) > args.sample_size:
            syn_df = syn_df.sample(n=args.sample_size, random_state=rng)

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

    out_csv = os.path.join(args.out_dir, "fidelity_mmd.csv")
    if mmd_rows:
        pd.DataFrame(mmd_rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
