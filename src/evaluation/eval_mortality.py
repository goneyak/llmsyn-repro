import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def run_rf(name, X_train, y_train, X_test, y_test, seed):
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    except ValueError:
        auc = np.nan
    print(name, acc, auc)
    return {"name": name, "ACC": acc, "AUROC": auc, "seed": seed}


if __name__ == "__main__":
    out_dir = "../outputs/eval"
    os.makedirs(out_dir, exist_ok=True)

    X_real = np.load("../outputs/eval/real_mortality_X.npy")
    y_real = np.load("../outputs/eval/real_mortality_y.npy")

    syn_files = {
        "syn_full": ("../outputs/eval/syn_full_mortality_X.npy", "../outputs/eval/syn_full_mortality_y.npy"),
        "syn_base": ("../outputs/eval/syn_base_mortality_X.npy", "../outputs/eval/syn_base_mortality_y.npy"),
        "syn_prior": ("../outputs/eval/syn_prior_mortality_X.npy", "../outputs/eval/syn_prior_mortality_y.npy"),
    }

    seeds = list(range(10))
    n_sample = 100
    rows = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        pos = np.where(y_real == 1)[0]
        neg = np.where(y_real == 0)[0]
        n_pos = max(1, int(round(len(pos) / len(y_real) * n_sample)))
        n_neg = n_sample - n_pos

        train_pos = rng.choice(pos, size=min(n_pos, len(pos)), replace=False)
        train_neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
        train_idx = np.concatenate([train_pos, train_neg])

        rest = np.setdiff1d(np.arange(len(y_real)), train_idx)
        pos_rest = [i for i in rest if y_real[i] == 1]
        neg_rest = [i for i in rest if y_real[i] == 0]
        test_pos = rng.choice(pos_rest, size=min(n_pos, len(pos_rest)), replace=False) if pos_rest else []
        test_neg = rng.choice(neg_rest, size=min(n_neg, len(neg_rest)), replace=False) if neg_rest else []
        test_idx = np.array(list(test_pos) + list(test_neg))
        if len(test_idx) < n_sample:
            extra = rng.choice(np.arange(len(y_real)), size=n_sample - len(test_idx), replace=False)
            test_idx = np.concatenate([test_idx, extra])

        X_train = X_real[train_idx]
        y_train = y_real[train_idx]
        X_test = X_real[test_idx]
        y_test = y_real[test_idx]

        rows.append(run_rf("REAL_train_REAL_test", X_train, y_train, X_test, y_test, seed))
        rows.append(run_rf("REAL_match_train_REALsub_test_REAL", X_train, y_train, X_test, y_test, seed))

        for name, (x_path, y_path) in syn_files.items():
            if not (os.path.exists(x_path) and os.path.exists(y_path)):
                print("missing", name)
                continue
            X_syn = np.load(x_path)
            y_syn = np.load(y_path)

            rows.append(run_rf(f"FS_{name}_train_SYN_test_REAL", X_syn, y_syn, X_test, y_test, seed))

            X_da = np.vstack([X_train, X_syn])
            y_da = np.concatenate([y_train, y_syn])
            rows.append(run_rf(f"DA_{name}_train_REALsub+SYN_test_REAL", X_da, y_da, X_test, y_test, seed))

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "utility_mortality_simple.csv"), index=False)
    print("saved utility_mortality_simple.csv")
