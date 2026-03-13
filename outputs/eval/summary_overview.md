# Evaluation Summary
## Utility - Mortality (mean±std over seeds)

```
name,ACC_mean,ACC_std,AUROC_mean,AUROC_std,seed_mean,seed_std
DA_syn_base_train_REALsub+SYN_test_REAL,0.885,0.0085,0.5895,0.0909,4.5,3.0277
DA_syn_full_train_REALsub+SYN_test_REAL,0.881,0.0074,0.6066,0.1118,4.5,3.0277
DA_syn_prior_train_REALsub+SYN_test_REAL,0.882,0.0103,0.5729,0.1141,4.5,3.0277
FS_syn_base_train_SYN_test_REAL,0.867,0.0134,0.5057,0.0602,4.5,3.0277
FS_syn_full_train_SYN_test_REAL,0.831,0.0428,0.5156,0.0787,4.5,3.0277
FS_syn_prior_train_SYN_test_REAL,0.866,0.0143,0.4253,0.1078,4.5,3.0277
REAL_match_train_REALsub_test_REAL,0.877,0.0157,0.6021,0.1081,4.5,3.0277
REAL_train_REAL_test,0.877,0.0157,0.6021,0.1081,4.5,3.0277

```

## Utility - Respiratory (mean±std over seeds)

```
name,ACC_mean,ACC_std,AUROC_mean,AUROC_std,seed_mean,seed_std
DA_syn_base_train_REALsub+SYN_test_REAL,0.96,0.0,0.5118,0.153,4.5,3.0277
DA_syn_full_train_REALsub+SYN_test_REAL,0.96,0.0,0.5398,0.1839,4.5,3.0277
DA_syn_prior_train_REALsub+SYN_test_REAL,0.96,0.0,0.5341,0.1753,4.5,3.0277
FS_syn_base_train_SYN_test_REAL,0.96,0.0,0.4518,0.1774,4.5,3.0277
FS_syn_full_train_SYN_test_REAL,0.958,0.0042,0.6241,0.1266,4.5,3.0277
FS_syn_prior_train_SYN_test_REAL,0.96,0.0,0.4988,0.1438,4.5,3.0277
REAL_train_REAL_test,0.958,0.0042,0.528,0.1639,4.5,3.0277

```

## Fidelity - KS/TVD (mean across features)

> Note: the fidelity script has been updated to use **TVD** for categorical features
> and **KS** for numeric features.  The values below were produced with the old
> KS-on-encoded-integers approach and are retained for historical reference.
> Re-run `fidelity_ks.py` to get updated TVD+KS results.

```
dataset,KS_stat,KS_pvalue
syn_base,0.4241,0.0987
syn_full,0.1976,0.5909
syn_prior,0.255,0.5157

```

## Fidelity - MMD (sample-size=5000)

```
dataset,pair,sigma,MMD2
syn_full,MAIN+PROC,1.0,0.038458
syn_full,MAIN+COMORB,1.0,0.176967
syn_full,COMORB+PROC,1.0,0.184235
syn_base,MAIN+PROC,1.0,0.025903
syn_base,MAIN+COMORB,1.0,0.159627
syn_base,COMORB+PROC,1.0,0.018054
syn_prior,MAIN+PROC,1.0,0.025971
syn_prior,MAIN+COMORB,1.0,0.160105
syn_prior,COMORB+PROC,1.0,0.018377

```
