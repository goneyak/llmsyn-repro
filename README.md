# llmsyn-repro

Synthetic EHR generation and evaluation pipeline for a portfolio/research setting.
The project compares LLM-generated variants against real-like processed data using utility and fidelity metrics.

## Working Directory Assumption

All commands below are intended to run from the repository root:

`/Users/cocoxoxo/llmsyn-repro`

## Pipeline Overview

1. Generate synthetic cohorts with an LLM (three variants: `full`, `prior`, `base`).
2. Align synthetic outputs to a common processed schema.
3. Build utility task features (mortality and respiratory tasks).
4. Evaluate utility with RandomForest (ACC, AUROC).
5. Evaluate fidelity with KS and MMD.

## Synthetic Variants

- `full`: prior-guided with **explicit frequency distributions** (demographic rates from `prior.json`).
- `prior`: prior-guided with **vocabulary constraints only** (allowed category values, no frequency target).
- `base`: **no prior knowledge** — model uses general clinical reasoning freely.

This gives three meaningfully different points on the prior-guidance spectrum.

## Required Inputs

- Already in repo:
   - `data/priors/top100_icd9.csv`
   - `data/priors/prior.json`
- Required for generation:
   - `data/input/D_ICD_PROCEDURES.csv` (MIMIC procedure dictionary)
- Optional, needed only to rebuild priors:
   - `data/input/ADMISSIONS.csv`
   - `data/input/PATIENTS.csv`
   - `data/input/DIAGNOSES_ICD.csv`
   - `data/input/D_ICD_DIAGNOSES.csv`
- Optional, needed only for real-data alignment and evaluation:
   - `data/input/real_mimic.csv` (user-prepared flat MIMIC extract)

## Setup

```bash
conda env create -f environment.yml
conda activate llmsyn
export OPENAI_API_KEY=your_key_here
```

The generation script currently uses OpenAI Responses API with model `gpt-5.1`, so network/model access and a valid API key are required.

## Generation

```bash
python src/generation/generate_variants.py
```

Expected output files:

- `outputs/synthetic/syn_full.csv`
- `outputs/synthetic/syn_prior.csv`
- `outputs/synthetic/syn_base.csv`

## Alignment

```bash
python src/preprocessing/align_input_data.py
```

Expected output files:

- `data/processed/syn_full.csv`
- `data/processed/syn_prior.csv`
- `data/processed/syn_base.csv`
- `data/processed/real_mimic.csv` (only if `data/input/real_mimic.csv` is present)

## Utility Evaluation

Build feature arrays first:

```bash
python src/evaluation/build_features_mortality.py --input data/processed/real_mimic.csv --out-prefix outputs/eval/real_mortality
python src/evaluation/build_features_mortality.py --input data/processed/syn_full.csv --out-prefix outputs/eval/syn_full_mortality
python src/evaluation/build_features_mortality.py --input data/processed/syn_prior.csv --out-prefix outputs/eval/syn_prior_mortality
python src/evaluation/build_features_mortality.py --input data/processed/syn_base.csv --out-prefix outputs/eval/syn_base_mortality

python src/evaluation/build_features_respiratory.py --input data/processed/real_mimic.csv --out-prefix outputs/eval/real_resp
python src/evaluation/build_features_respiratory.py --input data/processed/syn_full.csv --out-prefix outputs/eval/syn_full_resp
python src/evaluation/build_features_respiratory.py --input data/processed/syn_prior.csv --out-prefix outputs/eval/syn_prior_resp
python src/evaluation/build_features_respiratory.py --input data/processed/syn_base.csv --out-prefix outputs/eval/syn_base_resp
```

Then run utility metrics:

```bash
python src/evaluation/eval_mortality.py
python src/evaluation/eval_utility_respiratory.py
```

## Fidelity Evaluation

Fidelity metrics compare synthetic distributions against the real dataset:
- **Categorical features** (LANGUAGE, RELIGION, MARITAL_STATUS, ETHNICITY, INSURANCE, HOSPITAL_EXPIRE_FLAG, MAIN_DIAGNOSIS): **Total Variation Distance (TVD)**.
  TVD ∈ [0, 1]; lower is better.  KS is not used for nominal variables because its result depends on the arbitrary integer encoding order.
- **Numeric/count features** (AGE, OTHER_ICD_COUNT, PROCEDURE_COUNT): **Kolmogorov–Smirnov statistic**.

Output schema for `fidelity_ks.csv`:
`dataset, feature, metric_type (TVD|KS), stat, pvalue`

```bash
python src/evaluation/fidelity_ks.py \
   --real-path data/processed/real_mimic.csv \
   --dataset syn_full=data/processed/syn_full.csv \
   --dataset syn_base=data/processed/syn_base.csv \
   --dataset syn_prior=data/processed/syn_prior.csv

python src/evaluation/fidelity_mmd.py \
   --real-path data/processed/real_mimic.csv \
   --dataset syn_full=data/processed/syn_full.csv \
   --dataset syn_base=data/processed/syn_base.csv \
   --dataset syn_prior=data/processed/syn_prior.csv \
   --sample-size 5000
```

## Outputs

- Synthetic CSVs: `outputs/synthetic/*.csv`
- Processed datasets: `data/processed/*.csv`
- Utility outputs: `outputs/eval/utility_*.csv`
- Fidelity outputs: `outputs/eval/fidelity_ks.csv`, `outputs/eval/fidelity_mmd.csv`
- Summary: `outputs/eval/summary_overview.md`

Note: A plotting script is not currently included in this repository. Existing evaluation tables and summary files can be reviewed directly.

## Prototype Notes and Limitations

- This is a portfolio/research pipeline, not a clinical production system.
- Synthetic quality is evaluated only through the provided utility/fidelity tasks.
- Realism and privacy are not formally guaranteed by this repository alone.
- Generation depends on external API/model availability and stable model behavior.

## Troubleshooting

- Missing `data/input/D_ICD_PROCEDURES.csv`: generation step will fail until this file is provided.
- Missing `OPENAI_API_KEY`: generation step will fail by design.
- If matplotlib cache permission warnings appear, set `export MPLCONFIGDIR=/tmp/matplotlib`.
