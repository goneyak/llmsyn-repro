# llmsyn-repro

Synthetic EHR generation and evaluation pipeline (utility/fidelity) with minimal setup.

## Quick start
1. **Setup env**
   - `conda env create -f environment.yml` (includes `OPENAI_API_KEY` variable placeholder).
   - `conda activate llmsyn` and export your key: `export OPENAI_API_KEY=...`.
2. **Required data**
   - `data/priors/top100_icd9.csv` (already in repo).
   - `data/input/D_ICD_PROCEDURES.csv` (place here; MIMIC D_ICD_PROCEDURES).
   - Optional: MIMIC raw CSVs (`ADMISSIONS.csv`, `PATIENTS.csv`, `DIAGNOSES_ICD.csv`, `D_ICD_DIAGNOSES.csv`) for rebuilding priors.
3. **Generate synthetic variants (full/prior/base)**
   - `python src/generation/generate_variants.py`
   - Output: `outputs/synthetic/syn_full.csv`, `syn_prior.csv`, `syn_base.csv` (100 rows each).
4. **Align to processed format**
   - `python src/preprocessing/align_input_data.py`
   - Output: `data/processed/syn_full.csv`, `syn_prior.csv`, `syn_base.csv` plus `real_mimic.csv` if present.
5. **Evaluation**
   - Utility (mortality): `python src/evaluation/eval_mortality.py`
   - Utility (respiratory): `python src/evaluation/eval_utility_respiratory.py`
   - Fidelity (KS): `python src/evaluation/fidelity_ks.py --real-path ../data/processed/real_mimic.csv --dataset syn_full=../data/processed/syn_full.csv --dataset syn_base=../data/processed/syn_base.csv --dataset syn_prior=../data/processed/syn_prior.csv`
   - Fidelity (MMD, sample 5000): `python src/evaluation/fidelity_mmd.py --real-path ../data/processed/real_mimic.csv --dataset syn_full=../data/processed/syn_full.csv --dataset syn_base=../data/processed/syn_base.csv --dataset syn_prior=../data/processed/syn_prior.csv --sample-size 5000`
   - Summary table: `outputs/eval/summary_overview.md`
   - Plots: `python src/evaluation/plot_fidelity.py` → `outputs/eval/fidelity_ks.png`, `fidelity_mmd.png`

## Details
- **Generation logic**
  - `src/generation/generate_variants.py` runs 4 LLM steps (demographics → main diagnosis → comorbidities → procedures), enforces CSV parsing, and saves three variants:
    - `full`/`prior`: use prior distributions in prompt.
    - `base`: generic, no priors.
  - Requires network/model access (`gpt-5.1`) and `OPENAI_API_KEY`.
- **Priors rebuild (optional)**: `python src/preprocessing/build_priors.py` (reads MIMIC raw CSVs under `data/input`, writes `data/priors/prior.json` and `top100_icd9.csv`).
- **Evaluation metrics**: Utility uses RandomForest ACC/AUROC; Fidelity uses KS per feature and MMD² on diagnosis/procedure/comorbidity pairs.

## Outputs
- Synthetic CSVs: `outputs/synthetic/syn_full.csv`, `syn_prior.csv`, `syn_base.csv`
- Processed for eval: `data/processed/*.csv`
- Eval artifacts: `outputs/eval/*.csv`, `*.npy`, `fidelity_ks.png`, `fidelity_mmd.png`, `summary_overview.md`

## Troubleshooting
- Matplotlib cache warning: set `export MPLCONFIGDIR=/tmp/matplotlib` before plotting if cache dir is not writable.
- Missing `D_ICD_PROCEDURES.csv`: place the MIMIC procedures dictionary under `data/input`.
- LLM CSV formatting issues: rerun `generate_variants.py` (it enforces required columns and will error on malformed output).
