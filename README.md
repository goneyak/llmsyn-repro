# LLMSYN-Repro: Synthetic EHR Generation via LLM Reasoning

Reproduction/extension of *LLMSYN: Generating Synthetic Electronic Health Records Without Patient-Level Data* (MLHC 2024).

## Overview
LLMSYN is a controllable LLM-based pipeline that generates structured synthetic EHRs using:
- **Prior statistical knowledge** (e.g., disease and mortality distributions)
- **Relevant medical knowledge retrieval** (e.g., condition pages)
- **Markov-based generation** across 4 steps: demographics → main dx → comorbidities → procedures

This repo starts minimal on purpose — update each module incrementally.

## System Workflow
1. **Prior Knowledge Retrieval**
2. **Markov-based Generation (x0~x3)**
3. **Relevant Knowledge Retrieval**
4. **Prompt Adaptor**
5. **Evaluation** (Utility, Fidelity, Privacy)

## Repository Structure
```
llmsyn-repro/
├── data/
│   ├── mimic_stats.csv
│   ├── sample_prompts/
│   └── synthetic_ehr_output.json
├── generator/
│   ├── pipeline.py
│   ├── prompt_templates.py
│   ├── retrievers.py
│   └── utils.py
├── evaluation/
│   ├── utility_eval.md
│   ├── fidelity_eval.md
│   └── privacy_eval.md
├── docs/
│   ├── hao24a.pdf
│   ├── figures/
│   └── README_notebook.md
├── requirements.txt
├── LICENSE
└── README.md
```

## Quickstart
```bash
# (optional) conda create -n llmsyn python=3.11 -y && conda activate llmsyn
pip install -r requirements.txt

# run a no-op pipeline (writes an empty JSON you will fill later)
python -m generator.pipeline --model gpt-4 --data data/mimic_stats.csv --out data/synthetic_ehr_output.json
```

## Evaluation Plan
- **Utility**: RF for phenotype & mortality (report ACC/AUROC/F1)
- **Fidelity**: KS statistic (feature-wise), MMD (pair-wise)
- **Privacy**: k-anonymity over {ETHNICITY, ICD9_CODE}

## Reference
Hao Y., Ho J.C., He H. (2024). *LLMSYN: Generating Synthetic Electronic Health Records Without Patient-Level Data*, PMLR 252:1–27.
