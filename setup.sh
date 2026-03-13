#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Creating conda environment from environment.yml..."
conda env create -f environment.yml || echo "Environment may already exist."

echo "[2/3] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate llmsyn

echo "[3/3] Environment check"
python --version
python -c "import pandas, openai, matplotlib; print('Dependencies are available.')"

cat <<'EOF'

Setup complete.
Next steps:
	1) export OPENAI_API_KEY=your_key
	2) python src/generation/generate_variants.py
	3) python src/preprocessing/align_input_data.py

EOF

