import os
import io
from pathlib import Path
import pandas as pd
from openai import OpenAI


def require_csv(text: str, expected_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in model output")
    ordered = expected_cols + [c for c in df.columns if c not in expected_cols]
    return df[ordered]


def build_prompts(variant: str, priors_text: str, icd_top: str, proc_master: str):
    variant = variant.lower()
    use_priors = variant in {"full", "prior"}
    demo_note = (
        f"Match these priors: {priors_text}."
        if use_priors
        else "Generate realistic but generic distributions; do not assume priors."
    )

    # Prompt 1
    demo_cols = [
        "AGE",
        "LANGUAGE",
        "RELIGION",
        "MARITAL_STATUS",
        "ETHNICITY",
        "INSURANCE",
        "HOSPITAL_EXPIRE_FLAG",
    ]
    demo_prompt = f"""
You are a hospital intake expert generating realistic patient demographics for a synthetic EHR.
Use only reasoning and publicly available clinical knowledge. Do NOT use or infer real patient data.
{demo_note}
Rules: patients aged 1-17 cannot be married, widowed, or divorced.
Output format: return ONLY CSV with header and exactly 100 rows. Columns: {', '.join(demo_cols)}.
"""

    # Prompt 2
    diag_cols = demo_cols + ["MAIN_DIAGNOSIS"]
    diag_prompt = f"""
You are a professional medical coder assigning a primary diagnosis (ICD-9) to each synthetic patient based on demographics.
Use only public clinical knowledge (e.g., Mayo Clinic). Do not use real patient data.
Use ICD9 codes from this top-100 list:
{icd_top}

Patient demographics CSV:
{{demo_csv}}

Output format: return ONLY CSV with the same rows, columns {', '.join(diag_cols)} (add MAIN_DIAGNOSIS), no commentary.
"""

    # Prompt 3
    comorb_cols = diag_cols + ["COMORBIDITIES"]
    comorb_prompt = f"""
Assign 0-6 ICD-9 comorbidities per patient based on demographics and main diagnosis.
Use only public clinical knowledge (e.g., Mayo Clinic). Do not use real patient data.
ICD9 top 100 list:
{icd_top}

Patient info CSV (includes MAIN_DIAGNOSIS):
{{diag_csv}}

Output format: return ONLY CSV with the same rows, columns {', '.join(comorb_cols)} (COMORBIDITIES as pipe- or semicolon-separated list), no commentary.
"""

    # Prompt 4
    proc_cols = comorb_cols + ["PROCEDURES"]
    proc_prompt = f"""
Assign up to 6 ICD-9 procedure codes per patient based on demographics, main diagnosis, and comorbidities.
Use only public clinical knowledge (e.g., Mayo Clinic). Use codes from this procedures master:
{proc_master}

Patient info CSV (includes MAIN_DIAGNOSIS and COMORBIDITIES):
{{comorb_csv}}

Output format: return ONLY CSV with the same rows, columns {', '.join(proc_cols)} (PROCEDURES as pipe- or semicolon-separated list), no commentary.
"""

    return {
        "demo_prompt": demo_prompt,
        "diag_prompt": diag_prompt,
        "comorb_prompt": comorb_prompt,
        "proc_prompt": proc_prompt,
        "demo_cols": demo_cols,
        "diag_cols": diag_cols,
        "comorb_cols": comorb_cols,
        "proc_cols": proc_cols,
    }


def generate_variant(client: OpenAI, variant: str, priors_text: str, icd_top: str, proc_master: str, out_path: Path):
    prompts = build_prompts(variant, priors_text, icd_top, proc_master)

    resp = client.responses.create(
        model="gpt-5.1",
        input=prompts["demo_prompt"],
        reasoning={"effort": "medium"},
    ).output_text
    df_demo = require_csv(resp, prompts["demo_cols"])

    resp = client.responses.create(
        model="gpt-5.1",
        input=prompts["diag_prompt"].format(demo_csv=df_demo.to_csv(index=False)),
        reasoning={"effort": "medium"},
    ).output_text
    df_diag = require_csv(resp, prompts["diag_cols"])

    resp = client.responses.create(
        model="gpt-5.1",
        input=prompts["comorb_prompt"].format(diag_csv=df_diag.to_csv(index=False)),
        reasoning={"effort": "medium"},
    ).output_text
    df_comorb = require_csv(resp, prompts["comorb_cols"])

    resp = client.responses.create(
        model="gpt-5.1",
        input=prompts["proc_prompt"].format(comorb_csv=df_comorb.to_csv(index=False)),
        reasoning={"effort": "medium"},
    ).output_text
    df_proc = require_csv(resp, prompts["proc_cols"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_proc.to_csv(out_path, index=False)
    print(f"[OK] saved {variant} -> {out_path}")


def main():
    base_dir = Path(__file__).resolve().parents[2]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    priors_text = '{ "mortality_rate": 0.14778, "demographics": { "LANGUAGE": { "ENGL": 0.56, "None": 0.36, "OTHER": 0.08 }, "RELIGION": { "CATHOLIC": 0.35, "NOT SPECIFIED": 0.21, "UNOBTAINABLE": 0.13, "PROTESTANT QUAKER": 0.12, "JEWISH": 0.09, "OTHER": 0.10 }, "MARITAL_STATUS": { "MARRIED": 0.48, "SINGLE": 0.24, "WIDOWED": 0.14, "None": 0.06, "DIVORCED": 0.06, "OTHER": 0.02 }, "ETHNICITY": { "WHITE": 0.70, "UNKNOWN/NOT SPECIFIED": 0.10, "BLACK/AFRICAN AMERICAN": 0.07, "HISPANIC OR LATINO": 0.02, "OTHER": 0.11 }, "INSURANCE": { "Medicare": 0.5254320098745114, "Private": 0.34712507714462043, "Medicaid": 0.0829818967290681, "Government": 0.030292120962764863, "Self Pay": 0.014168895289035179 }, "GENDER": { "M": 0.5656757868751285, "F": 0.4343242131248714 }, "AGE_BIN": { "65-79": 0.30636700267434686, "50-64": 0.270854762394569, "80+": 0.15104916683809916, "35-49": 0.1378060069944456, "18-34": 0.07398169101008023, "1-17": 0.059941370088459164 } } }'

    icd_top = (base_dir / "data" / "priors" / "top100_icd9.csv").read_text()
    proc_master = (base_dir / "data" / "input" / "D_ICD_PROCEDURES.csv").read_text()
    out_dir = base_dir / "outputs" / "synthetic"

    variants = {
        "full": out_dir / "syn_full.csv",
        "prior": out_dir / "syn_prior.csv",
        "base": out_dir / "syn_base.csv",
    }
    for name, path in variants.items():
        generate_variant(client, name, priors_text, icd_top, proc_master, path)


if __name__ == "__main__":
    main()
