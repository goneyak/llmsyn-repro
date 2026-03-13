import os
import io
import re
import json
import time
import logging
from pathlib import Path

import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column schemas — defined once at module level, shared by prompt templates
# ---------------------------------------------------------------------------
DEMO_COLS = [
    "AGE", "LANGUAGE", "RELIGION", "MARITAL_STATUS",
    "ETHNICITY", "INSURANCE", "HOSPITAL_EXPIRE_FLAG",
]
DIAG_COLS = DEMO_COLS + ["MAIN_DIAGNOSIS"]
COMORB_COLS = DIAG_COLS + ["COMORBIDITIES"]
PROC_COLS = COMORB_COLS + ["PROCEDURES"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def fmt(template: str, **kwargs: str) -> str:
    """Safe single-pass string replacement.

    Uses plain str.replace() instead of str.format() so that data values
    containing curly braces (e.g. CSV cells) do not cause KeyErrors.
    """
    for key, val in kwargs.items():
        template = template.replace("{" + key + "}", str(val))
    return template


def strip_fences(text: str) -> str:
    """Remove markdown code fences that some models wrap CSV output in."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\r?\n?", "", text)
    text = re.sub(r"\r?\n?```\s*$", "", text)
    return text.strip()


def require_csv(text: str, expected_cols: list[str]) -> pd.DataFrame:
    text = strip_fences(text)
    df = pd.read_csv(io.StringIO(text))
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in model output.\nFirst 200 chars: {text[:200]}")
    ordered = expected_cols + [c for c in df.columns if c not in expected_cols]
    return df[ordered]


def call_api(client: OpenAI, prompt: str, max_retries: int = 3) -> str:
    """Call the OpenAI Responses API with exponential-backoff retry."""
    for attempt in range(max_retries):
        try:
            return client.responses.create(
                model="gpt-5.1",
                input=prompt,
                reasoning={"effort": "medium"},
            ).output_text
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "API call failed (attempt %d/%d): %s — retrying in %ds",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Prior summarisation — compact version for prompt injection
# ---------------------------------------------------------------------------

def summarize_priors(priors_data: dict, top_n: int = 6) -> dict:
    """Return a compact prior dict (top-N per field) suitable for a prompt.

    The full prior.json may contain 60+ language codes with tiny probabilities.
    Passing them all wastes tokens and adds noise; only the dominant entries
    are informative for the model.
    """
    out: dict = {"mortality_rate": round(priors_data.get("mortality_rate", 0.0), 4)}
    demo: dict = {}
    for field, dist in priors_data.get("demographics", {}).items():
        sorted_items = sorted(dist.items(), key=lambda x: -x[1])
        top = {k: round(v, 4) for k, v in sorted_items[:top_n]}
        rest = round(1.0 - sum(top.values()), 4)
        if rest > 0.01:
            top["OTHER"] = rest
        demo[field] = top
    out["demographics"] = demo
    return out


def build_prior_note(variant: str, priors_data: dict) -> str:
    """Return a variant-specific prior instruction block for the demographics prompt.

    full  — explicit frequency distributions from the real dataset.
            Model should match the stated rates as closely as possible.
    prior — categorical vocabulary only (which values are allowed), but no
            frequency target.  Model determines a plausible mixture freely.
    base  — no prior knowledge assumed; model uses general clinical reasoning.
    """
    if variant == "full":
        compact = summarize_priors(priors_data)
        return (
            "Match these demographic distributions from a real hospital dataset "
            f"(in-hospital mortality rate ≈ {compact['mortality_rate']:.1%}):\n"
            + json.dumps(compact["demographics"], indent=2)
        )
    elif variant == "prior":
        vocab = {
            field: sorted(dist.keys())
            for field, dist in priors_data.get("demographics", {}).items()
        }
        return (
            "Use only the following allowed values for each demographic field "
            "(no specific frequency target — generate a clinically plausible mixture):\n"
            + json.dumps(vocab, indent=2)
        )
    else:  # base
        return (
            "Generate realistic but generic demographics for a general inpatient hospital population. "
            "Do not assume any specific distributional priors."
        )


# ---------------------------------------------------------------------------
# Prompt loading and assembly
# ---------------------------------------------------------------------------

def load_prompt(name: str, prompt_dir: Path) -> str:
    return (prompt_dir / name).read_text(encoding="utf-8")


def build_prompts(
    variant: str,
    priors_data: dict,
    icd_top: str,
    proc_master: str,
    prompt_dir: Path,
) -> dict:
    """Load templates from prompts/ and perform the static substitutions.

    Dynamic CSV placeholders ({demo_csv}, {diag_csv}, {comorb_csv}) are left
    intact here and filled just-in-time inside generate_variant().
    """
    prior_note = build_prior_note(variant, priors_data)

    demo_prompt = fmt(
        load_prompt("demographics.txt", prompt_dir),
        prior_note=prior_note,
        demo_cols=", ".join(DEMO_COLS),
    )
    diag_prompt = fmt(
        load_prompt("diagnosis.txt", prompt_dir),
        icd_top=icd_top,
        diag_cols=", ".join(DIAG_COLS),
    )
    comorb_prompt = fmt(
        load_prompt("complications.txt", prompt_dir),
        icd_top=icd_top,
        comorb_cols=", ".join(COMORB_COLS),
    )
    proc_prompt = fmt(
        load_prompt("procedures.txt", prompt_dir),
        proc_master=proc_master,
        proc_cols=", ".join(PROC_COLS),
    )
    return {
        "demo_prompt": demo_prompt,
        "diag_prompt": diag_prompt,
        "comorb_prompt": comorb_prompt,
        "proc_prompt": proc_prompt,
    }


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_variant(
    client: OpenAI,
    variant: str,
    priors_data: dict,
    icd_top: str,
    proc_master: str,
    out_path: Path,
    prompt_dir: Path,
) -> None:
    prompts = build_prompts(variant, priors_data, icd_top, proc_master, prompt_dir)

    resp = call_api(client, prompts["demo_prompt"])
    df_demo = require_csv(resp, DEMO_COLS)

    resp = call_api(client, fmt(prompts["diag_prompt"], demo_csv=df_demo.to_csv(index=False)))
    df_diag = require_csv(resp, DIAG_COLS)

    resp = call_api(client, fmt(prompts["comorb_prompt"], diag_csv=df_diag.to_csv(index=False)))
    df_comorb = require_csv(resp, COMORB_COLS)

    resp = call_api(client, fmt(prompts["proc_prompt"], comorb_csv=df_comorb.to_csv(index=False)))
    df_proc = require_csv(resp, PROC_COLS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_proc.to_csv(out_path, index=False)
    print(f"[OK] saved {variant} -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base_dir = Path(__file__).resolve().parents[2]
    prompt_dir = base_dir / "prompts"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    prior_json_path = base_dir / "data" / "priors" / "prior.json"
    if not prior_json_path.exists():
        raise FileNotFoundError(f"Missing required prior file: {prior_json_path}")
    priors_data = json.loads(prior_json_path.read_text(encoding="utf-8"))

    icd_top = (base_dir / "data" / "priors" / "top100_icd9.csv").read_text(encoding="utf-8")

    proc_csv_path = base_dir / "data" / "input" / "D_ICD_PROCEDURES.csv"
    if not proc_csv_path.exists():
        raise FileNotFoundError(f"Missing required procedure dictionary: {proc_csv_path}")
    # Trim to the first 300 rows (ICD-9 procedure codes are ~3 700 rows in MIMIC).
    # Passing the full file wastes tokens without meaningfully improving code diversity.
    proc_df = pd.read_csv(proc_csv_path)
    proc_master = proc_df[["ICD9_CODE", "LONG_TITLE"]].head(300).to_csv(index=False)

    out_dir = base_dir / "outputs" / "synthetic"
    variants = {
        "full": out_dir / "syn_full.csv",
        "prior": out_dir / "syn_prior.csv",
        "base": out_dir / "syn_base.csv",
    }
    for name, path in variants.items():
        generate_variant(client, name, priors_data, icd_top, proc_master, path, prompt_dir)


if __name__ == "__main__":
    main()
