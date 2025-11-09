# generator/prompt_templates.py

def _fmt_line(name: str, d: dict) -> str:
    parts = [f"{k} {round(float(v), 4)}" for k, v in d.items()]
    return f"- {name}: " + ", ".join(parts)

def build_x0_prompt(prior: dict) -> str:
    dem = prior["demographics"]
    lines = [
        f"- Mortality (HOSPITAL_EXPIRE_FLAG=1): {round(float(prior['mortality_rate']), 4)}",
        _fmt_line("LANGUAGE", dem["LANGUAGE"]),
        _fmt_line("RELIGION", dem["RELIGION"]),
        _fmt_line("MARITAL_STATUS", dem["MARITAL_STATUS"]),
        _fmt_line("ETHNICITY", dem["ETHNICITY"]),
        _fmt_line("INSURANCE", dem["INSURANCE"]),
        _fmt_line("AGE_BIN", dem["AGE_BIN"]),
    ]
    priors_block = "\n".join(lines)
    return f"""
You are a hospital data expert generating realistic synthetic patient demographics.
Use only the given priors. Output JSON only.

PRIORS
{priors_block}

RULES
- Sample each field proportional to its probability.
- AGE: pick an integer within its AGE_BIN (80+ → 80–120).
- HOSPITAL_EXPIRE_FLAG: Bernoulli(p = Mortality above).
- Output valid JSON only.

SCHEMA
{{"AGE": int, "LANGUAGE": str, "RELIGION": str, "MARITAL_STATUS": str,
  "ETHNICITY": str, "INSURANCE": str, "HOSPITAL_EXPIRE_FLAG": int}}
""".strip()

__all__ = ["build_x0_prompt"]
