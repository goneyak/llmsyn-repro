# generator/sampler.py
import json, re, random, sys, time, os
from typing import List
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
JSON_RE = re.compile(r"\{.*?\}", re.S)

def extract_json(text: str) -> dict:
    m = JSON_RE.search(text.strip())
    if not m:
        raise ValueError("No JSON found")
    return json.loads(m.group(0))

def quick_fix(rec: dict) -> dict:
    try:
        age = int(rec.get("AGE", 0))
        if age < 18:
            if rec.get("MARITAL_STATUS") not in {"SINGLE","UNKNOWN"}:
                rec["MARITAL_STATUS"] = "SINGLE"
            if rec.get("INSURANCE") == "Medicare":
                rec["INSURANCE"] = random.choice(["Medicaid","Private"])
        rec["AGE"] = max(0, min(age, 120))
        rec["HOSPITAL_EXPIRE_FLAG"] = 1 if str(rec.get("HOSPITAL_EXPIRE_FLAG",0))=="1" else 0
    except Exception:
        pass
    return rec

def call_llm(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    res = model.generate_content(prompt)
    time.sleep(2)
    return res.text.strip()

def sample_x0(prompt: str, n: int = 100, retries: int = 2) -> List[dict]:
    out = []
    for i in range(n):
        ok = False
        for _ in range(retries + 1):
            try:
                txt = call_llm(prompt)
                rec = extract_json(txt)
                rec = quick_fix(rec)
                out.append(rec)
                ok = True
                break
            except Exception as e:
                print(f"[WARN] sample {i} failed: {e}", file=sys.stderr)
        if not ok:
            out.append({})
    return out
