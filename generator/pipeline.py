# generator/pipeline.py
import os, json
from .prompt_templates import build_x0_prompt
from .sampler import sample_x0

def main():
    prior_path = "outputs/priors/prior_non_newborn.json"
    output_path = "outputs/synth/synth_x0.json"

    os.makedirs("outputs/synth", exist_ok=True)
    with open(prior_path) as f:
        prior = json.load(f)

    prompt = build_x0_prompt(prior)
    samples = sample_x0(prompt, n=100)

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[OK] Saved {len(samples)} synthetic patients → {output_path}")

if __name__ == "__main__":
    main()
