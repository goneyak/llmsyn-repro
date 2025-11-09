"""
Prior and relevant knowledge retrievers (skeleton).
"""
from __future__ import annotations

def load_prior_stats(stats_path: str) -> dict:
    # TODO: parse CSV for your needs
    return {"mortality_rate": 0.175}

def retrieve_relevant_knowledge(main_dx_name: str) -> str:
    # TODO: scrape or cached text (avoid direct web calls in repo code)
    return f"[knowledge snippet for {main_dx_name}]"
