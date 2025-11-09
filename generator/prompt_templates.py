"""
Prompt templates (x0~x3) & prompt adaptor (skeleton).
Edit freely as you iterate.
"""
from __future__ import annotations

def x0_template(prior: dict) -> str:
    # TODO: read data/mimic_stats.csv and format prompt
    return "[x0 prompt here]"

def x1_template(prev: dict, prior: dict) -> str:
    return "[x1 prompt here]"

def x2_template(prev: dict, knowledge: str) -> str:
    return "[x2 prompt here]"

def x3_template(prev: dict, knowledge: str) -> str:
    return "[x3 prompt here]"

def adapt_prompt(prev_state: dict, prior: dict, knowledge: str) -> str:
    # TODO: combine previous outputs + prior + knowledge
    return "[adapted prompt]"
