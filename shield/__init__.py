"""
Component D — the Symbolic Validation Shield.

A hard, post-hoc gate: every GNN prediction is validated against the extracted
rule corpus before it is emitted. The GNN never sees the rules; the shield only
sees the prediction plus raw telemetry. There is no bypass path.

Layout:
    context.py    build_context()      raw record/obs -> the 6-variable namespace
    evaluator.py  evaluate_condition() condition + context -> Verdict
    shield.py     validate()           context + rules   -> ShieldResult
"""

from shield.context import CONTEXT_VARIABLES, build_context, load_base_kv
from shield.evaluator import Verdict, evaluate_condition
from shield.shield import (
    JsonlRuleProvider,
    RuleProvider,
    ShieldResult,
    build_explanation,
    validate,
)

__all__ = [
    "CONTEXT_VARIABLES",
    "build_context",
    "load_base_kv",
    "Verdict",
    "evaluate_condition",
    "RuleProvider",
    "JsonlRuleProvider",
    "ShieldResult",
    "validate",
    "build_explanation",
]
