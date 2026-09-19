"""
Component D — the Symbolic Validation Shield.

A hard, post-hoc gate: every GNN prediction is validated against the extracted
rule corpus before it is emitted. The GNN never sees the rules; the shield only
sees the prediction plus raw telemetry. There is no bypass path.

Layout:
    context.py    build_context()      raw record/obs -> the 14-variable namespace
    evaluator.py  evaluate_condition() condition + context -> Verdict
    channels.py   partition_by_channel() rules -> BLOCK / WARN / NORMAL / ...
    shield.py     validate()           context + rules   -> ShieldResult

v2 (SHIELD_VERSION) adds the explanation channels. v1 could only refuse: a rule
either vetoed a prediction or was discarded, so 54 of 58 expressible rules never
spoke. v2 renders the PASS path too -- warnings with the N-1 rate they earned,
and affirmations that say which standards currently hold. **Only the BLOCK
channel vetoes**, so every previously reported figure is unchanged.
"""

from shield.channels import (
    CHANNELS,
    SPEAKING,
    VETO_CHANNELS,
    assert_warnings_calibrated,
    channel_of,
    is_cross_vocabulary,
    partition_by_channel,
)
from shield.context import CONTEXT_VARIABLES, build_context, load_base_kv
from shield.evaluator import Verdict, evaluate_condition
from shield.shield import (
    SHIELD_VERSION,
    JsonlRuleProvider,
    RuleProvider,
    ShieldResult,
    build_explanation,
    render_explanation,
    validate,
    validate_n1,
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
    "validate_n1",
    "build_explanation",
    "render_explanation",
    "SHIELD_VERSION",
    "CHANNELS",
    "SPEAKING",
    "VETO_CHANNELS",
    "channel_of",
    "is_cross_vocabulary",
    "partition_by_channel",
    "assert_warnings_calibrated",
]
