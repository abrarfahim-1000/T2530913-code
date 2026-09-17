"""
channels.py — the four output channels, and the structural bar on the veto path.

Shield v1 had one channel. A rule either vetoed a prediction or it was discarded,
which is why 54 of the 58 expressible rules were on the floor (thesis_findings
19.3): not because they were wrong, but because vetoing was the only thing a rule
was permitted to do. A rule that faithfully reads a standard and evaluates on the
telemetry still has something to say when it cannot justify a veto.

v2 gives it somewhere to say it:

    BLOCK           validated CONSTRAINT. Vetoes an over-permissive prediction.
                    "Action blocked: <rule> is violated."
    WARN            fires, DISCRIMINATES ON THE N-1 TASK, not validated as
                    authoritative. "Let through, but per <rule>, that could
                    happen" -- and it must quote the rate it earned in 21.2.
    NORMAL          AFFIRMATION holding on its own class. "Let through;
                    environment normal per <rule>."
    NOT_APPLICABLE  correct, evaluable, but its calibration does not transfer to
                    these grids -- or it was measured and did not earn a voice
                    (21.2's two INVERTED power-factor predicates). Cited, never
                    rendered as a warning.
    INERT           cannot fire on any class on any grid. Excluded, and counted.

**Only BLOCK can veto, and that is enforced structurally, not by a convention.**
`partition_by_channel` hands the gate a dict, and the gate reads exactly one key
of it when deciding to block. A WARN rule cannot reach the veto path by being
mislabelled, mis-sorted or mis-ordered, because it is never in the list the veto
path iterates. This matters because 13's 92-94% intervention precision is
measured on the BLOCK set: anything that silently widens it moves a headline
number.

BACK-COMPATIBILITY. A rule with no `channel` key keeps its v1 behaviour exactly:
CONSTRAINT -> BLOCK, AFFIRMATION -> NORMAL. The served four-rule corpus carries no
`channel`, so every previously reported figure reproduces unchanged. This is
deliberate -- the channel corpus is a superset, not a replacement.
"""
from __future__ import annotations

from typing import Iterable

BLOCK = "BLOCK"
WARN = "WARN"
NORMAL = "NORMAL"
NOT_APPLICABLE = "NOT_APPLICABLE"
INERT = "INERT"

CHANNELS = (BLOCK, WARN, NORMAL, NOT_APPLICABLE, INERT)

#: Channels that produce output. NOT_APPLICABLE and INERT are carried so they can
#: be cited and counted, never rendered.
SPEAKING = (BLOCK, WARN, NORMAL)

#: The single channel permitted to veto. Kept as a one-element tuple rather than a
#: bare string so that widening it is a visible, reviewable diff.
VETO_CHANNELS = (BLOCK,)


#: The retired 4-class `classify` vocabulary an AFFIRMATION's `affirms` field was
#: written against, and the live N-1 vocabulary a prediction is issued in. They
#: overlap in meaning -- `normal` and `secure` describe the same physical state --
#: but they are NOT aliased. Aliasing them would move `supporting_rules`,
#: `unsupported`, and therefore the Option B counterfactual that eval_shield_n1
#: reports. What it does instead is let the explanation say which vocabulary a
#: rule speaks, rather than reporting a vocabulary mismatch as a contradiction.
CLASSIFY_LABELS = ("normal", "overload", "line_trip", "cascade")
N1_LABELS = ("secure", "violation")


def is_cross_vocabulary(affirms, predicted) -> bool:
    """True when the two labels come from different task vocabularies.

    Such a pair carries no information either way: `normal` neither supports nor
    contradicts `secure`, it was simply written for another task.
    """
    if affirms is None or predicted is None:
        return False
    a, p = str(affirms).lower(), str(predicted).lower()
    return (a in CLASSIFY_LABELS and p in N1_LABELS) or (a in N1_LABELS and p in CLASSIFY_LABELS)


def channel_of(rule: dict) -> str:
    """The channel this rule speaks on.

    An explicit `channel` wins. Without one, the v1 derivation applies, so a
    corpus predating v2 behaves identically. An unrecognised channel is treated
    as NOT_APPLICABLE -- the safe direction, since the failure mode of guessing
    wrong is a rule that does not speak, not a rule that vetoes.
    """
    declared = rule.get("channel")
    if declared is not None:
        declared = str(declared).upper()
        return declared if declared in CHANNELS else NOT_APPLICABLE
    role = str(rule.get("role", "CONSTRAINT")).upper()
    return NORMAL if role == "AFFIRMATION" else BLOCK


def partition_by_channel(rules: Iterable[dict]) -> dict[str, list[dict]]:
    """Bucket rules by channel, deduplicating by rule_id as v1 did.

    Dedup stays keyed on `rule_id` alone to reproduce v1 exactly. That is safe
    *here* and only here: the served corpus and the 58-rule channel corpus were
    both checked and carry no colliding id. The corpus builder asserts it, which
    is where a collision can actually be fixed. Never copy this key into new code
    that touches `rules_35b/` -- see 19.2, and the landmine in CLAUDE.md.
    """
    buckets: dict[str, list[dict]] = {c: [] for c in CHANNELS}
    seen: set = set()
    for rule in rules:
        rule_id = rule.get("rule_id")
        if rule_id is not None:
            if rule_id in seen:
                continue
            seen.add(rule_id)
        buckets[channel_of(rule)].append(rule)
    return buckets


def calibration_of(rule: dict) -> dict | None:
    """The N-1 calibration a WARN rule must carry before it may claim risk.

    Written by `evaluation/build_channel_corpus.py` from
    `results/audit/warn_n1_calibration.json`. `None` means the rule has not been
    calibrated -- it may still be reported, but not with a number, and
    `assert_warnings_calibrated` will refuse to serve it.
    """
    cal = rule.get("calibration")
    return cal if isinstance(cal, dict) else None


def assert_warnings_calibrated(rules: Iterable[dict]) -> None:
    """Refuse a corpus whose WARN rules cannot say how often they are right.

    21 is unambiguous about this: an uncalibrated warning is an alarm wearing a
    citation, and it is the one part of the four-channel design a reviewer can
    fairly attack. Two of the eleven WARN predicates measured INVERTED -- firing
    meant *lower* N-1 risk -- so this is not a hypothetical failure mode.
    """
    offenders = []
    for rule in rules:
        if channel_of(rule) != WARN:
            continue
        cal = calibration_of(rule)
        if cal is None or cal.get("p_violation_given_fires") is None:
            offenders.append(rule.get("rule_id", "?"))
        elif (cal.get("discrimination") or 0.0) <= 0.0:
            offenders.append(f"{rule.get('rule_id', '?')} (non-positive discrimination)")
    if offenders:
        raise ValueError(
            f"{len(offenders)} WARN rule(s) carry no usable N-1 calibration: "
            f"{', '.join(offenders)}. Build the corpus with "
            f"evaluation/build_channel_corpus.py, which demotes INVERTED and "
            f"INSUFFICIENT predicates out of WARN (thesis_findings 21.2)."
        )
