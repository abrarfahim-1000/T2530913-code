"""
evaluator.py — condition evaluation with an explicit verdict taxonomy.

The governing rule: **only VIOLATED blocks.** A condition the shield cannot
evaluate, or that blows up, must never gate a prediction — an unparseable rule is
the shield's problem, not the operator's, and failing closed on it would turn
every ruleset defect into a false block.
"""
from __future__ import annotations

import enum
from functools import lru_cache
from types import CodeType


class Verdict(str, enum.Enum):
    """Outcome of evaluating one rule condition against one context."""

    VIOLATED = "VIOLATED"            # condition true -> the rule is broken -> BLOCK
    SATISFIED = "SATISFIED"          # condition false -> the rule holds
    NOT_EVALUABLE = "NOT_EVALUABLE"  # a referenced variable is absent from the context
    ERROR = "ERROR"                  # malformed condition, or a non-boolean result

    @property
    def blocks(self) -> bool:
        return self is Verdict.VIOLATED


# Conditions are evaluated ~5k times each by the polarity guard and once per
# frame at inference; compiling once is the difference between seconds and minutes.
@lru_cache(maxsize=4096)
def compile_condition(condition: str) -> CodeType:
    return compile(condition, "<rule-condition>", "eval")


_NO_BUILTINS: dict = {"__builtins__": {}}


def evaluate_condition(condition: str, context: dict) -> Verdict:
    """Evaluate one rule condition against one context.

    The condition is expected to have already passed `lint_condition` at
    extraction time (AST whitelist over the closed vocabulary), so this is a
    second line of defence rather than the primary one. `NOT_EVALUABLE` should
    therefore be ~0 in practice; a nonzero count is a regression signal worth
    reporting, not a routine outcome.
    """
    try:
        code = compile_condition(condition)
    except (SyntaxError, ValueError, TypeError):
        return Verdict.ERROR

    try:
        result = eval(code, _NO_BUILTINS, context)  # noqa: S307 - AST-linted, no builtins
    except NameError:
        # Variable missing from the context — e.g. voltage_pu_min on a blackout
        # frame, where context.py deliberately omits it.
        return Verdict.NOT_EVALUABLE
    except Exception:
        return Verdict.ERROR

    if not isinstance(result, bool):
        # numpy.bool_ is boolean but does not subclass bool — unwrap it.
        item = getattr(result, "item", None)
        if callable(item):
            result = item()

    if not isinstance(result, bool):
        # A numeric or None result means the condition is not a predicate at all
        # (e.g. a bare "rho_max"). Malformed, not merely unevaluable.
        return Verdict.ERROR

    return Verdict.VIOLATED if result else Verdict.SATISFIED
