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


# ── THE THREE CALLABLE BUILT-INS ──────────────────────────────────────────────
# A magnitude bound — "the imbalance shall not exceed 5% in either direction" —
# has no natural form in this language without `abs`, and envelope clauses want
# `min`/`max`. Before 2026-09-20 the namespace had no builtins at all, so any such
# condition raised NameError and resolved to NOT_EVALUABLE on every frame: silent,
# total, and indistinguishable from a rule that simply never fired. The whitelist
# is enumerated, never inferred — these three are pure, total over numbers, and
# return no object with an attribute worth reaching for.
#
# `extraction.common.lint_condition` enforces the SAME three names at extraction
# time and imports this tuple, so the linter and the sandbox cannot drift apart.
CONDITION_BUILTINS: tuple[str, ...] = ("abs", "min", "max")

_SAFE_BUILTINS: dict = {"abs": abs, "min": min, "max": max}

# WHY THE GLOBALS AND NOT THE CONTEXT.
# `eval(code, globals, locals)` resolves a bare name locals -> globals -> builtins,
# and `context` is the LOCALS mapping here. Binding the callables into the globals'
# `__builtins__` slot therefore means:
#   * a context variable named `abs`/`min`/`max` still wins, because locals are
#     searched first — telemetry is never shadowed by a function, and the function
#     is never shadowed into a silent type error;
#   * the caller's `context` dict is neither mutated nor copied, which matters at
#     ~700k evaluations per grid;
#   * `__builtins__` bound to a plain dict IS the entire builtins namespace for
#     this eval, so `__import__`, `open`, `eval` and `getattr` remain absent. The
#     no-builtins posture is narrowed by exactly three names, not relaxed.
_EVAL_GLOBALS: dict = {"__builtins__": _SAFE_BUILTINS}

# Retained under its historical name for anything that imported it; the eval path
# uses _EVAL_GLOBALS.
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
        result = eval(code, _EVAL_GLOBALS, context)  # noqa: S307 - AST-linted, abs/min/max only
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
