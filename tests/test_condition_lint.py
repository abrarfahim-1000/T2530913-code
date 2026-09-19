"""
Tests for extraction.common.lint_condition — the deterministic guarantee that
every rule reaching the KG is machine-evaluable against Grid2Op telemetry.
Run: pytest tests/test_condition_lint.py -v
"""
import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extraction"))

from extraction.common import (
    ALLOWED_CONDITION_BUILTINS, CONDITION_VOCABULARY, EXTRACT_PROMPT,
    TRANSLATE_PROMPT, VALIDATE_PROMPT,
    Rule, RawRule, _strip_think, _vocabulary_block, extract_json_array,
    lint_condition, lint_condition_raw,
)
from shield.evaluator import (CONDITION_BUILTINS, Verdict, _EVAL_GLOBALS,
                              _SAFE_BUILTINS, evaluate_condition)
from pydantic import ValidationError


# ── ACCEPT ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cond", [
    "voltage_pu_min < 0.95 or voltage_pu_max > 1.05",
    "loading_pct > 100",
    "rho_max >= 1.0",
    "n_tripped_lines >= 2",
    "any_line_tripped",
    "not any_line_tripped",
    "rho_max > 1.0 and n_tripped_lines >= 1",
    "(voltage_pu_min < 0.9) or (loading_pct > 120 and any_line_tripped)",
    "voltage_pu_max > 1.1 or voltage_pu_max < -0.5",   # negative literal allowed
    "  loading_pct >= 110  ",                          # stripped
    # ── the three whitelisted built-ins (issue B5, 2026-09-20) ────────────────
    "abs(generation_load_imbalance_pct) > 5",          # magnitude bound
    "max(rho_max, 1.0) > 1.0",
    "min(voltage_pu_min, voltage_pu_max) < 0.95",
    "abs(generation_load_imbalance_pct) > 5 and loading_pct > 90",
    "not abs(generation_load_imbalance_pct) > 5",
])
def test_accepts_valid_conditions(cond):
    assert lint_condition(cond) == cond.strip()


# ── REJECT ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cond", [
    # non-Python syntax seen in the v1 ruleset
    "voltage_pu BETWEEN 0.2 AND 0.3",
    "voltage_pu < 0.90 within 80ms OR voltage_pu < 0.97 within 2s",
    # out-of-vocabulary variables (unobservable in Grid2Op)
    "frequency_hz < 49",
    "droop_pct > 5",
    "voltage_pu > 1.05",                 # v1 name — replaced by _min/_max forms
    "p_mw < 0.2 * rated_mw",
    "time_since_verification_months > 6",
    # disallowed constructs
    "sum(rho_max, 1.0) > 1.0",           # call to a NON-whitelisted function
    "rho_max + 0.1 > 1.0",               # arithmetic (BinOp not whitelisted)
    "loading_pct > '100'",               # non-numeric literal
    "__import__('os')",                  # sandbox escape attempt
    # not boolean / not an expression
    "rho_max",                           # evaluates to float, not bool
    "manual verification required",      # prose
    "",                                  # empty
])
def test_rejects_invalid_conditions(cond):
    with pytest.raises(ValueError):
        lint_condition(cond)


# ── INTEGRATION WITH Rule SCHEMA ──────────────────────────────────────────────

def _rule(condition: str) -> dict:
    return {
        "rule_id": "R_001",
        "source": "IEEE Std 1547-2018, Section 7.4",
        "entity": "Grid",
        "condition": condition,
        "action": "BLOCK",
        "severity": "high",
        "explanation": "test rule",
    }


def test_rule_schema_accepts_linted_condition():
    r = Rule(**_rule("voltage_pu_min < 0.95 or voltage_pu_max > 1.05"))
    assert r.condition == "voltage_pu_min < 0.95 or voltage_pu_max > 1.05"


def test_rule_schema_drops_unlintable_condition():
    with pytest.raises(ValidationError):
        Rule(**_rule("frequency_hz < 49"))


# ── PROMPT RENDERING ──────────────────────────────────────────────────────────

def test_prompts_contain_rendered_vocabulary():
    # VALIDATE_PROMPT and TRANSLATE_PROMPT contain the vocabulary (EXTRACT_PROMPT is raw)
    for name in CONDITION_VOCABULARY:
        assert name in VALIDATE_PROMPT
        assert name in TRANSLATE_PROMPT
    assert "{vocabulary}" not in VALIDATE_PROMPT
    assert "{vocabulary}" not in TRANSLATE_PROMPT
    # runtime placeholders must survive the pre-render
    assert "{chunk}" in EXTRACT_PROMPT
    assert "{chunk}" in VALIDATE_PROMPT and "{rules}" in VALIDATE_PROMPT


def test_extract_prompt_stays_open_vocabulary():
    """Stage 1 must NOT constrain the model to CONDITION_VOCABULARY.

    Regression guard: closing the vocabulary at extraction time makes the model
    answer '[]' for almost every chunk of frequency/timing-heavy standards
    (PRC-024, PRC-006, ...), starving translate.py — the stage that actually
    owns the vocabulary mapping. Assert on the rendered block, not on variable
    names: 'voltage_pu_min' legitimately appears in the rule-3 polarity example.
    """
    assert _vocabulary_block() not in EXTRACT_PROMPT
    assert "There is no pre-defined variable list." in EXTRACT_PROMPT


def test_translate_prompt_forbids_oring_an_equipment_rating_with_a_ratio():
    """Regression guard for R_769 (2026-08-19).

    The translator emitted
    `loading_pct > 100 or current_a_max > 580 or apparent_power_mva_max > 132`
    from an offshore circuit rating schedule that stated ONE limit in three units.
    The two absolute terms name grid-wide maxima, while 580 A / 132 MVA belong to
    one named circuit, so the rule fired on 100% of healthy frames on all three
    topologies. The prompt must forbid the ORed form explicitly.
    """
    assert 'PROXY SUBSTITUTION' in TRANSLATE_PROMPT
    assert 'REPLACES the absolute figure' in TRANSLATE_PROMPT
    assert 'never ORed with it' in TRANSLATE_PROMPT
    assert ('NEVER "loading_pct > 100 or current_a_max > 580 '
            'or apparent_power_mva_max > 132".') in TRANSLATE_PROMPT


def _self_check_healthy_values():
    """The healthy-grid substitutions the prompt tells the model to try."""
    import re
    body = TRANSLATE_PROMPT.split('SELF-CHECK before returning.', 1)[1]
    body = body.split('  - a CONSTRAINT', 1)[0]
    return {m.group(1): float(m.group(2))
            for m in re.finditer(r'(\w+) = (-?\d+(?:\.\d+)?)', body)}


def test_self_check_magnitudes_actually_expose_the_rating_defect():
    """The self-check is only load-bearing if its numbers are large enough to
    catch the bug. Evaluated against the healthy values the prompt itself
    supplies, the defective condition must come out True — so a model following
    the self-check literally is forced to reject it — while the corrected
    single-term form must come out False."""
    healthy = _self_check_healthy_values()
    for name in ('current_a_max', 'apparent_power_mva_max', 'loading_pct'):
        assert name in healthy, f'self-check gives no healthy value for {name}'

    defective = 'loading_pct > 100 or current_a_max > 580 or apparent_power_mva_max > 132'
    assert eval(defective, {'__builtins__': {}}, healthy) is True, (
        'self-check magnitudes are too small to expose the ORed-rating defect')
    assert eval('loading_pct > 100', {'__builtins__': {}}, healthy) is False


def test_self_check_covers_every_variable_in_the_vocabulary():
    """A variable with no healthy value in the self-check cannot be checked by the
    model at all — which is how the rating defect got through: the old self-check
    named seven variables and stopped."""
    healthy = _self_check_healthy_values()
    named = set(healthy) | {'any_line_tripped'}   # boolean, stated as False not `= n`
    missing = set(CONDITION_VOCABULARY) - named
    assert not missing, f'self-check supplies no healthy value for: {sorted(missing)}'

# ── THINK-TAG STRIPPING ───────────────────────────────────────────────────────

def test_strip_think_removes_well_formed_block():
    assert _strip_think("<think>reasoning here</think>[{\"a\": 1}]") == '[{"a": 1}]'


def test_strip_think_handles_multiline_block():
    raw = "<think>\nline one\nline two\n</think>\n[]"
    assert _strip_think(raw) == "[]"


def test_strip_think_drops_unclosed_block():
    """num_predict exhausted mid-reasoning — nothing after the tag is usable."""
    assert _strip_think("some preamble\n<think>cut off mid rea") == "some preamble"


def test_strip_think_leaves_think_free_text_alone():
    assert _strip_think('  [{"rule_id": "R_001"}]  ') == '[{"rule_id": "R_001"}]'


def test_extract_json_array_ignores_brackets_inside_think():
    """The exact shape that produced 'No JSON array found': a '[' inside the
    reasoning trace hijacks the find('[') slice unless the block is stripped."""
    raw = (
        "<think>The text mentions [Page 4] and a range [0.95, 1.05], so...</think>\n"
        '[{"rule_id": "R_001", "condition": "frequency_hz < 49"}]'
    )
    assert extract_json_array(raw) == [
        {"rule_id": "R_001", "condition": "frequency_hz < 49"}
    ]


# ── RAW RULE / lint_condition_raw ─────────────────────────────────────────────

@pytest.mark.parametrize("cond", [
    # Variables outside CONDITION_VOCABULARY are accepted by lint_condition_raw
    "frequency_hz < 49",
    "frequency_hz > 51.5",
    "power_factor < 0.9",
    "droop_pct > 5",
    "time_seconds > 0.16",
    "voltage_pu > 1.05",                   # old name — no _min/_max suffix needed in raw
    "n_tripped_lines >= 1 and voltage_pu_min < 0.95",  # mixed vocab + raw
])
def test_lint_condition_raw_accepts_any_variable(cond):
    """lint_condition_raw must accept any variable name (syntax-only check)."""
    assert lint_condition_raw(cond) == cond.strip()


@pytest.mark.parametrize("cond", [
    "voltage_pu BETWEEN 0.2 AND 0.3",      # non-Python syntax
    "sum(rho_max, 1.0) > 1.0",             # call to a NON-whitelisted function
    "manual verification required",         # prose
    "",                                     # empty
])
def test_lint_condition_raw_rejects_bad_syntax(cond):
    with pytest.raises(ValueError):
        lint_condition_raw(cond)


def test_lint_condition_raw_accepts_the_whitelisted_builtins():
    """Stage 1 is open-VOCABULARY, never open-SYNTAX. The raw linter must admit
    the same three calls the closed linter does — otherwise a standard stating a
    magnitude bound ('shall not exceed 5% in either direction') dies at stage 1,
    before the stage that owns the vocabulary mapping ever sees it."""
    for cond in ("abs(frequency_deviation_hz) > 0.2",
                 "max(rho_max, 1.0) > 1.0",
                 "min(droop_pct, 5) < 2"):
        assert lint_condition_raw(cond) == cond


def test_raw_rule_schema_accepts_out_of_vocab():
    """RawRule must accept conditions with variables outside Grid2Op vocabulary."""
    r = RawRule(**_rule("frequency_hz < 49"))
    assert r.condition == "frequency_hz < 49"


def test_raw_rule_schema_rejects_bad_syntax():
    """RawRule must still reject non-Python syntax."""
    with pytest.raises(ValidationError):
        RawRule(**_rule("voltage_pu BETWEEN 0.2 AND 0.3"))


def test_raw_rule_and_rule_differ_on_condition():
    """The same condition should pass RawRule but fail Rule when it uses
    out-of-vocabulary variables."""
    condition = "frequency_hz < 49"
    RawRule(**_rule(condition))             # should pass
    with pytest.raises(ValidationError):
        Rule(**_rule(condition))            # should fail


# ── THE CALL WHITELIST (issue B5) ─────────────────────────────────────────────
# Before 2026-09-20 the condition language had no calls at all, so a magnitude
# bound could not be written the natural way: the pre-registered expert rule
# X_010, `abs(generation_load_imbalance_pct) > 5`, resolved to NOT_EVALUABLE on
# every frame of every grid. Exactly three built-ins are now callable. The
# linter's job is to guarantee an LLM-authored string is safe to `eval`, so the
# widening must stay a widening of three names and nothing else.

_ATTACK_STRINGS = [
    "__import__('os')",                     # the classic escape
    "__import__('os').system('echo x')",
    "open('x')",
    "eval('1')",
    "exec('x=1')",
    "compile('1', '', 'eval')",
    "getattr(loading_pct, 'real') > 1",
    "globals() > 0",
    "locals() > 0",
    "vars() > 0",
    "dir() > 0",
    "type(1) > 0",
    "sum(loading_pct) > 1",                 # harmless-looking, still not whitelisted
    "print(1) > 0",
    "().__class__",                          # attribute access as a gadget chain
    "abs(1).__class__",                      # a WHITELISTED call, then an attribute
    "abs(1).__class__.__bases__[0] > 0",
    "loading_pct.real > 1",                  # attribute access of any kind
    "abs.__class__ > 0",
    "loading_pct[0] > 1",                    # subscript
    "__builtins__['abs'](1) > 0",            # call through a subscript
    "(lambda: 1)() > 0",                     # lambda
    "(lambda x: x)(loading_pct) > 1",
    "[x for x in (1,)] > 0",                 # comprehension
    "{x for x in (1,)} > 0",
    "abs(x=1) > 0",                          # keyword argument
    "max(loading_pct, key=abs) > 1",
    "abs(*[1]) > 0",                         # starred argument
    "max(*[1, 2]) > 0",
    "abs() > 0",                             # no argument at all
    "abs(abs)(1) > 0",                       # call whose callee is itself a call
]


@pytest.mark.parametrize("cond", _ATTACK_STRINGS)
def test_call_whitelist_rejects_every_escape(cond):
    """Every one of these must fail the CLOSED linter."""
    with pytest.raises(ValueError):
        lint_condition(cond)


@pytest.mark.parametrize("cond", _ATTACK_STRINGS)
def test_call_whitelist_rejects_every_escape_raw_too(cond):
    """...and the OPEN-vocabulary linter too. Stage 1 accepts any variable name,
    which would otherwise make `__import__` look like just another variable."""
    with pytest.raises(ValueError):
        lint_condition_raw(cond)


def test_whitelist_is_exactly_three_names():
    """A deliberately narrow widening. Growing this list is a security decision,
    not a convenience one — it must break this test first."""
    assert ALLOWED_CONDITION_BUILTINS == ("abs", "min", "max")


def test_linter_and_sandbox_share_one_whitelist():
    """The linter must not accept a call the gate cannot resolve, nor refuse one
    it can. They are the same tuple object, imported, not two copies."""
    assert ALLOWED_CONDITION_BUILTINS is CONDITION_BUILTINS
    assert set(_SAFE_BUILTINS) == set(CONDITION_BUILTINS)


def test_sandbox_globals_expose_nothing_but_the_three():
    """`__builtins__` bound to a plain dict IS the whole builtins namespace for
    that eval. Nothing else may be reachable by name."""
    builtins_ns = _EVAL_GLOBALS["__builtins__"]
    assert set(builtins_ns) == {"abs", "min", "max"}
    assert set(_EVAL_GLOBALS) == {"__builtins__"}
    for forbidden in ("__import__", "open", "eval", "exec", "getattr", "compile"):
        assert forbidden not in builtins_ns


def test_a_whitelisted_name_is_not_smuggled_into_the_vocabulary():
    """The callee exemption is by AST NODE, not by name. `abs` used as a VALUE is
    still an unknown variable — otherwise the whitelist would quietly widen
    CONDITION_VOCABULARY by three entries."""
    with pytest.raises(ValueError):
        lint_condition("abs > 5")
    with pytest.raises(ValueError):
        lint_condition("loading_pct > max")


def test_a_non_whitelisted_name_still_fails_the_vocabulary_check():
    with pytest.raises(ValueError):
        lint_condition("frequency_hz > 50")
    with pytest.raises(ValueError):
        lint_condition("abs(frequency_hz) > 5")     # whitelisted call, unknown arg


# ── THE SANDBOX SIDE OF THE SAME CHANGE ───────────────────────────────────────

def test_the_expert_rule_that_could_never_fire_now_fires():
    """X_010 of the pre-registered expert set. Both signs must reach VIOLATED —
    that is the whole content of a magnitude bound, and it was unreachable."""
    cond = "abs(generation_load_imbalance_pct) > 5"
    assert evaluate_condition(cond, {"generation_load_imbalance_pct": -7.0}) is Verdict.VIOLATED
    assert evaluate_condition(cond, {"generation_load_imbalance_pct": 7.0}) is Verdict.VIOLATED
    assert evaluate_condition(cond, {"generation_load_imbalance_pct": -1.0}) is Verdict.SATISFIED
    # ...and a missing variable is still NOT_EVALUABLE, not a crash.
    assert evaluate_condition(cond, {}) is Verdict.NOT_EVALUABLE


def test_the_sandbox_still_refuses_an_escape_at_runtime():
    """Belt and braces: the linter is the primary guard, but a condition that
    reached the evaluator unlinted must not execute either."""
    for cond in ("__import__('os').system('echo x') == 0",
                 "open('x') == 0",
                 "eval('1') == 1"):
        assert evaluate_condition(cond, {}) is Verdict.NOT_EVALUABLE  # NameError
    # attribute access on a whitelisted call's result raises, and ERROR never blocks
    assert not evaluate_condition("abs(1).__class__ == int", {}).blocks


def test_context_variables_win_over_the_builtins():
    """The callables live in the eval GLOBALS, so a context key of the same name
    shadows them (locals are searched first). Telemetry is never silently
    replaced by a function."""
    assert evaluate_condition("abs > 5", {"abs": 9.0}) is Verdict.VIOLATED
    assert evaluate_condition("max < 1", {"max": 0.5}) is Verdict.VIOLATED


def test_evaluating_a_condition_does_not_mutate_the_context():
    ctx = {"generation_load_imbalance_pct": -7.0}
    evaluate_condition("abs(generation_load_imbalance_pct) > 5", ctx)
    assert ctx == {"generation_load_imbalance_pct": -7.0}


# ── THE SERVED CORPORA ARE UNAFFECTED ─────────────────────────────────────────

_CORPUS_FILES = [
    "validated_translated/all_rules_deduped.jsonl",   # the 4-rule SERVED corpus
    "validated_strict/all_rules_deduped.jsonl",       # the counterfactual arm
    "shield_corpus/all_rules_channels.jsonl",         # the 58-rule v2 corpus
]


def test_no_shipped_rule_contains_a_call_so_no_reported_number_can_move():
    """The widening can only change the verdict of a condition that CONTAINS a
    call. No record in any shipped corpus does, so every shield delta, precision
    figure and block count reported to date is structurally untouched by it.

    This is the unit-test form of 're-run the harness and diff' — it proves the
    stronger statement (nothing *could* have moved) without a forward pass.
    """
    root = Path(__file__).resolve().parents[1]
    seen = 0
    for rel in _CORPUS_FILES:
        path = root / rel
        if not path.exists():                # data/ artifacts are gitignored
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cond = json.loads(line)["condition"]
            seen += 1
            tree = ast.parse(cond, mode="eval")
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
            assert not calls, f"{rel}: {cond!r} contains a call — verdict may move"
            # The other, subtler way the widening could reach a rule: a BARE name
            # `abs`/`min`/`max` used as a value. It used to fall through to a
            # NameError (NOT_EVALUABLE); it now resolves to a function and the
            # non-boolean result becomes ERROR. Neither blocks, but the harness
            # counts the two verdicts separately, so no shipped rule may do it.
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            shadowed = names & set(CONDITION_BUILTINS)
            assert not shadowed, f"{rel}: {cond!r} reads {shadowed} as a variable"
    assert seen >= 4, "no corpus was actually scanned"
