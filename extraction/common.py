"""
common.py — Shared constants, schemas, and utilities for the three-stage pipeline.
extract.py (raw) → translate.py (→ CONDITION_VOCABULARY) → validate.py all import from here.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ast
import json
import logging
import re
from pathlib import Path
from typing import Optional

import ollama
import pdfplumber
from pydantic import BaseModel, ValidationError, field_validator, model_validator
from training.config import DEVICE

# ── MODEL CONFIG ──────────────────────────────────────────────────────────────
# DEVICE is a torch.device — compare .type, never the object against a string
# (torch.device("cuda") == "cuda" is False, which silently selected the small model).
# Research PC (24GB VRAM) gets the 35B; personal PC runs the 9B for testing.
EXTRACTOR_MODEL  = os.environ.get(
    "EXTRACTOR_MODEL",
    "qwen3.6:35b" if DEVICE.type == "cuda" else "qwen3.5:9b",
)
VALIDATOR_MODEL  = "nemotron-3-nano:latest"   # update tag if yours differs
TRANSLATOR_MODEL = EXTRACTOR_MODEL            # same model, re-used after extraction

# ── CHUNKING CONFIG ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 1200   # chars — ~300–400 tokens for dense IEEE prose
CHUNK_OVERLAP = 200

# ── SCHEMA CONSTANTS ──────────────────────────────────────────────────────────
ENTITY_VALUES = {
    "bus", "line", "transformer", "generator",
    "load", "protectiondevice", "grid",
    "powergeneratingmodule", "module", "facility",
    "powergeneratingfacility", "powersystemfacility",
    "offshorepowerparkmodule", "parkmodule",
}
SEVERITY_VALUES = {"critical", "high", "medium", "low"}

# ── RULE ROLE ─────────────────────────────────────────────────────────────────
# A rule is not necessarily a prohibition. The shield cannot read the corpus at
# inference, so the KG must carry both directions of evidence:
#
#   CONSTRAINT  — condition TRUE means the grid state violates the standard.
#                 The shield BLOCKs.
#   AFFIRMATION — condition TRUE means the telemetry is consistent with the
#                 predicted class named in `affirms`. Supporting evidence, never a
#                 block on its own (Option A semantics, component_d_handoff.md).
#
# Forcing every rule into CONSTRAINT polarity is what mangled healthy-band clauses
# in v1 and v2: "voltage shall remain within 0.95-1.05" is an AFFIRMATION of
# `normal`, and inverting it produced predicates that fire at nominal.
ROLE_VALUES = {"CONSTRAINT", "AFFIRMATION"}
FAULT_CLASSES = {"normal", "overload", "line_trip", "cascade"}
ACTION_VALUES = {
    "BLOCK", "DISCONNECT", "ALERT", "REDISPATCH",
    "RECONNECT", "RESPOND_WITHIN_2S", "SHED_LOAD", "OTHER",
}

# ── CONDITION VOCABULARY ──────────────────────────────────────────────────────
# The ONLY variables allowed in a rule condition. Every one is directly derivable
# from a Grid2Op observation / dataset record (rho, v_or, line_status) — conditions
# written against this vocabulary are machine-evaluable by the symbolic shield.
# Single source of truth: rendered into both prompts AND enforced by the AST linter.
CONDITION_VOCABULARY = {
    # ── voltage / loading / topology (the original six) ────────────────────────
    "voltage_pu_min":  "lowest per-unit voltage across all energized lines (1.0 = nominal)",
    "voltage_pu_max":  "highest per-unit voltage across all energized lines (1.0 = nominal)",
    "loading_pct":     "maximum line loading as percent of thermal limit (100 = at limit)",
    "rho_max":         "maximum line loading ratio (1.0 = at thermal limit)",
    "n_tripped_lines": "number of disconnected transmission lines (integer, 0 = all in service)",
    "any_line_tripped": "boolean, True if at least one line is disconnected",
    # ── power flow (from p_or / q_or, present in every record) ─────────────────
    "active_power_mw_max":      "largest active power flow on any energized line, MW",
    "reactive_power_mvar_max":  "largest reactive power flow on any energized line, MVAr",
    "apparent_power_mva_max":   "largest apparent power flow on any energized line, MVA",
    "power_factor_at_max_load": "power factor on the most heavily loaded line (1.0 = unity)",
    "current_a_max":            "largest line current, amperes",
    # ── system balance (from gen_p / load_p) ───────────────────────────────────
    "total_generation_mw":      "total active generation across all generators, MW",
    "total_load_mw":            "total active demand across all loads, MW",
    "generation_load_imbalance_pct":
        "(generation - load) / load * 100; positive means surplus generation",
}

def _vocabulary_block() -> str:
    return "\n".join(f"- {name} : {desc}" for name, desc in CONDITION_VOCABULARY.items())


# ── PROMPTS ───────────────────────────────────────────────────────────────────
EXTRACT_PROMPT = """/no_think
You are a power systems engineer extracting operational constraints, thresholds, and
technical requirements from grid codes, network codes, and technical standards.
From the text below, extract all constraints, limits, and requirements as a JSON array.

Each rule MUST have exactly these keys:
- rule_id   : string, format "R_001" (sequential, unique within this response)
- source    : string, e.g. "ENTSO-E NC RfG, Article 10(2)(a), Table 2"
- entity    : string, one of: Bus, Line, Transformer, Generator, Load, ProtectionDevice, Grid, PowerGeneratingModule, Module, Facility, PowerGeneratingFacility, PowerSystemFacility, OffshorePowerParkModule, ParkModule
- condition : string, a Python boolean expression (see CONDITION RULES below)
- action    : string, one of: BLOCK, DISCONNECT, ALERT, REDISPATCH, RECONNECT, RESPOND_WITHIN_2S, SHED_LOAD, OTHER
- severity  : string, one of: critical, high, medium, low
- explanation : string, plain-English reason for the rule (one sentence)

CONDITION RULES (strict):
1. The condition must be a valid Python boolean expression. Allowed syntax:
   variable names, numeric literals, comparison operators (< <= > >= == !=),
   the keywords and / or / not, and parentheses. Nothing else.
   FORBIDDEN: BETWEEN, WITHIN, FOR, function calls, units inside the expression, prose.
2. Use variable names that naturally describe the engineering quantities in the
   standard — e.g. frequency_hz, voltage_pu, time_seconds, power_factor, loading_pct.
   There is no pre-defined variable list.
3. VIOLATION POLARITY: the condition must describe the UNSAFE / VIOLATING state —
   it must evaluate TRUE when the rule is violated. If the source text states a
   required or normal operating range, INVERT it.
   Example: "voltage shall remain within 0.95-1.05 pu"
   -> condition: "voltage_pu_min < 0.95 or voltage_pu_max > 1.05"
4. Extract every numeric limit, range, or threshold the text states. Be exhaustive — do not
   skip a constraint because it seems minor, or because it is hard to express.
   Return an empty array [] only when the text states no constraint at all: front-matter
   (title pages, tables of contents, address blocks), or purely administrative clauses about
   record-keeping, notification, or compliance reporting.
   Do not emit a condition whose bounds overlap (e.g. "x < 49.8 or x > 49.5") — that is
   always true. Inverting a range [a, b] gives "x < a or x > b" with a <= b.

Other rules:
- Output ONLY a valid JSON array. No preamble, no markdown, no explanation.

Text:
{chunk}
"""

VALIDATE_PROMPT = """You are a power systems safety auditor verifying extracted rules against source text.

For each rule below, verify:
1. Is this constraint actually stated in the source text?
2. Is the condition boundary (threshold value) correctly parsed?
3. Is the entity type correct?
4. Does the condition use ONLY these variables, with Python syntax
   (comparisons, and/or/not, parentheses — no BETWEEN, no units, no prose)?
{vocabulary}
   If the condition uses any other variable or cannot be expressed with these, REJECT.
5. ROLE CORRECTNESS. Each rule carries a `role`:
   - CONSTRAINT  : condition TRUE means the standard is VIOLATED.
   - AFFIRMATION : condition TRUE means the state is consistent with the class named
                   in `affirms` (one of: normal, overload, line_trip, cascade).
   Check the role matches what the source text asserts. "Voltage shall remain within
   0.95-1.05 pu" is an AFFIRMATION of `normal`, NOT a constraint to be inverted.
   "Loading shall not exceed the thermal rating" is a CONSTRAINT. If the role is
   wrong, return CORRECT with the right `role` (and `affirms`) in corrected_fields.
   Do NOT rewrite an affirmation into a violation predicate — both directions of
   evidence are wanted.
6. RIDE-THROUGH RANGES ARE NEITHER ROLE. If the source describes a
   fault-ride-through band, a "no trip zone", or any abnormal-but-must-survive
   range (e.g. "shall remain connected between 0.7 and 0.9 pu for 3 s"), the
   requirement is about *not tripping* during the excursion, which needs protection
   status and duration — neither is observable here. As a CONSTRAINT it fires at
   nominal; as an AFFIRMATION of `normal` it is false. REJECT such rules. A
   tell-tale sign is an upper bound below 0.95 or a lower bound above 1.0, e.g.
   "voltage_pu_min < 0.7 or voltage_pu_max > 0.9".
7. HEALTHY-GRID SELF-CHECK: substitute voltage_pu_min = 1.0, voltage_pu_max = 1.0,
   loading_pct = 40, rho_max = 0.4, n_tripped_lines = 0, any_line_tripped = False,
   power_factor_at_max_load = 0.93.
   - a CONSTRAINT that is TRUE for those values is wrong (a healthy grid violates
     nothing). REJECT, or CORRECT if the intended threshold is unambiguous.
   - an AFFIRMATION of `normal` that is FALSE for those values is wrong. Same remedy.

Output ONLY a JSON array. Each item must have:
- rule_id  : matching the input rule
- verdict  : one of "CONFIRM", "REJECT", "CORRECT"
- corrected_fields : object with corrected key-value pairs (only if verdict is "CORRECT", else omit)
- reason   : one sentence explaining the verdict

No preamble, no markdown.

Source text:
{chunk}

Extracted rules:
{rules}
"""

TRANSLATE_PROMPT = """You are a power systems engineer translating operational safety rules into a machine-evaluable format.

Each rule below was extracted from a grid code standard. Its "condition" field describes when the rule is violated (the UNSAFE state).

Your task: translate the condition to use ONLY these Grid2Op-observable variables:
{vocabulary}

For each rule, decide:
- If the condition CAN be expressed using the variables above:
  Return {{"translatable": true, "condition": "translated condition",
           "role": "CONSTRAINT" or "AFFIRMATION", "affirms": <class or null>, "reason": null}}
  The translated condition must be a valid Python boolean expression using ONLY:
  comparisons (< <= > >= == !=), and/or/not, parentheses, numeric literals, and the variables listed above.

  ROLE — decide which of these the rule is. Do NOT force everything into one shape.

  (a) CONSTRAINT — the condition describes a state that VIOLATES the standard.
      TRUE means something is wrong and the prediction should be blocked.
      "line loading shall not exceed its thermal rating"
      -> {{"role": "CONSTRAINT", "condition": "loading_pct > 100", "affirms": null}}

  (b) AFFIRMATION — the condition describes a state that CONFIRMS the grid is in a
      particular operating class. TRUE means the telemetry is consistent with that
      class. Set "affirms" to exactly one of: normal, overload, line_trip, cascade.
      "voltage shall remain within 0.95-1.05 pu" describes healthy operation:
      -> {{"role": "AFFIRMATION", "affirms": "normal",
           "condition": "voltage_pu_min >= 0.95 and voltage_pu_max <= 1.05"}}
      Do NOT invert this into a violation predicate. Both directions of evidence
      are wanted: what proves something is wrong, AND what proves it is fine.

  Choose by what the source text asserts, not by which shape is easier. Grid codes
  are written predominantly as "shall remain within" / "shall be capable of", which
  is AFFIRMATION-shaped. Expect many of them.

  RIDE-THROUGH RANGES ARE NEITHER. "the module shall remain connected for voltages
  between 0.7 and 0.9 pu for 3 seconds" and "shall not trip within the no trip zone"
  describe an abnormal-but-survivable band. The requirement is about *not tripping*
  during it, which needs protection status and duration — neither is observable here.
  As a CONSTRAINT it fires at nominal; as an AFFIRMATION of `normal` it is false.
  -> return {{"translatable": false, "condition": null,
             "reason": "ride-through: requires trip status and duration"}}

  TRANSLATION CASES. These are the transformations you ARE expected to perform:

  1. SYNONYM. The source variable names an observable quantity under a different
     name. Map it.
     "current_amps > 1200"        -> "current_a_max > 1200"
     "line_loading_percent > 95"  -> "loading_pct > 95"

  2. UNIT CONVERSION. Convert to the units the vocabulary uses.
     "voltage_kv < 130" on a 138 kV line   -> "voltage_pu_min < 0.94"
     "flow > 0.5 GW"                       -> "active_power_mw_max > 500"
     Only convert when the conversion is unambiguous from the source text. If the
     rule states an absolute kV threshold with no stated nominal, you cannot
     convert it to per-unit - mark it untranslatable.

  3. AGGREGATION BINDING. The standard speaks of "the voltage" or "the loading" as
     a scalar; the grid has many lines. Bind it to the right aggregate:
     an UNDER-limit becomes the minimum, an OVER-limit becomes the maximum.
     "voltage shall not fall below 0.9"  -> "voltage_pu_min < 0.9"
     "voltage shall not exceed 1.1"      -> "voltage_pu_max > 1.1"
     Never bind an under-limit to the maximum, or vice versa.

  4. PROXY SUBSTITUTION. A quantity expressed against an equipment rating is the
     loading ratio in disguise.
     "flow shall not exceed 100% of the continuous rating" -> "loading_pct > 100"
     "loaded beyond its emergency rating"                  -> "rho_max > 1.0"

  5. SCOPE vs PREDICATE. Many clauses combine WHICH equipment the rule covers with
     WHAT it requires. The scope half is NOT part of the condition.
     "Modules above 50 MW shall keep voltage within 0.95-1.05"
     The "above 50 MW" selects which modules are governed; it is not a violation.
     Translate ONLY the requirement:
     -> {{"role": "AFFIRMATION", "affirms": "normal",
          "condition": "voltage_pu_min >= 0.95 and voltage_pu_max <= 1.05"}}
     Never emit "active_power_mw_max > 50 and ..." - that would make the rule fire
     on the basis of equipment size rather than grid condition.

  COMPOUND CONDITIONS ARE DISCARDED, NOT TRIMMED. If a condition joins a measurable
  clause to an unmeasurable one, you must NOT keep the measurable half.
  "voltage_pu_min < 0.9 and time_seconds > 3" must NOT become "voltage_pu_min < 0.9":
  the standard tolerates a brief dip, so the trimmed rule would flag states the
  standard permits, while still citing that standard as its source.
  -> return {{"translatable": false, "condition": null,
             "reason": "compound: requires time_seconds, not observable in Grid2Op"}}
  Name the unmeasurable variable in the reason.

  SELF-CHECK before returning. Substitute a healthy grid: voltage_pu_min = 1.0,
  voltage_pu_max = 1.0, loading_pct = 40, rho_max = 0.4, n_tripped_lines = 0,
  any_line_tripped = False, power_factor_at_max_load = 0.93.
  - a CONSTRAINT must be FALSE for those values (a healthy grid violates nothing)
  - an AFFIRMATION of `normal` must be TRUE for those values
  If yours comes out the other way, fix it or mark it untranslatable.

- If the condition CANNOT be expressed (uses frequency, ROCOF, droop, protection relay
  settings, sub-second timing, or any variable not in the vocabulary):
  Return {{"translatable": false, "condition": null, "reason": "category: detail"}}
  Note that Grid2Op simulates NO frequency at all, so any frequency-dependent rule is
  untranslatable regardless of how it is worded.

Output ONLY a JSON array of translation results. No preamble, no markdown.

Source text:
{chunk}

Rules to translate:
{rules}
"""

# Pre-render the vocabulary so extract.py / validate.py / translate.py keep calling
# .format(chunk=...) / .format(chunk=..., rules=...) unchanged.
EXTRACT_PROMPT   = EXTRACT_PROMPT.replace("{vocabulary}", _vocabulary_block())
VALIDATE_PROMPT  = VALIDATE_PROMPT.replace("{vocabulary}", _vocabulary_block())
TRANSLATE_PROMPT = TRANSLATE_PROMPT.replace("{vocabulary}", _vocabulary_block())

# ── LOGGING ───────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# ── OLLAMA GENERATION ─────────────────────────────────────────────────────────
# Budgets: reasoning traces can eat the whole num_predict before any JSON is
# emitted. 4096/8192 leaves headroom for anything that slips past think=False.
GEN_NUM_PREDICT = 6144   # rule-dense chunks hit the 4096 ceiling mid-array (truncated JSON)
GEN_NUM_CTX     = 8192   # prompt+chunk is ~1.3k tokens, so this leaves predict full headroom


def generate_no_think(
    model: str,
    prompt: str,
    *,
    num_predict: int = GEN_NUM_PREDICT,
    num_ctx: int = GEN_NUM_CTX,
    keep_alive: int = -1,
) -> str:
    """ollama.generate with thinking suppressed at the API level.

    The `/no_think` prompt prefix is a Qwen3-specific convention that newer
    builds do not necessarily honour — `think=False` is the reliable switch
    (validate.py already uses it via ollama.chat). Falls back for older
    clients and models that reject the kwarg.
    """
    kwargs = dict(
        model=model,
        prompt=prompt,
        options=ollama.Options(temperature=0.0, num_predict=num_predict, num_ctx=num_ctx),
        keep_alive=keep_alive,
        stream=False,
    )
    try:
        return ollama.generate(**kwargs, think=False)["response"]
    except TypeError:
        # ollama client predates the `think` kwarg
        return ollama.generate(**kwargs)["response"]
    except ollama.ResponseError:
        # model does not support thinking toggles
        return ollama.generate(**kwargs)["response"]


# ── CONDITION LINTER ──────────────────────────────────────────────────────────
# Deterministic, LLM-independent guarantee that every rule reaching the KG is
# machine-evaluable: valid Python boolean expression over CONDITION_VOCABULARY only.

_ALLOWED_AST_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.USub,          # USub: negative numeric literals
    ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.Name, ast.Load, ast.Constant,
)

# Dummy namespace for the smoke-eval — values are arbitrary but type-correct.
_SMOKE_NAMESPACE = {
    "voltage_pu_min": 1.0, "voltage_pu_max": 1.0,
    "loading_pct": 50.0, "rho_max": 0.5,
    "n_tripped_lines": 0, "any_line_tripped": False,
    "active_power_mw_max": 80.0, "reactive_power_mvar_max": 20.0,
    "apparent_power_mva_max": 82.5, "power_factor_at_max_load": 0.93,
    "current_a_max": 320.0,
    "total_generation_mw": 250.0, "total_load_mw": 248.0,
    "generation_load_imbalance_pct": 0.8,
}


def lint_condition(condition: str) -> str:
    """
    Validate that `condition` is a machine-evaluable boolean expression over
    CONDITION_VOCABULARY. Returns the (stripped) condition, or raises ValueError.
    """
    condition = condition.strip()
    if not condition:
        raise ValueError("Empty condition")

    try:
        tree = ast.parse(condition, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Condition is not valid Python: {condition!r} ({exc.msg})")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(
                f"Disallowed syntax {type(node).__name__} in condition: {condition!r}"
            )
        if isinstance(node, ast.Name) and node.id not in CONDITION_VOCABULARY:
            raise ValueError(
                f"Unknown variable {node.id!r} in condition: {condition!r} "
                f"(allowed: {', '.join(CONDITION_VOCABULARY)})"
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(
                f"Non-numeric literal {node.value!r} in condition: {condition!r}"
            )

    # Smoke-eval: catches anything the AST walk missed at runtime.
    try:
        result = eval(condition, {"__builtins__": {}}, dict(_SMOKE_NAMESPACE))
    except Exception as exc:
        raise ValueError(f"Condition failed smoke evaluation: {condition!r} ({exc})")
    if not isinstance(result, bool):
        raise ValueError(f"Condition does not evaluate to a boolean: {condition!r}")

    return condition


def lint_condition_raw(condition: str) -> str:
    """Validate that condition is valid Python syntax, without checking vocabulary.
    Used by RawRule during Stage 1 extraction — accepts any variable name.
    """
    condition = condition.strip()
    if not condition:
        raise ValueError("Empty condition")
    try:
        tree = ast.parse(condition, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Condition is not valid Python: {condition!r} ({exc.msg})")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(
                f"Disallowed syntax {type(node).__name__} in condition: {condition!r}"
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(
                f"Non-numeric literal {node.value!r} in condition: {condition!r}"
            )
    return condition


# ── PYDANTIC SCHEMAS ──────────────────────────────────────────────────────────
class Rule(BaseModel):
    rule_id: str
    source: str
    entity: str
    condition: str
    action: str
    severity: str
    explanation: str
    # Defaults keep every pre-role artifact loadable: an unlabelled rule is a
    # CONSTRAINT, which is what the v1/v2 prompts forced it to be.
    role: str = "CONSTRAINT"
    affirms: Optional[str] = None

    @field_validator("role")
    @classmethod
    def check_role(cls, v):
        if v.upper() not in ROLE_VALUES:
            raise ValueError(f"Invalid role: {v} (expected one of {sorted(ROLE_VALUES)})")
        return v.upper()

    @field_validator("affirms")
    @classmethod
    def check_affirms(cls, v):
        if v is None:
            return v
        if v.lower() not in FAULT_CLASSES:
            raise ValueError(f"Invalid affirms class: {v} (expected one of {sorted(FAULT_CLASSES)})")
        return v.lower()

    @model_validator(mode="after")
    def check_role_class_pairing(self):
        """An affirmation with no class is meaningless — 'voltage in band' supports
        `normal` and refutes `cascade`, so the class is what makes it evaluable."""
        if self.role == "AFFIRMATION" and not self.affirms:
            raise ValueError("An AFFIRMATION rule must name the class it affirms")
        if self.role == "CONSTRAINT" and self.affirms:
            raise ValueError("A CONSTRAINT rule must not set `affirms`")
        return self

    @field_validator("entity")
    @classmethod
    def check_entity(cls, v):
        if v.lower() not in ENTITY_VALUES:
            raise ValueError(f"Invalid entity: {v}")
        return v

    @field_validator("severity")
    @classmethod
    def check_severity(cls, v):
        if v.lower() not in SEVERITY_VALUES:
            raise ValueError(f"Invalid severity: {v}")
        return v.lower()

    @field_validator("action")
    @classmethod
    def check_action(cls, v):
        if v.upper() not in ACTION_VALUES:
            return "OTHER"
        return v.upper()

    @field_validator("condition")
    @classmethod
    def check_condition(cls, v):
        return lint_condition(v)


class RawRule(BaseModel):
    """Rule with syntax-only condition validation (no vocabulary check).
    Used during Stage 1 extraction before conditions are translated to Grid2Op vars.
    """
    rule_id: str
    source: str
    entity: str
    condition: str
    action: str
    severity: str
    explanation: str

    @field_validator("entity")
    @classmethod
    def check_entity(cls, v):
        if v.lower() not in ENTITY_VALUES:
            raise ValueError(f"Invalid entity: {v}")
        return v

    @field_validator("severity")
    @classmethod
    def check_severity(cls, v):
        if v.lower() not in SEVERITY_VALUES:
            raise ValueError(f"Invalid severity: {v}")
        return v.lower()

    @field_validator("action")
    @classmethod
    def check_action(cls, v):
        if v.upper() not in ACTION_VALUES:
            return "OTHER"
        return v.upper()

    @field_validator("condition")
    @classmethod
    def check_condition(cls, v):
        return lint_condition_raw(v)


class Verdict(BaseModel):
    rule_id: str
    verdict: str   # CONFIRM | REJECT | CORRECT
    corrected_fields: Optional[dict] = None
    reason: Optional[str] = None

    @field_validator("verdict")
    @classmethod
    def check_verdict(cls, v):
        v = v.upper()
        if v not in {"CONFIRM", "REJECT", "CORRECT"}:
            raise ValueError(f"Invalid verdict: {v}")
        return v


class TranslationResult(BaseModel):
    rule_id: str
    translatable: bool
    condition: Optional[str] = None
    reason: Optional[str] = None
    role: Optional[str] = None      # CONSTRAINT | AFFIRMATION (defaults applied by Rule)
    affirms: Optional[str] = None   # required when role == AFFIRMATION


# ── PDF + CHUNKING ────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i + 1}]\n{text.strip()}")
    return "\n\n".join(pages)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            search_from = start + int(size * 0.8)
            para_break = text.rfind("\n\n", search_from, end)
            if para_break > search_from:
                end = para_break
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
    return [c for c in chunks if c]


# ── JSON PARSING ──────────────────────────────────────────────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove reasoning traces before JSON parsing.

    Thinking models emit <think>...</think>; if the block survives, the JSON
    slicer in extract_json_array() latches onto a '[' inside the reasoning.
    """
    text = _THINK_RE.sub("", text)
    # Unclosed <think>: generation was cut off mid-reasoning (num_predict
    # exhausted) — nothing from the tag onward is usable output.
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def extract_json_array(text: str) -> list:
    text = _strip_think(text)
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON array found. Raw (first 300 chars):\n{text[:300]}")
    return json.loads(text[start:end + 1])


# ── RULE ID REWRITING ─────────────────────────────────────────────────────────
def rewrite_rule_ids(rules: list[dict], base: int) -> list[dict]:
    for i, r in enumerate(rules):
        r["rule_id"] = f"R_{base + i:03d}"
    return rules


# ── DEDUPLICATION (called by validate.py at the end) ──────────────────────────
def deduplicate_rules(rules_dir: Path, log) -> int:
    """
    Merges all *_confirmed.jsonl files → all_rules_deduped.jsonl.
    Drops rules with identical (entity, condition); merges sources.
    """
    seen = {}
    dropped = 0

    for jf in sorted(rules_dir.glob("*_confirmed.jsonl")):
        with jf.open() as f:
            for line in f:
                try:
                    rule = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (rule.get("entity", ""), rule.get("condition", ""))
                if key not in seen:
                    seen[key] = rule
                else:
                    existing_src = seen[key]["source"]
                    new_src = rule["source"]
                    if new_src not in existing_src:
                        seen[key]["source"] = f"{existing_src}; {new_src}"
                    dropped += 1

    if dropped:
        log.info(f"  [Dedup] Dropped {dropped} duplicate rules.")

    out_path = rules_dir / "all_rules_deduped.jsonl"
    with out_path.open("w") as f:
        for rule in seen.values():
            f.write(json.dumps(rule) + "\n")

    return len(seen)