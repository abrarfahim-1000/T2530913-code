r"""
replicate_validation.py — is the served four-rule corpus REPRODUCIBLE?

╔══════════════════════════════════════════════════════════════════════════════╗
║  THIS SCRIPT HAS NOT BEEN RUN. It requires CUDA + Ollama and was written on   ║
║  a machine with neither. Every number it would produce is unmeasured. Do not  ║
║  cite anything from it until it has executed on the LLM host.                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHY
---
`thesis_findings.md` §12.3's central claim is:

    "The verdicts were stable across runs while the reasoning was fabricated,
     and only persisting the reasons exposed it."

The stability half of that sentence is established for the **`strict`** arm —
run 1 and the `strict` A/B arm returned the same 1 confirmed / 31 rejected with
the same survivor. It is NOT established for the **`translated`** arm, and the
`translated` arm is the one that produced `validated_translated/` — the served
four-rule corpus every shield number in §13 rests on. That arm ran once, at one
seed, at temperature 0.0.

So the load-bearing half of a load-bearing claim is unmeasured. This script
measures it.

WHAT IT MEASURES, AND THE DISTINCTION THAT IS THE WHOLE POINT
------------------------------------------------------------
**VERDICT stability** and **REASON stability** are reported separately and are
never combined into one "agreement" number. §12.3 is precisely the case where
they diverge: stable verdicts sitting on top of fabricated, unstable reasoning.
A single blended score would have hidden exactly the finding that cost a round
trip to discover. So:

  * verdict agreement — per (document, rule_id), do all N runs agree on
    CONFIRM / CORRECT / REJECT / FLAG?
  * reason agreement  — for the rules where a reason is persisted, is the text
    identical across runs? Reported exact and whitespace-normalised.
  * corpus reproduction — does each run's `all_rules_deduped.jsonl` reproduce
    the served corpus exactly, by (entity, condition, role)?

⚠️ **A KNOWN ASYMMETRY IN WHAT CAN BE MEASURED.** `validate.py` writes bare
rules to `*_confirmed.jsonl` — no verdict object, no `reason`. Reasons are
persisted only for REJECT and FLAG. So reason stability is measurable for the
21 rejections and NOT for the 11 confirmations. The script says so in its
output rather than quietly reporting reason agreement over a subset as though
it covered everything. CONFIRM vs CORRECT is *inferred* by diffing the emitted
rule against its guarded input, and is labelled `inferred_outcome`.

╭─ HOW TO RUN IT ON THE LLM HOST ──────────────────────────────────────────────╮
│                                                                              │
│  Prerequisites: CUDA + Ollama, with `nemotron-3-nano:30b` pulled.            │
│                                                                              │
│    ollama pull nemotron-3-nano:30b                                           │
│    ollama list                    # confirm the tag                          │
│                                                                              │
│  Then, from the repo root:                                                   │
│                                                                              │
│    python evaluation/replicate_validation.py --repeats 3                     │
│                                                                              │
│  Compare runs produced earlier without re-running the model:                 │
│                                                                              │
│    python evaluation/replicate_validation.py --compare-only \                │
│        --runs results/replication/runs/run_1 \                               │
│               results/replication/runs/run_2 \                               │
│               results/replication/runs/run_3                                 │
│                                                                              │
│  EXPECTED RUNTIME. The recorded 2026-08-20 `translated` run took             │
│  **48.9 s** end to end over 32 rules in 12 files                             │
│  (`validated_translated/validation_run_summary.json`). Three repeats is      │
│  therefore ~2–3 minutes of inference, plus a one-off model load on the       │
│  first call. Budget 10 minutes cold.                                         │
│                                                                              │
│  VRAM. The validator is the LARGER of the pipeline's two models and relies   │
│  on partial CPU offload. Rules that are not negotiable:                      │
│    * NEVER have Qwen3 and Nemotron resident at once. Qwen3 runs to           │
│      completion first; this script touches only the validator.               │
│    * Every Ollama call in the pipeline passes `keep_alive=0`, so VRAM is     │
│      released between stages. This script inherits that from validate.py     │
│      and does not override it.                                               │
│    * Repeats run SEQUENTIALLY on purpose. Two concurrent validator           │
│      processes would contend for the same offloaded weights, and any         │
│      timing or stability result from that is an artifact of the contention.  │
│                                                                              │
│  MODEL TAG. ⚠️ CLAUDE.md says "Set VALIDATOR_MODEL if your Ollama tag        │
│  differs". That instruction is currently WRONG: `extraction/common.py`       │
│  line 30 pins `VALIDATOR_MODEL = "nemotron-3-nano:30b"` as a literal and     │
│  reads no environment variable (`EXTRACTOR_MODEL` does; the validator does   │
│  not). This script reads `$VALIDATOR_MODEL` for its preflight and REFUSES    │
│  TO RUN if it disagrees with the pinned constant, because running the wrong  │
│  model while believing the override took effect is worse than not running.   │
│  To change the tag, edit that one line in `extraction/common.py`.            │
╰──────────────────────────────────────────────────────────────────────────────╯

THE TRAP THIS SCRIPT EXISTS TO AVOID
------------------------------------
`deduplicate_rules()` merges **every** `*_confirmed.jsonl` it finds in the out
directory. Point two repeats at one directory and run 2 silently absorbs run 1:
`all_rules_deduped.jsonl` reconciles, no error is raised, and the "replication"
is a blend. Every repeat therefore gets its own directory, and
`assert_isolated_output_dirs()` refuses duplicates, nesting, and any directory
that already holds a `*_confirmed.jsonl` — before a single token is generated.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

DEFAULT_CANDIDATES = REPO / "translated_rules" / "guarded"
DEFAULT_ROOT = REPO / "results" / "replication"
REFERENCE_CORPUS = REPO / "validated_translated" / "all_rules_deduped.jsonl"

# The recorded 2026-08-20 `translated` run, for the comparator to reconcile
# against. Source: validated_translated/validation_run_summary.json.
RECORDED_RUN = {
    "confirmed": 11,
    "rejected": 21,
    "flagged": 0,
    "distinct_after_dedup": 4,
    "reject_rate": 0.65625,
    "total_time_sec": 48.9,
    "validator_model": "nemotron-3-nano:30b",
    "prompt_variant": "translated",
}

OUTCOMES = ("CONFIRM_OR_CORRECT", "REJECT", "FLAG", "ABSENT")


class PreflightError(RuntimeError):
    """Raised before any inference. A partial run must never look like a finding."""


# ---------------------------------------------------------------------------
# Preflight — fail loudly, fail early, fail before spending a single token
# ---------------------------------------------------------------------------

def check_candidates(candidates_dir: Path) -> List[Path]:
    files = sorted(candidates_dir.glob("*_translated.jsonl"))
    if not files:
        raise PreflightError(
            f"no *_translated.jsonl in {candidates_dir}. The stage-2.5 guarded corpus "
            f"is the input; run extraction/polarity_guard.py first."
        )
    n = sum(1 for f in files for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip())
    if n == 0:
        raise PreflightError(f"{candidates_dir} holds {len(files)} files but 0 rules")
    return files


def check_model_tag() -> str:
    """Resolve the validator tag, and refuse a silently-ignored override.

    `extraction/common.py` pins the tag as a literal. If someone sets
    $VALIDATOR_MODEL expecting it to take effect — which CLAUDE.md tells them
    to do — the run would quietly use the pinned model instead. A replication
    study that validated with a different model than the operator believed is
    worse than no replication study.
    """
    from extraction.common import VALIDATOR_MODEL  # noqa: PLC0415

    override = os.environ.get("VALIDATOR_MODEL", "").strip()
    if override and override != VALIDATOR_MODEL:
        raise PreflightError(
            f"$VALIDATOR_MODEL is '{override}' but extraction/common.py pins "
            f"VALIDATOR_MODEL = '{VALIDATOR_MODEL}' as a literal and reads no environment "
            f"variable. The override would be silently ignored. Edit that line in "
            f"extraction/common.py, or unset $VALIDATOR_MODEL to run with the pinned tag."
        )
    return VALIDATOR_MODEL


def check_ollama(model_tag: str) -> dict:
    """Daemon reachable AND the tag present. Both, before anything runs."""
    try:
        import ollama  # noqa: PLC0415
    except ImportError as exc:
        raise PreflightError(
            "the `ollama` package is not installed — pip install -r requirements.txt"
        ) from exc

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        listing = ollama.list()
    except Exception as exc:
        raise PreflightError(
            f"Ollama is not reachable at {host} ({type(exc).__name__}: {exc}). "
            f"Start it with `ollama serve`. This script must run on the LLM host "
            f"(CUDA + Ollama); it cannot run on the workstation."
        ) from exc

    models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
    tags = []
    for m in models:
        tag = m.get("model") or m.get("name") if isinstance(m, dict) else (
            getattr(m, "model", None) or getattr(m, "name", None))
        if tag:
            tags.append(tag)

    if not any(t == model_tag or t.split(":")[0] == model_tag.split(":")[0] for t in tags):
        raise PreflightError(
            f"model '{model_tag}' is not pulled on this host. Available: "
            f"{', '.join(sorted(tags)) or '(none)'}. Run: ollama pull {model_tag}"
        )
    return {"host": host, "available_tags": sorted(tags), "resolved": model_tag}


def assert_isolated_output_dirs(dirs: Sequence[Path]) -> None:
    """Refuse anything that could let two repeats blend.

    `deduplicate_rules()` merges EVERY `*_confirmed.jsonl` in the output
    directory. Two repeats sharing one directory produce a merged
    `all_rules_deduped.jsonl` that reconciles perfectly and is a blend of both
    — no error, no warning, and a replication result that is not one. All three
    of these are that failure wearing a different hat.
    """
    resolved = [Path(d).resolve() for d in dirs]

    dupes = [str(p) for p, c in Counter(resolved).items() if c > 1]
    if dupes:
        raise PreflightError(
            f"repeats share an output directory ({', '.join(dupes)}). "
            f"deduplicate_rules() would merge them into one corpus."
        )

    for i, a in enumerate(resolved):
        for j, b in enumerate(resolved):
            if i != j and (a == b or b in a.parents):
                raise PreflightError(
                    f"output directory {a} is nested inside {b}; a glob from the outer "
                    f"directory would pick up the inner run's files."
                )

    for d in resolved:
        existing = list(d.glob("*_confirmed.jsonl")) if d.exists() else []
        if existing:
            raise PreflightError(
                f"{d} already contains {len(existing)} *_confirmed.jsonl file(s) "
                f"({existing[0].name}, ...). deduplicate_rules() would merge them into "
                f"this repeat's corpus. Use --force to clear, or pick an empty root."
            )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_one(candidates_dir: Path, out_dir: Path, python: str = sys.executable) -> dict:
    """One `validate.py --prompt-variant translated` into its own directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python, str(REPO / "extraction" / "validate.py"),
        "--candidates", str(candidates_dir),
        "--out", str(out_dir),
        "--prompt-variant", "translated",
    ]
    started = datetime.now(timezone.utc)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    # stdin is closed deliberately. validate.py has a `input("Continue anyway?")`
    # prompt on its stale-output guard. assert_isolated_output_dirs() guarantees
    # a fresh directory so the prompt cannot trigger — and if that guarantee ever
    # breaks, this turns an invisible hang into an immediate EOFError.
    proc = subprocess.run(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True)
    finished = datetime.now(timezone.utc)

    if proc.returncode != 0:
        raise RuntimeError(
            f"validate.py exited {proc.returncode} for {out_dir.name}.\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )
    return {
        "out_dir": str(out_dir),
        "returncode": proc.returncode,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "seconds": (finished - started).total_seconds(),
        "stdout_tail": proc.stdout[-2000:],
    }


# ---------------------------------------------------------------------------
# Loading a completed run
# ---------------------------------------------------------------------------

@dataclass
class RuleVerdict:
    document: str
    rule_id: str
    outcome: str                       # one of OUTCOMES
    reason: Optional[str]              # persisted for REJECT/FLAG only
    condition: Optional[str]
    role: Optional[str]
    # CONFIRM vs CORRECT is not written per-rule; it is inferred by diffing the
    # emitted rule against its guarded input. Labelled, never asserted.
    inferred_outcome: Optional[str] = None
    changed_fields: List[str] = field(default_factory=list)


def _stem(path: Path, suffix: str) -> str:
    return path.name[: -len(suffix)]


def load_guarded_inputs(candidates_dir: Path) -> Dict[Tuple[str, str], dict]:
    """The stage-2.5 input, keyed on (document, rule_id) — rule_id is not unique."""
    out: Dict[Tuple[str, str], dict] = {}
    for f in sorted(candidates_dir.glob("*_translated.jsonl")):
        doc = _stem(f, "_translated.jsonl")
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[(doc, rec["rule"]["rule_id"])] = rec["rule"]
    return out


def load_run(run_dir: Path,
             guarded: Optional[Dict[Tuple[str, str], dict]] = None
             ) -> Dict[Tuple[str, str], RuleVerdict]:
    """Every verdict a run produced, keyed on (document, rule_id).

    Keyed on the composite because `rule_id` is NOT globally unique — five
    source documents restarted numbering (`thesis_findings.md` §19.2).
    """
    run_dir = Path(run_dir)
    out: Dict[Tuple[str, str], RuleVerdict] = {}

    for f in sorted(run_dir.glob("*_confirmed.jsonl")):
        doc = _stem(f, "_confirmed.jsonl")
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rule = json.loads(line)
            key = (doc, rule["rule_id"])
            changed: List[str] = []
            inferred = None
            if guarded is not None and key in guarded:
                src = guarded[key]
                changed = sorted(k for k in set(src) | set(rule)
                                 if src.get(k) != rule.get(k))
                inferred = "CORRECT" if changed else "CONFIRM"
            out[key] = RuleVerdict(doc, rule["rule_id"], "CONFIRM_OR_CORRECT", None,
                                   rule.get("condition"), rule.get("role"),
                                   inferred, changed)

    for suffix, outcome in (("_rejected.jsonl", "REJECT"), ("_flagged.jsonl", "FLAG")):
        for f in sorted(run_dir.glob(f"*{suffix}")):
            doc = _stem(f, suffix)
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                rule, verdict = rec["rule"], rec.get("verdict") or {}
                out[(doc, rule["rule_id"])] = RuleVerdict(
                    doc, rule["rule_id"], outcome, verdict.get("reason"),
                    rule.get("condition"), rule.get("role"))
    return out


def load_corpus(path: Path) -> List[Tuple[str, str, str]]:
    """A deduped corpus as a comparable set of (entity, condition, role) triples.

    Compared on the predicate, not on `rule_id`: dedup merges sources, so which
    id survives a merge is an ordering artifact and not a property of the
    corpus.
    """
    path = Path(path)
    if not path.exists():
        return []
    triples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            triples.append((r.get("entity"), r.get("condition"), r.get("role")))
    return sorted(triples)


# ---------------------------------------------------------------------------
# Comparison — VERDICT stability and REASON stability, never merged
# ---------------------------------------------------------------------------

def _normalise(text: Optional[str]) -> Optional[str]:
    return " ".join(text.split()).lower() if text else text


def compare_runs(runs: Sequence[Dict[Tuple[str, str], RuleVerdict]],
                 labels: Sequence[str]) -> dict:
    """Per-rule verdict and reason agreement across N runs.

    The two are reported side by side and never combined. §12.3's finding IS
    the divergence between them; a blended score erases it.
    """
    if len(runs) < 2:
        raise ValueError("need at least 2 runs to compare")

    keys = sorted(set().union(*(set(r) for r in runs)))
    per_rule = []
    verdict_stable = reason_stable = reason_measurable = 0

    for key in keys:
        outcomes = [runs[i].get(key).outcome if key in runs[i] else "ABSENT"
                    for i in range(len(runs))]
        v_stable = len(set(outcomes)) == 1

        reasons = [runs[i].get(key).reason if key in runs[i] else None
                   for i in range(len(runs))]
        # A reason is only persisted for REJECT/FLAG. If any run did not
        # persist one, reason stability is UNMEASURABLE for that rule — which
        # is a different statement from "the reasons matched", and is kept
        # distinct on purpose.
        measurable = all(r is not None for r in reasons)
        r_exact = measurable and len(set(reasons)) == 1
        r_norm = measurable and len({_normalise(r) for r in reasons}) == 1

        verdict_stable += int(v_stable)
        if measurable:
            reason_measurable += 1
            reason_stable += int(r_exact)

        per_rule.append({
            "document": key[0],
            "rule_id": key[1],
            "outcomes": dict(zip(labels, outcomes)),
            "verdict_stable": v_stable,
            "reason_measurable": measurable,
            "reason_stable_exact": r_exact if measurable else None,
            "reason_stable_normalised": r_norm if measurable else None,
            "reasons": dict(zip(labels, reasons)) if measurable else None,
            "inferred_outcomes": {
                lab: (runs[i][key].inferred_outcome if key in runs[i] else None)
                for i, lab in enumerate(labels)
            },
        })

    n = len(keys)
    flips = [p for p in per_rule if not p["verdict_stable"]]
    reason_drift = [p for p in per_rule
                    if p["reason_measurable"] and not p["reason_stable_exact"]]

    return {
        "n_rules": n,
        "n_runs": len(runs),
        "labels": list(labels),
        "verdict_stability": {
            "n_stable": verdict_stable,
            "n_unstable": n - verdict_stable,
            "rate": verdict_stable / n if n else 0.0,
            "unstable_rules": [{"document": p["document"], "rule_id": p["rule_id"],
                                "outcomes": p["outcomes"]} for p in flips],
        },
        "reason_stability": {
            "n_measurable": reason_measurable,
            "n_unmeasurable": n - reason_measurable,
            "unmeasurable_because": "validate.py writes bare rules to *_confirmed.jsonl "
                                    "— no verdict object, so no reason. Reasons exist "
                                    "only for REJECT and FLAG.",
            "n_stable_exact": reason_stable,
            "rate_over_measurable": (reason_stable / reason_measurable)
                                    if reason_measurable else None,
            "drifting_rules": [{"document": p["document"], "rule_id": p["rule_id"],
                                "reasons": p["reasons"]} for p in reason_drift],
        },
        "per_rule": per_rule,
    }


def compare_corpora(run_dirs: Sequence[Path], reference: Path) -> dict:
    """Does each run reproduce the served corpus exactly?"""
    ref = load_corpus(reference)
    per_run = []
    for d in run_dirs:
        got = load_corpus(Path(d) / "all_rules_deduped.jsonl")
        per_run.append({
            "run": str(d),
            "n_distinct": len(got),
            "matches_reference": got == ref,
            "missing_from_run": [t for t in ref if t not in got],
            "extra_in_run": [t for t in got if t not in ref],
        })
    all_runs_identical = len({json.dumps(load_corpus(Path(d) / "all_rules_deduped.jsonl"))
                              for d in run_dirs}) == 1
    return {
        "reference": str(reference),
        "reference_n_distinct": len(ref),
        "reference_triples": ref,
        "all_runs_identical_to_each_other": all_runs_identical,
        "all_runs_match_reference": all(r["matches_reference"] for r in per_run),
        "per_run": per_run,
    }


def overall_verdict(comparison: dict, corpora: dict) -> dict:
    """The one sentence a reader needs, plus what it does and does not cover."""
    v_ok = comparison["verdict_stability"]["n_unstable"] == 0
    c_ok = corpora["all_runs_match_reference"]
    r = comparison["reason_stability"]
    r_rate = r["rate_over_measurable"]

    if v_ok and c_ok:
        headline = ("REPRODUCIBLE — every verdict agreed across runs and every run "
                    "reproduced the served four-rule corpus exactly.")
    elif c_ok:
        headline = (f"CORPUS REPRODUCIBLE BUT VERDICTS ARE NOT — "
                    f"{comparison['verdict_stability']['n_unstable']} rule(s) flipped "
                    f"across runs while the deduped corpus still matched. The corpus is "
                    f"robust to the flips that occurred; it is not robust in principle.")
    else:
        headline = (f"NOT REPRODUCIBLE — the served corpus was not recovered. "
                    f"{comparison['verdict_stability']['n_unstable']} verdict(s) flipped.")

    reason_line = (
        "reason stability UNMEASURED (no rule had a reason in every run)"
        if r_rate is None else
        f"reasons identical on {r['n_stable_exact']}/{r['n_measurable']} measurable rules "
        f"({r_rate:.0%}); {r['n_unmeasurable']} rules have no persisted reason to compare"
    )

    return {
        "headline": headline,
        "verdicts_stable": v_ok,
        "corpus_reproduced": c_ok,
        "reason_note": reason_line,
        "caveat": "Verdict stability and reason stability are SEPARATE results. §12.3 "
                  "found stable verdicts resting on fabricated reasoning; a run that is "
                  "verdict-stable says nothing about whether its justifications are sound.",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-run the translated-arm validator N times and compare "
                    "(RUNS ON THE LLM HOST ONLY — needs CUDA + Ollama).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Expected runtime ~1 min per repeat (the recorded run took 48.9 s). "
               "Repeats are sequential: never load two models at once.",
    )
    ap.add_argument("--repeats", type=int, default=3,
                    help="number of independent validator runs (default 3)")
    ap.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES,
                    help="stage-2.5 guarded corpus (the input)")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="where run directories and the artifact are written")
    ap.add_argument("--reference", type=Path, default=REFERENCE_CORPUS,
                    help="the served corpus each run must reproduce")
    ap.add_argument("--compare-only", action="store_true",
                    help="skip inference; compare run directories that already exist")
    ap.add_argument("--runs", type=Path, nargs="*", default=None,
                    help="with --compare-only: the run directories to compare")
    ap.add_argument("--force", action="store_true",
                    help="delete existing run directories first")
    ap.add_argument("--json", type=Path, default=None,
                    help="artifact path (default <root>/validator_replication.json)")
    args = ap.parse_args()

    root = args.root
    artifact = args.json or (root / "validator_replication.json")

    if args.compare_only:
        if not args.runs or len(args.runs) < 2:
            print("--compare-only needs at least two --runs directories", file=sys.stderr)
            return 2
        run_dirs = [Path(d) for d in args.runs]
        missing = [d for d in run_dirs if not d.exists()]
        if missing:
            print(f"missing run directories: {missing}", file=sys.stderr)
            return 2
        model_tag = None
        run_meta: List[dict] = []
    else:
        # ── Preflight. Nothing below spends a token until all of this passes. ──
        try:
            check_candidates(args.candidates)
            model_tag = check_model_tag()
            ollama_info = check_ollama(model_tag)
        except PreflightError as exc:
            print("\n" + "!" * 76, file=sys.stderr)
            print("PREFLIGHT FAILED — nothing was run.", file=sys.stderr)
            print("!" * 76, file=sys.stderr)
            print(f"{exc}\n", file=sys.stderr)
            return 2

        run_dirs = [root / "runs" / f"run_{i + 1}" for i in range(args.repeats)]
        if args.force:
            for d in run_dirs:
                if d.exists():
                    shutil.rmtree(d)
        try:
            assert_isolated_output_dirs(run_dirs)
        except PreflightError as exc:
            print("\n" + "!" * 76, file=sys.stderr)
            print("OUTPUT ISOLATION CHECK FAILED — nothing was run.", file=sys.stderr)
            print("!" * 76, file=sys.stderr)
            print(f"{exc}\n", file=sys.stderr)
            return 2

        print(f"validator model : {model_tag}")
        print(f"ollama host     : {ollama_info['host']}")
        print(f"repeats         : {args.repeats} (sequential)")
        print()

        run_meta = []
        for i, d in enumerate(run_dirs, 1):
            print(f"[{i}/{args.repeats}] validating -> {d}")
            run_meta.append(run_one(args.candidates, d))
            print(f"          done in {run_meta[-1]['seconds']:.1f}s")

    # ── Compare ───────────────────────────────────────────────────────────────
    guarded = load_guarded_inputs(args.candidates) if args.candidates.exists() else None
    labels = [d.name for d in run_dirs]
    runs = [load_run(d, guarded) for d in run_dirs]
    comparison = compare_runs(runs, labels)
    corpora = compare_corpora(run_dirs, args.reference)
    verdict = overall_verdict(comparison, corpora)

    payload = {
        "what": "replication of the stage-3 validator, translated arm",
        "why": "§12.3's stability claim is established for the `strict` arm only; the "
               "`translated` arm produced the served corpus and ran once.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "validator_model": model_tag,
        "prompt_variant": "translated",
        "candidates": str(args.candidates),
        "runs": run_meta,
        "recorded_run_2026_08_20": RECORDED_RUN,
        "verdict": verdict,
        "verdict_stability": comparison["verdict_stability"],
        "reason_stability": comparison["reason_stability"],
        "corpus_reproduction": corpora,
        "per_rule": comparison["per_rule"],
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Report ────────────────────────────────────────────────────────────────
    vs, rs = comparison["verdict_stability"], comparison["reason_stability"]
    print()
    print("=" * 76)
    print("  VALIDATOR REPLICATION — translated arm")
    print("=" * 76)
    print(f"  rules compared : {comparison['n_rules']} across {comparison['n_runs']} runs")
    print()
    print(f"  VERDICT stability : {vs['n_stable']}/{comparison['n_rules']} stable "
          f"({vs['rate']:.1%})")
    for u in vs["unstable_rules"]:
        print(f"      FLIP {u['rule_id']:8} {u['outcomes']}")
    print()
    print(f"  REASON stability  : {rs['n_stable_exact']}/{rs['n_measurable']} identical "
          f"({rs['n_unmeasurable']} unmeasurable — no persisted reason)")
    for d in rs["drifting_rules"][:5]:
        print(f"      DRIFT {d['rule_id']}")
    print()
    print(f"  distinct rules per run : "
          f"{[r['n_distinct'] for r in corpora['per_run']]} "
          f"(recorded run: {RECORDED_RUN['distinct_after_dedup']})")
    print(f"  reproduces served corpus : {corpora['all_runs_match_reference']}")
    print()
    print(f"  >> {verdict['headline']}")
    print(f"  >> {verdict['reason_note']}")
    print()
    print(f"  {verdict['caveat']}")
    print()
    print(f"  artifact: {artifact}")
    return 0 if verdict["corpus_reproduced"] else 1


if __name__ == "__main__":
    sys.exit(main())
