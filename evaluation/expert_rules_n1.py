"""§24.3 Arm 2 — the hand-written expert rules, as a third arm.

`thesis_findings.md` §24.3. Fourteen rules written over the same 14 variables of
`shield/context.py`, from power-system doctrine and the numeric limits the source
standards state, and run against the same target as §28:

    the model predicted secure AND the contingency was a violation

🚨 THIS IS NOT A BLIND BASELINE, AND §24.3 REQUIRES THAT BE SAID PLAINLY.
§24.3 demands the rules be written BEFORE consulting `results/audit/*`, the fire
rates, or the §21 calibration, because rules written afterwards are "the extracted
corpus laundered through a human". That ordering was NOT available: the author had
already read the served corpus, §13's precision figures and §28's tree splits in the
same session. §24.3's own fallback therefore applies — report it as **"a hand-written
comparison, not a blind baseline"** and claim less. What it can still show is whether
a broader, doctrine-derived rule set does better than the four that survived
extraction; what it cannot show is that a human working blind would have written
these.

Two mitigations are in place and neither repairs the above:
  * The rules were committed to `expert_rules/expert_rules.jsonl` and fingerprinted
    (md5) BEFORE any evaluation ran. This script asserts the hash, so the rules
    cannot have been tuned against the result.
  * Every family the 14 variables can express is included rather than a chosen few,
    and thresholds are the standards' own wherever one exists. Each record carries a
    `basis` field marking it `standard-stated`, `operating-practice` or `doctrine`.

⚠️ The rules are NEVER merged into the corpus (§24.3). They live in their own
directory, and the filename deliberately avoids `*_confirmed.jsonl` so
`deduplicate_rules` cannot pick them up.

GATE SEMANTICS. A gate can only act where the model predicted secure, so every arm
here is scored as `rule fires AND model predicted secure`. §28's tree is re-scored the
same way (`tree_held_gate`) so the three arms are like for like.

Usage:
    python evaluation/expert_rules_n1.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.append(".")

from evaluation.ceiling_tree_n1 import (CONTEXT_VARS, RECORDED_THRESHOLD, HOME,
                                        TAGS, frame_contexts, frame_count,
                                        load_meta, prf, score_model,
                                        shield_reference)
from evaluation.eval_n1_cross_topology import best_f1_and_thr
from scripts.pyg_data import PreloadedGridDataset
from shield.evaluator import evaluate_condition
from training.config import DATA_DIR, DEVICE
from training.train_gnn import compute_normalization_stats

RULES_PATH = os.path.join("expert_rules", "expert_rules.jsonl")
# Fingerprint taken at 2026-09-19T08:14:58Z, before any evaluation was run. If this
# does not match, the rules were edited after seeing a result and the arm is void.
RULES_MD5 = "05a29bd74e0ba1722e64f0c3579f6cd8"
OUT = os.path.join("results", "ceiling", "expert_rules.json")


def load_rules(path: str, expect_md5: str | None) -> list[dict]:
    with open(path, "rb") as fh:
        digest = hashlib.md5(fh.read()).hexdigest()
    if expect_md5 and digest != expect_md5:
        raise SystemExit(
            f"{path} has md5 {digest}, expected {expect_md5}.\n"
            f"The rules were pre-registered before evaluation; a changed file means "
            f"they were tuned against the answer, which voids the arm (§24.3)."
        )
    rules = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rules.append(json.loads(line))
    return rules, digest


def rule_fires_per_frame(rule: dict, contexts: list[dict]
                         ) -> tuple[np.ndarray, dict[str, int]]:
    """Evaluate one rule over every frame, through the SHIELD's own evaluator.

    A rule that cannot be evaluated on a frame is False, which is the gate's own
    behaviour: NOT_EVALUABLE never blocks.

    ⚠️ HISTORY (issue B5). Until 2026-09-20 the evaluator ran with NO builtins, so
    X_010's `abs(generation_load_imbalance_pct) > 5` resolved to NOT_EVALUABLE on
    every frame of every grid and could never fire. `results/ceiling/expert_rules.json`
    is the record of THAT run and is kept as-run: 13 of 14 rules effective.

    The constraint has since been repaired in the language itself — `abs`, `min`
    and `max` are whitelisted in `shield/evaluator.py` and `extraction.common.
    lint_condition` — WITHOUT touching the pre-registered rules file, whose md5 is
    still asserted above. Re-running this script now evaluates all 14. Write the
    result to a NEW path (`results/ceiling/expert_rules_builtins_repaired.json`),
    never over the original:

        python evaluation/expert_rules_n1.py \\
            --json results/ceiling/expert_rules_builtins_repaired.json

    Measured 2026-09-20: X_010 fires on 0.00% / 0.03% / 0.18% of frames and scores
    F1 0.0000 / 0.0000 / 0.0084 (neurips2020 / case14 / wcci2022) against X_001
    `loading_pct > 100` at 0.2765 / 0.0102 / 0.4644. Every other rule-grid cell is
    bit-identical to the original run, and `expert_best_single_rule` is unchanged
    on all three grids. The repair strengthens §29's conclusion; it does not move it.
    """
    cond = rule["condition"]
    verdicts = [evaluate_condition(cond, dict(c)) for c in contexts]
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.value] = counts.get(v.value, 0) + 1
    return np.array([v.blocks for v in verdicts]), counts


def contexts_for(tag: str, indices: np.ndarray, meta) -> tuple[list[dict], np.ndarray]:
    """Per-frame context dicts plus each frame's contingency count."""
    X, counts = frame_contexts(tag, indices, meta)
    out = []
    for row in X:
        ctx = {v: float(val) for v, val in zip(CONTEXT_VARS, row) if np.isfinite(val)}
        if "any_line_tripped" in ctx:
            ctx["any_line_tripped"] = bool(ctx["any_line_tripped"])
        out.append(ctx)
    return out, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=OUT)
    ap.add_argument("--rules", default=RULES_PATH)
    args = ap.parse_args()
    t0 = time.time()

    rules, digest = load_rules(args.rules, RULES_MD5)
    print(f"{len(rules)} expert rules, md5 {digest} (matches the pre-registered hash)")

    home_meta = load_meta(HOME)
    print("Computing 36-bus train-split normalization stats...")
    home_pt = PreloadedGridDataset(os.path.join(DATA_DIR, "processed_grid_data_n1.pt"),
                                   device=DEVICE)
    train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_train_idx.npy"))
    val_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_val_idx.npy"))
    test_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_test_idx.npy"))
    norm = compute_normalization_stats(home_pt, train_idx)
    del home_pt

    print("Re-deriving the decision threshold on the neurips2020 VAL split...")
    val_logits, val_y = score_model(HOME, val_idx, home_meta, norm)
    _, thr = best_f1_and_thr(val_logits, val_y)
    if abs(thr - RECORDED_THRESHOLD) > 5e-4:
        raise SystemExit(f"threshold {thr:.6f} != recorded {RECORDED_THRESHOLD}")
    print(f"  threshold {thr:.6f}")

    report = {
        "protocol": {
            "target": "model predicted secure AND the contingency was a violation",
            "semantics": "gate — a rule can only act where the model predicted secure",
            "rules_file": args.rules, "rules_md5": digest, "n_rules": len(rules),
            "model_threshold": float(thr),
            "blind": False,
            "blind_note": ("NOT a blind baseline. The author had already seen the served "
                           "corpus, the shield's precision figures and the §28 tree splits. "
                           "§24.3's fallback applies: report as a hand-written comparison "
                           "and claim less."),
        },
        "rules": [{k: r[k] for k in ("rule_id", "family", "condition", "basis")}
                  for r in rules],
        "grids": {},
    }

    for tag in TAGS:
        meta = home_meta if tag == HOME else load_meta(tag)
        idx = test_idx if tag == HOME else np.arange(frame_count(tag))
        scope = "held-out test split" if tag == HOME else "all frames (unseen topology)"
        print(f"Scoring {tag} ({len(idx):,} frames)...")

        logits, y = score_model(tag, idx, meta, norm)
        ctxs, counts = contexts_for(tag, idx, meta)
        if counts.sum() != len(y):
            raise ValueError(f"{tag}: {counts.sum()} context rows vs {len(y)} contingencies")
        eligible = logits <= thr
        target = (eligible & (y == 1)).astype(int)

        per_rule, any_fire = {}, np.zeros(len(y), dtype=bool)
        for rule in rules:
            frame_fires, verdicts = rule_fires_per_frame(rule, ctxs)
            fires = np.repeat(frame_fires, counts)
            any_fire |= fires
            per_rule[rule["rule_id"]] = {
                "condition": rule["condition"], "basis": rule["basis"],
                "frames_firing": float(frame_fires.mean()),
                "verdicts": verdicts,
                **prf(target, fires & eligible),
            }

        best_id = max(per_rule, key=lambda k: per_rule[k]["f1"])
        # "block every contingency the model called secure" -- no rule at all.
        # Any rule firing on ~100% of frames collapses onto this, so it must be
        # visible or a degenerate rule reads as a winner.
        flag_all = prf(target, eligible)
        report["grids"][tag] = {
            "scope": scope,
            "contingencies": int(len(y)),
            "target_contingencies": int(target.sum()),
            "all_positive_f1_on_target": float(
                2 * target.mean() / (1 + target.mean())) if target.mean() else 0.0,
            "flag_all_eligible_baseline": flag_all,
            "expert_union": prf(target, any_fire & eligible),
            "expert_best_single_rule": {"rule_id": best_id, **per_rule[best_id]},
            "per_rule": per_rule,
            "shield_on_same_target": shield_reference(tag),
        }

    report["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=2)
    _print(report)
    print(f"\n  written: {args.json}")


def _print(rep: dict) -> None:
    print(f"\n{'=' * 80}")
    print("  §24.3 expert rules — a hand-written comparison, NOT a blind baseline")
    print(f"  {rep['protocol']['n_rules']} rules, md5 {rep['protocol']['rules_md5'][:12]}, "
          f"gate semantics")
    print(f"{'=' * 80}")
    for tag, g in rep["grids"].items():
        s = g["shield_on_same_target"]
        u, b = g["expert_union"], g["expert_best_single_rule"]
        print(f"\n  {tag}  —  {g['scope']}")
        print(f"    target                 : {g['target_contingencies']:,} of "
              f"{g['contingencies']:,}   all-positive F1 {g['all_positive_f1_on_target']:.4f}")
        fa = g["flag_all_eligible_baseline"]
        print(f"    flag-all-eligible      : F1 {fa['f1']:.4f}   "
              f"prec {fa['precision']:.3f}  rec {fa['recall']:.3f}"
              "   <- no rule at all")
        print(f"    EXPERT, all 14 (OR)    : F1 {u['f1']:.4f}   "
              f"prec {u['precision']:.3f}  rec {u['recall']:.3f}")
        print(f"    EXPERT, best single    : F1 {b['f1']:.4f}   "
              f"prec {b['precision']:.3f}  rec {b['recall']:.3f}   "
              f"[{b['rule_id']}: {b['condition']}]")
        if s:
            print(f"    EXTRACTED SHIELD       : F1 {s['f1']:.4f}   "
                  f"prec {s['precision']:.3f}  rec {s['recall']:.3f}")
    bad = sorted({rid for g in rep["grids"].values()
                  for rid, r in g["per_rule"].items()
                  if set(r["verdicts"]) & {"NOT_EVALUABLE", "ERROR"}})
    if bad:
        print("")
        print("  !! NOT EVALUABLE ON SOME FRAME: " + ", ".join(bad))
        print("     Before 2026-09-20 this listed X_010 on all three grids (the "
              "sandbox had no builtins at all); abs/min/max are now whitelisted.")
    print("\n  top expert rules by F1, per grid:")
    for tag, g in rep["grids"].items():
        ranked = sorted(g["per_rule"].items(), key=lambda kv: -kv[1]["f1"])[:4]
        print(f"    {tag}:")
        for rid, r in ranked:
            print(f"      {rid}  F1 {r['f1']:.4f}  p {r['precision']:.3f} "
                  f"r {r['recall']:.3f}  fires on {r['frames_firing']:.1%} of frames"
                  f"   {r['condition']}")


if __name__ == "__main__":
    main()
