"""
P(N-1 violation | rule fires AND the model predicted secure) -- the MODEL-CONDITIONAL
calibration of the WARN channel. thesis_findings.md §21.4 / §22.5.

`evaluation/warn_rule_calibration.py` measures P(violation | rule fires) over every
contingency of every base-case frame. That is a GRID-conditional statement: "on this
topology, when this predicate holds, this fraction of single-line contingencies
violate a limit."

The WARN channel does not say that. It says:

    "this prediction was let through, but according to this rule, that could happen."

`let through` is the operative word. A WARN line is only ever rendered beside a
prediction the model called SECURE, so the conditioning set is contingencies where
`logit <= threshold`, not all contingencies. That is a strictly smaller set, and a
systematically easier one -- the model is filtering out the contingencies it can
already see are dangerous, so whatever violations remain are the ones it missed.
The honest question is whether the predicate still discriminates INSIDE the model's
own blind spot.

WHAT IS REUSED, AND WHY
-----------------------
Everything statistical comes from `warn_rule_calibration.py` by import, not by
copy: `rates`, `verdict_of`, `stratified`, `blank`, `condition_groups`, MIN_FRAMES
and MIN_LIFT. If the verdict rule changes there it changes here, and the two passes
cannot drift into reporting different things under the same word.

Everything about the model comes from `evaluation/eval_shield_n1.py` by import:
`score_topology` (forward pass, normalization, per-contingency frame/line mapping),
`score_topology_indices` + `best_f1_and_thr` (the held threshold), and
`EVAL_BATCH_SIZE` (64, load-bearing -- BatchNorm uses live batch statistics, so the
batch size IS part of the measurement; see gnn_n1_tightening.md §8).

THE THREE METHODOLOGY DECISIONS THIS FILE MAKES
-----------------------------------------------
1. **Frames with no model-secure contingency are dropped from both arms**, and
   counted separately as `frames_no_secure_contingency`. Such a frame contributes
   zero contingencies to either rate; counting it as a "firing frame" would inflate
   the sample-size guard with frames carrying no data. It is dropped symmetrically,
   so it cannot bias the discrimination -- but it does shrink the frame counts
   relative to §21, and those are exactly the most dangerous frames (the model
   called every contingency unsafe), so the restriction is not random.

2. **neurips2020 is scanned over ALL 12,000 frames by default, not the held-out
   test split.** `eval_shield_n1.score_topology` restricts the home topology to its
   test split, which is right for reporting model performance and wrong here for
   one reason: §21 measured over all 12,000 frames, and a 1,800-frame subset would
   confound "model-conditional" with "different frames". At MIN_FRAMES=30 that
   confound is not cosmetic -- the voltage predicates fire on 71 of 12,000 neurips
   frames, so a test-split-only pass would report INSUFFICIENT for reasons of
   arithmetic rather than of the model. The cost is that 70% of those frames are
   in-sample for the checkpoint. `--neurips-scope test` runs the uncontaminated
   arm; both are reported, and the artifact carries whichever was run.

3. **The threshold is selected on the neurips2020 val split and held fixed across
   all three grids** -- the protocol from CLAUDE.md, unchanged. It is never
   re-selected per grid and never chosen with the answer key. Under
   `--neurips-scope all` the val frames are inside the scanned set; that is a known
   and small leak, disclosed rather than engineered around.

Inherited invariants, all load-bearing, none rediscovered here:
  * the statistic is `P|fires - P|silent`, NOT lift against the base rate -- a
    predicate firing on 98% of frames IS the base rate (§21.1);
  * pooled values are STRATIFIED BY GRID and weighted by FRAMES, and a grid whose
    own verdict is INSUFFICIENT gets no vote (§21.1);
  * `-1` in `n1_violation` means NOT EVALUATED -- excluded, never counted secure.
    `build_line_targets` already drops them via `line_mask`, which `score_topology`
    applies; asserted below rather than assumed;
  * NOT_EVALUABLE / ERROR rule verdicts are excluded from BOTH arms, never silent.

Run:  .venv\\Scripts\\python.exe evaluation\\warn_rule_calibration_conditional.py
      .venv\\Scripts\\python.exe evaluation\\warn_rule_calibration_conditional.py --neurips-scope test
      .venv\\Scripts\\python.exe evaluation\\warn_rule_calibration_conditional.py --max-frames 400
"""
from __future__ import annotations

import argparse
import json
import linecache
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.eval_shield_n1 import (  # noqa: E402
    CKPT,
    EVAL_BATCH_SIZE,
    best_f1_and_thr,
    score_topology,
    score_topology_indices,
)
from evaluation.warn_rule_calibration import (  # noqa: E402
    MIN_FRAMES,
    MIN_LIFT,
    TAGS,
    blank,
    condition_groups,
    fmt,
    rates,
    stratified,
    verdict_of,
)
from shield.context import build_context, load_base_kv  # noqa: E402
from shield.evaluator import Verdict, evaluate_condition  # noqa: E402
from training.config import DATA_DIR, DEVICE  # noqa: E402
from training.train_gnn import compute_normalization_stats  # noqa: E402

HOME = "neurips2020"
GRID_CONDITIONAL = "results/audit/warn_n1_calibration.json"


def held_threshold(checkpoint: str, batch_size: int, norm: tuple) -> float:
    """The threshold from CLAUDE.md: chosen on the neurips2020 VAL split, held.

    Not re-selected per grid, not an oracle. This is `eval_shield_n1.main`'s own
    default path, called directly so the two harnesses cannot disagree.
    """
    val_path = os.path.join(DATA_DIR, "split_neurips2020_n1_val_idx.npy")
    if not os.path.exists(val_path):
        sys.exit("No neurips2020 val split on disk; cannot hold a threshold.")
    val = score_topology_indices(HOME, checkpoint, batch_size, norm,
                                 np.load(val_path))
    _, thr = best_f1_and_thr(val["logits"], val["y"])
    return float(thr)


def score_grid(tag: str, checkpoint: str, batch_size: int, norm: tuple,
               neurips_scope: str) -> dict:
    """One forward pass over a topology, via eval_shield_n1.score_topology.

    That function restricts `tag == HOME` to the held-out test split. To scan every
    frame of the home grid -- which is what makes this run comparable with §21 --
    its module-level HOME is temporarily rebound to a tag that matches nothing, so
    the identical code path takes its `else` branch (all frames). No behaviour is
    reimplemented and nothing on disk is modified; the binding is restored in a
    `finally`.
    """
    import evaluation.eval_shield_n1 as esn

    original = esn.HOME
    try:
        if tag == HOME and neurips_scope == "all":
            esn.HOME = "\0__scan_every_frame__"
        return score_topology(tag, checkpoint, batch_size, norm)
    finally:
        esn.HOME = original


def scan_conditional(tag: str, conditions: list[str], scored: dict,
                     pred: np.ndarray, data_dir: Path, max_frames) -> dict:
    """Accumulate per-condition counts over MODEL-SECURE contingencies only.

    `scored["frame"]` is the JSONL row of each contingency and `pred` is the
    thresholded model call, so the two index the same axis. Per frame:

        n_secure  contingencies the model let through (pred == 0)
        n_vio     of those, how many actually violate a limit (y == 1)

    A frame with `n_secure == 0` carries no information for either arm and is
    dropped from the frame counts rather than credited to one of them.
    """
    y = scored["y"]
    frames = scored["frame"].astype(np.int64)
    # -1 labels never reach here: build_line_targets masks them out and
    # score_topology applies that mask. Assert rather than trust.
    assert y.min() >= 0, "NOT-EVALUATED (-1) labels leaked into the scored set"

    secure = pred == 0
    n_frames_total = int(frames.max()) + 1 if frames.size else 0
    n_secure = np.bincount(frames[secure], minlength=n_frames_total)
    n_vio = np.bincount(frames[secure & (y == 1)], minlength=n_frames_total)

    base_kv = load_base_kv(tag, data_dir=str(data_dir))
    jsonl = scored["jsonl"]

    acc = blank(conditions)
    n_frames = n_skipped = n_no_secure = 0
    cont_total = int(y.size)
    cont_secure = int(secure.sum())

    for frame_idx in sorted(set(int(f) for f in frames)):
        if max_frames is not None and n_frames >= max_frames:
            break
        if n_secure[frame_idx] == 0:
            n_no_secure += 1
            continue
        raw = linecache.getline(jsonl, frame_idx + 1).strip()
        if not raw:
            n_skipped += 1
            continue
        try:
            ctx = build_context(json.loads(raw), base_kv)
        except (KeyError, ValueError, json.JSONDecodeError):
            n_skipped += 1
            continue
        n_frames += 1

        c, v = int(n_secure[frame_idx]), int(n_vio[frame_idx])
        for cond in conditions:
            a = acc[cond]
            verdict = evaluate_condition(cond, dict(ctx))
            if verdict is Verdict.VIOLATED:
                a["frames_fired"] += 1
                a["cont_fired"] += c
                a["vio_fired"] += v
            elif verdict is Verdict.SATISFIED:
                a["frames_silent"] += 1
                a["cont_silent"] += c
                a["vio_silent"] += v
            else:
                # NOT_EVALUABLE / ERROR -> neither arm. Counting it silent would
                # credit the rule with a quiet it never made.
                a["frames_unevaluable"] += 1

    return {
        "n_frames": n_frames,
        "n_skipped": n_skipped,
        "frames_no_secure_contingency": n_no_secure,
        "contingencies_total": cont_total,
        "contingencies_model_secure": cont_secure,
        "model_secure_fraction": (cont_secure / cont_total) if cont_total else None,
        "violation_rate_all": float((y == 1).mean()) if cont_total else None,
        "violation_rate_model_secure": (
            float((y[secure] == 1).mean()) if cont_secure else None),
        "acc": acc,
    }


def pooled_row(rows: dict, pooled_acc: dict, grids: list[str]) -> dict:
    """Stratified pooled row -- identical construction to warn_rule_calibration."""
    row = rates(pooled_acc)
    row["discrimination_unstratified"] = row["discrimination"]
    row["lift_vs_base_unstratified"] = row["lift_vs_base"]
    row["discrimination"] = stratified(rows, "discrimination")
    row["lift_vs_base"] = stratified(rows, "lift_vs_base")
    row["stratified_by_grid"] = True
    contributing = [t for t in grids if rows[t]["verdict"] != "INSUFFICIENT"]
    row["n_grids_contributing"] = len(contributing)
    row["grids_contributing"] = contributing
    row["verdict"] = (
        verdict_of(row["frames_fired"], row["frames_silent"], row["discrimination"])
        if contributing else "INSUFFICIENT")
    # The rate a warning is allowed to quote must come from the grids that actually
    # scored the predicate, not from the confounded pooled counts (§22.4 defect 1).
    num_f = den_f = num_s = den_s = 0.0
    for t in contributing:
        r = rows[t]
        if r["p_violation_given_fires"] is not None:
            num_f += r["p_violation_given_fires"] * r["frames_fired"]
            den_f += r["frames_fired"]
        if r["p_violation_given_silent"] is not None:
            num_s += r["p_violation_given_silent"] * r["frames_fired"]
            den_s += r["frames_fired"]
    row["p_fires_scoring_grids"] = (num_f / den_f) if den_f else None
    row["p_silent_scoring_grids"] = (num_s / den_s) if den_s else None
    return row


def load_grid_conditional(path: Path) -> dict:
    """The §21 artifact, keyed by condition, for the side-by-side column."""
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {c["condition"]: c for c in doc.get("conditions", [])}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Model-conditional N-1 calibration of a shield channel")
    ap.add_argument("--readmission", default="results/audit/readmission.json")
    ap.add_argument("--channel", default="WARN")
    ap.add_argument("--corpus", default="shield_corpus/all_rules_channels.jsonl",
                    help="Shipped corpus; used only to mark which conditions ship")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE,
                    help=f"LOAD-BEARING, default {EVAL_BATCH_SIZE}; BatchNorm uses "
                         "live batch statistics (gnn_n1_tightening.md §8)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override the held threshold. Do not use for reported runs.")
    ap.add_argument("--neurips-scope", choices=("all", "test"), default="all",
                    help="'all' = every neurips2020 frame (comparable with §21, but "
                         "70%% in-sample for the checkpoint); 'test' = the held-out "
                         "split only (uncontaminated, ~1,800 frames)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap frames per grid. SMOKE TESTING ONLY.")
    ap.add_argument("--grid-conditional", default=GRID_CONDITIONAL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.batch_size != EVAL_BATCH_SIZE:
        print(f"\n  !! --batch-size {args.batch_size} != {EVAL_BATCH_SIZE}: BatchNorm "
              f"uses live batch statistics, so the model's calls change and no "
              f"recorded figure is comparable.\n")

    groups = condition_groups(Path(args.readmission), args.channel)
    if not groups:
        print(f"no {args.channel} rules in {args.readmission}")
        return 1
    conditions = list(groups)
    n_rules = sum(len(v) for v in groups.values())

    shipped: set[str] = set()
    corpus = Path(args.corpus)
    if corpus.exists():
        for line in corpus.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("channel") == args.channel:
                shipped.add((rec.get("condition") or "").strip())

    print(f"\nmodel-conditional calibration of {len(conditions)} distinct "
          f"{args.channel} conditions ({n_rules} rule records)")
    print(f"  conditioning set: contingencies the model PREDICTED SECURE")
    print(f"  {len(shipped)} of these conditions ship in {args.corpus}")

    if not os.path.exists(args.checkpoint):
        sys.exit(f"Missing {args.checkpoint} — train the N-1 model first.")

    print("\nComputing 36-bus N-1 train-split normalization stats...")
    from scripts.pyg_data import PreloadedGridDataset

    home_pt = os.path.join(DATA_DIR, "processed_grid_data_n1.pt")
    train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_train_idx.npy"))
    home = PreloadedGridDataset(home_pt, device=DEVICE)
    norm = compute_normalization_stats(home, train_idx)
    del home

    thr = args.threshold
    thr_source = "supplied on the command line — NOT the reported protocol"
    if thr is None:
        print("Selecting the decision threshold on the neurips2020 VAL split...")
        thr = held_threshold(args.checkpoint, args.batch_size, norm)
        thr_source = "selected on the neurips2020 val split, held fixed across grids"
    print(f"  threshold {thr:.4f}  ({thr_source})")

    per_grid = {}
    scopes = {}
    for tag in TAGS:
        try:
            print(f"  [{tag}] forward pass + scan ...")
            scored = score_grid(tag, args.checkpoint, args.batch_size, norm,
                                args.neurips_scope)
            pred = (scored["logits"] > thr).astype(int)
            scopes[tag] = scored["scope"]
            per_grid[tag] = scan_conditional(tag, conditions, scored, pred,
                                             Path(args.data_dir), args.max_frames)
            g = per_grid[tag]
            print(f"      {g['n_frames']:,} frames scored "
                  f"({g['frames_no_secure_contingency']:,} dropped: no secure "
                  f"contingency) | model-secure {g['contingencies_model_secure']:,}"
                  f"/{g['contingencies_total']:,} "
                  f"({g['model_secure_fraction']:.1%}) | violation rate "
                  f"{g['violation_rate_all']:.3f} all -> "
                  f"{g['violation_rate_model_secure']:.3f} model-secure")
        except (FileNotFoundError, SystemExit) as exc:
            print(f"  [{tag}] skipped: {exc}")
    if not per_grid:
        print("no dataset scored")
        return 1

    pooled = blank(conditions)
    for res in per_grid.values():
        for cond, a in res["acc"].items():
            for k, v in a.items():
                pooled[cond][k] += v

    prior = load_grid_conditional(Path(args.grid_conditional))
    grids = list(per_grid)

    print("\n" + "=" * 112)
    print(f"P(N-1 violation | rule fires, MODEL PREDICTED SECURE) — {args.channel}")
    print("=" * 112)
    print(f"\n  {'condition':<50}{'grid':<13}{'fires':>7}{'cover':>8}"
          f"{'P|fire':>8}{'P|quiet':>9}{'P|f-P|q':>9}{'grids':>7}  verdict")
    print("  " + "-" * 110)

    out_rules = []
    changed = []
    for cond in conditions:
        ids = sorted({r["rule_id"] for r in groups[cond]})
        rows = {tag: rates(per_grid[tag]["acc"][cond]) for tag in grids}
        rows["pooled"] = pooled_row(rows, pooled[cond], grids)

        first = True
        for tag, r in rows.items():
            label = cond[:48] if first else ""
            first = False
            mark = "*" if tag == "pooled" else " "
            print(f"  {label:<50}{mark}{tag:<12}{r['frames_fired']:>7}"
                  f"{fmt(r['coverage'], 8, 3)}{fmt(r['p_violation_given_fires'])}"
                  f"{fmt(r['p_violation_given_silent'], 9)}"
                  f"{fmt(r['discrimination'], 9, 3)}"
                  f"{r.get('n_grids_contributing', ''):>7}  {r['verdict']}")

        before = prior.get(cond, {}).get("pooled", {})
        v_before = before.get("verdict")
        v_after = rows["pooled"]["verdict"]
        d_before = before.get("discrimination")
        d_after = rows["pooled"]["discrimination"]
        if v_before is not None:
            flag = "  <-- VERDICT CHANGED" if v_before != v_after else ""
            print(f"    grid-conditional (§21): {v_before:<12} "
                  f"disc {fmt(d_before, 7, 3)}   ->  model-conditional: "
                  f"{v_after:<12} disc {fmt(d_after, 7, 3)}{flag}")
            if v_before != v_after:
                changed.append({"condition": cond, "from": v_before, "to": v_after,
                                "discrimination_before": d_before,
                                "discrimination_after": d_after})
        print(f"    -> {', '.join(ids)}"
              f"{'   [SHIPPED]' if cond in shipped else '   [not shipped]'}")

        out_rules.append({
            "condition": cond,
            "rule_ids": ids,
            "n_records": len(groups[cond]),
            "role": groups[cond][0].get("role"),
            "shipped_in_corpus": cond in shipped,
            "per_grid": {t: rows[t] for t in grids},
            "pooled": rows["pooled"],
            "grid_conditional": {
                "verdict": v_before,
                "discrimination": d_before,
                "p_violation_given_fires": before.get("p_violation_given_fires"),
                "p_violation_given_silent": before.get("p_violation_given_silent"),
                "grids_contributing": before.get("grids_contributing"),
            },
            "verdict_changed": bool(v_before is not None and v_before != v_after),
        })

    tally: dict = {}
    for r in out_rules:
        v = r["pooled"]["verdict"]
        tally[v] = tally.get(v, 0) + 1
    print("\n  pooled verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    if changed:
        print(f"\n  !! {len(changed)} VERDICT(S) CHANGED against §21:")
        for c in changed:
            print(f"     {c['from']:>12} -> {c['to']:<12}  {c['condition']}")
    else:
        print("\n  no verdict changed against the grid-conditional pass")

    dest = Path(args.json or
                f"results/audit/{args.channel.lower()}_n1_calibration_conditional.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "channel": args.channel,
        "conditioning": ("contingencies the model predicted SECURE "
                         "(logit <= held threshold)"),
        "n_distinct_conditions": len(conditions),
        "n_rule_records": n_rules,
        "threshold": thr,
        "threshold_source": thr_source,
        "eval_batch_size": args.batch_size,
        "checkpoint": args.checkpoint,
        "neurips_scope": args.neurips_scope,
        "scopes": scopes,
        "max_frames": args.max_frames,
        "min_frames_for_a_rate": MIN_FRAMES,
        "min_discrimination": MIN_LIFT,
        "frames_scanned": {t: per_grid[t]["n_frames"] for t in grids},
        "grid_stats": {t: {k: per_grid[t][k] for k in (
            "n_frames", "n_skipped", "frames_no_secure_contingency",
            "contingencies_total", "contingencies_model_secure",
            "model_secure_fraction", "violation_rate_all",
            "violation_rate_model_secure")} for t in grids},
        "verdict_tally_pooled": tally,
        "verdicts_changed_vs_grid_conditional": changed,
        "compared_against": str(args.grid_conditional),
        "note": (
            "MODEL-CONDITIONAL calibration: the conditioning set is contingencies "
            "the model predicted SECURE, which is the WARN channel's actual "
            "semantic ('let through, but this could happen'). Statistic, verdict "
            "thresholds and grid stratification are imported from "
            "evaluation/warn_rule_calibration.py so the two passes cannot disagree "
            "about what a word means. The verdict is taken on `discrimination` "
            "(P|fire - P|silent), never on lift against the base rate. Pooled "
            "values are STRATIFIED BY GRID, weighted by FRAMES, and a grid whose "
            "own verdict is INSUFFICIENT gets no vote; *_unstratified is kept for "
            "inspection and must not be quoted. -1 labels are NOT EVALUATED and are "
            "masked out upstream by build_line_targets. NOT_EVALUABLE/ERROR rule "
            "verdicts are excluded from both arms. Frames where the model predicted "
            "secure on NO contingency contribute to neither arm and are counted as "
            "frames_no_secure_contingency. Eval batch size 64 is part of the "
            "measurement (BatchNorm live batch statistics). Threshold selected on "
            "the neurips2020 val split and held fixed."),
        "conditions": out_rules,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
