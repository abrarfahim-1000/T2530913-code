"""
summarize_shield_results.py — turn the per-topology shield runs into thesis tables.

⚠ ORPHANED 2026-08-19. It reads `results/shield_summary_<tag>.json` — a directory that
does not exist in this layout, in a schema produced by the retired `eval_shield.py`.
The live harness is `evaluation/eval_shield_n1.py`, which writes `shield_<tag>.json` at
the repo root with a different schema. Nothing imports this module. Rewrite it against
the new schema or delete it; do not run it expecting output.

Reads results/shield_summary_<tag>.json + results/failures_<tag>.jsonl for every
tag given, and writes supplimentary_docs/shield_final_results.md — the companion
to gnn_final_results.md.

Also dumps N random BLOCK explanations per topology to a review sheet, for the
manual 1-5 explanation-quality rating.

Usage:
    python evaluation/summarize_shield_results.py --tag neurips2020 --tag case14 --tag wcci2022
"""
import argparse
import json
import os
import random

RESULTS_DIR = "results"
OUT_DOC = "supplimentary_docs/shield_final_results.md"
REVIEW_SHEET = "supplimentary_docs/shield_explanation_review.md"
SEED = 42

FAILURE_MODES = [
    "overconfident_wrong",
    "class_confusion",
    "threshold_failure",
    "novel_topology_state",
]


def _fmt(value, spec="{:.4f}", dash="—"):
    return dash if value is None else spec.format(value)


def _pct(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def load_summary(tag: str, results_dir: str) -> dict | None:
    path = os.path.join(results_dir, f"shield_summary_{tag}.json")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found - run eval_shield.py --tag {tag} first")
        return None
    with open(path) as f:
        return json.load(f)


def load_failures(tag: str, results_dir: str) -> list[dict]:
    path = os.path.join(results_dir, f"failures_{tag}.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def metric_table(summaries: dict[str, dict]) -> str:
    header = (
        "| Topology | Records | Rules | Macro F1 (raw) | Macro F1 (Lever A) | "
        "False block rate | Missed-fault catch rate | NOT_EVALUABLE | ERROR |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for tag, s in summaries.items():
        rows.append(
            f"| `{tag}` | {s['n_records']:,} | {s['n_rules']} | "
            f"{_fmt(s.get('macro_f1_raw'))} | {_fmt(s.get('macro_f1_calibrated'))} | "
            f"{_pct(s.get('false_block_rate'))} | {_pct(s.get('missed_fault_catch_rate'))} | "
            f"{s.get('n_not_evaluable', 0):,} | {s.get('n_error', 0):,} |"
        )
    return header + "\n".join(rows)


def failure_mode_table(summaries: dict[str, dict]) -> str:
    header = "| Topology | " + " | ".join(m.replace("_", " ") for m in FAILURE_MODES) + " |\n"
    header += "|---" * (len(FAILURE_MODES) + 1) + "|\n"
    rows = []
    for tag, s in summaries.items():
        dist = s.get("failure_mode_distribution", {})
        rows.append(f"| `{tag}` | " + " | ".join(str(dist.get(m, 0)) for m in FAILURE_MODES) + " |")
    return header + "\n".join(rows)


def catch_rate_table(summaries: dict[str, dict]) -> str:
    header = "| Topology | " + " | ".join(m.replace("_", " ") for m in FAILURE_MODES) + " |\n"
    header += "|---" * (len(FAILURE_MODES) + 1) + "|\n"
    rows = []
    for tag, s in summaries.items():
        by_mode = s.get("catch_rate_by_failure_mode", {})
        rows.append(f"| `{tag}` | " + " | ".join(_pct(by_mode.get(m)) for m in FAILURE_MODES) + " |")
    return header + "\n".join(rows)


def severity_table(failures_by_tag: dict[str, list[dict]]) -> str:
    severities = ["critical", "high", "medium", "low"]
    header = "| Topology | " + " | ".join(severities) + " | total |\n"
    header += "|---" * (len(severities) + 2) + "|\n"
    rows = []
    for tag, rows_ in failures_by_tag.items():
        counts = {s: 0 for s in severities}
        for r in rows_:
            sev = str(r.get("highest_severity", "")).lower()
            if sev in counts:
                counts[sev] += 1
        rows.append(f"| `{tag}` | " + " | ".join(str(counts[s]) for s in severities)
                    + f" | {len(rows_)} |")
    return header + "\n".join(rows)


def write_review_sheet(failures_by_tag: dict[str, list[dict]], n: int, path: str) -> None:
    rng = random.Random(SEED)
    lines = [
        "# Shield explanation review sheet",
        "",
        f"{n} randomly sampled BLOCK explanations per topology (seed {SEED}).",
        "Rate each 1-5 for whether the cited rules actually justify the block.",
        "",
    ]
    for tag, rows in failures_by_tag.items():
        lines += [f"## `{tag}`", ""]
        if not rows:
            lines += ["_No blocks recorded._", ""]
            continue
        for i, row in enumerate(rng.sample(rows, min(n, len(rows))), 1):
            lines += [
                f"### {i}. record {row.get('record_idx')} — "
                f"true `{row.get('true_label')}`, predicted `{row.get('pred_raw')}` "
                f"(conf {row.get('confidence')})",
                f"- failure mode: `{row.get('failure_mode')}`",
                f"- context: `{row.get('context')}`",
                "",
                "```",
                str(row.get("explanation", "")).strip(),
                "```",
                "",
                "Rating (1-5): ",
                "",
            ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate shield runs into thesis tables")
    ap.add_argument("--tag", action="append", dest="tags",
                    default=None, help="Repeat per topology")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--out", default=OUT_DOC)
    ap.add_argument("--review-sheet", default=REVIEW_SHEET)
    ap.add_argument("--n-review", type=int, default=30)
    args = ap.parse_args()

    tags = args.tags or ["neurips2020", "case14", "wcci2022"]

    summaries, failures_by_tag = {}, {}
    for tag in tags:
        s = load_summary(tag, args.results_dir)
        if s is None:
            continue
        summaries[tag] = s
        failures_by_tag[tag] = load_failures(tag, args.results_dir)

    if not summaries:
        raise SystemExit("No shield summaries found. Run evaluation/eval_shield.py first.")

    doc = f"""# Component D — Shield Results

Generated by `evaluation/summarize_shield_results.py` from
`{args.results_dir}/shield_summary_*.json`. Companion to `gnn_final_results.md`.

## 1. Metric grid

{metric_table(summaries)}

`neurips2020` is the held-out test split (in-distribution baseline); `case14` and
`wcci2022` are unseen topologies with no retraining. The Lever A column is blank off
the 36-bus grid by design — the margin is val-tuned in-distribution and does not
transfer.

**Reading the two headline columns.** *False block rate* is the shield's
over-constraint: ground-truth `normal` frames it blocks anyway. *Missed-fault catch
rate* is its value: of the frames the GNN got wrong, the fraction the shield stopped.
The GNN-only vs GNN+shield delta is the difference between shipping every raw
prediction and shipping only the ones that pass.

## 2. Failure-mode distribution (blocked frames)

{failure_mode_table(summaries)}

## 3. Catch rate by failure mode

Of the frames the GNN predicted incorrectly, the share the shield blocked:

{catch_rate_table(summaries)}

## 4. Blocks by highest violated severity

{severity_table(failures_by_tag)}

## 5. Verification notes

- `NOT_EVALUABLE` should be ~0: every rule is pre-linted against the closed
  vocabulary. A nonzero count means rules reached the corpus without passing
  `lint_condition`.
- `ERROR` must be 0. Nonzero indicates malformed conditions or an evaluator bug.
- Rule compliance on PASS states is 100% by construction — a PASS means no
  evaluable rule was violated.
- {args.n_review} sampled explanations per topology are in
  `{os.path.basename(args.review_sheet)}` for manual 1-5 quality rating.
"""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    write_review_sheet(failures_by_tag, args.n_review, args.review_sheet)

    print(f"Wrote {args.out}")
    print(f"Wrote {args.review_sheet}")


if __name__ == "__main__":
    main()
