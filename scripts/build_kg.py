"""
build_kg.py — build Component C, the provenance knowledge graph.

    python scripts/build_kg.py                      # build + save + stats
    python scripts/build_kg.py --figures            # also write the thesis figures

Reads only stage-3 output. No grid, no checkpoint, no topology — the graph is the
same on all three grids because the rules are.

The shield does not need this to run: `evaluation/eval_shield_n1.py` defaults to
the JSONL corpus and takes `--rules-kg` to read from the graph instead. Those two
paths are pinned equal by `tests/test_kg.py`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# ── determinism ───────────────────────────────────────────────────────────────
# The GRAPH is reproducible byte-for-byte; the FIGURES were not. `nx.multipartite_
# layout` orders the nodes inside each layer through a hash-backed collection, and
# Python randomizes string hashing per process, so every rebuild reshuffled the six
# columns vertically while drawing the identical graph. That turns `git diff` on
# kg/*.svg into noise and hides a real change behind a cosmetic one.
#
# Hash randomization is fixed when the interpreter STARTS, so setting the variable
# here would be too late — the only fix is to re-exec once with it set. subprocess
# rather than os.execv: execv on Windows spawns and detaches instead of replacing,
# which loses the exit code.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    sys.exit(subprocess.run([sys.executable, *sys.argv], env=os.environ).returncode)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from kg.build import (  # noqa: E402
    CANDIDATES_DIR,
    CHANNELS_CORPUS,
    CONFIRMED_DIR,
    DEDUPED_RULES,
    TRANSLATED_DIR,
    add_explanatory_layer,
    build_knowledge_graph,
    kg_stats,
    save_kg,
)

DEFAULT_OUT = "kg/knowledge_graph.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the provenance knowledge graph")
    ap.add_argument("--confirmed", default=CONFIRMED_DIR,
                    help=f"folder of *_confirmed.jsonl (default: {CONFIRMED_DIR})")
    ap.add_argument("--deduped", default=DEDUPED_RULES,
                    help=f"the served corpus (default: {DEDUPED_RULES})")
    ap.add_argument("--candidates", default=CANDIDATES_DIR,
                    help=f"stage-1 folder, for candidate counts (default: {CANDIDATES_DIR})")
    ap.add_argument("--channels", nargs="?", const=CHANNELS_CORPUS, default=None,
                    help="Also layer the four-channel corpus in as ExplanatoryRule\n"
                         f"nodes so every rule that can SPEAK is citable, not just\n"
                         f"the four that can veto (default path: {CHANNELS_CORPUS}).\n"
                         "Strictly additive: the served set is untouched, so no\n"
                         "reported number can move."),
    ap.add_argument("--translated", default=TRANSLATED_DIR,
                    help="Stage-2 output, which is where a channel rule's DOCUMENT\n"
                         f"is recovered from (default: {TRANSLATED_DIR}). NOT the\n"
                         "guarded/ subfolder — that is a 32-rule subset."),
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--figures", action="store_true",
                    help="also write kg_provenance and kg_corroboration (SVG + PNG)")
    args = ap.parse_args()

    kg = build_knowledge_graph(args.confirmed, args.deduped, args.candidates)
    if args.channels:
        before = kg.number_of_nodes(), kg.number_of_edges()
        kg = add_explanatory_layer(kg, args.channels, args.translated)
        after = kg.number_of_nodes(), kg.number_of_edges()
        print(f"[kg] explanatory layer: +{after[0] - before[0]} nodes, "
              f"+{after[1] - before[1]} edges")
    path = save_kg(kg, args.out)
    stats = kg_stats(kg)

    print(f"\nKnowledge graph -> {path}")
    print(f"  {stats['nodes']} nodes, {stats['edges']} edges")
    for node_type, n in stats["by_type"].items():
        print(f"    {node_type:<12} {n:>3}")

    print("\n  Predicates, and what corroborates them:")
    for p in stats["predicates"]:
        print(f"\n    {p['canonical']}   [{p['role']}]")
        print(f"      {p['plain_english']}")
        extra = (f" (+{p['n_explanatory_rules']} explanatory)"
                 if p.get("n_explanatory_rules") else "")
        print(f"      served rules {p['n_served_rules']}{extra} · "
              f"clauses {p['n_clauses']} · "
              f"documents {p['n_documents']} · identified bodies {p['n_bodies']}")
        for d in p["documents"]:
            print(f"        - {d}")

    if args.figures:
        from kg.figures import write_all

        print()
        for f in write_all(kg, os.path.dirname(args.out) or "kg"):
            print(f"  figure -> {f}")

    print()


if __name__ == "__main__":
    main()
