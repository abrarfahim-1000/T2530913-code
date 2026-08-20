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
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from kg.build import (  # noqa: E402
    CANDIDATES_DIR,
    CONFIRMED_DIR,
    DEDUPED_RULES,
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
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--figures", action="store_true",
                    help="also write kg_provenance and kg_corroboration (SVG + PNG)")
    args = ap.parse_args()

    kg = build_knowledge_graph(args.confirmed, args.deduped, args.candidates)
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
        print(f"      served rules {p['n_served_rules']} · clauses {p['n_clauses']} · "
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
