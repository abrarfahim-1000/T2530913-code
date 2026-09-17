"""
build.py — assemble the provenance knowledge graph from the stage-3 corpus.

Inputs, all produced by `extraction/validate.py`:

    validated_translated/*_confirmed.jsonl    one file per source document
    validated_translated/all_rules_deduped.jsonl   what the shield is served
    rules_35b/*_candidates.jsonl              stage-1 counts, for the funnel figure

Nothing here reads a grid, a topology, or a checkpoint. The graph is the same on
all three grids because the rules are.

⚠ The Rule -> ServedRule edge recomputes the dedup key `(entity, condition)` used
by `extraction.common.deduplicate_rules`. If that key ever changes, this build
raises rather than silently dropping a rule — see `_link_to_served`.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Iterable

import networkx as nx

from extraction.common import CONDITION_VOCABULARY
from kg.schema import (
    CLAUSE,
    CONTAINS,
    DEDUPED_INTO,
    DOCUMENT,
    EXPLANATORY_RULE,
    INSTANTIATES,
    PREDICATE,
    READS,
    RULE,
    SERVED_RULE,
    STATES,
    VARIABLE,
    clause_id,
    doc_id,
    explanatory_id,
    issuing_body,
    predicate_id,
    rule_id,
    served_id,
    variable_id,
)

CONFIRMED_DIR = "validated_translated"
DEDUPED_RULES = "validated_translated/all_rules_deduped.jsonl"
CANDIDATES_DIR = "rules_35b"

#: Shield v2's four-channel corpus. A SUPERSET of the served rules: 58 records
#: of which only the BLOCK channel can veto.
CHANNELS_CORPUS = "shield_corpus/all_rules_channels.jsonl"

#: Stage-2 output, which is where a channel-corpus rule's DOCUMENT lives - the
#: channel record carries a `source` clause but not the document it came from.
#: NOT `translated_rules/guarded/`: that is the 32-rule post-guard subset and
#: would leave most of the corpus uncited.
TRANSLATED_DIR = "translated_rules"


# ── PREDICATE NORMALIZATION ───────────────────────────────────────────────────
# Dedup keys on (entity, condition), so one physical check fragments whenever the
# extractor labelled it `Line` in one standard and `Facility` in another, or wrote
# `100` in one and `100.0` in another. Both are noise in LLM output, not physics.

class _CanonicalNumbers(ast.NodeTransformer):
    """1.0 -> 1, 100.0 -> 100. Leaves genuinely fractional literals alone."""

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        v = node.value
        if isinstance(v, bool):          # bool is a subclass of int — keep it
            return node
        if isinstance(v, float) and v.is_integer():
            return ast.Constant(value=int(v))
        return node


def normalize_predicate(condition: str) -> str:
    """Canonical form of a condition: same physics -> same string.

    Whitespace and numeric formatting are erased; variable names, operators and
    structure are not. A condition that will not parse is returned stripped, so a
    malformed rule gets its own predicate rather than crashing the build.
    """
    try:
        tree = ast.parse(condition, mode="eval")
    except SyntaxError:
        return condition.strip()
    tree = _CanonicalNumbers().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def used_variables(condition: str) -> list[str]:
    """Vocabulary variables the condition reads, in source order."""
    try:
        tree = ast.parse(condition, mode="eval")
    except SyntaxError:
        return []
    seen: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in seen:
            seen.append(node.id)
    return seen


_OP_WORDS = {
    ast.Gt: "exceeds",
    ast.GtE: "is at least",
    ast.Lt: "is below",
    ast.LtE: "is at most",
    ast.Eq: "equals",
    ast.NotEq: "differs from",
}


def _describe(name: str) -> str:
    """Vocabulary description with the parenthetical calibration note removed."""
    desc = CONDITION_VOCABULARY.get(name, name)
    return desc.split("(")[0].strip()


def _plain(node: ast.AST) -> str:
    if isinstance(node, ast.Expression):
        return _plain(node.body)
    if isinstance(node, ast.BoolOp):
        joiner = " and " if isinstance(node.op, ast.And) else " or "
        return joiner.join(_plain(v) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return f"not ({_plain(node.operand)})"
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left = _plain(node.left)
        word = _OP_WORDS.get(type(node.ops[0]), "compares to")
        return f"{left} {word} {_plain(node.comparators[0])}"
    if isinstance(node, ast.Name):
        return _describe(node.id)
    if isinstance(node, ast.Constant):
        return str(node.value)
    return ast.unparse(node)


def plain_english(condition: str) -> str:
    """One sentence a non-specialist can read, built from the vocabulary."""
    try:
        tree = ast.parse(condition, mode="eval")
    except SyntaxError:
        return condition.strip()
    return _plain(_CanonicalNumbers().visit(tree))


# ── LOADING ───────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _dedup_key(rule: dict) -> tuple[str, str]:
    """The key `extraction.common.deduplicate_rules` uses. Kept in lockstep."""
    return (rule.get("entity", ""), rule.get("condition", ""))


def _candidate_counts(candidates_dir: Path | None) -> dict[str, int]:
    if candidates_dir is None or not candidates_dir.is_dir():
        return {}
    counts: dict[str, int] = {}
    for f in sorted(candidates_dir.glob("*_candidates.jsonl")):
        counts[f.name.replace("_candidates.jsonl", "")] = len(_read_jsonl(f))
    return counts


# ── BUILD ─────────────────────────────────────────────────────────────────────

def build_knowledge_graph(
    confirmed_dir: str | Path = CONFIRMED_DIR,
    deduped_path: str | Path = DEDUPED_RULES,
    candidates_dir: str | Path | None = CANDIDATES_DIR,
) -> nx.DiGraph:
    """Document -> Clause -> Rule -> ServedRule -> Predicate -> Variable."""
    confirmed_dir = Path(confirmed_dir)
    deduped_path = Path(deduped_path)
    candidates_dir = Path(candidates_dir) if candidates_dir else None

    served = _read_jsonl(deduped_path)
    if not served:
        raise ValueError(f"No served rules in {deduped_path} — run stage 3 first.")

    kg = nx.DiGraph()

    # ── ServedRule + Predicate + Variable ─────────────────────────────────────
    served_by_key: dict[tuple[str, str], str] = {}
    predicate_ids: dict[str, str] = {}

    for rec in served:
        rid = rec["rule_id"]
        node = served_id(rid)
        kg.add_node(
            node,
            type=SERVED_RULE,
            rule_id=rid,
            label=rid,
            rule=dict(rec),          # verbatim — this is what the shield receives
        )
        served_by_key[_dedup_key(rec)] = node

        canonical = normalize_predicate(rec.get("condition", ""))
        if canonical not in predicate_ids:
            pid = predicate_id(len(predicate_ids))
            predicate_ids[canonical] = pid
            kg.add_node(
                pid,
                type=PREDICATE,
                canonical=canonical,
                plain_english=plain_english(canonical),
                role=str(rec.get("role", "CONSTRAINT")).upper(),
                variables=used_variables(canonical),
                label=canonical,
            )
            for var in used_variables(canonical):
                vnode = variable_id(var)
                if vnode not in kg:
                    kg.add_node(
                        vnode,
                        type=VARIABLE,
                        name=var,
                        description=CONDITION_VOCABULARY.get(var, ""),
                        label=var,
                    )
                kg.add_edge(pid, vnode, type=READS)
        kg.add_edge(node, predicate_ids[canonical], type=INSTANTIATES)

    # ── Document -> Clause -> Rule ────────────────────────────────────────────
    counts = _candidate_counts(candidates_dir)
    clause_ids: dict[tuple[str, str], str] = {}
    n_clauses = 0

    for cf in sorted(confirmed_dir.glob("*_confirmed.jsonl")):
        stem = cf.name.replace("_confirmed.jsonl", "")
        rules = _read_jsonl(cf)
        if not rules:
            continue

        dnode = doc_id(stem)
        kg.add_node(
            dnode,
            type=DOCUMENT,
            stem=stem,
            issuing_body=issuing_body(stem),
            n_candidates=counts.get(stem),
            label=_short(stem),
        )

        for rule in rules:
            text = str(rule.get("source", "")).strip() or "(no clause given)"
            ckey = (stem, text)
            if ckey not in clause_ids:
                clause_ids[ckey] = clause_id(n_clauses)
                kg.add_node(
                    clause_ids[ckey],
                    type=CLAUSE,
                    text=text,
                    document=stem,
                    label=_short(text, 44),
                )
                kg.add_edge(dnode, clause_ids[ckey], type=CONTAINS)
                n_clauses += 1

            rnode = rule_id(rule["rule_id"])
            kg.add_node(
                rnode,
                type=RULE,
                rule_id=rule["rule_id"],
                condition=rule.get("condition", ""),
                entity=rule.get("entity", ""),
                severity=rule.get("severity", ""),
                role=str(rule.get("role", "CONSTRAINT")).upper(),
                affirms=rule.get("affirms"),
                action=rule.get("action", ""),
                explanation=rule.get("explanation", ""),
                label=rule["rule_id"],
            )
            kg.add_edge(clause_ids[ckey], rnode, type=STATES)
            _link_to_served(kg, rnode, rule, served_by_key)

    return kg


def _link_to_served(kg: nx.DiGraph, rnode: str, rule: dict, served_by_key: dict) -> None:
    """Attach a validated rule to the deduped record that represents it.

    Raises rather than skipping: a rule with no served representative means the
    dedup key drifted, and silently dropping it would understate corroboration —
    which is the one number this graph exists to report.
    """
    key = _dedup_key(rule)
    target = served_by_key.get(key)
    if target is None:
        raise ValueError(
            f"{rule.get('rule_id')} has no deduped representative for key {key!r}. "
            f"The dedup key in extraction.common.deduplicate_rules has changed; "
            f"update _dedup_key() to match."
        )
    kg.add_edge(rnode, target, type=DEDUPED_INTO)


def _short(text: str, limit: int = 34) -> str:
    text = str(text).replace("_", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ── PERSISTENCE ───────────────────────────────────────────────────────────────
# JSON, not pickle. The first iteration persisted a .pkl and it vanished from disk
# without anything noticing; a 36-node graph costs nothing to store readably, and
# JSON survives a networkx version bump that would break an unpickle.

def save_kg(kg: nx.DiGraph, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(kg, edges="edges")
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_kg(path: str | Path) -> nx.DiGraph:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, directed=True, multigraph=False, edges="edges")


# ── STATS ─────────────────────────────────────────────────────────────────────

def nodes_of(kg: nx.DiGraph, node_type: str) -> list[str]:
    return [n for n, a in kg.nodes(data=True) if a.get("type") == node_type]


def kg_stats(kg: nx.DiGraph) -> dict:
    """Counts worth printing, plus the corroboration figure per predicate."""
    stats: dict = {
        "nodes": kg.number_of_nodes(),
        "edges": kg.number_of_edges(),
        "by_type": {t: len(nodes_of(kg, t)) for t in
                    (DOCUMENT, CLAUSE, RULE, SERVED_RULE, EXPLANATORY_RULE,
                     PREDICATE, VARIABLE)},
        "predicates": [],
    }
    for pid in sorted(nodes_of(kg, PREDICATE)):
        clauses, docs, bodies = _provenance_of_predicate(kg, pid)
        stats["predicates"].append({
            "canonical": kg.nodes[pid]["canonical"],
            "plain_english": kg.nodes[pid]["plain_english"],
            "role": kg.nodes[pid]["role"],
            # Split, not pooled: a predicate can now be instantiated by a
            # ServedRule (can veto) and by ExplanatoryRule records (cannot).
            # Pooling them would inflate the corroboration figure the graph
            # exists to report.
            "n_served_rules": len([
                n for n in kg.predecessors(pid)
                if kg.nodes[n].get("type") == SERVED_RULE]),
            "n_explanatory_rules": len([
                n for n in kg.predecessors(pid)
                if kg.nodes[n].get("type") == EXPLANATORY_RULE]),
            "n_clauses": len(clauses),
            "n_documents": len(docs),
            "n_bodies": len(bodies),
            "documents": sorted(docs),
        })
    return stats


def _provenance_of_predicate(kg: nx.DiGraph, pid: str) -> tuple[set, set, set]:
    clauses: set[str] = set()
    docs: set[str] = set()
    bodies: set[str] = set()
    for snode in kg.predecessors(pid):
        # ServedRule sits one hop further from its clauses than an
        # ExplanatoryRule does: Clause -> Rule -> ServedRule against
        # Clause -> ExplanatoryRule. Walk whichever shape the node has rather
        # than assuming the served one, or every explanatory predicate reports
        # zero corroboration.
        for cnode in _clauses_behind(kg, snode):
            clauses.add(cnode)
            stem = kg.nodes[cnode]["document"]
            docs.add(stem)
            body = kg.nodes.get(doc_id(stem), {}).get("issuing_body")
            if body:
                bodies.add(body)
    return clauses, docs, bodies


def _clauses_behind(kg: nx.DiGraph, node: str) -> Iterable[str]:
    """Every Clause node that states `node`, for either rule shape."""
    for parent in kg.predecessors(node):
        ptype = kg.nodes[parent].get("type")
        if ptype == CLAUSE:
            yield parent
        elif ptype == RULE:
            for cnode in kg.predecessors(parent):
                if kg.nodes[cnode].get("type") == CLAUSE:
                    yield cnode


def iter_served_rules(kg: nx.DiGraph) -> Iterable[dict]:
    """The rule dicts the shield is served, verbatim from all_rules_deduped.jsonl."""
    for node in sorted(nodes_of(kg, SERVED_RULE)):
        yield dict(kg.nodes[node]["rule"])


# -- EXPLANATORY LAYER (shield v2) ---------------------------------------------
# The served corpus is four rules. The four-channel corpus is 58, of which 54
# cannot veto anything: they warn, they affirm, or they are carried only so the
# corpus can be accounted for. Before this layer existed the graph could cite 11
# of the 58 and the other 47 raised KeyError, which meant the shield could render
# a warning it was unable to attribute. A warning without a citation is the exact
# thing the WARN channel was built to avoid (thesis_findings 21).
#
# STRICTLY ADDITIVE. A graph built without it is a subgraph of one built with it,
# no ServedRule node is touched, and `KgRuleProvider` ignores the layer unless it
# is constructed with `include_explanatory=True`. That is what keeps every
# measured number attached to the served four.

def _index_translated(translated_dir: Path) -> dict[tuple, str]:
    """(rule_id, source, condition) -> document stem, from stage-2 output.

    Keyed on the triple rather than on `rule_id` alone. Stage-1 ids are NOT
    globally unique - five documents restarted numbering, so R_001..R_172 are
    reused (thesis_findings 19.2). The triple is unique across stage 2 today and
    this asserts it, so a future collision fails loudly instead of quietly
    attributing a rule to the wrong standard.
    """
    index: dict[tuple, str] = {}
    collisions: list[tuple] = []
    for f in sorted(translated_dir.glob("*_translated.jsonl")):
        stem = f.name.replace("_translated.jsonl", "")
        for rec in _read_jsonl(f):
            rule = rec.get("rule") or rec
            key = (rule.get("rule_id"), rule.get("source"), rule.get("condition"))
            if index.setdefault(key, stem) != stem:
                collisions.append(key)
    if collisions:
        raise ValueError(
            f"{len(collisions)} rule(s) in {translated_dir} resolve to more than "
            f"one document on (rule_id, source, condition), e.g. {collisions[0]!r}. "
            f"Citing them would attribute a rule to the wrong standard."
        )
    return index


def add_explanatory_layer(
    kg: nx.DiGraph,
    channels_path: str | Path = CHANNELS_CORPUS,
    translated_dir: str | Path = TRANSLATED_DIR,
) -> nx.DiGraph:
    """Layer the four-channel corpus in as ExplanatoryRule nodes.

    Document -> Clause -> ExplanatoryRule -> Predicate -> Variable, reusing the
    Clause and Predicate nodes the served chain already built wherever the clause
    text and the normalized condition match. That reuse is the point: it is what
    makes a WARN rule and the BLOCK rule beside it visibly two instances of one
    physical check, on the occasions when they are.
    """
    channels_path = Path(channels_path)
    translated_dir = Path(translated_dir)
    records = _read_jsonl(channels_path)
    if not records:
        raise ValueError(
            f"No rules in {channels_path} - build the channel corpus first.")

    index = _index_translated(translated_dir)

    # Reuse the Clause and Predicate nodes already present rather than twinning.
    clause_ids = {(a["document"], a["text"]): n
                  for n, a in kg.nodes(data=True) if a.get("type") == CLAUSE}
    predicate_ids = {a["canonical"]: n
                     for n, a in kg.nodes(data=True) if a.get("type") == PREDICATE}
    n_clauses = len(clause_ids)
    unresolved: list[str] = []

    for rec in records:
        rid = rec.get("rule_id")
        key = (rid, rec.get("source"), rec.get("condition"))
        stem = index.get(key)
        if stem is None:
            unresolved.append(str(rid))
            continue

        dnode = doc_id(stem)
        if dnode not in kg:
            kg.add_node(dnode, type=DOCUMENT, stem=stem,
                        issuing_body=issuing_body(stem), n_candidates=None,
                        label=_short(stem))

        text = str(rec.get("source", "")).strip() or "(no clause given)"
        ckey = (stem, text)
        if ckey not in clause_ids:
            clause_ids[ckey] = clause_id(n_clauses)
            kg.add_node(clause_ids[ckey], type=CLAUSE, text=text, document=stem,
                        label=_short(text, 44))
            kg.add_edge(dnode, clause_ids[ckey], type=CONTAINS)
            n_clauses += 1

        node = explanatory_id(rid)
        # Structural attributes stay free of grid tags: the channel corpus carries
        # `fire_rate_<tag>` and a justification naming all three grids, and
        # test_graph_is_topology_agnostic exempts only the `rule` payload.
        kg.add_node(
            node,
            type=EXPLANATORY_RULE,
            rule_id=rid,
            channel=str(rec.get("channel", "")).upper(),
            label=rid,
            rule=dict(rec),          # verbatim, exactly as ServedRule carries it
        )
        kg.add_edge(clause_ids[ckey], node, type=STATES)

        canonical = normalize_predicate(rec.get("condition", ""))
        if canonical not in predicate_ids:
            pid = predicate_id(len(predicate_ids))
            predicate_ids[canonical] = pid
            kg.add_node(pid, type=PREDICATE, canonical=canonical,
                        plain_english=plain_english(canonical),
                        role=str(rec.get("role", "CONSTRAINT")).upper(),
                        variables=used_variables(canonical), label=canonical)
            for var in used_variables(canonical):
                vnode = variable_id(var)
                if vnode not in kg:
                    kg.add_node(vnode, type=VARIABLE, name=var,
                                description=CONDITION_VOCABULARY.get(var, ""),
                                label=var)
                kg.add_edge(pid, vnode, type=READS)
        kg.add_edge(node, predicate_ids[canonical], type=INSTANTIATES)

    # Fingerprint what was layered in. The graph carries each rule dict
    # VERBATIM, so it is a snapshot, not a view: restamp the corpus without
    # rebuilding and the shield is served the old records while every count
    # still looks right. That failure is silent - it cost a full three-grid
    # run that silenced nothing and reported success. `_assert_corpus_current`
    # turns it into an error at load time.
    kg.graph["explanatory_corpus"] = {
        "path": channels_path.as_posix(),
        "translated_dir": translated_dir.as_posix(),
        "sha256": corpus_fingerprint(channels_path),
        "n_records": len(records),
    }

    if unresolved:
        raise ValueError(
            f"{len(unresolved)} channel-corpus rule(s) have no stage-2 record in "
            f"{translated_dir} on (rule_id, source, condition): "
            f"{', '.join(unresolved[:8])}. Without one the document is unknown and "
            f"the rule cannot be cited - which is the whole point of the layer."
        )
    return kg


def iter_explanatory_rules(kg: nx.DiGraph) -> Iterable[dict]:
    """The explanatory layer's rule dicts, verbatim. Empty if never layered."""
    for node in sorted(nodes_of(kg, EXPLANATORY_RULE)):
        yield dict(kg.nodes[node]["rule"])


def corpus_fingerprint(path: str | Path) -> str:
    """SHA-256 of a corpus file, as stored in the graph and checked on load."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def explanatory_corpus_status(kg: nx.DiGraph) -> dict:
    """What the graph was built from, and whether that file still matches.

    `state` is one of:
        "absent"    no explanatory layer in this graph
        "missing"   the corpus file recorded at build time is no longer there,
                    so nothing can be checked - not an error, the repo may have
                    moved
        "current"   the file on disk is byte-for-byte what was layered in
        "stale"     the file changed since the build; the graph is serving old
                    records
    """
    meta = kg.graph.get("explanatory_corpus")
    if not meta:
        return {"state": "absent"}
    path = Path(meta["path"])
    if not path.exists():
        return {"state": "missing", **meta}
    actual = corpus_fingerprint(path)
    return {
        "state": "current" if actual == meta["sha256"] else "stale",
        "actual_sha256": actual,
        **meta,
    }


def _structure_key(rule: dict) -> tuple:
    """What the layer's SHAPE depends on: which clause, and which predicate.

    A change to any of these moves edges, so it needs a real rebuild. A change to
    anything else - a stamped coverage, a severity, a justification - only moves a
    payload, and that can be refreshed in place.
    """
    return (rule.get("rule_id"), rule.get("source"), rule.get("condition"))


def refresh_explanatory_payloads(kg: nx.DiGraph) -> dict:
    """Re-read the rule payloads from the corpus, in place.

    THE BUG THIS EXISTS FOR. The layer stores each rule dict verbatim, so it is a
    snapshot of a file that keeps being regenerated - stamping per-grid coverage
    onto the corpus left the graph serving pre-stamp records while every count
    still reconciled, and a three-grid run demoted nothing and reported success.
    Detecting that only converted a silent wrong answer into a chore.

    So: when the corpus has changed but its SHAPE has not - same rules, same
    clauses, same conditions - the payloads are simply refreshed and the caller
    gets current data with nothing to remember. When the shape HAS changed, edges
    would have to move, and that is a real rebuild this must not fake.

    Returns {"state": ..., "refreshed": n}. States are those of
    `explanatory_corpus_status`, plus "rebuild_required".
    """
    status = explanatory_corpus_status(kg)
    if status["state"] != "stale":
        return {**status, "refreshed": 0}

    corpus = {r.get("rule_id"): r for r in _read_jsonl(Path(status["path"]))}
    nodes = {kg.nodes[n]["rule_id"]: n for n in nodes_of(kg, EXPLANATORY_RULE)}

    if set(corpus) != set(nodes):
        return {**status, "state": "rebuild_required", "refreshed": 0,
                "reason": "the corpus gained or lost rules"}
    for rid, node in nodes.items():
        if _structure_key(corpus[rid]) != _structure_key(kg.nodes[node]["rule"]):
            return {**status, "state": "rebuild_required", "refreshed": 0,
                    "reason": f"{rid} changed its clause or its condition"}

    for rid, node in nodes.items():
        kg.nodes[node]["rule"] = dict(corpus[rid])
        kg.nodes[node]["channel"] = str(corpus[rid].get("channel", "")).upper()
    kg.graph["explanatory_corpus"]["sha256"] = status["actual_sha256"]
    return {**status, "state": "refreshed", "refreshed": len(nodes)}
