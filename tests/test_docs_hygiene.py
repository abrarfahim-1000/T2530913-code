"""
test_docs_hygiene.py — the retired-claims list (`issues.md` §C1), made executable.

WHAT THIS GUARDS, AND WHY IT EXISTS
-----------------------------------
Several claims in this project were true of an earlier revision and are now known
to be wrong. They are not typos: each was measured, published, and then overturned
by a later measurement, so each is *quotable* — a reader holding an intermediate
copy will find them, and a later editing pass can reintroduce one by copying a
paragraph forward. Until now the only thing standing between the documents and a
restatement was human memory.

This test scans the live documents and fails if a retired claim reappears.

WHAT IT DOES **NOT** DO
-----------------------
It checks for the ABSENCE of retired phrasings. It never asserts that the current
wording is present. The documents are still being edited; a test that pinned the
replacement text would break on every legitimate revision and would be deleted
within a week, which is worse than no test.

THE WITHDRAWAL-NOTICE PROBLEM
-----------------------------
Some of these phrases legitimately appear in the live documents *because they are
being retracted there*: `results_and_analysis.md` §5.6 quotes the retired sentences
verbatim in order to withdraw them, and `thesis_findings.md` §17.1 carries a
correction banner that quotes "never run" for the same reason. A guard that fired on
those would punish the documents for doing the honourable thing.

Two allowances, both deliberately narrow, both documented below:

  1. WITHDRAWAL MARKERS (the general mechanism). A hit is permitted when a
     withdrawal marker — "withdrawn", "superseded", "corrected", "must not be
     quoted", "no longer true", "an earlier version", ... — appears within
     `_WINDOW` lines of it. The marker list is explicit and the window is small, so
     a bare restatement in ordinary prose is never excused.

  2. PINNED LIVE OCCURRENCES (the exception list). A handful of retired claims are
     still stated live in `supplimentary_docs/`, which a separate documentation
     pass owns. Each is pinned by file, claim and a distinctive fragment of the
     line, with the reason written out. A pin excuses exactly that line: the same
     claim anywhere else — or on a line whose wording has drifted — still fails.
     Removing a pin once the documentation pass fixes the line is the intended
     end state; nothing here requires a pinned line to keep existing.

Run: pytest tests/test_docs_hygiene.py -v
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The live documents. `issues.md` itself is deliberately NOT scanned: its whole
# job is to enumerate the retired claims, so scanning it would be self-defeating.
SCANNED_FILES = (
    "supplimentary_docs/results_and_analysis.md",
    "supplimentary_docs/thesis_findings.md",
    "CLAUDE.md",
)

# How far from a hit a withdrawal marker may sit, in lines. Small on purpose: a
# retraction sits next to the thing it retracts. Widening this is how the guard
# stops guarding.
_WINDOW = 6


# ── NORMALISATION ─────────────────────────────────────────────────────────────
# The documents are Markdown, so the same sentence appears as `precision *rises*
# off-distribution`, `precision **rises** off-distribution` and `precision rises
# off-distribution`. Emphasis is stripped and whitespace collapsed before matching,
# so each pattern below can be written the way a human would say it.

def _normalise(line: str) -> str:
    return re.sub(r"\s+", " ", line.replace("*", "").replace("`", "").replace("_", " "))


# ── THE RETIRED CLAIMS ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetiredClaim:
    """One claim that must never be restated."""

    claim_id: str
    pattern: str          # matched case-insensitively against the normalised line
    retired: str          # what the document used to say
    replacement: str      # what a writer should say instead
    authority: str        # where the correction is recorded

    @property
    def regex(self) -> re.Pattern:
        return re.compile(self.pattern, re.IGNORECASE)


RETIRED_CLAIMS: tuple[RetiredClaim, ...] = (
    RetiredClaim(
        claim_id="greedy-vs-enumerated-ceiling",
        pattern=r"fitted ceiling does not beat",
        retired='"the fitted ceiling does not beat the four extracted rules"',
        replacement=("the GREEDY ceiling does not; the ENUMERATED ceiling DOES, by "
                     "0.04-0.05 F1 at roughly half the precision (0.566 vs 0.938 on "
                     "neurips2020, 0.744 vs 0.934 on wcci2022). Say 'greedy'."),
        authority="thesis_findings.md §30.5; results_and_analysis.md §5.2.11, §5.6",
    ),
    RetiredClaim(
        claim_id="reading-a-argued-not-demonstrated",
        pattern=r"argued rather than demonstrated",
        retired='"reading (a) is argued rather than demonstrated"',
        replacement=("reading (a) — the task is rule-poor — is DEMONSTRATED, by "
                     "exhaustive enumeration of the whole rule language, and "
                     "corroborated twice."),
        authority="thesis_findings.md §30; results_and_analysis.md §5.5.3, §5.6",
    ),
    RetiredClaim(
        claim_id="controls-never-run",
        pattern=r"never (been )?run|specified and not (run|executed)|never executed",
        retired='"the control that was never run" / "specified and not executed"',
        replacement=("all THREE controls ran on 2026-09-19: the greedy tree (§28), "
                     "the pre-registered expert arm (§29) and the exhaustive "
                     "enumeration (§30)."),
        authority="thesis_findings.md §17.1, §28, §29, §30; issues.md §C2",
    ),
    RetiredClaim(
        claim_id="precision-rises-off-distribution",
        pattern=r"precision rises off-?distribution",
        retired='"intervention precision RISES off-distribution"',
        replacement=("VOID — an averaging artefact of mixing voltage rules with "
                     "thermal rules. Precision is FLAT at 0.938 / 0.922 / 0.934. "
                     "What changes off-distribution is how OFTEN the gate can act."),
        authority="thesis_findings.md §13.2; results_and_analysis.md §5.2.6",
    ),
    RetiredClaim(
        claim_id="all-frames-strictly-mixed",
        pattern=r"100% of frames are (strictly )?mixed",
        retired='"100% of frames are strictly mixed"',
        replacement=("96-99% (98.80 / 98.48 / 96.43). The 100% figure belongs to a "
                     "DIFFERENT statistic: no frame anywhere is entirely secure."),
        authority="thesis_findings.md §9; CLAUDE.md 'Why no global pooling'",
    ),
    RetiredClaim(
        claim_id="model-f1-8872",
        pattern=r"0\.8872",
        retired='the model F1 "0.8872"',
        replacement=("0.8872 does not reproduce from the tracked checkpoint at any "
                     "batch size tested. Quote 0.8972 AND its eval batch size (64); "
                     "the same checkpoint scores 0.9255 at batch 512."),
        authority="gnn_n1_tightening.md §8; thesis_findings.md §14",
    ),
    RetiredClaim(
        claim_id="no-larger-rulebook",
        pattern=r"no larger rulebook (closes|lifts)|rulebook closes that gap",
        retired='"no larger rulebook closes that gap"',
        replacement=("the 84% / 69% figures are the SERVED rule's, not the rule "
                     "LANGUAGE's. The best rule in the language leaves about "
                     "77% / 62% unreachable."),
        authority="results_and_analysis.md §5.4.3, §5.6; thesis_findings.md §30",
    ),
    RetiredClaim(
        claim_id="six-convergence-procedures",
        pattern=r"six (unrelated|independent) procedures",
        retired='"six unrelated procedures" converged on one predicate',
        replacement="SEVEN — §30.6's exhaustive enumeration is the seventh.",
        authority="results_and_analysis.md §5.4.7",
    ),
    RetiredClaim(
        claim_id="two-controls",
        pattern=r"\btwo controls\b",
        retired='"the two controls that would settle it"',
        replacement="THREE controls, and all three have run (§28, §29, §30).",
        authority="thesis_findings.md §17.1, §24; results_and_analysis.md §5.2.11",
    ),
)


# ── ALLOWANCE 1: WITHDRAWAL MARKERS ───────────────────────────────────────────
# A hit is forgiven when one of these sits within _WINDOW lines of it. Each is a
# phrase a writer uses when RETRACTING something, not when asserting it. Matched
# case-insensitively as a substring of the normalised line.
WITHDRAWAL_MARKERS: tuple[str, ...] = (
    # explicit retraction
    "withdraw",                 # withdrawn / withdrawal
    "supersede",                # superseded / supersedes / superseding
    "corrected",
    "correction",
    "retired claim",
    "must not be quoted",
    "must not be restated",
    "no longer true",
    "no longer applies",
    "is void",
    "reading is void",
    "claim is void",
    "measured to be false",
    "does not reproduce",
    "not a result",
    "qualified",                # the §5.6 change table's status word
    # "an earlier X said ..." — the standard way this project quotes a dead claim
    "an earlier note",
    "an earlier version",
    "an earlier analysis",
    "an earlier revision",
    "earlier draft",
    "a second revision",
    "a third revision",
    "the paragraph that stood here",
    "used to read",
    "used to quote",
    "this read",
    "previously read",
    # "...and it HAS since run / been replaced" — the §17.1 and §5.2.11 banners
    "have run",
    "have been run",
    "has since run",
    "has since replaced",
    "replaces this instrument",
    "narrower than it first appears",
    # a spec stating what WOULD have to be said in a branch that did not occur
    "if neither is run",
)


# ── ALLOWANCE 2: PINNED LIVE OCCURRENCES ──────────────────────────────────────
# Known-live restatements in files this task may not edit. Each pin excuses ONE
# line, identified by a distinctive fragment, and carries its reason. When the
# documentation pass rewrites the line, delete the pin — nothing asserts a pinned
# line still exists, so a stale pin is inert rather than a false failure.

@dataclass(frozen=True)
class Pin:
    path: str
    claim_id: str
    line_fragment: str      # distinctive substring of the NORMALISED offending line
    reason: str = field(default="")


# EMPTY, and that is the intended end state. Both original pins were cleared by the
# 2026-09-20 documentation pass: `thesis_findings.md`'s structural-ceiling paragraph
# now quotes the retired sentence only inside a dated withdrawal notice (so the
# withdrawal-marker allowance covers it, and no pin is needed), and the §5.4.3
# heading in `results_and_analysis.md` was retitled "a bound that rule quantity does
# not lift". Add a pin here only for a retired claim that is knowingly left live in a
# file the current task may not edit, and delete it as soon as that line is fixed.
PINNED_LIVE_OCCURRENCES: tuple[Pin, ...] = ()


# ── THE SCAN ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Hit:
    path: str
    lineno: int
    claim: RetiredClaim
    raw_line: str

    def message(self) -> str:
        return (
            f"\n  {self.path}:{self.lineno}"
            f"\n    retired claim  : {self.claim.retired}"
            f"\n    matched        : {self.raw_line.strip()[:160]}"
            f"\n    say instead    : {self.claim.replacement}"
            f"\n    authority      : {self.claim.authority}"
            f"\n    if this line IS a withdrawal notice, put a marker within "
            f"{_WINDOW} lines of it (see WITHDRAWAL_MARKERS)."
        )


def _is_withdrawal_notice(normalised: list[str], lineno: int) -> bool:
    """True when a withdrawal marker sits within _WINDOW lines of `lineno` (1-based)."""
    lo = max(0, lineno - 1 - _WINDOW)
    hi = min(len(normalised), lineno + _WINDOW)
    window = " \n ".join(normalised[lo:hi]).lower()
    return any(marker in window for marker in WITHDRAWAL_MARKERS)


def _is_pinned(path: str, claim_id: str, normalised_line: str) -> bool:
    return any(
        pin.path == path and pin.claim_id == claim_id
        and pin.line_fragment.lower() in normalised_line.lower()
        for pin in PINNED_LIVE_OCCURRENCES
    )


def scan_text(path: str, text: str) -> list[Hit]:
    """Every unforgiven restatement in `text`. Pure, so the tests can feed it
    synthetic documents without touching the tree."""
    raw = text.splitlines()
    normalised = [_normalise(line) for line in raw]
    hits: list[Hit] = []
    for idx, norm in enumerate(normalised):
        lineno = idx + 1
        for claim in RETIRED_CLAIMS:
            if not claim.regex.search(norm):
                continue
            if _is_pinned(path, claim.claim_id, norm):
                continue
            if _is_withdrawal_notice(normalised, lineno):
                continue
            hits.append(Hit(path, lineno, claim, raw[idx]))
    return hits


def scan_file(rel_path: str) -> list[Hit]:
    return scan_text(rel_path, (ROOT / rel_path).read_text(encoding="utf-8"))


# ── TESTS ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rel_path", SCANNED_FILES)
def test_no_retired_claim_is_restated(rel_path):
    """THE test. A retired claim reappearing in a live document fails here."""
    path = ROOT / rel_path
    assert path.exists(), f"{rel_path} is missing — update SCANNED_FILES"
    hits = scan_file(rel_path)
    if hits:
        pytest.fail(
            f"{len(hits)} retired claim(s) restated in {rel_path}:"
            + "".join(h.message() for h in hits)
        )


# ── TESTS OF THE TEST ─────────────────────────────────────────────────────────
# A guard nobody has seen fail is a guard nobody should trust.

@pytest.mark.parametrize("claim", RETIRED_CLAIMS, ids=lambda c: c.claim_id)
def test_every_claim_fires_on_a_bare_restatement(claim):
    """Each pattern must catch its own retired sentence in ordinary prose — no
    withdrawal marker, no pin, nothing nearby to excuse it."""
    bare = {
        "greedy-vs-enumerated-ceiling":
            "The fitted ceiling does not beat the four extracted rules.",
        "reading-a-argued-not-demonstrated":
            "Reading (a) is argued rather than demonstrated.",
        "controls-never-run":
            "The controls were specified and not executed; they were never run.",
        "precision-rises-off-distribution":
            "Intervention precision *rises* off-distribution, from 0.514 to 0.86.",
        "all-frames-strictly-mixed":
            "100% of frames are strictly mixed, so no per-frame label works.",
        "model-f1-8872":
            "Held-out test F1 0.8872 on the neurips2020 split.",
        "no-larger-rulebook":
            "The bound is structural and no larger rulebook closes that gap.",
        "six-convergence-procedures":
            "Six unrelated procedures converged on one predicate.",
        "two-controls":
            "Section 24 specifies the two controls that would settle it.",
    }[claim.claim_id]
    hits = scan_text("scratch.md", "context line\n" + bare + "\ncontext line\n")
    assert [h.claim.claim_id for h in hits] == [claim.claim_id], (
        f"pattern for {claim.claim_id!r} did not fire on its own retired sentence")


@pytest.mark.parametrize("claim", RETIRED_CLAIMS, ids=lambda c: c.claim_id)
def test_a_withdrawal_notice_is_not_a_restatement(claim):
    """The same sentence, retracted, must pass. This is the allowlist working."""
    bare = {
        "greedy-vs-enumerated-ceiling": "the fitted ceiling does not beat the rules",
        "reading-a-argued-not-demonstrated": "reading (a) is argued rather than demonstrated",
        "controls-never-run": "the controls were never run",
        "precision-rises-off-distribution": "precision rises off-distribution",
        "all-frames-strictly-mixed": "100% of frames are strictly mixed",
        "model-f1-8872": "held-out test F1 0.8872",
        "no-larger-rulebook": "no larger rulebook closes that gap",
        "six-convergence-procedures": "six unrelated procedures, one predicate",
        "two-controls": "the two controls that would settle it",
    }[claim.claim_id]
    doc = (f'An earlier version of this document said: "{bare}".\n'
           "That sentence is withdrawn and must not be quoted.\n")
    assert scan_text("scratch.md", doc) == []


def test_the_marker_window_does_not_reach_across_a_whole_section():
    """A withdrawal notice must sit NEXT to what it retracts. A marker eight lines
    away is a different paragraph and must not launder a fresh restatement."""
    filler = "\n".join(f"unrelated prose line {i}" for i in range(_WINDOW + 2))
    doc = ("That claim is withdrawn and superseded.\n" + filler
           + "\nThe fitted ceiling does not beat the four extracted rules.\n")
    hits = scan_text("scratch.md", doc)
    assert [h.claim.claim_id for h in hits] == ["greedy-vs-enumerated-ceiling"]


def test_a_pin_excuses_exactly_one_line_and_not_the_claim_at_large(monkeypatch):
    """Pins are keyed on file + claim + line fragment. The same retired claim on a
    different line, or in a different file, still fails.

    PINNED_LIVE_OCCURRENCES is empty in the committed tree, so this exercises the
    mechanism through a synthetic pin rather than depending on a real one existing.
    That keeps the guarantee under test after the last real pin is retired."""
    pinned_line = "exact computation. No larger rulebook closes that gap; it is"
    synthetic = Pin(
        path="supplimentary_docs/thesis_findings.md",
        claim_id="no-larger-rulebook",
        line_fragment="No larger rulebook closes that gap; it is",
        reason="synthetic, for this test only",
    )
    monkeypatch.setitem(globals(), "PINNED_LIVE_OCCURRENCES", (synthetic,))
    assert scan_text(synthetic.path, pinned_line + "\n") == []
    # same claim, same file, different sentence -> NOT excused
    assert len(scan_text(synthetic.path, "and no larger rulebook lifts the bound\n")) == 1
    # same sentence, different file -> NOT excused
    assert len(scan_text("some/other/doc.md", pinned_line + "\n")) == 1


def test_the_pin_list_is_empty_so_no_retired_claim_is_knowingly_live():
    """A non-empty pin list is a debt marker. This test does not forbid pins — it
    makes adding one a deliberate act that shows up in the diff, with its reason."""
    assert PINNED_LIVE_OCCURRENCES == (), (
        "A retired claim is knowingly live in a document:\n  "
        + "\n  ".join(f"{p.path} [{p.claim_id}] - {p.reason}"
                      for p in PINNED_LIVE_OCCURRENCES))


def test_markdown_emphasis_cannot_smuggle_a_retired_claim_through():
    """`precision *rises* off-distribution` is the form the documents actually
    use, so matching must survive emphasis, backticks and doubled asterisks."""
    for variant in ("precision rises off-distribution",
                    "precision *rises* off-distribution",
                    "precision **rises** off-distribution",
                    "`precision rises off-distribution`",
                    "precision   rises    off-distribution"):
        hits = scan_text("scratch.md", f"The gate's override {variant} on both grids.\n")
        assert [h.claim.claim_id for h in hits] == ["precision-rises-off-distribution"], variant


def test_the_guard_actually_fails_when_a_claim_is_injected_into_a_real_document():
    """End-to-end on the live text, in memory: the tree is clean, and the SAME
    text with one retired sentence appended is not. This is the check that the
    green result above means something."""
    rel = "supplimentary_docs/results_and_analysis.md"
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert scan_text(rel, text) == []
    # Padded clear of the document's own tail: the artifacts table there carries
    # withdrawal markers, and a sentence appended right after it would be forgiven
    # by allowance 1 — correctly, but that is not what this test is measuring.
    pad = "\n".join("injected filler" for _ in range(_WINDOW + 2))
    injected = (text + "\n" + pad
                + "\nThe fitted ceiling does not beat the four extracted rules.\n"
                + pad + "\n")
    hits = scan_text(rel, injected)
    assert len(hits) == 1 and hits[0].claim.claim_id == "greedy-vs-enumerated-ceiling"
    assert "say instead" in hits[0].message()
    assert str(hits[0].lineno) in hits[0].message()


def test_every_claim_and_pin_is_documented():
    """A pattern with no stated replacement is a rule nobody can act on, and a pin
    with no stated reason is a silently disabled rule."""
    ids = [c.claim_id for c in RETIRED_CLAIMS]
    assert len(ids) == len(set(ids)), "duplicate claim_id"
    for claim in RETIRED_CLAIMS:
        assert claim.replacement.strip() and claim.authority.strip(), claim.claim_id
        re.compile(claim.pattern)      # the pattern must be a valid regex
    known = set(ids)
    for pin in PINNED_LIVE_OCCURRENCES:
        assert pin.claim_id in known, f"pin references unknown claim {pin.claim_id!r}"
        assert pin.path in SCANNED_FILES, f"pin on an unscanned file: {pin.path}"
        assert pin.reason.strip(), f"pin {pin.claim_id}/{pin.path} has no reason"
