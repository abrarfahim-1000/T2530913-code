"""
Tests for evaluation/replicate_validation.py.

⚠️ THE RUNNER ITSELF IS UNEXECUTED. It needs CUDA + Ollama and was written on a
machine with neither. These tests cover everything that does NOT need a model:
the comparator logic against synthetic fixtures, the directory-isolation guard,
and the preflight refusals.

The fixtures are built to the three cases that matter:
  * identical runs                       -> stable verdicts, stable reasons
  * one verdict flips                    -> caught, and named
  * same verdicts, different reasons     -> caught SEPARATELY from the verdicts

That last one is the whole point of the script. §12.3 found stable verdicts
resting on fabricated reasoning; a comparator that blends the two into one
"agreement" score would have reported that run as clean.

Run: pytest tests/test_replicate_validation.py -v
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from evaluation.replicate_validation import (  # noqa: E402
    RECORDED_RUN,
    PreflightError,
    assert_isolated_output_dirs,
    check_candidates,
    compare_corpora,
    compare_runs,
    load_corpus,
    load_guarded_inputs,
    load_run,
    overall_verdict,
)

REPO = os.path.join(os.path.dirname(__file__), "..")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _rule(rule_id, condition="loading_pct > 100", entity="Line", role="CONSTRAINT"):
    return {
        "rule_id": rule_id, "source": "Std, Sec 1", "entity": entity,
        "condition": condition, "action": "ALERT", "severity": "high",
        "explanation": "e", "role": role, "affirms": None,
    }


def _make_run(root, name, confirmed=(), rejected=(), doc="doc_alpha"):
    """Write one run directory in validate.py's own output shape.

    Note the asymmetry being reproduced deliberately: *_confirmed.jsonl holds
    BARE rules (no verdict, no reason); *_rejected.jsonl holds {rule, verdict}.
    """
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{doc}_confirmed.jsonl").open("w", encoding="utf-8") as fh:
        for r in confirmed:
            fh.write(json.dumps(r) + "\n")
    with (d / f"{doc}_rejected.jsonl").open("w", encoding="utf-8") as fh:
        for r, reason in rejected:
            fh.write(json.dumps({
                "rule": r,
                "verdict": {"rule_id": r["rule_id"], "verdict": "REJECT",
                            "corrected_fields": None, "reason": reason},
            }) + "\n")
    with (d / "all_rules_deduped.jsonl").open("w", encoding="utf-8") as fh:
        for r in confirmed:
            fh.write(json.dumps(r) + "\n")
    return d


# ---------------------------------------------------------------------------
# 1. DIRECTORY ISOLATION — the trap the script exists to avoid
# ---------------------------------------------------------------------------

def test_two_repeats_sharing_a_directory_is_refused(tmp_path):
    """deduplicate_rules() merges every *_confirmed.jsonl in the out dir."""
    shared = tmp_path / "runs"
    with pytest.raises(PreflightError, match="share an output directory"):
        assert_isolated_output_dirs([shared, shared])


def test_a_nested_run_directory_is_refused(tmp_path):
    outer = tmp_path / "runs"
    inner = outer / "run_1"
    with pytest.raises(PreflightError, match="nested inside"):
        assert_isolated_output_dirs([inner, outer])


def test_a_directory_already_holding_confirmed_output_is_refused(tmp_path):
    """The blend can also come from a PREVIOUS run's leftovers, not just a shared dir."""
    d = tmp_path / "run_1"
    _make_run(tmp_path, "run_1", confirmed=[_rule("R_1")])
    with pytest.raises(PreflightError, match="already contains"):
        assert_isolated_output_dirs([d, tmp_path / "run_2"])


def test_distinct_empty_directories_pass(tmp_path):
    assert_isolated_output_dirs([tmp_path / "run_1", tmp_path / "run_2",
                                 tmp_path / "run_3"])


def test_the_guard_triggers_before_any_inference(tmp_path):
    """It must raise on paths alone — nothing created, nothing called."""
    a, b = tmp_path / "x", tmp_path / "x"
    with pytest.raises(PreflightError):
        assert_isolated_output_dirs([a, b])
    assert not a.exists(), "the guard created a directory before refusing"


# ---------------------------------------------------------------------------
# 2. COMPARATOR — identical runs
# ---------------------------------------------------------------------------

def test_identical_runs_are_stable_on_both_axes(tmp_path):
    conf = [_rule("R_858"), _rule("R_916")]
    rej = [(_rule("R_116", "voltage_pu_min >= 0.9"), "time-bound requirement")]
    dirs = [_make_run(tmp_path, f"run_{i}", conf, rej) for i in (1, 2, 3)]

    cmp = compare_runs([load_run(d) for d in dirs], [d.name for d in dirs])

    assert cmp["n_rules"] == 3
    assert cmp["verdict_stability"]["n_unstable"] == 0
    assert cmp["verdict_stability"]["rate"] == 1.0
    # Only the rejection carries a reason; the two confirmations cannot be compared.
    assert cmp["reason_stability"]["n_measurable"] == 1
    assert cmp["reason_stability"]["n_unmeasurable"] == 2
    assert cmp["reason_stability"]["n_stable_exact"] == 1


# ---------------------------------------------------------------------------
# 3. COMPARATOR — a verdict flips
# ---------------------------------------------------------------------------

def test_a_single_verdict_flip_is_caught_and_named(tmp_path):
    keep = _rule("R_858")
    flipper = _rule("R_916")
    d1 = _make_run(tmp_path, "run_1", confirmed=[keep, flipper])
    d2 = _make_run(tmp_path, "run_2", confirmed=[keep],
                   rejected=[(flipper, "on reflection, no")])
    d3 = _make_run(tmp_path, "run_3", confirmed=[keep, flipper])

    cmp = compare_runs([load_run(d) for d in (d1, d2, d3)], ["r1", "r2", "r3"])

    vs = cmp["verdict_stability"]
    assert vs["n_unstable"] == 1
    assert vs["n_stable"] == 1
    assert [u["rule_id"] for u in vs["unstable_rules"]] == ["R_916"]
    assert vs["unstable_rules"][0]["outcomes"] == {
        "r1": "CONFIRM_OR_CORRECT", "r2": "REJECT", "r3": "CONFIRM_OR_CORRECT"}


def test_a_rule_missing_entirely_from_one_run_counts_as_unstable(tmp_path):
    """ABSENT is not agreement. A dropped rule is a difference."""
    d1 = _make_run(tmp_path, "run_1", confirmed=[_rule("R_1"), _rule("R_2")])
    d2 = _make_run(tmp_path, "run_2", confirmed=[_rule("R_1")])

    cmp = compare_runs([load_run(d1), load_run(d2)], ["r1", "r2"])
    assert cmp["verdict_stability"]["n_unstable"] == 1
    assert cmp["verdict_stability"]["unstable_rules"][0]["outcomes"]["r2"] == "ABSENT"


# ---------------------------------------------------------------------------
# 4. COMPARATOR — same verdicts, different reasons. THE POINT.
# ---------------------------------------------------------------------------

def test_identical_verdicts_with_drifting_reasons_are_reported_separately(tmp_path):
    """§12.3's exact shape: stable verdicts, unstable justification.

    A comparator that blended the two axes would call this run clean.
    """
    r = _rule("R_1926", "voltage_pu_min < 0.95")
    d1 = _make_run(tmp_path, "run_1", rejected=[(r, "the opposite logical requirement")])
    d2 = _make_run(tmp_path, "run_2", rejected=[(r, "uses an invalid variable name")])
    d3 = _make_run(tmp_path, "run_3", rejected=[(r, "the source text does not state it")])

    cmp = compare_runs([load_run(d) for d in (d1, d2, d3)], ["r1", "r2", "r3"])

    assert cmp["verdict_stability"]["n_unstable"] == 0, "verdicts did not move"
    assert cmp["reason_stability"]["n_measurable"] == 1
    assert cmp["reason_stability"]["n_stable_exact"] == 0, "reason drift went unreported"
    assert cmp["reason_stability"]["drifting_rules"][0]["rule_id"] == "R_1926"
    assert len(cmp["reason_stability"]["drifting_rules"][0]["reasons"]) == 3


def test_whitespace_only_reason_differences_are_distinguished_from_real_ones(tmp_path):
    r = _rule("R_1")
    d1 = _make_run(tmp_path, "run_1", rejected=[(r, "the  source   says\nno")])
    d2 = _make_run(tmp_path, "run_2", rejected=[(r, "The source says no")])

    cmp = compare_runs([load_run(d1), load_run(d2)], ["r1", "r2"])
    p = cmp["per_rule"][0]
    assert p["reason_stable_exact"] is False
    assert p["reason_stable_normalised"] is True, \
        "a whitespace/case difference should not read as fabricated reasoning"


def test_reason_stability_is_null_not_true_when_unmeasurable(tmp_path):
    """Confirmed rules carry no reason. 'Unmeasurable' must never read as 'agreed'."""
    d1 = _make_run(tmp_path, "run_1", confirmed=[_rule("R_1")])
    d2 = _make_run(tmp_path, "run_2", confirmed=[_rule("R_1")])

    cmp = compare_runs([load_run(d1), load_run(d2)], ["r1", "r2"])
    p = cmp["per_rule"][0]
    assert p["reason_measurable"] is False
    assert p["reason_stable_exact"] is None
    assert cmp["reason_stability"]["rate_over_measurable"] is None


# ---------------------------------------------------------------------------
# 5. KEYING — rule_id is not globally unique
# ---------------------------------------------------------------------------

def test_the_same_rule_id_in_two_documents_stays_two_rules(tmp_path):
    d = tmp_path / "run_1"
    d.mkdir()
    for doc in ("doc_alpha", "doc_beta"):
        (d / f"{doc}_confirmed.jsonl").write_text(
            json.dumps(_rule("R_042")) + "\n", encoding="utf-8")
    loaded = load_run(d)
    assert len(loaded) == 2
    assert ("doc_alpha", "R_042") in loaded and ("doc_beta", "R_042") in loaded


# ---------------------------------------------------------------------------
# 6. CORPUS REPRODUCTION
# ---------------------------------------------------------------------------

def test_a_run_reproducing_the_reference_is_recognised(tmp_path):
    ref = tmp_path / "reference.jsonl"
    rules = [_rule("R_858"), _rule("R_1154", "voltage_pu_min >= 0.9",
                                   "PowerGeneratingModule", "AFFIRMATION")]
    ref.write_text("".join(json.dumps(r) + "\n" for r in rules), encoding="utf-8")
    dirs = [_make_run(tmp_path, f"run_{i}", confirmed=rules) for i in (1, 2)]

    got = compare_corpora(dirs, ref)
    assert got["all_runs_match_reference"] is True
    assert got["all_runs_identical_to_each_other"] is True
    assert got["reference_n_distinct"] == 2


def test_a_run_missing_a_rule_is_flagged_with_which_one(tmp_path):
    ref = tmp_path / "reference.jsonl"
    a, b = _rule("R_858"), _rule("R_916", "loading_pct > 105")
    ref.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n", encoding="utf-8")
    dirs = [_make_run(tmp_path, "run_1", confirmed=[a, b]),
            _make_run(tmp_path, "run_2", confirmed=[a])]

    got = compare_corpora(dirs, ref)
    assert got["all_runs_match_reference"] is False
    assert got["per_run"][1]["missing_from_run"] == [("Line", "loading_pct > 105",
                                                      "CONSTRAINT")]


def test_corpora_compare_on_the_predicate_not_the_rule_id(tmp_path):
    """dedup merges sources, so which id survives is an ordering artifact."""
    ref = tmp_path / "reference.jsonl"
    ref.write_text(json.dumps(_rule("R_858")) + "\n", encoding="utf-8")
    d = _make_run(tmp_path, "run_1", confirmed=[_rule("R_1443")])  # same predicate

    got = compare_corpora([d], ref)
    assert got["all_runs_match_reference"] is True


# ---------------------------------------------------------------------------
# 7. OVERALL VERDICT
# ---------------------------------------------------------------------------

def test_verdict_separates_a_reproduced_corpus_from_stable_verdicts(tmp_path):
    """A corpus can survive flips that did not reach it. Say that, don't round it up."""
    ref = tmp_path / "reference.jsonl"
    keep = _rule("R_858")
    ref.write_text(json.dumps(keep) + "\n", encoding="utf-8")
    flipper = _rule("R_916", "voltage_pu_min < 0.95")
    d1 = _make_run(tmp_path, "run_1", confirmed=[keep],
                   rejected=[(flipper, "no")])
    d2 = _make_run(tmp_path, "run_2", confirmed=[keep],
                   rejected=[(flipper, "no")])
    # run_2 has the same corpus; force a verdict flip on a third run
    d3 = _make_run(tmp_path, "run_3", confirmed=[keep, flipper])
    # but strip flipper from run_3's corpus so the corpus still matches
    (d3 / "all_rules_deduped.jsonl").write_text(json.dumps(keep) + "\n", encoding="utf-8")

    cmp = compare_runs([load_run(d) for d in (d1, d2, d3)], ["r1", "r2", "r3"])
    cor = compare_corpora([d1, d2, d3], ref)
    v = overall_verdict(cmp, cor)

    assert v["corpus_reproduced"] is True
    assert v["verdicts_stable"] is False
    assert "CORPUS REPRODUCIBLE BUT VERDICTS ARE NOT" in v["headline"]


def test_a_clean_replication_reads_as_reproducible(tmp_path):
    ref = tmp_path / "reference.jsonl"
    rules = [_rule("R_858")]
    ref.write_text(json.dumps(rules[0]) + "\n", encoding="utf-8")
    dirs = [_make_run(tmp_path, f"run_{i}", confirmed=rules) for i in (1, 2, 3)]

    v = overall_verdict(compare_runs([load_run(d) for d in dirs], ["a", "b", "c"]),
                        compare_corpora(dirs, ref))
    assert v["corpus_reproduced"] and v["verdicts_stable"]
    assert v["headline"].startswith("REPRODUCIBLE")
    assert "UNMEASURED" in v["reason_note"]


def test_the_verdict_always_carries_the_two_axes_caveat(tmp_path):
    ref = tmp_path / "r.jsonl"
    ref.write_text(json.dumps(_rule("R_1")) + "\n", encoding="utf-8")
    dirs = [_make_run(tmp_path, f"run_{i}", confirmed=[_rule("R_1")]) for i in (1, 2)]
    v = overall_verdict(compare_runs([load_run(d) for d in dirs], ["a", "b"]),
                        compare_corpora(dirs, ref))
    assert "SEPARATE" in v["caveat"]


# ---------------------------------------------------------------------------
# 8. PREFLIGHT
# ---------------------------------------------------------------------------

def test_an_empty_candidates_directory_is_refused(tmp_path):
    with pytest.raises(PreflightError, match="no \\*_translated.jsonl"):
        check_candidates(tmp_path)


def test_a_candidates_directory_with_files_but_no_rules_is_refused(tmp_path):
    (tmp_path / "doc_translated.jsonl").write_text("\n\n", encoding="utf-8")
    with pytest.raises(PreflightError, match="0 rules"):
        check_candidates(tmp_path)


def test_a_silently_ignored_model_override_is_refused(monkeypatch):
    """CLAUDE.md tells the operator to set $VALIDATOR_MODEL; common.py ignores it."""
    from evaluation.replicate_validation import check_model_tag
    monkeypatch.setenv("VALIDATOR_MODEL", "some-other-model:7b")
    with pytest.raises(PreflightError, match="silently ignored"):
        check_model_tag()


def test_the_pinned_tag_is_returned_when_no_override_is_set(monkeypatch):
    from evaluation.replicate_validation import check_model_tag
    monkeypatch.delenv("VALIDATOR_MODEL", raising=False)
    assert check_model_tag() == "nemotron-3-nano:30b"


# ---------------------------------------------------------------------------
# 9. Import-clean and --help work with no Ollama on the machine
# ---------------------------------------------------------------------------

def test_help_works_without_a_model_host():
    """The comparator must be usable and inspectable off the LLM host."""
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "evaluation", "replicate_validation.py"),
         "--help"],
        capture_output=True, text=True, cwd=REPO,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
    )
    assert proc.returncode == 0, proc.stderr
    assert "--repeats" in proc.stdout
    assert "--compare-only" in proc.stdout


def test_the_recorded_run_constants_match_the_summary_on_disk():
    """Guards the reconciliation baseline against a silent edit."""
    path = os.path.join(REPO, "validated_translated", "validation_run_summary.json")
    if not os.path.exists(path):
        pytest.skip("validated_translated/ absent")
    with open(path, encoding="utf-8") as fh:
        s = json.load(fh)
    assert RECORDED_RUN["confirmed"] == s["total_confirmed"]
    assert RECORDED_RUN["rejected"] == s["total_rejected"]
    assert RECORDED_RUN["flagged"] == s["total_flagged"]
    assert RECORDED_RUN["distinct_after_dedup"] == s["n_unique_rules_after_dedup"]
    assert RECORDED_RUN["prompt_variant"] == s["prompt_variant"]


def test_the_real_guarded_corpus_loads_as_32_rules():
    path = os.path.join(REPO, "translated_rules", "guarded")
    if not os.path.isdir(path):
        pytest.skip("guarded corpus absent")
    from pathlib import Path
    assert len(load_guarded_inputs(Path(path))) == 32


def test_the_served_corpus_loads_as_four_distinct_predicates():
    path = os.path.join(REPO, "validated_translated", "all_rules_deduped.jsonl")
    if not os.path.exists(path):
        pytest.skip("served corpus absent")
    from pathlib import Path
    assert len(load_corpus(Path(path))) == 4
