"""
Tests for the N-1 contingency-screening labeller in scripts/generate_dataset.py.

The property that matters most here is the **-1 sentinel**. `n1_violation[k]`
is -1 when the contingency was not evaluated (line already out, or the power
flow diverged). Those are MISSING labels, not secure ones — training must mask
them out. Silently treating -1 as 0 would teach the model that every already-
tripped line is safe to lose, which is exactly backwards.

Run: pytest tests/test_n1_labels.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from generate_dataset import N1_LABEL_MAP, label_n1


class _SimObs:
    def __init__(self, rho):
        self.rho = np.asarray(rho, dtype=float)


class _Obs:
    """Minimal stand-in for a Grid2Op observation.

    `post` maps line_id -> (sim_obs, done). A line absent from `post` simulates
    as a solver failure (sim_obs None), which must produce the -1 sentinel.
    """

    def __init__(self, line_status, post):
        self.line_status = np.asarray(line_status, dtype=bool)
        self._post = post
        self.simulated = []

    def simulate(self, act):
        self.simulated.append(act)
        entry = self._post.get(act, (None, False))
        return entry[0], 0.0, entry[1], {}


class _ActionSpace:
    def __call__(self, d):
        return d["set_line_status"][0][0]      # the line id, used as the key


class _Env:
    def __init__(self, n_line):
        self.n_line = n_line
        self.action_space = _ActionSpace()


def test_violation_when_post_contingency_rho_reaches_limit():
    obs = _Obs([True, True, True], {
        0: (_SimObs([0.4, 0.5, 0.6]), False),     # secure
        1: (_SimObs([0.4, 1.02, 0.6]), False),    # thermal violation
        2: (_SimObs([0.99, 0.5, 0.6]), False),    # just under the limit
    })
    viol, post = label_n1(obs, _Env(3))
    assert viol == [0, 1, 0]
    assert post == pytest.approx([0.6, 1.02, 0.99])


def test_game_over_counts_as_a_violation_not_a_missing_label():
    """A contingency that ends the episode is the most severe outcome there is."""
    obs = _Obs([True, True], {
        0: (_SimObs([0.1]), True),               # done -> violation
        1: (_SimObs([0.2]), False),
    })
    viol, _ = label_n1(obs, _Env(2))
    assert viol == [1, 0]


def test_already_disconnected_lines_are_not_evaluated():
    obs = _Obs([True, False, True], {
        0: (_SimObs([0.3]), False),
        2: (_SimObs([0.3]), False),
    })
    viol, post = label_n1(obs, _Env(3))
    assert viol[1] == -1, "an out-of-service line is a missing label, not 'secure'"
    assert post[1] == -1.0
    assert 1 not in obs.simulated, "must not waste a solve on a line already out"


def test_solver_failure_yields_the_missing_sentinel():
    obs = _Obs([True, True], {0: (_SimObs([0.3]), False)})   # line 1 -> None
    viol, post = label_n1(obs, _Env(2))
    assert viol == [0, -1]
    assert post[1] == -1.0


def test_simulate_exception_does_not_abort_the_frame():
    class _Boom(_Obs):
        def simulate(self, act):
            if act == 1:
                raise RuntimeError("backend blew up")
            return super().simulate(act)

    obs = _Boom([True, True, True], {
        0: (_SimObs([0.3]), False),
        2: (_SimObs([1.5]), False),
    })
    viol, _ = label_n1(obs, _Env(3))
    assert viol == [0, -1, 1], "one bad contingency must not lose the other labels"


def test_every_energized_line_gets_exactly_one_solve():
    obs = _Obs([True, True, False, True],
               {k: (_SimObs([0.5]), False) for k in (0, 1, 3)})
    label_n1(obs, _Env(4))
    assert sorted(obs.simulated) == [0, 1, 3]


def test_vectors_are_aligned_to_line_id_and_full_length():
    obs = _Obs([False] * 5, {})
    viol, post = label_n1(obs, _Env(5))
    assert len(viol) == 5 and len(post) == 5
    assert set(viol) == {-1}


def test_label_map_is_binary_secure_violation():
    assert N1_LABEL_MAP == {"secure": 0, "violation": 1}
    assert set(N1_LABEL_MAP.values()) == {0, 1}
