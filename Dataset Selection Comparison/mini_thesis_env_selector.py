import json
import time
from collections import Counter

import numpy as np
import grid2op
from lightsim2grid import LightSimBackend

import ssl
# This restores the old behavior of not verifying certificates
ssl._create_default_https_context = ssl._create_unverified_context

ENV_NAMES = [
    "l2rpn_neurips_2020_track1_small",
    "l2rpn_wcci_2022",
]

# Keep this small. This is a screening pass, not full data collection.
N_EPISODES = 8
MAX_STEPS_PER_EPISODE = 288   # 1 day at 5-min resolution


def safe_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def safe_step(env, action):
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
    else:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    return obs, reward, done, info


def make_env(env_name):
    env = grid2op.make(env_name, backend=LightSimBackend())
    if hasattr(env, "deactivate_forecast"):
        env.deactivate_forecast()
    return env


def static_profile(env_name, env):
    return {
        "env_name": env_name,
        "n_sub": int(getattr(env, "n_sub", -1)),
        "n_line": int(getattr(env, "n_line", -1)),
        "n_gen": int(getattr(env, "n_gen", -1)),
        "n_load": int(getattr(env, "n_load", -1)),
        "n_storage": int(getattr(env, "n_storage", 0)),
    }


def classify_step(obs):
    """
    Coarse screening labels only.
    This is NOT your final thesis label logic.
    """

    rho = np.asarray(getattr(obs, "rho", []), dtype=float)
    line_status = np.asarray(getattr(obs, "line_status", []), dtype=bool)

    # Maintenance visibility is environment-specific.
    # If present, we treat "maintenance now" as any line with time_next_maintenance == 0.
    maint_now = False
    if hasattr(obs, "time_next_maintenance"):
        tnm = np.asarray(getattr(obs, "time_next_maintenance", []))
        if tnm.size > 0 and np.any(tnm == 0):
            maint_now = True

    overload = bool(rho.size > 0 and np.nanmax(rho) > 1.0)
    line_down = bool(line_status.size > 0 and np.any(~line_status))

    if overload:
        return "overload"
    if line_down:
        return "line_down"
    if maint_now:
        return "maintenance"
    return "normal"


def short_probe(env):
    counts = Counter()
    overload_steps = 0
    line_down_steps = 0
    maintenance_steps = 0
    total_steps = 0

    start = time.time()

    for _ in range(N_EPISODES):
        obs = safe_reset(env)
        done = False

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = env.action_space({})  # do nothing
            obs, reward, done, info = safe_step(env, action)

            total_steps += 1
            label = classify_step(obs)
            counts[label] += 1

            rho = np.asarray(getattr(obs, "rho", []), dtype=float)
            if rho.size > 0 and np.nanmax(rho) > 1.0:
                overload_steps += 1

            line_status = np.asarray(getattr(obs, "line_status", []), dtype=bool)
            if line_status.size > 0 and np.any(~line_status):
                line_down_steps += 1

            if hasattr(obs, "time_next_maintenance"):
                tnm = np.asarray(getattr(obs, "time_next_maintenance", []))
                if tnm.size > 0 and np.any(tnm == 0):
                    maintenance_steps += 1

            if done:
                break

    elapsed = time.time() - start
    steps_per_sec = total_steps / elapsed if elapsed > 0 else 0.0
    non_normal_steps = total_steps - counts["normal"]

    unique_non_normal = sum(1 for k in ["overload", "line_down", "maintenance"] if counts[k] > 0)

    return {
        "total_steps": total_steps,
        "steps_per_sec": steps_per_sec,
        "overload_rate": overload_steps / total_steps if total_steps else 0.0,
        "line_down_rate": line_down_steps / total_steps if total_steps else 0.0,
        "maintenance_rate": maintenance_steps / total_steps if total_steps else 0.0,
        "non_normal_rate": non_normal_steps / total_steps if total_steps else 0.0,
        "class_diversity": unique_non_normal / 3.0,  # scaled to [0,1]
        "screening_counts": dict(counts),
    }


def scope_fit_score(static_metrics):
    """
    Thesis-specific fit, not universal truth.

    Higher is better for YOUR current thesis:
    - storage is treated as a separate capability, not a selection penalty
    - the comparison is rubric-based rather than tied to a fixed hierarchy
    - we want a serious benchmark that is still manageable end-to-end
    """
    score = 1.0

    if static_metrics["n_sub"] > 50:
        score -= 0.20

    if static_metrics["n_line"] > 100:
        score -= 0.15

    return max(0.0, min(1.0, score))


def normalize_bigger_better(values):
    max_v = max(values)
    if max_v <= 0:
        return [0.0 for _ in values]
    return [v / max_v for v in values]


def normalize_smaller_better(values):
    min_v = min(values)
    return [min_v / v if v > 0 else 0.0 for v in values]


def compute_scores(results):
    steps_scores = normalize_bigger_better([r["probe"]["steps_per_sec"] for r in results])
    graph_size_scores = normalize_smaller_better(
        [r["static"]["n_sub"] + r["static"]["n_line"] for r in results]
    )

    for i, r in enumerate(results):
        sf = scope_fit_score(r["static"])
        ef = steps_scores[i]

        # Event richness: we want the environment to produce useful non-normal states quickly.
        # This is capped so that one weird rollout does not dominate the decision.
        non_normal_component = min(r["probe"]["non_normal_rate"] / 0.10, 1.0)
        overload_component = min(r["probe"]["overload_rate"] / 0.03, 1.0)
        diversity_component = r["probe"]["class_diversity"]

        erf = 0.45 * non_normal_component + 0.25 * overload_component + 0.30 * diversity_component
        erf = max(0.0, min(1.0, erf))

        gsf = graph_size_scores[i]

        final_score = (
            0.40 * sf +
            0.25 * ef +
            0.20 * erf +
            0.15 * gsf
        )

        r["scores"] = {
            "scope_fit_score": round(sf, 4),
            "efficiency_score": round(ef, 4),
            "event_richness_score": round(erf, 4),
            "graph_simplicity_score": round(gsf, 4),
            "final_thesis_score": round(final_score, 4),
            "integration_score": round(0.40 * sf + 0.35 * ef + 0.25 * gsf, 4),
            "scale_score": round(0.45 * erf + 0.30 * r["probe"]["class_diversity"] + 0.25 * ef, 4),
        }

    return results


def build_recommendation(results):
    integration_best = max(results, key=lambda x: x["scores"]["integration_score"])
    scale_best = max(results, key=lambda x: x["scores"]["scale_score"])

    rec = {
        "decision_mode": "phase_specific",
        "integration_phase": {
            "recommended_env": integration_best["static"]["env_name"],
            "score": round(integration_best["scores"]["integration_score"], 4),
        },
        "scale_phase": {
            "recommended_env": scale_best["static"]["env_name"],
            "score": round(scale_best["scores"]["scale_score"], 4),
        },
        "note": "No single environment dominates across all criteria; the recommendation depends on whether the goal is initial integration or scale validation.",
    }
    return rec


def main():
    all_results = []

    for env_name in ENV_NAMES:
        print(f"\n=== Running screening on: {env_name} ===")
        env = make_env(env_name)
        s = static_profile(env_name, env)
        p = short_probe(env)
        env.close()

        result = {
            "static": s,
            "probe": p,
        }
        all_results.append(result)

    all_results = compute_scores(all_results)
    recommendation = build_recommendation(all_results)

    output = {
        "results": all_results,
        "recommendation": recommendation,
    }

    with open("mini_thesis_env_selection_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== FINAL RESULTS ===")
    for r in all_results:
        print(f"\nEnvironment: {r['static']['env_name']}")
        print("Static:", r["static"])
        print("Probe:", r["probe"])
        print("Scores:", r["scores"])

    print("\n=== RECOMMENDATION ===")
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()