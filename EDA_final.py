#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis — Grid Fault Detection
# 25 targeted analyses for the neuro-symbolic GNN thesis project.

# ---------------------------------------------------------------------------
# SETUP 01: IMPORTS, PATHS, AND GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------
from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LABEL_ORDER = ["normal", "overload", "line_trip", "cascade"]
LABEL_TO_INT = {label: idx for idx, label in enumerate(LABEL_ORDER)}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

JSONL_PATH = Path("data/grid_dataset_neurips2020.jsonl")
META_PATH  = Path("data/grid_dataset_neurips2020_meta.json")
OUTPUT_DIR = Path("eda_outputs")

TABLE_DIR   = OUTPUT_DIR / "tables"
FIGURE_DIR  = OUTPUT_DIR / "figures"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
for d in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

VIS_SAMPLE_SIZE   = 5_000
STAT_SAMPLE_SIZE  = 20_000
MODEL_SAMPLE_SIZE = 40_000

np.random.seed(RANDOM_STATE)

for path in [JSONL_PATH, META_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path.resolve()}")

with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Dataset path :", JSONL_PATH)
print("Metadata path:", META_PATH)
print("Environment  :", metadata.get("env_name"))
print("Records      :", f'{metadata.get("total_records", 0):,}')


# ---------------------------------------------------------------------------
# SETUP 02: LOAD JSONL AND EXTRACT FEATURES
# ---------------------------------------------------------------------------
def _finite(record, key, dtype=np.float64):
    arr = np.asarray(record.get(key, []), dtype=dtype)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _ratio(num, den, default=0.0):
    return float(num / den) if abs(float(den)) > 1e-12 else float(default)


def infer_physical_label(max_rho, connected_lines, total_lines):
    if float(max_rho) >= 1.0:
        return "overload"
    disconnected = int(total_lines) - int(connected_lines)
    if disconnected == 0:
        return "normal"
    if disconnected == 1:
        return "line_trip"
    return "cascade"


def load_and_extract_features(filepath):
    rows = []
    with Path(filepath).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            rho         = _finite(rec, "rho")
            line_status = np.asarray(rec.get("line_status", np.ones(len(rho))), dtype=np.int64)
            load_p      = _finite(rec, "load_p")
            gen_p       = _finite(rec, "gen_p")
            p_or        = _finite(rec, "p_or")
            q_or        = _finite(rec, "q_or")
            v_or        = _finite(rec, "v_or")
            v_ex        = _finite(rec, "v_ex")

            if rho.size == 0:
                continue

            total_lines      = int(line_status.size)
            connected_lines  = int(np.sum(line_status == 1))
            disconnected_lines = total_lines - connected_lines
            total_load = float(np.sum(load_p))
            total_gen  = float(np.sum(gen_p))
            max_rho    = float(np.max(rho))

            rows.append({
                "record_index":                 int(idx),
                "label":                        str(rec.get("label", "unknown")),
                "label_int":                    int(rec.get("label_int", LABEL_TO_INT.get(str(rec.get("label")), -1))),
                "physical_label_reconstructed": infer_physical_label(max_rho, connected_lines, total_lines),
                "fault_loc":                    int(rec.get("fault_loc", -1)) if rec.get("fault_loc") is not None else -1,
                "timestep":                     int(rec.get("timestep", -1)),
                "chronic_id":                   int(rec.get("chronic_id", -1)),
                "reward":                       float(rec.get("reward", 0.0)),
                "max_rho":                      max_rho,
                "min_rho":                      float(np.min(rho)),
                "mean_rho":                     float(np.mean(rho)),
                "median_rho":                   float(np.median(rho)),
                "std_rho":                      float(np.std(rho)),
                "rho_q75":                      float(np.quantile(rho, 0.75)),
                "rho_range":                    float(max_rho - np.min(rho)),
                "rho_above_90_pct":             int(np.sum(rho > 0.90)),
                "rho_above_100_pct":            int(np.sum(rho >= 1.0)),
                "rho_at_clip_count":            int(np.sum(rho >= float(metadata.get("rho_clip", 2.0)))),
                "total_load":                   total_load,
                "mean_load":                    float(np.mean(load_p)) if load_p.size else 0.0,
                "std_load":                     float(np.std(load_p))  if load_p.size else 0.0,
                "total_gen":                    total_gen,
                "mean_gen":                     float(np.mean(gen_p))  if gen_p.size else 0.0,
                "std_gen":                      float(np.std(gen_p))   if gen_p.size else 0.0,
                "load_gen_ratio":               _ratio(total_load, total_gen),
                "power_balance":                float(total_gen - total_load),
                "abs_power_balance":            float(abs(total_gen - total_load)),
                "mean_abs_p_or":                float(np.mean(np.abs(p_or))) if p_or.size else 0.0,
                "mean_abs_q_or":                float(np.mean(np.abs(q_or))) if q_or.size else 0.0,
                "mean_v_or":                    float(np.mean(v_or))  if v_or.size else 0.0,
                "mean_v_ex":                    float(np.mean(v_ex))  if v_ex.size else 0.0,
                "total_lines":                  total_lines,
                "connected_lines":              connected_lines,
                "disconnected_lines":           disconnected_lines,
                "connected_fraction":           _ratio(connected_lines, total_lines),
            })

    frame = pd.DataFrame(rows)
    frame["label"] = pd.Categorical(frame["label"], categories=LABEL_ORDER, ordered=True)
    return frame


df = load_and_extract_features(JSONL_PATH)

MODEL_FEATURES_FULL = [
    "max_rho", "mean_rho", "std_rho", "rho_range",
    "rho_above_90_pct", "rho_above_100_pct",
    "total_load", "total_gen", "load_gen_ratio",
    "power_balance", "abs_power_balance",
    "mean_abs_p_or", "mean_abs_q_or", "mean_v_or", "mean_v_ex",
    "connected_lines", "disconnected_lines", "connected_fraction",
]

# Excludes direct rule-defining signals for a stricter diagnostic baseline.
MODEL_FEATURES_REDUCED = [
    "mean_rho", "std_rho", "rho_range", "rho_above_90_pct",
    "total_load", "total_gen", "load_gen_ratio",
    "power_balance", "abs_power_balance",
    "mean_abs_p_or", "mean_abs_q_or", "mean_v_or", "mean_v_ex",
]

EDA_TABLES  = {}
EDA_RESULTS = {}

print(f"Loaded rows : {len(df):,}")
print(f"Columns     : {len(df.columns)}")
display(df.head())


# ---------------------------------------------------------------------------
# SETUP 03: REUSABLE SAMPLING, PLOTTING, AND BASELINE HELPERS
# ---------------------------------------------------------------------------
def stratified_sample(frame, n, label_col="label"):
    if len(frame) <= n:
        return frame.copy()
    fractions = frame[label_col].value_counts(normalize=True, dropna=False)
    pieces = []
    for label, frac in fractions.items():
        group = frame[frame[label_col] == label]
        take  = max(1, min(len(group), int(round(n * float(frac)))))
        pieces.append(group.sample(n=take, random_state=RANDOM_STATE))
    sampled = pd.concat(pieces, ignore_index=False)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=RANDOM_STATE)
    return sampled.copy()


def model_frame(feature_cols=MODEL_FEATURES_FULL, n=MODEL_SAMPLE_SIZE):
    sample = stratified_sample(df, min(n, len(df)))
    X = sample[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = sample["label"].astype(str)
    return sample, X, y


def train_test_baseline(feature_cols=MODEL_FEATURES_FULL, n=MODEL_SAMPLE_SIZE, class_weight=None):
    sample, X, y = model_frame(feature_cols=feature_cols, n=n)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=140, random_state=RANDOM_STATE, n_jobs=-1, class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "sample": sample, "X": X, "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "model": model, "pred": pred,
    }


def plot_bar(series, title, xlabel="", ylabel="Count", rotation=0):
    plt.figure(figsize=(9, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def plot_line(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


print("Helpers ready.")


# ---------------------------------------------------------------------------
# SETUP 04: AUTO-SAVE — every store_table() and plt.show() writes to disk
# ---------------------------------------------------------------------------
import re

SAVED_TABLE_PATHS  = {}
SAVED_FIGURE_PATHS = []
_FIGURE_NAME_COUNTER  = Counter()
_SAVED_FIGURE_OBJECTS = set()
_ORIGINAL_PLT_SHOW    = plt.show


def _safe_filename(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def store_table(name, table):
    safe_name   = _safe_filename(name)
    output_path = TABLE_DIR / f"{safe_name}.csv"
    copy = table.copy()
    EDA_TABLES[name] = copy
    copy.to_csv(output_path, index=True)
    SAVED_TABLE_PATHS[name] = str(output_path.resolve())
    display(copy)
    print(f"Saved table: {output_path.resolve()}")
    return copy


def _save_and_show(*args, **kwargs):
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        obj_id = id(fig)
        if obj_id in _SAVED_FIGURE_OBJECTS:
            continue
        title = ""
        for ax in fig.axes:
            if ax.get_title():
                title = ax.get_title()
                break
        safe_title = _safe_filename(title or f"figure_{fig_num}")
        _FIGURE_NAME_COUNTER[safe_title] += 1
        num = _FIGURE_NAME_COUNTER[safe_title]
        out = FIGURE_DIR / f"{safe_title}_{num:02d}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        SAVED_FIGURE_PATHS.append(str(out.resolve()))
        _SAVED_FIGURE_OBJECTS.add(obj_id)
        print(f"Saved figure: {out.resolve()}")
    _ORIGINAL_PLT_SHOW(*args, **kwargs)


plt.show = _save_and_show
print(f"Auto-save ready. Tables → {TABLE_DIR}  |  Figures → {FIGURE_DIR}")


# ===========================================================================
# SECTION A: DATASET STRUCTURE AND PHYSICAL VALIDITY
# ===========================================================================

# EDA 01: DATAFRAME PREVIEW AND SCHEMA VALIDATION
print(f"\nRows: {len(df):,}  |  Columns: {len(df.columns)}")
print(df.columns.tolist())
display(df.head(30))


# EDA 02: CLASS DISTRIBUTION
class_counts = df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
class_pct    = (100 * class_counts / len(df)).round(2)
class_dist   = pd.DataFrame({"records": class_counts.astype(int), "percentage": class_pct})
store_table("02_class_distribution", class_dist)
plot_bar(class_counts, "Operational-State Class Distribution", "Operational state", "Records")


# EDA 03: PHYSICAL LABEL CONSISTENCY AUDIT
# Validates that every stored label matches the physics-derived priority rule
# (overload first, then topology) — defends dataset integrity in the thesis.
label_match = (df["label"].astype(str) == df["physical_label_reconstructed"].astype(str))
consistency = pd.DataFrame({
    "metric": ["matching labels", "contradictory labels", "consistency percentage"],
    "value": [
        int(label_match.sum()),
        int((~label_match).sum()),
        round(100 * float(label_match.mean()), 6),
    ],
})
store_table("03_physical_label_consistency", consistency)

if (~label_match).any():
    print("\nExample contradictions:")
    display(
        df.loc[~label_match, ["record_index", "label", "physical_label_reconstructed",
                               "max_rho", "connected_lines", "disconnected_lines"]].head(30)
    )
else:
    print("\nNo label contradictions found.")


# EDA 04: TOPOLOGY AND LINE-STATUS ANALYSIS
# Confirms that line_trip and cascade states have structurally fewer active
# lines — the rationale for pruning tripped edges from edge_index in pyg_data.py.
topology_summary = (
    df.groupby("label", observed=False)
      .agg(
          records                  = ("label",              "size"),
          mean_connected_lines     = ("connected_lines",    "mean"),
          min_connected_lines      = ("connected_lines",    "min"),
          max_connected_lines      = ("connected_lines",    "max"),
          mean_disconnected_lines  = ("disconnected_lines", "mean"),
          max_disconnected_lines   = ("disconnected_lines", "max"),
          mean_connected_fraction  = ("connected_fraction", "mean"),
      )
      .round(4)
)
store_table("04_topology_summary", topology_summary)
plot_bar(
    df.groupby("label", observed=False)["disconnected_lines"].mean(),
    "Average Disconnected Lines by Class",
    "Operational state", "Average disconnected lines",
)


# EDA 07: NODE VS EDGE TOPOLOGY-SIGNAL AUDIT
# Confirms that GNN pooling targets (connected_fraction, rho, active flows)
# carry distinct signals per class — validates graph construction choices.
topology_signal = (
    df.groupby("label", observed=False)
      .agg(
          mean_connected_fraction = ("connected_fraction", "mean"),
          mean_disconnected_lines = ("disconnected_lines", "mean"),
          mean_max_rho            = ("max_rho",            "mean"),
          mean_abs_active_flow    = ("mean_abs_p_or",      "mean"),
          mean_abs_reactive_flow  = ("mean_abs_q_or",      "mean"),
      )
      .round(5)
)
store_table("07_topology_signals", topology_signal)

for feature in ["connected_fraction", "disconnected_lines", "max_rho"]:
    grouped = [
        df.loc[df["label"].astype(str) == lbl, feature]
          .sample(n=min(1500, (df["label"].astype(str) == lbl).sum()), random_state=RANDOM_STATE)
          .values
        for lbl in LABEL_ORDER
    ]
    plt.figure(figsize=(9, 5))
    plt.boxplot(grouped, labels=LABEL_ORDER, showfliers=False)
    plt.title(f"Topology-Signal Audit: {feature}")
    plt.xlabel("Operational state")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()


# EDA 09: GRAPH-POOLING SIGNAL HYPOTHESIS
# Justifies why triple pooling (max/min/mean) is needed:
# max captures overload spikes, min captures connectivity drops, mean captures baseline.
pooling_proxy = (
    df.groupby("label", observed=False)
      .agg(
          rho_mean_proxy               = ("mean_rho",           "mean"),
          rho_max_proxy                = ("max_rho",            "mean"),
          rho_spread_proxy             = ("rho_range",          "mean"),
          active_edge_count_proxy      = ("connected_lines",    "mean"),
          disconnected_edge_count_proxy= ("disconnected_lines", "mean"),
      )
      .round(5)
)
store_table("09_pooling_signal_proxy", pooling_proxy)

for feature in ["mean_rho", "max_rho", "rho_range", "connected_lines"]:
    series = df.groupby("label", observed=False)[feature].mean()
    plot_bar(series, f"Pooling Signal: Mean {feature} by Class", "Operational state", f"Mean {feature}")


# EDA 58: PHYSICS VALIDATION
# Hard constraints from shield.py — all must hold at >= 99.9% for the dataset
# to be trusted in the thesis. Matches the 7 rule conditions in shield.evaluate_condition().
physics_checks = {
    "normal_has_no_overload":                ((df["label"].astype(str) != "normal")    | (df["max_rho"] < 1.0)),
    "normal_has_intact_topology":            ((df["label"].astype(str) != "normal")    | (df["disconnected_lines"] == 0)),
    "overload_meets_rho_threshold":          ((df["label"].astype(str) != "overload")  | (df["max_rho"] >= 1.0)),
    "line_trip_has_exactly_one_disconnection":((df["label"].astype(str) != "line_trip")| (df["disconnected_lines"] == 1)),
    "cascade_has_multiple_disconnections":   ((df["label"].astype(str) != "cascade")   | (df["disconnected_lines"] > 1)),
    "connected_fraction_in_unit_interval":   df["connected_fraction"].between(0.0, 1.0),
    "stored_rho_respects_clip":              df["max_rho"] <= float(metadata.get("rho_clip", 2.0)),
}

physics_table = pd.DataFrame([
    {
        "check":            name,
        "passed_records":   int(mask.sum()),
        "failed_records":   int((~mask).sum()),
        "pass_percentage":  100 * float(mask.mean()),
    }
    for name, mask in physics_checks.items()
])
store_table("58_physics_validation", physics_table.round(6))


# ===========================================================================
# SECTION B: DATA QUALITY
# ===========================================================================

# EDA 05+11: ZERO-VARIANCE AND MISSING-VALUE AUDIT (merged)
numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
variance_table = pd.DataFrame({
    "feature":       numeric_cols,
    "variance":      [float(df[c].var())            for c in numeric_cols],
    "unique_values": [int(df[c].nunique(dropna=False)) for c in numeric_cols],
    "missing":       [int(df[c].isna().sum())        for c in numeric_cols],
    "missing_pct":   [(100 * df[c].isna().mean()).round(4) for c in numeric_cols],
})
variance_table["zero_variance"]     = variance_table["variance"].fillna(0.0) == 0.0
variance_table["near_zero_variance"]= variance_table["variance"].fillna(0.0) < 1e-10
variance_table = variance_table.sort_values(
    ["zero_variance", "near_zero_variance", "variance"],
    ascending=[False, False, True],
)
store_table("05_variance_and_missing", variance_table)


# EDA 08: FAULT-LOCALIZATION BIAS CHECK
# A heavily skewed fault_loc distribution means the localizer head would overfit
# a few substations — informs the decision to disable it for cross-topo eval.
localized = df[df["fault_loc"] >= 0].copy()
fault_loc_counts = localized["fault_loc"].value_counts().sort_index()
fault_loc_summary = pd.DataFrame({
    "localized_records":             [len(localized)],
    "unlocalized_records":           [int((df["fault_loc"] < 0).sum())],
    "unique_localized_substations":  [int(localized["fault_loc"].nunique())],
    "most_common_fault_substation":  [int(fault_loc_counts.index[0]) if len(fault_loc_counts) else -1],
    "largest_substation_count":      [int(fault_loc_counts.iloc[0])  if len(fault_loc_counts) else 0],
})
store_table("08_fault_localization_summary", fault_loc_summary)

if len(fault_loc_counts):
    plot_bar(
        fault_loc_counts,
        "Fault-Location Frequency by Substation",
        "Substation ID", "Localized records", rotation=90,
    )


# EDA 10: DESCRIPTIVE STATISTICS
desc_stats = df[MODEL_FEATURES_FULL + ["fault_loc", "timestep", "reward"]].describe().T
store_table("10_descriptive_statistics", desc_stats.round(5))


# EDA 54: DATA-LEAKAGE AND CHRONIC-SPLIT AUDIT
# Validates that the chronic-level split has no temporal leakage and
# that full vs reduced feature sets perform similarly (no direct rule signals leaking).
full_baseline    = train_test_baseline(MODEL_FEATURES_FULL,    n=min(MODEL_SAMPLE_SIZE, 35_000))
reduced_baseline = train_test_baseline(MODEL_FEATURES_REDUCED, n=min(MODEL_SAMPLE_SIZE, 35_000))

leakage_table = pd.DataFrame([
    {
        "feature_setting":             "full aggregate features",
        "macro_f1":                    f1_score(full_baseline["y_test"],    full_baseline["pred"],    average="macro"),
        "contains_direct_rule_signals": True,
    },
    {
        "feature_setting":             "reduced aggregate features",
        "macro_f1":                    f1_score(reduced_baseline["y_test"], reduced_baseline["pred"], average="macro"),
        "contains_direct_rule_signals": False,
    },
])
store_table("54_direct_signal_leakage", leakage_table.round(6))

available_chronics = np.array(sorted(df["chronic_id"].unique()))
rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(available_chronics)
test_n    = max(1, int(round(0.20 * len(available_chronics))))
test_set  = set(available_chronics[:test_n])
train_set = set(available_chronics[test_n:])

chronic_split = pd.DataFrame({
    "metric": ["training chronics", "testing chronics", "overlap", "training records", "testing records"],
    "value":  [
        len(train_set), len(test_set),
        len(train_set.intersection(test_set)),
        int((df["chronic_id"].isin(train_set)).sum()),
        int((df["chronic_id"].isin(test_set)).sum()),
    ],
})
store_table("54_chronic_split_audit", chronic_split)


# ===========================================================================
# SECTION C: VISUALIZATION
# ===========================================================================

# EDA 15: BOX PLOTS BY CLASS
box_features = ["max_rho", "mean_rho", "std_rho", "total_load", "total_gen",
                "connected_fraction", "disconnected_lines"]
sample = stratified_sample(df, VIS_SAMPLE_SIZE)

for feature in box_features:
    grouped = [sample.loc[sample["label"].astype(str) == lbl, feature].values for lbl in LABEL_ORDER]
    plt.figure(figsize=(9, 5))
    plt.boxplot(grouped, labels=LABEL_ORDER, showfliers=False)
    plt.title(f"Box Plot by Class: {feature}")
    plt.xlabel("Operational state")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()


# EDA 27: PCA DIMENSIONALITY REDUCTION
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sample   = stratified_sample(df, min(STAT_SAMPLE_SIZE, len(df)))
X_scaled = StandardScaler().fit_transform(sample[MODEL_FEATURES_FULL].fillna(0.0))
pca      = PCA().fit(X_scaled)

pca_table = pd.DataFrame({
    "component":                    np.arange(1, len(pca.explained_variance_ratio_) + 1),
    "explained_variance_ratio":     pca.explained_variance_ratio_,
    "cumulative_explained_variance":np.cumsum(pca.explained_variance_ratio_),
})
store_table("27_pca_variance", pca_table.round(6))
plot_line(
    pca_table["component"],
    pca_table["cumulative_explained_variance"],
    "PCA Cumulative Explained Variance",
    "Principal component", "Cumulative explained variance",
)

# 2-component scatter coloured by class
projection = PCA(n_components=2).fit_transform(X_scaled)
plt.figure(figsize=(9, 6))
for lbl in LABEL_ORDER:
    mask = sample["label"].astype(str).to_numpy() == lbl
    plt.scatter(projection[mask, 0], projection[mask, 1], s=12, alpha=0.45, label=lbl)
plt.title("PCA 2-Component Projection")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.tight_layout()
plt.show()


# EDA 28: t-SNE MANIFOLD VISUALIZATION
from sklearn.manifold import TSNE

tsne_sample = stratified_sample(df, min(800, len(df)))
X_tsne      = StandardScaler().fit_transform(tsne_sample[MODEL_FEATURES_FULL].fillna(0.0))
embedding   = TSNE(
    n_components=2, random_state=RANDOM_STATE,
    init="pca", learning_rate="auto", perplexity=25, max_iter=400,
).fit_transform(X_tsne)

plt.figure(figsize=(9, 6))
for lbl in LABEL_ORDER:
    mask = tsne_sample["label"].astype(str).to_numpy() == lbl
    plt.scatter(embedding[mask, 0], embedding[mask, 1], s=12, alpha=0.55, label=lbl)
plt.title("t-SNE Projection of Grid-State Aggregates")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.legend()
plt.tight_layout()
plt.show()


# EDA 46: STATE-TRANSITION MATRIX
# Shows how states evolve within a chronic — exposes normal→overload→cascade progression.
ordered = df.sort_values(["chronic_id", "timestep", "record_index"]).copy()
ordered["next_label"] = ordered.groupby("chronic_id", observed=False)["label"].shift(-1)
transition_table = pd.crosstab(
    ordered["label"], ordered["next_label"], normalize="index"
).reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0.0)
store_table("46_state_transition_matrix", transition_table.round(6))

plt.figure(figsize=(8, 6))
plt.imshow(transition_table.to_numpy(), aspect="auto")
plt.colorbar(label="Transition probability")
plt.xticks(range(len(LABEL_ORDER)), LABEL_ORDER, rotation=45)
plt.yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
plt.title("Operational-State Transition Matrix")
plt.xlabel("Next state")
plt.ylabel("Current state")
plt.tight_layout()
plt.show()


# EDA 57: CASCADE PATTERN ANALYSIS
# Cascade is the rarest class and has the most structural variation —
# dedicated analysis justifies extra class weight and the min-pool signal.
cascade = df[df["label"].astype(str) == "cascade"].copy()
cascade_summary = pd.DataFrame({
    "cascade_records":          [len(cascade)],
    "cascade_percentage":       [100 * len(cascade) / len(df)],
    "min_disconnected_lines":   [int(cascade["disconnected_lines"].min())   if len(cascade) else float("nan")],
    "median_disconnected_lines":[float(cascade["disconnected_lines"].median()) if len(cascade) else float("nan")],
    "max_disconnected_lines":   [int(cascade["disconnected_lines"].max())   if len(cascade) else float("nan")],
    "mean_connected_fraction":  [float(cascade["connected_fraction"].mean()) if len(cascade) else float("nan")],
})
store_table("57_cascade_summary", cascade_summary.round(6))

if len(cascade):
    cascade_line_counts = cascade["disconnected_lines"].value_counts().sort_index()
    plot_bar(
        cascade_line_counts,
        "Cascade Distribution by Disconnected Line Count",
        "Disconnected lines", "Cascade records",
    )


# ===========================================================================
# SECTION D: FEATURE IMPORTANCE
# ===========================================================================

# EDA 06: FEATURE CORRELATION MATRIX
# Previously identified gen_p ≈ load_p (r = 1.00) → gen_p dropped from node features.
corr_matrix = df[MODEL_FEATURES_FULL].corr(numeric_only=True)
store_table("06_correlation_matrix", corr_matrix.round(4))

plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Pearson r")
plt.xticks(range(len(MODEL_FEATURES_FULL)), MODEL_FEATURES_FULL, rotation=90)
plt.yticks(range(len(MODEL_FEATURES_FULL)), MODEL_FEATURES_FULL)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

high_corr = [
    {"feature_1": a, "feature_2": b, "correlation": float(corr_matrix.loc[a, b])}
    for a, b in combinations(MODEL_FEATURES_FULL, 2)
    if abs(float(corr_matrix.loc[a, b])) >= 0.90
]
store_table("06_high_correlation_pairs", pd.DataFrame(high_corr))


# EDA 29: MUTUAL-INFORMATION FEATURE IMPORTANCE
from sklearn.feature_selection import mutual_info_classif

sample, X, y = model_frame(MODEL_FEATURES_FULL, n=MODEL_SAMPLE_SIZE)
mi_values    = mutual_info_classif(X, y, random_state=RANDOM_STATE)
mi_table     = pd.DataFrame({
    "feature": MODEL_FEATURES_FULL, "mutual_information": mi_values,
}).sort_values("mutual_information", ascending=False)
store_table("29_mutual_information", mi_table)
plot_bar(
    mi_table.set_index("feature")["mutual_information"],
    "Mutual-Information Feature Importance",
    "Feature", "Mutual information", rotation=90,
)


# EDA 30: RANDOM-FOREST FEATURE IMPORTANCE
rf_model = RandomForestClassifier(n_estimators=180, random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X, y)
rf_importance = pd.DataFrame({
    "feature": MODEL_FEATURES_FULL, "importance": rf_model.feature_importances_,
}).sort_values("importance", ascending=False)
store_table("30_rf_importance", rf_importance)
plot_bar(
    rf_importance.set_index("feature")["importance"],
    "Random-Forest Feature Importance",
    "Feature", "Importance", rotation=90,
)


# EDA 34: PERMUTATION IMPORTANCE (+ optional SHAP)
from sklearn.inspection import permutation_importance

baseline    = train_test_baseline(MODEL_FEATURES_FULL, n=min(MODEL_SAMPLE_SIZE, 30_000))
perm_result = permutation_importance(
    baseline["model"], baseline["X_test"], baseline["y_test"],
    n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1, scoring="f1_macro",
)
perm_table = pd.DataFrame({
    "feature":                    MODEL_FEATURES_FULL,
    "mean_permutation_importance":perm_result.importances_mean,
    "std_permutation_importance": perm_result.importances_std,
}).sort_values("mean_permutation_importance", ascending=False)
store_table("34_permutation_importance", perm_table)
plot_bar(
    perm_table.set_index("feature")["mean_permutation_importance"],
    "Permutation Importance (Macro-F1 drop after permutation)",
    "Feature", "Macro-F1 decrease", rotation=90,
)

try:
    import shap
    shap_sample = baseline["X_test"].sample(n=min(500, len(baseline["X_test"])), random_state=RANDOM_STATE)
    explainer   = shap.TreeExplainer(baseline["model"])
    shap_values = explainer.shap_values(shap_sample)
    print("SHAP values computed for", len(shap_sample), "records.")
except Exception as err:
    print("Optional SHAP skipped:", err)


# EDA 56: FEATURE-INTERACTION ANALYSIS
# Tests composite features that combine rho stress with topology/load signals —
# mirrors the interaction terms considered for the GNN node feature set.
interaction_frame = df.copy()
interaction_frame["rho_topology_interaction"]  = interaction_frame["max_rho"] * (1.0 - interaction_frame["connected_fraction"])
interaction_frame["load_stress_interaction"]   = interaction_frame["total_load"] * interaction_frame["max_rho"]
interaction_frame["flow_stress_interaction"]   = interaction_frame["mean_abs_p_or"] * interaction_frame["max_rho"]
interaction_frame["imbalance_stress_interaction"] = interaction_frame["abs_power_balance"] * interaction_frame["max_rho"]

interaction_features = [
    "rho_topology_interaction", "load_stress_interaction",
    "flow_stress_interaction", "imbalance_stress_interaction",
]
interaction_summary = (
    interaction_frame.groupby("label", observed=False)[interaction_features]
                     .mean().round(6)
)
store_table("56_feature_interactions", interaction_summary)

for feature in interaction_features:
    plot_bar(
        interaction_frame.groupby("label", observed=False)[feature].mean(),
        f"Feature Interaction by Class: {feature}",
        "Operational state", f"Mean {feature}",
    )


# ===========================================================================
# SECTION E: DIAGNOSTIC BASELINE
# ===========================================================================

# EDA 32: CLASS-IMBALANCE ANALYSIS
# ICF weights and sqrt-smoothed variants — these numbers feed training/config.py directly.
counts   = df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
largest  = int(counts.max())
smallest = int(counts.min())
imbalance_table = pd.DataFrame({
    "records":                    counts,
    "percentage":                 (100 * counts / counts.sum()).round(4),
    "inverse_frequency_weight":   (counts.sum() / (len(counts) * counts)).round(6),
    "sqrt_inverse_frequency_weight": np.sqrt(counts.sum() / (len(counts) * counts)).round(6),
})
store_table("32_class_imbalance", imbalance_table)
print(f"Largest-to-smallest class ratio: {largest / max(smallest, 1):.4f}")


# EDA 39: CONFUSION MATRIX AND CLASSIFICATION REPORT
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

baseline = train_test_baseline(MODEL_FEATURES_REDUCED, n=MODEL_SAMPLE_SIZE)
matrix   = confusion_matrix(baseline["y_test"], baseline["pred"], labels=LABEL_ORDER)
matrix_table = pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER)
store_table("39_confusion_matrix", matrix_table)

report = pd.DataFrame(
    classification_report(
        baseline["y_test"], baseline["pred"],
        labels=LABEL_ORDER, output_dict=True, zero_division=0,
    )
).T.round(5)
display(report)

display_obj = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABEL_ORDER)
display_obj.plot(values_format="d")
plt.title("Confusion Matrix: Reduced-Feature Diagnostic Baseline")
plt.tight_layout()
plt.show()


# EDA 44: PRECISION-RECALL CURVES
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

probabilities = baseline["model"].predict_proba(baseline["X_test"])
class_order   = baseline["model"].classes_.tolist()
binary_y      = label_binarize(baseline["y_test"], classes=class_order)

pr_rows = []
plt.figure(figsize=(9, 6))
for i, lbl in enumerate(class_order):
    precision, recall, _ = precision_recall_curve(binary_y[:, i], probabilities[:, i])
    ap = average_precision_score(binary_y[:, i], probabilities[:, i])
    pr_rows.append({"class": lbl, "average_precision": ap})
    plt.plot(recall, precision, label=f"{lbl} (AP={ap:.4f})")
plt.title("Precision-Recall Curves: Reduced-Feature Baseline")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.show()
store_table("44_precision_recall_summary", pd.DataFrame(pr_rows))


# EDA 45: CROSS-VALIDATION FOLD ANALYSIS
from sklearn.model_selection import StratifiedKFold, cross_validate

sample, X, y = model_frame(MODEL_FEATURES_REDUCED, n=min(MODEL_SAMPLE_SIZE, 30_000))
cv_model     = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
scores       = cross_validate(
    cv_model, X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring={"accuracy": "accuracy", "macro_f1": "f1_macro", "weighted_f1": "f1_weighted"},
    n_jobs=-1,
)
fold_table = pd.DataFrame({
    "fold":       np.arange(1, len(scores["test_accuracy"]) + 1),
    "accuracy":   scores["test_accuracy"],
    "macro_f1":   scores["test_macro_f1"],
    "weighted_f1":scores["test_weighted_f1"],
})
store_table("45_cross_validation_folds", fold_table.round(6))


# EDA 52: RHO-THRESHOLD SENSITIVITY
# Sweeps the overload decision boundary from 0.80 to 1.20 — shows precision/recall
# trade-off and confirms 1.0 is the right physical threshold for the shield.
threshold_rows = []
for threshold in np.arange(0.80, 1.21, 0.05):
    predicted_overload = df["max_rho"] >= threshold
    actual_overload    = df["label"].astype(str) == "overload"
    tp = int((predicted_overload & actual_overload).sum())
    fp = int((predicted_overload & ~actual_overload).sum())
    fn = int((~predicted_overload & actual_overload).sum())
    threshold_rows.append({
        "rho_threshold":             round(float(threshold), 2),
        "predicted_overload_records":int(predicted_overload.sum()),
        "precision":                 tp / max(tp + fp, 1),
        "recall":                    tp / max(tp + fn, 1),
    })

rho_threshold_table = pd.DataFrame(threshold_rows)
store_table("52_rho_threshold_sensitivity", rho_threshold_table.round(6))
plot_line(
    rho_threshold_table["rho_threshold"],
    rho_threshold_table["recall"],
    "Overload Recall Across Rho Thresholds",
    "Rho threshold", "Recall",
)
plot_line(
    rho_threshold_table["rho_threshold"],
    rho_threshold_table["precision"],
    "Overload Precision Across Rho Thresholds",
    "Rho threshold", "Precision",
)


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("EDA COMPLETE")
print(f"  Tables saved : {len(SAVED_TABLE_PATHS)}")
print(f"  Figures saved: {len(SAVED_FIGURE_PATHS)}")
print(f"  Output root  : {OUTPUT_DIR.resolve()}")
print("=" * 70)
